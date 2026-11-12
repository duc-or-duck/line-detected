"""YAML-configured lane detection and departure warning pipeline."""

from __future__ import annotations

import collections
import copy
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from .baseline import detect_baseline
from .departure import (DepartureWarning, calculate_offset,
                       calculate_straight_lines_offset, compute_lateral_velocity_and_tlc)
from .kalman import LaneKalmanFilter
from .lane_fit import fit_support_overlaps, sliding_window_fit, validate_and_recover_lanes
from .preprocess import canny_edges, lane_mask, roi_mask
from .warp import (unwarp_lane_fit, unwarp_y_range, vehicle_x_in_bev,
                   warp_image)


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("configuration must be a YAML mapping")
    return config


class LanePipeline:
    def __init__(self, config: dict, data_source: str | None = None):
        self.config = copy.deepcopy(config)
        self.data_source = data_source or self.config.get("pipeline", {}).get("default_data_source")
        self.previous_fits = None
        self.previous_baseline_fits = None
        self.previous_display_fits = None
        self.previous_display_ranges = None
        self.last_meters_per_pixel = None
        self.frame_index = 0
        self.filters = None
        self.warning = DepartureWarning(self.config.get("departure", {}))
        # Baseline temporal smoothing & anti-jump
        k = int(self.config.get("preprocess", {}).get("smoothing_k", 5))
        self.baseline_history = collections.deque(maxlen=k)
        self.baseline_smoothed = None
        self.baseline_hold_count = 0
        self.baseline_hold_max = int(self.config.get("preprocess", {}).get("hold_max_frames", 8))
        self.baseline_jump_px = float(self.config.get("preprocess", {}).get("jump_px", 80))
        self.baseline_d_history = collections.deque(maxlen=max(k, 3))
        self.baseline_last_valid_xm = None
        self.cam_offset_px = float(self.config.get("cam_offset_px", 0))
        self.csv_records = []

    def process(self, frame: np.ndarray, method: str = "proposed",
                fps: float | None = None) -> dict:
        blocks: dict[str, float] = {}
        height, width = frame.shape[:2]
        pipeline = self.config.get("pipeline", {})
        preprocess_cfg = self.config.get("preprocess", {})
        roi_polygon = preprocess_cfg.get("roi_polygon_by_source", {}).get(
            self.data_source, preprocess_cfg.get("roi_polygon"))
        reference_size = preprocess_cfg.get("roi_reference_size_by_source", {}).get(
            self.data_source, preprocess_cfg.get("roi_reference_size"))
        if roi_polygon and reference_size:
            ref_width, ref_height = map(float, reference_size)
            roi_polygon = [[round(point[0] * width / ref_width),
                            round(point[1] * height / ref_height)] for point in roi_polygon]
        source_preprocess_cfg = preprocess_cfg | {"roi_polygon": roi_polygon,
                                                   "data_source": self.data_source}
        line_ranges = None
        matrix = None
        valid_left = valid_right = single_line = False
        lane_measurement_available = False
        offset_left = offset_right = None
        lost = False
        active_source = self.data_source
        if method == "baseline":
            # Condition auto-switch (SPEC: dark_thr -> night; low contrast -> rain)
            if self.config.get("condition_auto"):
                gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
                if roi_polygon:
                    mask_full = np.zeros(gray_full.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(mask_full, [np.asarray(roi_polygon, dtype=np.int32)], 255)
                    rvals = gray_full[mask_full > 0]
                    rmed = float(np.median(rvals)) if rvals.size else float(np.median(gray_full))
                    rstd = float(np.std(rvals)) if rvals.size else float(np.std(gray_full))
                else:
                    rmed, rstd = float(np.median(gray_full)), float(np.std(gray_full))
                dark_thr = float(self.config.get("condition_auto", {}).get("dark_threshold", 70))
                cont_thr = float(self.config.get("condition_auto", {}).get("contrast_threshold", 45))
                if rmed < dark_thr:
                    active_source = "night"
                elif rstd < cont_thr:
                    active_source = "rain"
                else:
                    active_source = self.data_source or "day"
                if active_source != self.data_source and preprocess_cfg.get("roi_polygon_by_source", {}).get(active_source):
                    roi_polygon = preprocess_cfg["roi_polygon_by_source"][active_source]
                    if reference_size:
                        ref_w, ref_h = map(float, reference_size)
                        roi_polygon = [[round(p[0] * width / ref_w), round(p[1] * height / ref_h)] for p in roi_polygon]
                    source_preprocess_cfg["roi_polygon"] = roi_polygon
                source_preprocess_cfg["data_source"] = active_source

            start = time.perf_counter()
            found = detect_baseline(frame, source_preprocess_cfg)
            blocks["preprocess_detection"] = (time.perf_counter() - start) * 1000
            binary = found["edges"]
            line_ranges = found.get("line_ranges")
            raw_left, raw_right = found["left"], found["right"]

            use_anti_jump = preprocess_cfg.get("smoothing_k") is not None
            if use_anti_jump:
                raw_valid = raw_left is not None and raw_right is not None
                if raw_valid:
                    # (a) Validity check: opposite signs, left < right at 10 rows, width within [w_min, w_max]
                    sl_l, sl_r = raw_left[1], raw_right[1]
                    if sl_l >= 0 or sl_r <= 0:
                        raw_valid = False
                    else:
                        y_test = np.linspace(0.6, 1.0, 10)
                        xl_test = np.polyval(raw_left, y_test)
                        xr_test = np.polyval(raw_right, y_test)
                        if np.any(xl_test >= xr_test):
                            raw_valid = False
                        else:
                            bot_w = (xr_test[-1] - xl_test[-1]) * (width - 1)
                            w_min = float(preprocess_cfg.get("validity_width_min_frac", 0.20)) * width
                            w_max = float(preprocess_cfg.get("validity_width_max_frac", 0.90)) * width
                            if bot_w < w_min or bot_w > w_max:
                                raw_valid = False

                    # (b) Jump gate: reject if bottom x differs from smoothed previous by > jump_px
                    if raw_valid and self.baseline_smoothed is not None:
                        prev_l, prev_r = self.baseline_smoothed
                        curr_bot_l = np.polyval(raw_left, 1.0) * (width - 1)
                        curr_bot_r = np.polyval(raw_right, 1.0) * (width - 1)
                        prev_bot_l = np.polyval(prev_l, 1.0) * (width - 1)
                        prev_bot_r = np.polyval(prev_r, 1.0) * (width - 1)
                        if (abs(curr_bot_l - prev_bot_l) > self.baseline_jump_px or
                                abs(curr_bot_r - prev_bot_r) > self.baseline_jump_px):
                            raw_valid = False

                if raw_valid:
                    # (c) Smoothing: median of last K accepted pairs
                    self.baseline_history.append((raw_left.copy(), raw_right.copy()))
                    sm_l = np.median([h[0] for h in self.baseline_history], axis=0)
                    sm_r = np.median([h[1] for h in self.baseline_history], axis=0)
                    self.baseline_smoothed = (sm_l, sm_r)
                    self.baseline_hold_count = 0
                    left, right = sm_l.copy(), sm_r.copy()
                    lane_measurement_available = valid_left = valid_right = True
                else:
                    # (d) Hold on reject: hold last accepted lines up to hold_max frames
                    if self.baseline_smoothed is not None and self.baseline_hold_count < self.baseline_hold_max:
                        self.baseline_hold_count += 1
                        left, right = self.baseline_smoothed[0].copy(), self.baseline_smoothed[1].copy()
                        lane_measurement_available = valid_left = valid_right = True
                    else:
                        left, right = None, None
                        lane_measurement_available = valid_left = valid_right = False
                        lost = True
                        self.baseline_hold_count = 0

                if lane_measurement_available:
                    self.previous_baseline_fits = (left.copy(), right.copy())
                    self.previous_display_fits = (left.copy(), right.copy())
                    self.previous_display_ranges = line_ranges
                    offset_left, offset_right = left, right
                else:
                    offset_left, offset_right = None, None
            else:
                valid_left, valid_right = found["left"] is not None, found["right"] is not None
                lane_measurement_available = valid_left and valid_right
                if lane_measurement_available:
                    left, right = found["left"], found["right"]
                    self.previous_baseline_fits = (left.copy(), right.copy())
                    self.previous_display_fits = (left.copy(), right.copy())
                    self.previous_display_ranges = line_ranges
                    offset_left, offset_right = left, right
                else:
                    left, right = None, None
                    line_ranges = None
                    offset_left, offset_right = (None, None)
            if pipeline.get("kalman", False):
                if lane_measurement_available:
                    left, right = self._filter_fits(left, right)
                else:
                    self._filter_fits(None, None)
        elif method == "proposed":
            fit_cfg = dict(self.config.get("fit", {}))
            fit_cfg.update(self.config.get("fit_by_source", {}).get(self.data_source, {}))
            start = time.perf_counter()
            if pipeline.get("color_threshold", False):
                colors = lane_mask(frame, preprocess_cfg)
                hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
                kernel_size = int(preprocess_cfg.get("sobel_x_kernel", 3))
                sobel = cv2.Sobel(hls[:, :, 1], cv2.CV_64F, 1, 0, ksize=kernel_size)
                magnitude = cv2.convertScaleAbs(sobel)
                threshold = preprocess_cfg.get("sobel_x_threshold")
                if threshold is None:
                    threshold, sobel_mask = cv2.threshold(
                        magnitude, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                else:
                    _, sobel_mask = cv2.threshold(magnitude, int(threshold), 255, cv2.THRESH_BINARY)
                binary = cv2.bitwise_or(colors, sobel_mask)
                close_kernel = preprocess_cfg.get("dash_close_kernel")
                if close_kernel is not None:
                    close_size = tuple(map(int, close_kernel))
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, close_size)
                    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                binary = cv2.Canny(gray, int(preprocess_cfg.get("canny_low", 50)),
                                   int(preprocess_cfg.get("canny_high", 150)))
            if pipeline.get("roi", False):
                binary = roi_mask(binary, roi_polygon)
            blocks["preprocess_detection"] = (time.perf_counter() - start) * 1000

            homography = self.config.get("homography", {}).get(self.data_source or "")
            start = time.perf_counter()
            if pipeline.get("perspective", False) and homography:
                warped, matrix = warp_image(binary, homography.get("source"), homography.get("destination"))
            else:
                warped = binary
            blocks["perspective"] = (time.perf_counter() - start) * 1000

            start = time.perf_counter()
            previous = self.previous_fits if pipeline.get("previous_fit_search", False) else None
            bev_left, bev_right, fit_diagnostics = sliding_window_fit(warped, fit_cfg, previous)
            bev_left, bev_right, validity = validate_and_recover_lanes(
                bev_left, bev_right, width, height, fit_cfg)
            valid_left, valid_right = validity["valid_left"], validity["valid_right"]
            single_line = validity["single_line"]
            lane_measurement_available = validity["valid"]
            minimum_windows = fit_cfg.get("minimum_common_windows")
            if (lane_measurement_available and not single_line and minimum_windows is not None and
                    not fit_support_overlaps(fit_diagnostics, fit_cfg.get("window_count", 1),
                                             minimum_windows)):
                bev_left = bev_right = None
                valid_left = valid_right = lane_measurement_available = False
            if not lane_measurement_available and pipeline.get("hough", False):
                hough_result = detect_baseline(frame, source_preprocess_cfg)
                if hough_result["left"] is not None and hough_result["right"] is not None:
                    hough_fit_cfg = fit_cfg
                    if roi_polygon:
                        roi_top_norm = min(point[1] for point in roi_polygon) / max(height - 1, 1)
                        hough_fit_cfg = fit_cfg | {"valid_y_top_norm": roi_top_norm}
                    seed_fits = (hough_result["left"], hough_result["right"])
                    validity = {"valid": False}
                    refine_margin = fit_cfg.get("refinement_search_margin")
                    if refine_margin is None:
                        refine_margin = fit_cfg.get("window_margin")
                    if refine_margin is not None:
                        refine_cfg = fit_cfg | {"search_margin": refine_margin}
                        curved_left, curved_right, curved_diagnostics = sliding_window_fit(
                            binary, refine_cfg, seed_fits)
                        curved_left, curved_right, curved_validity = validate_and_recover_lanes(
                            curved_left, curved_right, width, height, hough_fit_cfg)
                        curved_validity["valid"] = (curved_validity["valid"] and
                            fit_support_overlaps(curved_diagnostics, fit_cfg.get("window_count", 1),
                                                 minimum_windows or 0))
                        if curved_validity["valid"]:
                            bev_left, bev_right = curved_left, curved_right
                            fit_diagnostics = curved_diagnostics
                            validity = curved_validity
                    if not validity["valid"]:
                        bev_left, bev_right, validity = validate_and_recover_lanes(
                            seed_fits[0], seed_fits[1], width, height, hough_fit_cfg)
                        hough_ranges = hough_result.get("line_ranges", {})
                        fit_diagnostics = {
                            "y_min_left": (hough_ranges.get("left") or (None, None))[0],
                            "y_max_left": (hough_ranges.get("left") or (None, None))[1],
                            "y_min_right": (hough_ranges.get("right") or (None, None))[0],
                            "y_max_right": (hough_ranges.get("right") or (None, None))[1],
                        }
                    if validity["valid"]:
                        valid_left = valid_right = lane_measurement_available = True
                        single_line = False
            if lane_measurement_available:
                line_ranges = {}
                for side, fit in (("left", bev_left), ("right", bev_right)):
                    other = "right" if side == "left" else "left"
                    use_other = ((side == "left" and not valid_left) or
                                 (side == "right" and not valid_right))
                    range_side = other if single_line and use_other else side
                    low, high = (fit_diagnostics.get(f"y_min_{range_side}"),
                                 fit_diagnostics.get(f"y_max_{range_side}"))
                    if low is None or high is None:
                        low, high = fit_diagnostics.get(f"y_min_{other}"), fit_diagnostics.get(f"y_max_{other}")
                    margin = fit_cfg.get("fit_y_margin_norm") or 0.0
                    bounds = None if low is None else (max(0.0, low - margin), min(1.0, high + margin))
                    if matrix is None and roi_polygon and bounds is not None:
                        roi_bottom_norm = max(point[1] for point in roi_polygon) / max(height - 1, 1)
                        bounds = (bounds[0], max(bounds[1], roi_bottom_norm))
                    line_ranges[side] = unwarp_y_range(fit, bounds, matrix, height, width)
                offset_left, offset_right = bev_left, bev_right
                if not single_line:
                    self.previous_fits = (bev_left.copy(), bev_right.copy())
                if pipeline.get("kalman", False):
                    if single_line:
                        self._filter_fits(None, None)
                    else:
                        filtered_left, filtered_right = self._filter_fits(bev_left, bev_right)
                        bev_left = bev_left if filtered_left is None else filtered_left
                        bev_right = bev_right if filtered_right is None else filtered_right
                left = unwarp_lane_fit(bev_left, matrix, height, width)
                right = unwarp_lane_fit(bev_right, matrix, height, width)
                self.previous_display_fits = (left.copy(), right.copy())
                self.previous_display_ranges = line_ranges
            else:
                if pipeline.get("kalman", False):
                    self._filter_fits(None, None)
                left, right = self.previous_display_fits or (None, None)
                line_ranges = self.previous_display_ranges
                offset_left, offset_right = None, None
            blocks["lane_fit"] = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            blocks["kalman"] = (time.perf_counter() - start) * 1000
        else:
            raise ValueError("method must be 'baseline' or 'proposed'")

        if method == "baseline":
            offset_left, offset_right = (left, right)
        departure_start = time.perf_counter()
        departure_cfg = self.config.get("departure", {})
        lane_width_m = departure_cfg.get("lane_width_m")
        current_fps = fps or self.config.get("dataset_metadata", {}).get("dashcam_fps")
        meter_scale_override = self.last_meters_per_pixel if single_line else None
        vehicle_x = vehicle_x_in_bev(width, height, matrix)
        if method == "baseline" and self.cam_offset_px != 0:
            vehicle_x += self.cam_offset_px
        y_eval_norm = (min(bounds[1] for bounds in line_ranges.values()
                           if bounds is not None)
                       if method == "baseline" and line_ranges and
                       any(bounds is not None for bounds in line_ranges.values()) else 1.0)
        if method == "baseline" and preprocess_cfg.get("smoothing_k") is not None:
            y_eval_px = y_eval_norm * max(height - 1, 1)
            if left is not None and right is not None:
                left_line = self._fit_to_slope_intercept(left, height, width)
                right_line = self._fit_to_slope_intercept(right, height, width)
                offset = calculate_straight_lines_offset(
                    left_line, right_line, y_eval_px, width,
                    float(lane_width_m) if lane_width_m is not None else None,
                    cam_offset_px=self.cam_offset_px,
                    last_meters_per_pixel=self.baseline_last_valid_xm)
            else:
                offset = calculate_straight_lines_offset(
                    None, None, y_eval_px, width,
                    float(lane_width_m) if lane_width_m is not None else None,
                    cam_offset_px=self.cam_offset_px,
                    last_meters_per_pixel=self.baseline_last_valid_xm)
            if lane_measurement_available and offset.get("meters_per_pixel") is not None:
                self.baseline_last_valid_xm = offset["meters_per_pixel"]
                self.last_meters_per_pixel = offset["meters_per_pixel"]
            offset.update({"left_x_px": offset.get("x_left"), "right_x_px": offset.get("x_right"),
                           "center_px": ((offset.get("x_left") + offset.get("x_right")) / 2.0
                                          if offset.get("x_left") is not None and offset.get("x_right") is not None else None)})
        else:
            offset = calculate_offset(offset_left, offset_right, width,
                                      float(lane_width_m) if lane_width_m is not None else None,
                                      vehicle_x_px=vehicle_x, y_eval=y_eval_norm,
                                      meters_per_pixel=meter_scale_override,
                                      reuse_scale=single_line)
            if lane_measurement_available and not single_line:
                self.last_meters_per_pixel = offset.get("meters_per_pixel")
                self.baseline_last_valid_xm = offset.get("meters_per_pixel")
            elif lost and self.baseline_last_valid_xm is not None:
                offset["meters_per_pixel"] = self.baseline_last_valid_xm

        baseline_tlc = None
        if method == "baseline":
            d_curr = offset.get("offset_m")
            if not lost:
                self.baseline_d_history.append(d_curr)
            velocity, baseline_tlc = compute_lateral_velocity_and_tlc(
                list(self.baseline_d_history), float(current_fps or 30.0),
                float(lane_width_m or 3.7))
        else:
            velocity = self._offset_velocity(offset_left, offset_right,
                                             offset.get("meters_per_pixel"), current_fps, width)

        if pipeline.get("warning", True):
            if lost:
                warning = {"warning": self.warning.active, "state": self.warning.state,
                           "tlc_s": None, "available": False}
            else:
                warning_offset = offset.get("offset_m") if lane_measurement_available else None
                warning = self.warning.update(warning_offset, velocity, lane_width_m)
                if baseline_tlc is not None and (warning.get("tlc_s") is None or not np.isfinite(warning.get("tlc_s", float("inf")))):
                    warning["tlc_s"] = baseline_tlc
        else:
            warning = {"warning": False, "state": "SAFE" if lane_measurement_available else self.warning.state,
                       "tlc_s": None, "available": lane_measurement_available}
        blocks["offset_warning"] = (time.perf_counter() - departure_start) * 1000
        output = frame.copy()
        self.frame_index += 1
        display_state = warning.get("state")
        if pipeline.get("warning", True) and not warning.get("available", False):
            display_state = "UNAVAILABLE"
        draw_vehicle_center = (width - 1) / 2.0
        draw_lane_center = None
        marker_y = height - 18
        if left is not None and right is not None:
            left_bounds = line_ranges.get("left") if line_ranges else None
            right_bounds = line_ranges.get("right") if line_ranges else None
            y_low = max(left_bounds[0], right_bounds[0]) if left_bounds and right_bounds else 0.0
            y_high = min(left_bounds[1], right_bounds[1]) if left_bounds and right_bounds else 1.0
            draw_y_norm = y_high
            marker_rows = np.linspace(y_low, y_high, 101)
            left_norm = np.polyval(left, marker_rows)
            right_norm = np.polyval(right, marker_rows)
            usable = ((left_norm >= 0) & (left_norm < right_norm) & (right_norm <= 1))
            if np.any(usable):
                draw_y_norm = float(marker_rows[np.flatnonzero(usable)[-1]])
                marker_y = int(draw_y_norm * (height - 1))
            left_at_y = float(np.polyval(left, draw_y_norm)) * (width - 1)
            right_at_y = float(np.polyval(right, draw_y_norm)) * (width - 1)
            if 0 <= left_at_y < right_at_y < width:
                draw_lane_center = (left_at_y + right_at_y) / 2.0
        self._draw(output, left, right, warning.get("warning"), roi_polygon, line_ranges,
                   draw_lane_center, draw_vehicle_center,
                   offset.get("offset_px") if lane_measurement_available else None,
                   display_state, clip_roi=True, marker_y=marker_y,
                   lost=lost, offset_m=offset.get("offset_m"), tlc_s=warning.get("tlc_s"))
        departure_record = {"frame": self.frame_index, "d_m": offset.get("offset_m"),
                            "TLC": warning.get("tlc_s"),
                            "warning": warning.get("warning") if warning.get("available") else None,
                            "valid_left": valid_left, "valid_right": valid_right,
                            "single_line": single_line}
        csv_record = {
            "frame": self.frame_index,
            "d_m": offset.get("offset_m"),
            "tlc_s": warning.get("tlc_s"),
            "warning": warning.get("warning") if not lost else None,
            "lost": lost,
            "x_left": offset.get("left_x_px"),
            "x_right": offset.get("right_x_px"),
        }
        self.csv_records.append(csv_record)
        return {"frame": output, "binary": binary, "left": left, "right": right,
                "offset_m": offset.get("offset_m"), "offset_px": offset.get("offset_px"),
                "x_vehicle": vehicle_x, "x_left": offset.get("left_x_px"),
                "x_right": offset.get("right_x_px"), "x_m": offset.get("meters_per_pixel"),
                "lane_center_px": draw_lane_center, "vehicle_center_px": draw_vehicle_center,
                "lane_measurement_available": lane_measurement_available,
                "valid_left": valid_left, "valid_right": valid_right,
                "single_line": single_line, "warning": warning,
                "departure_record": departure_record, "timings_ms": blocks, "lost": lost}

    @staticmethod
    def _fit_to_slope_intercept(fit, height: int, width: int) -> tuple[float, float]:
        """Convert normalized x(y) linear fit into image-space y = slope*x + intercept."""
        slope_xy = float(fit[1]) * max(width - 1, 1) / max(height - 1, 1)
        if abs(slope_xy) < 1e-12:
            return 0.0, 0.0
        intercept_xy = float(fit[2]) * max(width - 1, 1)
        return 1.0 / slope_xy, -intercept_xy / slope_xy
    def _filter_fits(self, left, right):
        cfg = self.config.get("kalman", {})
        needed = ("dt", "process_noise", "measurement_noise", "initial_covariance")
        if any(cfg.get(key) is None for key in needed):
            return left, right
        if self.filters is None:
            self.filters = tuple(LaneKalmanFilter(cfg["dt"], cfg["process_noise"],
                                                   cfg["measurement_noise"], cfg["initial_covariance"],
                                                   gating_threshold=cfg.get("gating_threshold", 11.34))
                                 for _ in range(2))
        outputs = []
        for filt, measurement in zip(self.filters, (left, right)):
            result = filt.step(measurement)
            outputs.append(result if measurement is not None or np.any(filt.x) else None)
        return outputs[0], outputs[1]

    def _offset_velocity(self, left, right, meters_per_pixel, fps, frame_width):
        if (self.filters is None or left is None or right is None or
                meters_per_pixel is None or fps is None):
            return None
        center_velocity = (np.sum(self.filters[0].x[3:, 0]) + np.sum(self.filters[1].x[3:, 0])) / 2.0
        pixels_per_frame = float(center_velocity) * max(frame_width - 1, 1)
        return -pixels_per_frame * float(meters_per_pixel) * float(fps)

    @staticmethod
    def _draw(frame: np.ndarray, left, right, warning, roi_polygon=None, line_ranges=None,
              lane_center_px=None, vehicle_center_px=None, offset_px=None, warning_state=None,
              clip_roi=True, marker_y=None, lost=False, offset_m=None, tlc_s=None) -> None:
        height, width = frame.shape[:2]
        original = frame.copy() if roi_polygon else None
        region = None
        if roi_polygon:
            region = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(region, [np.asarray(roi_polygon, dtype=np.int32)], 255)
        if left is not None and right is not None:
            y = np.linspace(0, 1, height)
            ys = np.arange(height)
            left_x = np.rint(np.polyval(left, y) * (width - 1)).astype(int)
            right_x = np.rint(np.polyval(right, y) * (width - 1)).astype(int)
            ordered = (left_x >= 0) & (left_x < right_x) & (right_x < width)
            for side, points, xs in (("left", np.stack([left_x, ys], axis=1), left_x),
                                     ("right", np.stack([right_x, ys], axis=1), right_x)):
                valid = ordered.copy()
                bounds = line_ranges.get(side) if line_ranges else None
                if bounds is not None:
                    valid &= (y >= bounds[0]) & (y <= bounds[1])
                if region is not None and clip_roi:
                    valid &= region[ys, np.clip(xs, 0, width - 1)] > 0
                indices = np.flatnonzero(valid)
                for run in np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1):
                    if len(run) >= 2:
                        cv2.polylines(frame, [points[run]], False, (0, 255, 0), 3)
        if region is not None and clip_roi:
            frame[region == 0] = original[region == 0]
        if lost:
            color = (128, 128, 128)
            label = "MẤT LÀN"
        elif warning is None or warning_state == "UNAVAILABLE":
            color = (0, 165, 255)
            label = "CHUA THAY DU 2 VACH" if offset_px is None else "CHUA CAU HINH CANH BAO"
        elif warning is True:
            color = (0, 0, 255) if warning_state == "DANGER" else (0, 100, 255)
            label = "CẢNH BÁO CHỆCH LÀN"
        else:
            color = (0, 200, 0)
            label = warning_state or "SAFE"
        cv2.putText(frame, label, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, .8, color, 2, cv2.LINE_AA)

        if warning is True and offset_px is not None:
            arrow_start_x = width // 2
            arrow_y = height - 50
            arrow_end_x = arrow_start_x + (120 if offset_px > 0 else -120)
            cv2.arrowedLine(frame, (arrow_start_x, arrow_y), (arrow_end_x, arrow_y),
                            (0, 0, 255), 4, tipLength=0.3)

        if offset_m is not None:
            d_text = f"d = {offset_m:+.2f} m ({offset_px:+.1f} px)" if offset_px is not None else f"d = {offset_m:+.2f} m"
            cv2.putText(frame, d_text, (24, 76), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2, cv2.LINE_AA)
        if tlc_s is not None and np.isfinite(float(tlc_s)):
            cv2.putText(frame, f"TLC = {float(tlc_s):.1f} s", (24, 106),
                        cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2, cv2.LINE_AA)

        if lane_center_px is not None and vehicle_center_px is not None:
            y = height - 18 if marker_y is None else int(marker_y)
            cv2.line(frame, (int(lane_center_px), y - 12), (int(lane_center_px), y + 8),
                     (255, 0, 0), 3)
            cv2.circle(frame, (int(vehicle_center_px), y), 6, (0, 0, 255), -1)
