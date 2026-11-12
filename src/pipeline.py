"""YAML-configured lane detection and departure warning pipeline."""

from __future__ import annotations

import copy
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml

from .baseline import detect_baseline, detect_hough
from .departure import DepartureWarning, calculate_offset
from .kalman import LaneKalmanFilter
from .lane_fit import (constrain_lane_pair, fit_support_overlaps,
                       output_geometry_is_valid, sliding_window_fit,
                       validate_and_recover_lanes)
from .preprocess import lane_mask, roi_mask
from .warp import (unwarp_lane_fit, unwarp_y_range, vehicle_x_in_bev,
                   warp_image)


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("configuration must be a YAML mapping")
    return config


class LanePipeline:
    def __init__(self, config: dict, data_source: str | None = None,
                 video_id: str | None = None):
        self.config = copy.deepcopy(config)
        self.data_source = data_source or self.config.get("pipeline", {}).get("default_data_source")
        self.video_id = Path(video_id).name if video_id else None
        self.calibration_status = "source"
        self.previous_fits = None
        self.previous_display_fits = None
        self.previous_display_ranges = None
        self.last_meters_per_pixel = None
        self.frame_index = 0
        self.filters = None
        self.missed_frames = 0
        self.warning = DepartureWarning(self.config.get("departure", {}))
        self.csv_records = []

    def _homography_for_frame(self) -> dict | None:
        """Prefer the explicitly calibrated matrix for this video."""
        if self.video_id:
            path = Path(__file__).resolve().parents[1] / "configs" / "homography_by_video.yaml"
            if path.is_file():
                with path.open("r", encoding="utf-8") as stream:
                    by_video = yaml.safe_load(stream) or {}
                calibrated = by_video.get(self.video_id)
                if (calibrated and calibrated.get("accepted") is True and
                        calibrated.get("source") and calibrated.get("destination")):
                    self.calibration_status = "video"
                    return calibrated
            self.calibration_status = "uncalibrated"
            return None
        source = self.config.get("homography", {}).get(self.data_source or "")
        self.calibration_status = "source" if source and source.get("source") else "uncalibrated"
        return source

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
        detection_mode = "sliding"
        matrix = None
        valid_left = valid_right = single_line = False
        lane_measurement_available = False
        offset_left = offset_right = None
        bev_left = bev_right = None
        lost = False
        if method == "baseline":
            start = time.perf_counter()
            found = detect_baseline(frame, source_preprocess_cfg)
            blocks["preprocess_detection"] = (time.perf_counter() - start) * 1000
            binary = found["edges"]
            line_ranges = found.get("line_ranges")
            left, right = found["left"], found["right"]
            valid_left, valid_right = left is not None, right is not None
            lane_measurement_available = valid_left and valid_right
            lost = not lane_measurement_available
            if not lane_measurement_available:
                line_ranges = None
                left = right = None
            offset_left, offset_right = left, right
        elif method == "proposed":
            fit_cfg = dict(self.config.get("fit", {}))
            fit_cfg.update(self.config.get("fit_by_source", {}).get(self.data_source, {}))
            fit_reference = preprocess_cfg.get("roi_reference_size", [1280, 720])
            fit_scale = width / max(float(fit_reference[0]), 1.0)
            for key in ("window_margin", "search_margin", "expected_lane_width_px",
                        "lane_width_min_px", "lane_width_max_px"):
                if fit_cfg.get(key) is not None:
                    fit_cfg[key] = float(fit_cfg[key]) * fit_scale
            start = time.perf_counter()
            if pipeline.get("color_threshold", False):
                hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
                colors = lane_mask(frame, source_preprocess_cfg)
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

            homography = self._homography_for_frame()
            start = time.perf_counter()
            use_video_calibration = self.calibration_status == "video"
            if (pipeline.get("perspective", False) or use_video_calibration) and homography:
                source_points = homography.get("source")
                destination_points = homography.get("destination")
                if source_points is not None and destination_points is not None and fit_reference:
                    ref_width, ref_height = map(float, fit_reference)
                    source_points = [[round(x * width / ref_width), round(y * height / ref_height)]
                                     for x, y in source_points]
                    destination_points = [[round(x * width / ref_width), round(y * height / ref_height)]
                                          for x, y in destination_points]
                warped, matrix = warp_image(binary, source_points, destination_points)
            else:
                warped = binary
            blocks["perspective"] = (time.perf_counter() - start) * 1000

            start = time.perf_counter()
            previous = self.previous_fits if pipeline.get("previous_fit_search", False) else None
            if pipeline.get("hough", False):
                hough = detect_hough(warped, warped.shape[:2], source_preprocess_cfg)
                bev_left, bev_right = hough["left"], hough["right"]
                fit_diagnostics = {}
                for side, bounds in hough["line_ranges"].items():
                    fit_diagnostics[f"y_min_{side}"] = None if bounds is None else bounds[0]
                    fit_diagnostics[f"y_max_{side}"] = None if bounds is None else bounds[1]
            else:
                bev_left, bev_right, fit_diagnostics = sliding_window_fit(warped, fit_cfg, previous)
            fit_validation_cfg = dict(fit_cfg)
            observed = [(fit_diagnostics.get(f"y_min_{side}"),
                         fit_diagnostics.get(f"y_max_{side}"))
                        for side in ("left", "right")]
            if all(low is not None and high is not None for low, high in observed):
                observed_top = max(float(low) for low, _ in observed)
                observed_bottom = min(float(high) for _, high in observed)
                if observed_bottom <= observed_top:
                    bev_left = bev_right = None
                else:
                    fit_validation_cfg["valid_y_top_norm"] = observed_top
                    fit_validation_cfg["valid_y_bottom_norm"] = observed_bottom
            bev_left, bev_right, validity = validate_and_recover_lanes(
                bev_left, bev_right, width, height, fit_validation_cfg)
            valid_left, valid_right = validity["valid_left"], validity["valid_right"]
            single_line = validity["single_line"]
            lane_measurement_available = validity["valid"]
            minimum_windows = fit_cfg.get("minimum_common_windows")
            if (lane_measurement_available and not single_line and minimum_windows is not None and
                    not fit_support_overlaps(fit_diagnostics, fit_cfg.get("window_count", 1),
                                             minimum_windows)):
                bev_left = bev_right = None
                valid_left = valid_right = lane_measurement_available = False
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
                    margin = fit_cfg.get("fit_y_margin_norm")
                    margin = 0.015 if margin is None else float(margin)
                    bounds = None if low is None else (max(0.0, low - margin), min(1.0, high + margin))
                    line_ranges[side] = unwarp_y_range(fit, bounds, matrix, height, width)
                offset_left, offset_right = bev_left, bev_right
                if pipeline.get("kalman", False):
                    if single_line:
                        self._filter_fits(None, None)
                    else:
                        measured_left, measured_right = bev_left.copy(), bev_right.copy()
                        filtered_left, filtered_right = self._filter_fits(bev_left, bev_right)
                        if filtered_left is not None and filtered_right is not None:
                            smooth_left, smooth_right, smooth_validity = validate_and_recover_lanes(
                                filtered_left, filtered_right, width, height, fit_validation_cfg)
                            if smooth_validity["valid"]:
                                bev_left, bev_right = smooth_left, smooth_right
                            else:
                                # Keep a valid current measurement if the
                                # temporal estimate lags a curve or lane change.
                                bev_left, bev_right = measured_left, measured_right
                left = unwarp_lane_fit(bev_left, matrix, height, width)
                right = unwarp_lane_fit(bev_right, matrix, height, width)
            else:
                if pipeline.get("kalman", False):
                    self._filter_fits(None, None)
                self.missed_frames += 1
                max_hold = int(pipeline.get("max_hold_frames", 5))
                if self.missed_frames <= max_hold:
                    left, right = self.previous_display_fits or (None, None)
                    line_ranges = self.previous_display_ranges
                else:
                    left = right = None
                    line_ranges = None
                offset_left, offset_right = None, None
            if lane_measurement_available:
                self.missed_frames = 0
            lost = not lane_measurement_available
            blocks["lane_fit"] = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            blocks["kalman"] = (time.perf_counter() - start) * 1000
        else:
            raise ValueError("method must be 'baseline' or 'proposed'")

        if method == "proposed" and lane_measurement_available and (left is None or right is None):
            # A homography can push an inferred edge outside the source frame;
            # never let a partial inverse transform crash the video loop.
            lane_measurement_available = False
            lost = True
            line_ranges = None
            offset_left = offset_right = None

        if left is not None and right is not None and line_ranges:
            min_width_norm = fit_cfg.get("min_lane_width_norm") if method == "proposed" else None
            max_width_norm = fit_cfg.get("max_lane_width_norm") if method == "proposed" else None
            safe_range = constrain_lane_pair(
                left, right, line_ranges, width,
                .05 if min_width_norm is None else float(min_width_norm),
                .95 if max_width_norm is None else float(max_width_norm))
            if safe_range is None:
                left = right = None
                line_ranges = None
                lane_measurement_available = False
                lost = True
                offset_left = offset_right = None
            else:
                line_ranges = {"left": safe_range, "right": safe_range}
        if (method == "proposed" and
                (not lane_measurement_available or
                 not output_geometry_is_valid(left, right, line_ranges, width, fit_cfg))):
            # Sliding windows can follow a seam, reflection, or guardrail and
            # produce a smooth but wrong quadratic.  For an uncalibrated
            # camera, use only a verified straight Hough pair as a fallback.
            # It deliberately never reuses the rejected curve.
            fallback = None
            if matrix is None and pipeline.get("straight_line_fallback", False):
                # A filename is often insufficient to classify rain or a dark
                # scene.  Try the three declared source presets and choose a
                # geometrically valid pair with the largest observed span.
                candidates = []
                for source in (self.data_source, "day", "night", "rain"):
                    candidate_cfg = dict(source_preprocess_cfg)
                    candidate_cfg["data_source"] = source
                    candidate_roi = preprocess_cfg.get("roi_polygon_by_source", {}).get(
                        source, preprocess_cfg.get("roi_polygon"))
                    if candidate_roi and reference_size:
                        ref_width, ref_height = map(float, reference_size)
                        candidate_roi = [[round(x * width / ref_width),
                                          round(y * height / ref_height)]
                                         for x, y in candidate_roi]
                    candidate_cfg["roi_polygon"] = candidate_roi
                    candidate = detect_baseline(frame, candidate_cfg)
                    ranges = candidate["line_ranges"]
                    fallback_gate_cfg = dict(fit_cfg)
                    fallback_gate_cfg["minimum_support_norm"] = float(
                        pipeline.get("fallback_minimum_support_norm",
                                     fit_cfg.get("minimum_support_norm", .15)))
                    if (candidate["left"] is not None and candidate["right"] is not None and
                            output_geometry_is_valid(candidate["left"], candidate["right"],
                                                     ranges, width, fallback_gate_cfg)):
                        shared = min(value[1] for value in ranges.values()) - max(
                            value[0] for value in ranges.values())
                        candidates.append((shared, candidate))
                if candidates:
                    fallback = max(candidates, key=lambda item: item[0])[1]
            if fallback is not None:
                left, right = fallback["left"], fallback["right"]
                line_ranges = fallback["line_ranges"]
                offset_left, offset_right = left, right
                valid_left = valid_right = lane_measurement_available = True
                lost = False
                detection_mode = "straight_fallback"
                # The following common output gate must use the same support
                # requirement that admitted this observed Hough pair.
                fit_cfg["minimum_support_norm"] = float(
                    pipeline.get("fallback_minimum_support_norm",
                                 fit_cfg.get("minimum_support_norm", .15)))
            else:
                left = right = None
                line_ranges = None
                lane_measurement_available = False
                lost = True
                offset_left = offset_right = None
        if method == "proposed" and lane_measurement_available:
            if not single_line and bev_left is not None and bev_right is not None:
                self.previous_fits = (bev_left.copy(), bev_right.copy())
            self.previous_display_fits = (left.copy(), right.copy())
            self.previous_display_ranges = line_ranges

        if method == "baseline":
            offset_left, offset_right = (left, right)
        departure_start = time.perf_counter()
        departure_cfg = self.config.get("departure", {})
        lane_width_m = departure_cfg.get("lane_width_m")
        current_fps = fps or self.config.get("dataset_metadata", {}).get("dashcam_fps")
        meter_scale_override = self.last_meters_per_pixel if single_line else None
        vehicle_x = vehicle_x_in_bev(width, height, matrix)
        if method == "proposed" and matrix is not None:
            y_eval_norm = 1.0  # Fits and vehicle position are both in bird's-eye coordinates.
        else:
            y_eval_norm = (min(bounds[1] for bounds in line_ranges.values() if bounds is not None)
                           if line_ranges and
                           any(bounds is not None for bounds in line_ranges.values()) else 1.0)
        offset = calculate_offset(offset_left, offset_right, width,
                                  float(lane_width_m) if lane_width_m is not None else None,
                                  vehicle_x_px=vehicle_x, y_eval=y_eval_norm,
                                  meters_per_pixel=meter_scale_override,
                                  reuse_scale=single_line)
        if lane_measurement_available and not single_line:
            self.last_meters_per_pixel = offset.get("meters_per_pixel")
        elif lost and self.last_meters_per_pixel is not None:
            offset["meters_per_pixel"] = self.last_meters_per_pixel
        velocity = self._offset_velocity(offset_left, offset_right,
                                         offset.get("meters_per_pixel"), current_fps, width)

        if pipeline.get("warning", True):
            if lost:
                warning = {"warning": self.warning.active, "state": self.warning.state,
                           "tlc_s": None, "available": False}
            else:
                warning_offset = offset.get("offset_m") if lane_measurement_available else None
                warning = self.warning.update(warning_offset, velocity, lane_width_m)
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
                "departure_record": departure_record, "timings_ms": blocks, "lost": lost,
                "calibration_status": self.calibration_status,
                "detection_mode": detection_mode}

    def _filter_fits(self, left, right):
        cfg = self.config.get("kalman", {})
        needed = ("dt", "process_noise", "measurement_noise", "initial_covariance")
        if any(cfg.get(key) is None for key in needed):
            return left, right
        if self.filters is None:
            if left is None or right is None:
                return left, right
            self.filters = tuple(LaneKalmanFilter(cfg["dt"], cfg["process_noise"],
                                                   cfg["measurement_noise"], cfg["initial_covariance"],
                                                   initial_state=np.r_[fit, [0.0, 0.0, 0.0]],
                                                   gating_threshold=cfg.get("gating_threshold", 11.34))
                                 for fit in (left, right))
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
            cv2.copyTo(original, cv2.bitwise_not(region), frame)
        if lost:
            color = (128, 128, 128)
            label = "MAT LAN"
        elif warning is None or warning_state == "UNAVAILABLE":
            color = (0, 165, 255)
            label = "CHUA THAY DU 2 VACH" if offset_px is None else "CHUA CAU HINH CANH BAO"
        elif warning is True:
            color = (0, 0, 255) if warning_state == "DANGER" else (0, 100, 255)
            label = "CẢNH BÁO CHỆCH LÀN"
        else:
            color = (0, 200, 0)
            label = warning_state or "SAFE"
        _draw_unicode_text(frame, label, (24, 12), color, 26)

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


def _draw_unicode_text(frame: np.ndarray, text: str, origin: tuple[int, int],
                       color: tuple[int, int, int], size: int) -> None:
    """Draw labels with a Unicode font; OpenCV Hershey fonts lack Vietnamese glyphs."""
    font_path = next((p for p in (Path("C:/Windows/Fonts/arial.ttf"),
                                  Path("C:/Windows/Fonts/segoeui.ttf")) if p.is_file()), None)
    if font_path is None:
        cv2.putText(frame, text.encode("ascii", "replace").decode("ascii"),
                    (origin[0], origin[1] + size), cv2.FONT_HERSHEY_SIMPLEX,
                    .8, color, 2, cv2.LINE_AA)
        return
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(image).text(origin, text, font=ImageFont.truetype(str(font_path), size),
                               fill=color[::-1], stroke_width=1, stroke_fill=(0, 0, 0))
    frame[:] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
