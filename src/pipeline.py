"""YAML-configured lane detection and departure warning pipeline."""

from __future__ import annotations

import copy
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from .baseline import detect_baseline
from .departure import DepartureWarning, calculate_offset
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

    def process(self, frame: np.ndarray, method: str = "proposed",
                fps: float | None = None) -> dict:
        blocks: dict[str, float] = {}
        height, width = frame.shape[:2]
        pipeline = self.config.get("pipeline", {})
        preprocess_cfg = self.config.get("preprocess", {})
        roi_polygon = preprocess_cfg.get("roi_polygon_by_source", {}).get(
            self.data_source, preprocess_cfg.get("roi_polygon"))
        source_preprocess_cfg = preprocess_cfg | {"roi_polygon": roi_polygon}
        line_ranges = None
        matrix = None
        valid_left = valid_right = single_line = False
        lane_measurement_available = False
        offset_left = offset_right = None
        if method == "baseline":
            start = time.perf_counter()
            found = detect_baseline(frame, source_preprocess_cfg)
            blocks["preprocess_detection"] = (time.perf_counter() - start) * 1000
            binary = found["edges"]
            line_ranges = found.get("line_ranges")
            valid_left, valid_right = found["left"] is not None, found["right"] is not None
            lane_measurement_available = valid_left and valid_right
            if lane_measurement_available:
                left, right = found["left"], found["right"]
                self.previous_baseline_fits = (left.copy(), right.copy())
                self.previous_display_fits = (left.copy(), right.copy())
                self.previous_display_ranges = line_ranges
                offset_left, offset_right = left, right
            else:
                left, right = self.previous_baseline_fits or (None, None)
                line_ranges = self.previous_display_ranges
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
                    bev_left, bev_right, validity = validate_and_recover_lanes(
                        hough_result["left"], hough_result["right"], width, height, hough_fit_cfg)
                    if validity["valid"]:
                        valid_left = valid_right = lane_measurement_available = True
                        single_line = False
                        hough_ranges = hough_result.get("line_ranges", {})
                        fit_diagnostics = {
                            "y_min_left": (hough_ranges.get("left") or (None, None))[0],
                            "y_max_left": (hough_ranges.get("left") or (None, None))[1],
                            "y_min_right": (hough_ranges.get("right") or (None, None))[0],
                            "y_max_right": (hough_ranges.get("right") or (None, None))[1],
                        }
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
            offset_left, offset_right = (found["left"], found["right"])
        departure_start = time.perf_counter()
        departure_cfg = self.config.get("departure", {})
        lane_width_m = departure_cfg.get("lane_width_m")
        current_fps = fps or self.config.get("dataset_metadata", {}).get("dashcam_fps")
        meter_scale_override = self.last_meters_per_pixel if single_line else None
        vehicle_x = vehicle_x_in_bev(width, height, matrix)
        offset = calculate_offset(offset_left, offset_right, width,
                                  float(lane_width_m) if lane_width_m is not None else None,
                                  vehicle_x_px=vehicle_x, meters_per_pixel=meter_scale_override,
                                  reuse_scale=single_line)
        if lane_measurement_available and not single_line:
            self.last_meters_per_pixel = offset.get("meters_per_pixel")
        velocity = self._offset_velocity(offset_left, offset_right,
                                         offset.get("meters_per_pixel"), current_fps, width)
        if pipeline.get("warning", True):
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
                   display_state, clip_roi=(method == "proposed"), marker_y=marker_y)
        departure_record = {"frame": self.frame_index, "d_m": offset.get("offset_m"),
                            "TLC": warning.get("tlc_s"),
                            "warning": warning.get("warning") if warning.get("available") else None,
                            "valid_left": valid_left, "valid_right": valid_right,
                            "single_line": single_line}
        return {"frame": output, "binary": binary, "left": left, "right": right,
                "offset_m": offset.get("offset_m"), "offset_px": offset.get("offset_px"),
                "x_vehicle": vehicle_x, "x_left": offset.get("left_x_px"),
                "x_right": offset.get("right_x_px"), "x_m": offset.get("meters_per_pixel"),
                "lane_center_px": draw_lane_center, "vehicle_center_px": draw_vehicle_center,
                "lane_measurement_available": lane_measurement_available,
                "valid_left": valid_left, "valid_right": valid_right,
                "single_line": single_line, "warning": warning,
                "departure_record": departure_record, "timings_ms": blocks}

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
              clip_roi=True, marker_y=None) -> None:
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
        if warning is None or warning_state == "UNAVAILABLE":
            color = (0, 165, 255)
        elif warning is True:
            color = (0, 0, 255) if warning_state == "DANGER" else (0, 165, 255)
        else:
            color = (0, 200, 0)
        label = warning_state or ("WARNING" if warning else "SAFE")
        if label == "UNAVAILABLE":
            label = "CHUA THAY DU 2 VACH" if offset_px is None else "CHUA CAU HINH CANH BAO"
        cv2.putText(frame, label, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, .8, color, 2, cv2.LINE_AA)
        if lane_center_px is not None and vehicle_center_px is not None:
            y = height - 18 if marker_y is None else int(marker_y)
            cv2.line(frame, (int(lane_center_px), y - 12), (int(lane_center_px), y + 8),
                     (255, 0, 0), 3)
            cv2.circle(frame, (int(vehicle_center_px), y), 6, (0, 0, 255), -1)
