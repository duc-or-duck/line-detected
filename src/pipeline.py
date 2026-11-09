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
from .lane_fit import sliding_window_fit
from .preprocess import enhance, lane_mask, roi_mask
from .warp import warp_image


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("configuration must be a YAML mapping")
    return config


class LanePipeline:
    def __init__(self, config: dict, data_source: str | None = None):
        self.config = copy.deepcopy(config)
        self.data_source = data_source
        self.previous_fits = None
        self.filters = None
        self.warning = DepartureWarning(self.config.get("departure", {}))

    def process(self, frame: np.ndarray, method: str = "proposed") -> dict:
        blocks: dict[str, float] = {}
        height, width = frame.shape[:2]
        pipeline = self.config.get("pipeline", {})
        preprocess_cfg = self.config.get("preprocess", {})
        if method == "baseline":
            start = time.perf_counter()
            found = detect_baseline(frame, preprocess_cfg)
            blocks["preprocess_detection"] = (time.perf_counter() - start) * 1000
            left, right, binary = found["left"], found["right"], found["edges"]
        elif method == "proposed":
            start = time.perf_counter()
            enhanced = enhance(frame, pipeline.get("enhancement"), self.config.get("enhancement", {}))
            if pipeline.get("color_threshold", False):
                binary = lane_mask(enhanced, preprocess_cfg)
            else:
                gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
                binary = cv2.Canny(gray, int(preprocess_cfg.get("canny_low", 50)),
                                   int(preprocess_cfg.get("canny_high", 150)))
            if pipeline.get("roi", False):
                binary = roi_mask(binary, preprocess_cfg.get("roi_polygon"))
            blocks["preprocess_detection"] = (time.perf_counter() - start) * 1000

            homography = self.config.get("homography", {}).get(self.data_source or "")
            start = time.perf_counter()
            if pipeline.get("perspective", False) and homography:
                warped, _ = warp_image(binary, homography.get("source"), homography.get("destination"))
            else:
                warped = binary
            blocks["perspective"] = (time.perf_counter() - start) * 1000

            start = time.perf_counter()
            previous = self.previous_fits if pipeline.get("previous_fit_search", False) else None
            left, right, _ = sliding_window_fit(warped, self.config.get("fit", {}), previous)
            if left is not None and right is not None:
                fit_cfg = self.config.get("fit", {})
                lane_width = float(np.polyval(right, 1) - np.polyval(left, 1))
                minimum, maximum = fit_cfg.get("min_lane_width_norm"), fit_cfg.get("max_lane_width_norm")
                if ((minimum is None or lane_width >= float(minimum)) and
                        (maximum is None or lane_width <= float(maximum))):
                    self.previous_fits = (left.copy(), right.copy())
                else:
                    left, right = None, None
            blocks["lane_fit"] = (time.perf_counter() - start) * 1000

            start = time.perf_counter()
            if pipeline.get("kalman", False):
                left, right = self._filter_fits(left, right)
            blocks["kalman"] = (time.perf_counter() - start) * 1000
        else:
            raise ValueError("method must be 'baseline' or 'proposed'")

        start = time.perf_counter()
        departure_cfg = self.config.get("departure", {})
        lane_width_m = departure_cfg.get("lane_width_m")
        offset = calculate_offset(left, right, width, float(lane_width_m)) if lane_width_m is not None else {"offset_m": None}
        velocity = self._offset_velocity(left, right, lane_width_m) if self.filters else None
        if pipeline.get("warning", True):
            warning = self.warning.update(offset.get("offset_m"), velocity, lane_width_m)
        else:
            warning = {"warning": False, "tlc_s": None, "available": True}
        blocks["offset_warning"] = (time.perf_counter() - start) * 1000
        output = frame.copy()
        self._draw(output, left, right, warning.get("warning"))
        return {"frame": output, "binary": binary, "left": left, "right": right,
                "offset_m": offset.get("offset_m"), "warning": warning, "timings_ms": blocks}

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

    def _offset_velocity(self, left, right, lane_width_m):
        if self.filters is None or left is None or right is None or lane_width_m is None:
            return None
        center_velocity = (np.sum(self.filters[0].x[3:, 0]) + np.sum(self.filters[1].x[3:, 0])) / 6.0
        width = float(np.polyval(right, 1) - np.polyval(left, 1))
        return None if width <= 0 else -float(center_velocity) * float(lane_width_m) / width

    @staticmethod
    def _draw(frame: np.ndarray, left, right, warning) -> None:
        height, width = frame.shape[:2]
        if left is not None and right is not None:
            y = np.linspace(0, 1, height)
            ys = np.arange(height)
            left_points = np.stack([(np.polyval(left, y) * (width - 1)).astype(int), ys], axis=1)
            right_points = np.stack([(np.polyval(right, y) * (width - 1)).astype(int), ys], axis=1)
            cv2.polylines(frame, [left_points], False, (0, 255, 0), 3)
            cv2.polylines(frame, [right_points], False, (0, 255, 0), 3)
        if warning is True:
            cv2.putText(frame, "LANE DEPARTURE WARNING", (24, 48), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 0, 255), 2, cv2.LINE_AA)
