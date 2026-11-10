"""Lane offset, time-to-line-crossing, and hysteretic warning state."""

from __future__ import annotations

import numpy as np


def calculate_offset(left: np.ndarray | None, right: np.ndarray | None,
                     frame_width: int, lane_width_m: float | None,
                     vehicle_x_px: float | None = None, y_eval: float = 1.0,
                     meters_per_pixel: float | None = None,
                     reuse_scale: bool = False) -> dict:
    if left is None or right is None:
        return {"offset_m": None, "offset_px": None, "lane_width_norm": None,
                "center_norm": None, "center_px": None, "vehicle_center_px": None}
    y = float(y_eval)
    left_x, right_x = float(np.polyval(left, y)), float(np.polyval(right, y))
    width = right_x - left_x
    if not np.isfinite(width) or width <= 0:
        return {"offset_m": None, "offset_px": None, "lane_width_norm": None,
                "center_norm": None, "center_px": None, "vehicle_center_px": None}
    center = (left_x + right_x) / 2.0
    pixel_scale = max(frame_width - 1, 1)
    vehicle_x = (frame_width - 1) / 2.0 if vehicle_x_px is None else float(vehicle_x_px)
    center_px = center * pixel_scale
    offset_px = vehicle_x - center_px
    if meters_per_pixel is None and not reuse_scale and lane_width_m is not None:
        meters_per_pixel = float(lane_width_m) / (width * pixel_scale)
    offset_m = offset_px * meters_per_pixel if meters_per_pixel is not None else None
    return {"offset_m": offset_m,
            "offset_px": offset_px, "lane_width_norm": width, "center_norm": center,
            "center_px": center_px, "vehicle_center_px": vehicle_x,
            "meters_per_pixel": meters_per_pixel,
            "left_x_px": left_x * pixel_scale, "right_x_px": right_x * pixel_scale}


class DepartureWarning:
    """Config-driven warning logic; thresholds remain unavailable until supplied."""

    def __init__(self, config: dict):
        self.config = config
        self.active = False
        self.candidate_count = 0
        self.state = "SAFE"

    def update(self, offset_m: float | None, offset_velocity_mps: float | None,
               lane_width_m: float | None) -> dict:
        cfg = self.config
        required = ("on_offset_m", "off_offset_m", "tlc_threshold_s", "consecutive_frames")
        if (offset_m is None or lane_width_m is None or
                any(cfg.get(key) is None for key in required)):
            self.candidate_count = 0
            return {"warning": self.active, "state": self.state, "tlc_s": None, "available": False}
        tlc = None
        if offset_velocity_mps is not None:
            if abs(offset_velocity_mps) <= 1e-9:
                tlc = float("inf")
            elif np.sign(offset_velocity_mps) == np.sign(offset_m):
                remaining = max(float(lane_width_m) / 2.0 - abs(offset_m), 0.0)
                tlc = remaining / abs(offset_velocity_mps)
        danger_threshold = cfg.get("danger_offset_m")
        danger_condition = (danger_threshold is not None and
                            abs(offset_m) >= float(danger_threshold))
        offset_condition = abs(offset_m) > float(cfg["on_offset_m"]) or danger_condition
        tlc_condition = tlc is not None and tlc < float(cfg["tlc_threshold_s"])
        if self.active:
            if abs(offset_m) < float(cfg["off_offset_m"]):
                self.active = False
                self.candidate_count = 0
                self.state = "SAFE"
            elif danger_condition:
                self.state = "DANGER"
            else:
                self.state = "WARNING"
        elif tlc_condition:
            self.active = True
            self.candidate_count = 0
            self.state = "DANGER" if danger_condition else "WARNING"
        elif offset_condition:
            self.candidate_count += 1
            if self.candidate_count >= int(cfg["consecutive_frames"]):
                self.active = True
                self.state = "DANGER" if danger_condition else "WARNING"
        else:
            self.candidate_count = 0
            self.state = "SAFE"
        return {"warning": self.active, "state": self.state, "tlc_s": tlc, "available": True}


def offset_only_warning(offset_m: float | None, threshold_m: float | None) -> bool | None:
    if offset_m is None or threshold_m is None:
        return None
    return abs(offset_m) >= threshold_m


def false_alarms_per_minute(warnings: list[bool], true_departures: list[bool], fps: float) -> float:
    """Count false warning onsets per observed minute."""
    if len(warnings) != len(true_departures):
        raise ValueError("warnings and true_departures must have equal length")
    if fps <= 0:
        raise ValueError("fps must be positive")
    duration_minutes = len(warnings) / float(fps) / 60.0
    if duration_minutes == 0:
        return 0.0
    false_onsets = sum(bool(active) and (index == 0 or not warnings[index - 1])
                       and not true_departures[index]
                       for index, active in enumerate(warnings))
    return false_onsets / duration_minutes


def compute_lateral_velocity_and_tlc(d_history: list[float | None], fps: float,
                                     lane_width_m: float = 3.7) -> tuple[float | None, float | None]:
    """Compute lateral velocity v = (d_t - d_{t-k}) / (k/fps) and TLC.

    TLC = distance_to_marking / |v| when moving toward the marking (v * d > 0), else inf.
    """
    valid = [float(d) for d in d_history if d is not None]
    if len(valid) < 2 or fps <= 0:
        return None, None
    k = len(valid) - 1
    dt = k / float(fps)
    if dt <= 1e-9:
        return None, None
    v = (valid[-1] - valid[0]) / dt
    d_curr = valid[-1]
    if abs(v) <= 1e-9:
        tlc = float("inf")
    elif np.sign(v) == np.sign(d_curr):
        distance = max(float(lane_width_m) / 2.0 - abs(d_curr), 0.0)
        tlc = distance / abs(v)
    else:
        tlc = float("inf")
    return v, tlc


def calculate_straight_lines_offset(left_line: tuple[float, float] | None,
                                    right_line: tuple[float, float] | None,
                                    y_eval: float, frame_width: int,
                                    lane_width_m: float | None = 3.7,
                                    cam_offset_px: float = 0.0,
                                    last_meters_per_pixel: float | None = None) -> dict:
    """Straight-line adapter for baseline departure warning calculation.

    left_line, right_line: (slope, intercept) where y = slope * x + intercept,
    or None if lost.
    y_eval: y row (pixels) at which to evaluate offset (typically ROI bottom).
    """
    if left_line is None or right_line is None:
        return {"offset_m": None, "offset_px": None, "d": None,
                "x_left": None, "x_right": None, "meters_per_pixel": last_meters_per_pixel}
    sl_l, ic_l = left_line
    sl_r, ic_r = right_line
    if abs(sl_l) < 1e-9 or abs(sl_r) < 1e-9:
        return {"offset_m": None, "offset_px": None, "d": None,
                "x_left": None, "x_right": None, "meters_per_pixel": last_meters_per_pixel}
    x_left = (y_eval - ic_l) / sl_l
    x_right = (y_eval - ic_r) / sl_r
    lane_width_px = x_right - x_left
    if lane_width_px <= 0:
        return {"offset_m": None, "offset_px": None, "d": None,
                "x_left": x_left, "x_right": x_right, "meters_per_pixel": last_meters_per_pixel}

    x_vehicle = (frame_width - 1) / 2.0 + float(cam_offset_px)
    x_m = float(lane_width_m) / lane_width_px if lane_width_m is not None else last_meters_per_pixel
    d = (x_vehicle - (x_left + x_right) / 2.0) * x_m if x_m is not None else None
    offset_px = x_vehicle - (x_left + x_right) / 2.0
    return {
        "offset_m": d,
        "d": d,
        "offset_px": offset_px,
        "x_left": x_left,
        "x_right": x_right,
        "meters_per_pixel": x_m,
        "lane_width_px": lane_width_px,
        "x_vehicle": x_vehicle,
    }
