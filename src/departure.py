"""Lane offset, time-to-line-crossing, and hysteretic warning state."""

from __future__ import annotations

import numpy as np


def calculate_offset(left: np.ndarray | None, right: np.ndarray | None,
                     frame_width: int, lane_width_m: float) -> dict:
    if left is None or right is None:
        return {"offset_m": None, "lane_width_norm": None, "center_norm": None}
    y = 1.0
    left_x, right_x = float(np.polyval(left, y)), float(np.polyval(right, y))
    width = right_x - left_x
    if not np.isfinite(width) or width <= 0:
        return {"offset_m": None, "lane_width_norm": None, "center_norm": None}
    center = (left_x + right_x) / 2.0
    return {"offset_m": (0.5 - center) * lane_width_m / width,
            "lane_width_norm": width, "center_norm": center}


class DepartureWarning:
    """Config-driven warning logic; thresholds remain unavailable until supplied."""

    def __init__(self, config: dict):
        self.config = config
        self.active = False
        self.candidate_count = 0

    def update(self, offset_m: float | None, offset_velocity_mps: float | None,
               lane_width_m: float | None) -> dict:
        cfg = self.config
        required = ("on_offset_m", "off_offset_m", "tlc_threshold_s", "consecutive_frames")
        if (offset_m is None or lane_width_m is None or
                any(cfg.get(key) is None for key in required)):
            return {"warning": None, "tlc_s": None, "available": False}
        tlc = None
        if offset_velocity_mps is not None and abs(offset_velocity_mps) > 0:
            toward = np.sign(offset_velocity_mps) == np.sign(offset_m)
            if toward:
                remaining = max(float(lane_width_m) / 2.0 - abs(offset_m), 0.0)
                tlc = remaining / abs(offset_velocity_mps)
        condition = (abs(offset_m) >= float(cfg["on_offset_m"]) or
                     (tlc is not None and tlc <= float(cfg["tlc_threshold_s"])))
        if self.active:
            if abs(offset_m) <= float(cfg["off_offset_m"]):
                self.active = False
                self.candidate_count = 0
        elif condition:
            self.candidate_count += 1
            if self.candidate_count >= int(cfg["consecutive_frames"]):
                self.active = True
        else:
            self.candidate_count = 0
        return {"warning": self.active, "tlc_s": tlc, "available": True}


def offset_only_warning(offset_m: float | None, threshold_m: float | None) -> bool | None:
    if offset_m is None or threshold_m is None:
        return None
    return abs(offset_m) >= threshold_m
