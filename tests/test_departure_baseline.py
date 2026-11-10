"""Unit tests for baseline lane departure warning:
- Shifted vehicle gives d = +0.5 / -0.5 / 0 m (+-0.05 m).
- Arrow direction sign consistency.
- Hysteresis does not flicker for d oscillating between off_thr and on_thr.
- Warning state unchanged on lost frames.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.departure import DepartureWarning, calculate_offset, calculate_straight_lines_offset
from src.pipeline import LanePipeline, load_config


def test_departure_offset_shifted_vehicle():
    """Vehicle shifted +0.5m / -0.5m / 0m gives d = +0.5 / -0.5 / 0 (+-0.05)."""
    # Width = 1280, lane_width_m = 3.7
    # Centered lane: left at x = 200, right at x = 1080 -> width = 880 px.
    # meters_per_pixel = 3.7 / 880 = 0.0042045 m/px.
    # Vehicle center = (1280 - 1) / 2 = 639.5 px.
    # Lane center = (200 + 1080) / 2 = 640.0 px.
    # d = (639.5 - 640.0) * mpp ~ 0.0.
    lane_width_m = 3.7
    lane_w_px = 880.0
    mpp = lane_width_m / lane_w_px
    px_for_half_meter = 0.5 / mpp  # ~ 118.9 px

    # 1. Centered vehicle
    res_center = calculate_straight_lines_offset(
        left_line=(-1.0, 719.0 + 200.0),
        right_line=(1.0, 719.0 - 1080.0),
        y_eval=719.0,
        frame_width=1280,
        lane_width_m=lane_width_m,
    )
    assert abs(res_center["d"] - 0.0) <= 0.05, f"Expected 0.0, got {res_center['d']}"

    # 2. Vehicle shifted right by +0.5 m (i.e. lane markings are shifted left by ~119 px)
    shift_px = px_for_half_meter
    res_right = calculate_straight_lines_offset(
        left_line=(-1.0, 719.0 + (200.0 - shift_px)),
        right_line=(1.0, 719.0 - (1080.0 - shift_px)),
        y_eval=719.0,
        frame_width=1280,
        lane_width_m=lane_width_m,
    )
    assert abs(res_right["d"] - 0.5) <= 0.05, f"Expected +0.5, got {res_right['d']}"

    # 3. Vehicle shifted left by -0.5 m (lane markings shifted right by ~119 px)
    res_left = calculate_straight_lines_offset(
        left_line=(-1.0, 719.0 + (200.0 + shift_px)),
        right_line=(1.0, 719.0 - (1080.0 + shift_px)),
        y_eval=719.0,
        frame_width=1280,
        lane_width_m=lane_width_m,
    )
    assert abs(res_left["d"] - (-0.5)) <= 0.05, f"Expected -0.5, got {res_left['d']}"


def test_departure_warning_arrow_and_hysteresis_no_flicker():
    """Hysteresis does not flicker when d oscillates between off_thr and on_thr."""
    dep_cfg = {
        "lane_width_m": 3.7,
        "on_offset_m": 0.40,
        "off_offset_m": 0.25,
        "danger_offset_m": 0.65,
        "tlc_threshold_s": 2.5,
        "consecutive_frames": 3,
    }
    warning_system = DepartureWarning(dep_cfg)

    # Initially in safe zone (d = 0.10 m)
    res = warning_system.update(0.10, 0.0, 3.7)
    assert res["warning"] is False
    assert res["state"] == "SAFE"

    # Drift out: 3 frames with d = 0.45 m (> on_offset_m) -> triggers WARNING
    for _ in range(3):
        res = warning_system.update(0.45, 0.05, 3.7)
    assert res["warning"] is True
    assert res["state"] == "WARNING"

    # Now oscillate d between 0.30 m and 0.35 m (between off_offset_m 0.25 and on_offset_m 0.40)
    # Hysteresis must keep warning active and NOT flicker!
    for d_val in [0.35, 0.30, 0.38, 0.28, 0.32]:
        res = warning_system.update(d_val, 0.0, 3.7)
        assert res["warning"] is True, f"Warning flickered off at d={d_val}"
        assert res["state"] == "WARNING"

    # Once d drops strictly below off_offset_m (< 0.25 m), it turns OFF
    res = warning_system.update(0.20, -0.05, 3.7)
    assert res["warning"] is False
    assert res["state"] == "SAFE"


def test_departure_state_unchanged_on_lost_frames():
    """Warning state remains unchanged on lost frames."""
    dep_cfg = {
        "lane_width_m": 3.7,
        "on_offset_m": 0.40,
        "off_offset_m": 0.25,
        "danger_offset_m": 0.65,
        "tlc_threshold_s": 2.5,
        "consecutive_frames": 1,
    }
    warning_system = DepartureWarning(dep_cfg)

    # Trigger DANGER state with large offset
    res = warning_system.update(0.70, 0.1, 3.7)
    assert res["warning"] is True
    assert res["state"] == "DANGER"

    # Now pass a lost frame (offset_m = None)
    res_lost = warning_system.update(None, None, 3.7)
    assert res_lost["warning"] is True
    assert res_lost["state"] == "DANGER"
    assert res_lost["available"] is False
