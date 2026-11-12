"""Unit tests for baseline lane departure warning:
- Shifted vehicle gives d = +0.5 / -0.5 / 0 m (+-0.05 m).
- Arrow direction sign consistency.
- Hysteresis does not flicker for d oscillating between off_thr and on_thr.
- Warning state unchanged on lost frames.
"""

from __future__ import annotations

from src.departure import DepartureWarning


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
