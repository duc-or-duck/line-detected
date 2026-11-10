import numpy as np

from src.departure import (DepartureWarning, calculate_offset, false_alarms_per_minute,
                           offset_only_warning)


def test_offset_from_normalized_boundaries():
    left = np.array([0, 0, .2])
    right = np.array([0, 0, .8])
    result = calculate_offset(left, right, 1280, 3.7)
    assert result["offset_m"] == 0
    assert result["offset_px"] == 0
    assert result["center_px"] == (1280 - 1) / 2
    assert np.isclose(result["lane_width_norm"], .6)


def test_offset_sign_and_magnitude_use_vehicle_position_in_bev():
    left, right = np.array([0, 0, .4]), np.array([0, 0, .6])
    scale = 99
    lane_center = .5 * scale
    px_per_meter = .2 * scale / 3.7
    for shift, expected in ((0, 0), (.5, .5), (-.5, -.5)):
        vehicle_x = lane_center + shift * px_per_meter
        result = calculate_offset(left, right, 100, 3.7, vehicle_x_px=vehicle_x)
        assert np.isclose(result["offset_m"], expected, atol=.05 if expected else .02)


def test_pixel_offset_remains_available_without_meter_calibration():
    result = calculate_offset(np.array([0, 0, .2]), np.array([0, 0, .8]), 100, None)
    assert result["offset_m"] is None
    assert result["offset_px"] == 0


def test_unknown_thresholds_are_not_guessed():
    warning = DepartureWarning({"on_offset_m": None, "off_offset_m": None,
                                "tlc_threshold_s": None, "consecutive_frames": None})
    assert warning.update(.2, .1, 3.7)["warning"] is False


def test_warning_requires_consecutive_frames_and_hysteresis():
    warning = DepartureWarning({"on_offset_m": .5, "off_offset_m": .3,
                                "tlc_threshold_s": .8, "consecutive_frames": 2})
    assert warning.update(.55, .1, 3.7)["warning"] is False
    assert warning.update(.55, .1, 3.7)["warning"] is True
    assert warning.update(.4, 0, 3.7)["warning"] is True
    assert warning.update(.2, 0, 3.7)["warning"] is False
    assert offset_only_warning(.55, .5) is True
    assert warning.state == "SAFE"


def test_missing_measurement_resets_consecutive_warning_counter():
    warning = DepartureWarning({"on_offset_m": .5, "off_offset_m": .3,
                                "tlc_threshold_s": .8, "consecutive_frames": 2})
    assert warning.update(.55, 0, 3.7)["warning"] is False
    assert warning.update(None, None, 3.7)["state"] == "SAFE"
    assert warning.update(.55, 0, 3.7)["warning"] is False


def test_invalid_lane_frame_does_not_change_warning_state():
    warning = DepartureWarning({"on_offset_m": .5, "off_offset_m": .3,
                                "tlc_threshold_s": .8, "consecutive_frames": 1})
    assert warning.update(.6, 0, 3.7)["warning"] is True
    state = warning.state
    result = warning.update(None, None, None)
    assert warning.state == state
    assert result["warning"] is True


def test_warning_hysteresis_does_not_flicker_between_thresholds():
    warning = DepartureWarning({"on_offset_m": .5, "off_offset_m": .3,
                                "tlc_threshold_s": .8, "consecutive_frames": 1})
    assert warning.update(.55, 0, 3.7)["warning"] is True
    assert warning.update(.4, 0, 3.7)["warning"] is True
    assert warning.update(.32, 0, 3.7)["warning"] is True
    assert warning.update(.29, 0, 3.7)["warning"] is False


def test_warning_turns_off_below_offset_release_threshold():
    warning = DepartureWarning({"on_offset_m": .3, "off_offset_m": .2,
                                "tlc_threshold_s": .8, "consecutive_frames": 1})
    assert warning.update(.4, 0, 3.7)["warning"] is True
    result = warning.update(.1, 4.0, 3.7)
    assert result["warning"] is False
    assert result["tlc_s"] < .8


def test_tlc_trigger_does_not_wait_for_offset_frame_count():
    warning = DepartureWarning({"on_offset_m": .5, "off_offset_m": .3,
                                "tlc_threshold_s": .8, "consecutive_frames": 4})
    result = warning.update(.1, 3.0, 3.7)
    assert result["warning"] is True
    assert result["state"] == "WARNING"


def test_warning_reports_safe_warning_danger_and_unavailable_states():
    warning = DepartureWarning({"on_offset_m": .3, "off_offset_m": .2,
                                "tlc_threshold_s": 1.0, "consecutive_frames": 1,
                                "danger_offset_m": .8})
    assert warning.update(.4, 0, 3.7)["state"] == "WARNING"
    assert warning.update(.9, 0, 3.7)["state"] == "DANGER"
    assert warning.update(.1, 0, 3.7)["state"] == "SAFE"
    assert warning.update(None, None, 3.7)["state"] == "SAFE"


def test_false_alarm_metric_counts_warning_onsets_per_minute():
    assert np.isclose(false_alarms_per_minute([False, True, True, False, True],
                                              [False, False, False, True, False], 60), 1440)
