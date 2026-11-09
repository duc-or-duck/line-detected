import numpy as np

from src.departure import (DepartureWarning, calculate_offset, false_alarms_per_minute,
                           offset_only_warning)


def test_offset_from_normalized_boundaries():
    left = np.array([0, 0, .2])
    right = np.array([0, 0, .8])
    result = calculate_offset(left, right, 1280, 3.7)
    assert result["offset_m"] == 0
    assert np.isclose(result["lane_width_norm"], .6)


def test_unknown_thresholds_are_not_guessed():
    warning = DepartureWarning({"on_offset_m": None, "off_offset_m": None,
                                "tlc_threshold_s": None, "consecutive_frames": None})
    assert warning.update(.2, .1, 3.7)["warning"] is None


def test_warning_requires_consecutive_frames_and_hysteresis():
    warning = DepartureWarning({"on_offset_m": .5, "off_offset_m": .3,
                                "tlc_threshold_s": .8, "consecutive_frames": 2})
    assert warning.update(.55, .1, 3.7)["warning"] is False
    assert warning.update(.55, .1, 3.7)["warning"] is True
    assert warning.update(.4, 0, 3.7)["warning"] is True
    assert warning.update(.2, 0, 3.7)["warning"] is False
    assert offset_only_warning(.55, .5) is True


def test_false_alarm_metric_counts_warning_onsets_per_minute():
    assert np.isclose(false_alarms_per_minute([False, True, True, False, True],
                                              [False, False, False, True, False], 60), 1440)
