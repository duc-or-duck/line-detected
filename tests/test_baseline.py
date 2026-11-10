import cv2
import numpy as np

from src.baseline import detect_baseline


def test_hough_baseline_uses_side_of_segment_midpoint_and_never_crosses(monkeypatch):
    segments = np.asarray([
        [15, 90, 40, 40], [20, 90, 45, 40],
        [85, 90, 60, 40], [80, 90, 55, 40],
        [75, 80, 95, 40],
    ], dtype=np.int32)[:, None, :]
    monkeypatch.setattr(cv2, "HoughLinesP", lambda *args, **kwargs: segments)
    config = {
        "roi_polygon": [[0, 25], [99, 25], [99, 99], [0, 99]],
        "hough": {"threshold": 1, "min_line_length": 1, "max_line_gap": 1},
        "min_abs_slope": 0.4,
        "max_abs_slope": 2.0,
    }

    result = detect_baseline(np.zeros((100, 100, 3), dtype=np.uint8), config)

    assert result["left"] is not None
    assert result["right"] is not None
    y = np.linspace(result["line_ranges"]["left"][0], result["line_ranges"]["left"][1], 10)
    assert np.all(np.polyval(result["left"], y) < np.polyval(result["right"], y))
    for bounds in result["line_ranges"].values():
        assert bounds is not None
        assert np.isclose(bounds[0], 25 / 99)
        assert bounds[1] == 1.0


def test_baseline_rejects_crossing_fits_inside_roi(monkeypatch):
    segments = np.asarray([[55, 20, 25, 80], [45, 20, 75, 80]], dtype=np.int32)[:, None, :]
    monkeypatch.setattr(cv2, "HoughLinesP", lambda *args, **kwargs: segments)
    config = {"roi_polygon": [[0, 20], [99, 20], [99, 80], [0, 80]],
              "hough": {"threshold": 1, "min_line_length": 1, "max_line_gap": 1},
              "min_abs_slope": .4}
    result = detect_baseline(np.zeros((100, 100, 3), dtype=np.uint8), config)
    assert result["left"] is None and result["right"] is None
