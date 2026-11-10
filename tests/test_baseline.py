import cv2
import numpy as np

from src.baseline import detect_baseline
from src.preprocess import canny_edges


def test_adaptive_clahe_canny_recovers_edges_in_dark_frame():
    image = np.full((100, 160, 3), 8, dtype=np.uint8)
    cv2.line(image, (40, 99), (72, 35), (38, 38, 38), 4)
    cv2.line(image, (120, 99), (86, 35), (38, 38, 38), 4)
    edges = canny_edges(image, {"auto_canny": True, "canny_sigma": .4,
                                "clahe_clip_limit": 3.0, "clahe_tile_grid": [8, 8]})
    assert np.count_nonzero(edges) > 0


def test_baseline_uses_roi_center_when_frame_is_not_lane_centered(monkeypatch):
    segments = np.asarray([[5, 90, 25, 30], [40, 30, 50, 90]], dtype=np.int32)[:, None, :]
    monkeypatch.setattr(cv2, "HoughLinesP", lambda *args, **kwargs: segments)
    config = {
        "roi_polygon": [[10, 20], [40, 20], [70, 99], [0, 99]],
        "hough": {"threshold": 1, "min_line_length": 1, "max_line_gap": 1},
        "min_abs_slope": .4,
        "max_abs_slope": 8.0,
    }
    result = detect_baseline(np.zeros((100, 100, 3), dtype=np.uint8), config)
    assert result["left"] is not None and result["right"] is not None


def test_baseline_uses_source_slope_threshold_for_gently_sloped_lanes(monkeypatch):
    segments = np.asarray([[25, 95, 58, 85], [80, 85, 94, 99]], dtype=np.int32)[:, None, :]
    monkeypatch.setattr(cv2, "HoughLinesP", lambda *args, **kwargs: segments)
    config = {
        "data_source": "rain",
        "roi_polygon": [[0, 85], [99, 85], [99, 99], [0, 99]],
        "hough": {"threshold": 1, "min_line_length": 1, "max_line_gap": 1},
        "min_abs_slope": .4,
        "min_abs_slope_by_source": {"rain": .25},
        "max_abs_slope": 4.0,
    }

    result = detect_baseline(np.zeros((100, 100, 3), dtype=np.uint8), config)

    assert result["left"] is not None and result["right"] is not None


def test_baseline_clips_small_vanishing_point_crossing(monkeypatch):
    segments = np.asarray([[55, 20, 25, 80], [45, 20, 75, 80]], dtype=np.int32)[:, None, :]
    monkeypatch.setattr(cv2, "HoughLinesP", lambda *args, **kwargs: segments)
    config = {"roi_polygon": [[0, 20], [99, 20], [99, 80], [0, 80]],
              "hough": {"threshold": 1, "min_line_length": 1, "max_line_gap": 1},
              "min_abs_slope": .4, "vanishing_point_tolerance": .2}
    result = detect_baseline(np.zeros((100, 100, 3), dtype=np.uint8), config)
    assert result["left"] is not None and result["right"] is not None
    top, bottom = result["line_ranges"]["left"]
    rows = np.linspace(top, bottom, 10)
    assert top > 20 / 99
    assert np.all(np.polyval(result["left"], rows) < np.polyval(result["right"], rows))


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
