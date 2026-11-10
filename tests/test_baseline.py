import cv2
import numpy as np

from src.baseline import _manual_kmeans, detect_baseline


def test_manual_kmeans_is_deterministic_and_separates_two_groups():
    features = np.asarray([
        [-1.0, 0.2, 0.2], [-0.9, 0.21, 0.22],
        [1.0, -0.2, 0.8], [0.9, -0.21, 0.78],
    ])

    labels = _manual_kmeans(features, 2)

    assert labels is not None
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]
    np.testing.assert_array_equal(labels, _manual_kmeans(features, 2))


def test_hough_baseline_uses_kmeans_clusters_for_left_and_right(monkeypatch):
    segments = np.asarray([
        [10, 90, 40, 30], [12, 88, 42, 28],
        [90, 90, 60, 30], [88, 88, 58, 28],
        [70, 90, 71, 10],
    ], dtype=np.int32)[:, None, :]
    monkeypatch.setattr(cv2, "HoughLinesP", lambda *args, **kwargs: segments)
    config = {
        "roi_polygon": None,
        "hough": {"threshold": 1, "min_line_length": 1, "max_line_gap": 1},
        "kmeans": {"clusters": 2, "reference_y_fraction": 0.85},
        "min_abs_slope": 0.1,
        "max_abs_slope": 2.0,
    }

    result = detect_baseline(np.zeros((100, 100, 3), dtype=np.uint8), config)

    assert result["left"] is not None
    assert result["right"] is not None
    assert set(result["cluster_labels"]) == {"left", "right"}
    assert len(result["cluster_labels"]["left"]) == 2
    assert len(result["cluster_labels"]["right"]) == 2
    for bounds in result["line_ranges"].values():
        assert bounds is not None
        assert bounds[1] == 1.0
