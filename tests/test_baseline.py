import cv2
import numpy as np

from src.baseline import detect_baseline


def test_hough_baseline_fits_curved_lane_boundaries(monkeypatch):
    left = [(22, 90, 28, 70), (28, 70, 38, 50),
            (38, 50, 50, 30), (50, 30, 58, 10)]
    right = [(78, 90, 72, 70), (72, 70, 62, 50),
             (62, 50, 50, 30), (50, 30, 42, 10)]
    segments = np.asarray([[*line] for line in left + right], dtype=np.int32)[:, None, :]
    monkeypatch.setattr(cv2, "HoughLinesP", lambda *args, **kwargs: segments)
    config = {
        "roi_polygon": None,
        "hough": {"threshold": 1, "min_line_length": 1, "max_line_gap": 1},
        "min_abs_slope": 0.1,
    }

    result = detect_baseline(np.zeros((100, 100, 3), dtype=np.uint8), config)

    assert result["left"] is not None and result["right"] is not None
    assert result["left"][0] > 0.01
    assert result["right"][0] < -0.01
