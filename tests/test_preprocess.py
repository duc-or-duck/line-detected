import numpy as np

from src.preprocess import canny_edges, lane_mask, roi_mask


def test_masks_are_binary_and_roi_respects_polygon():
    image = np.full((40, 60, 3), 220, dtype=np.uint8)
    assert set(np.unique(lane_mask(image, {}))).issubset({0, 255})
    roi = roi_mask(np.ones((40, 60), dtype=np.uint8) * 255, [[0, 0], [30, 0], [30, 39], [0, 39]])
    assert np.count_nonzero(roi[:, 40:]) == 0
    assert canny_edges(image, {}).shape == image.shape[:2]
