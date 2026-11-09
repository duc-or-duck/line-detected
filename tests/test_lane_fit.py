import numpy as np

from src.lane_fit import sliding_window_fit


def test_sliding_windows_fit_normalized_quadratics():
    height, width = 120, 160
    image = np.zeros((height, width), dtype=np.uint8)
    y = np.arange(height)
    x_left = (.15 * (y / (height - 1)) ** 2 + .15) * (width - 1)
    x_right = (.15 * (y / (height - 1)) ** 2 + .65) * (width - 1)
    image[y, x_left.astype(int)] = 255
    image[y, x_right.astype(int)] = 255
    cfg = {"window_count": 6, "window_margin": 8, "recenter_min_pixels": 2,
           "min_fit_points": 10}
    left, right, counts = sliding_window_fit(image, cfg)
    assert left is not None and right is not None
    np.testing.assert_allclose(np.polyval(left, 1), .30, atol=.03)
    np.testing.assert_allclose(np.polyval(right, 1), .80, atol=.03)
    assert counts["left_count"] >= 10 and counts["right_count"] >= 10


def test_missing_search_parameters_do_not_guess():
    left, right, _ = sliding_window_fit(np.ones((10, 10), dtype=np.uint8), {})
    assert left is None and right is None
