import numpy as np

from src.lane_fit import fit_support_overlaps, sliding_window_fit, validate_and_recover_lanes


def test_fit_support_requires_shared_evidence_span():
    diagnostics = {"y_min_left": .70, "y_max_left": .78,
                   "y_min_right": .70, "y_max_right": 1.0}
    assert not fit_support_overlaps(diagnostics, window_count=12, minimum_windows=2)
    diagnostics["y_max_left"] = .90
    assert fit_support_overlaps(diagnostics, window_count=12, minimum_windows=2)


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


def test_dashed_side_is_recovered_from_solid_side_with_expected_width():
    height, width = 100, 101
    image = np.zeros((height, width), dtype=np.uint8)
    y = np.arange(height)
    image[y, 25] = 255
    image[(y % 20 < 8), 75] = 255
    cfg = {"window_count": 20, "window_margin": 8, "recenter_min_pixels": 2,
           "min_fit_points": 60, "expected_lane_width_px": 50,
           "lane_width_min_px": 45, "lane_width_max_px": 55,
           "parallel_tolerance": .05}
    left, right, diagnostics = sliding_window_fit(image, cfg)
    assert diagnostics["missing_windows_right"] > 0
    left, right, validity = validate_and_recover_lanes(left, right, width, height, cfg)
    assert validity["single_line"] is True
    for y_norm in np.linspace(0, 1, 10):
        assert np.polyval(left, y_norm) < np.polyval(right, y_norm)
    assert abs((np.polyval(right, 1) - np.polyval(left, 1)) * (width - 1) - 50) <= 1


def test_crossed_or_nonparallel_fit_is_rejected():
    cfg = {"lane_width_min_px": 40, "lane_width_max_px": 60,
           "parallel_tolerance": .05}
    left = np.array([0, .4, .2])
    right = np.array([0, -.4, .8])
    out_left, out_right, flags = validate_and_recover_lanes(left, right, 101, 100, cfg)
    assert out_left is None and out_right is None
    assert flags["valid"] is False
