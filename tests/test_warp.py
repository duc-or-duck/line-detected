import numpy as np

from src.warp import unwarp_points, warp_image


def test_warp_disabled_when_points_are_unknown():
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    output, matrix = warp_image(image, None, None)
    assert matrix is None
    np.testing.assert_array_equal(output, image)


def test_rectangular_homography_round_trip():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    points = [[10, 10], [90, 10], [90, 90], [10, 90]]
    output, matrix = warp_image(image, points, points)
    assert output.shape == image.shape
    restored = unwarp_points(np.array([[20, 20], [70, 70]], dtype=np.float32), matrix)
    np.testing.assert_allclose(restored, [[20, 20], [70, 70]], atol=1e-4)
