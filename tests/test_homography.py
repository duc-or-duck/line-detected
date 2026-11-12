import cv2
import numpy as np


def test_homography_round_trip_and_parallel_lanes():
    src = np.float32([[300, 300], [700, 300], [100, 700], [900, 700]])
    dst = np.float32([[300, 80], [700, 80], [300, 680], [700, 680]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    restored = cv2.perspectiveTransform(cv2.perspectiveTransform(src.reshape(-1, 1, 2), matrix), np.linalg.inv(matrix)).reshape(-1, 2)
    assert np.max(np.abs(restored - src)) < 1
    assert abs((dst[2, 0] - dst[0, 0]) - (dst[3, 0] - dst[1, 0])) < 1e-6
