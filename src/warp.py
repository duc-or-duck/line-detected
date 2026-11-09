"""Perspective transformation utilities."""

from __future__ import annotations

import cv2
import numpy as np


def warp_image(image: np.ndarray, source_points: list | None,
               destination_points: list | None) -> tuple[np.ndarray, np.ndarray | None]:
    if source_points is None or destination_points is None:
        return image.copy(), None
    source = np.asarray(source_points, dtype=np.float32)
    destination = np.asarray(destination_points, dtype=np.float32)
    if source.shape != (4, 2) or destination.shape != (4, 2):
        raise ValueError("homography points must each contain four (x, y) pairs")
    matrix = cv2.getPerspectiveTransform(source, destination)
    height, width = image.shape[:2]
    return cv2.warpPerspective(image, matrix, (width, height)), matrix


def unwarp_points(points: np.ndarray, matrix: np.ndarray | None) -> np.ndarray:
    if matrix is None:
        return np.asarray(points)
    return cv2.perspectiveTransform(np.asarray(points, dtype=np.float32).reshape(-1, 1, 2),
                                    np.linalg.inv(matrix)).reshape(-1, 2)
