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


def unwarp_lane_fit(fit: np.ndarray | None, matrix: np.ndarray | None,
                    height: int, width: int) -> np.ndarray | None:
    """Map a normalized bird's-eye quadratic back to original image coordinates."""
    if fit is None or matrix is None:
        return fit
    y_norm = np.linspace(0.0, 1.0, max(16, height // 8))
    x_norm = np.polyval(fit, y_norm)
    points = np.column_stack((x_norm * max(width - 1, 1),
                              y_norm * max(height - 1, 1)))
    mapped = unwarp_points(points, matrix)
    mapped_y = mapped[:, 1] / max(height - 1, 1)
    mapped_x = mapped[:, 0] / max(width - 1, 1)
    valid = (np.isfinite(mapped_x) & np.isfinite(mapped_y) &
             (mapped_x >= 0) & (mapped_x <= 1) & (mapped_y >= 0) & (mapped_y <= 1))
    if np.count_nonzero(valid) < 3:
        return None
    result = np.polyfit(mapped_y[valid], mapped_x[valid], 2)
    return result if np.isfinite(result).all() else None
