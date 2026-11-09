"""Lane feature extraction and image preprocessing."""

from __future__ import annotations

import cv2
import numpy as np


def lane_mask(image: np.ndarray, params: dict) -> np.ndarray:
    """Extract white and yellow lane markings with HLS thresholds."""
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    yellow_low, yellow_high = params.get("yellow_h_range", [15, 35])
    saturation = params.get("yellow_s_min", 100)
    white = params.get("white_l_min", 200)
    yellow = cv2.inRange(hls, np.array([yellow_low, 0, saturation], dtype=np.uint8),
                         np.array([yellow_high, 255, 255], dtype=np.uint8))
    white_mask = cv2.inRange(hls[:, :, 1], int(white), 255)
    return cv2.bitwise_or(yellow, white_mask)


def canny_edges(image: np.ndarray, params: dict) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, tuple(params.get("gaussian_kernel", [5, 5])), 0)
    return cv2.Canny(blurred, int(params.get("canny_low", 50)), int(params.get("canny_high", 150)))


def roi_mask(image: np.ndarray, polygon: list[list[int]] | None) -> np.ndarray:
    if not polygon:
        return image.copy()
    mask = np.zeros_like(image)
    points = np.asarray(polygon, dtype=np.int32)
    if image.ndim == 2:
        cv2.fillPoly(mask, [points], 255)
    else:
        cv2.fillPoly(mask, [points], (255,) * image.shape[2])
    return cv2.bitwise_and(image, mask)
