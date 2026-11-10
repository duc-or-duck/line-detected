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
    settings = dict(params)
    settings.update(params.get("canny_by_source", {}).get(params.get("data_source"), {}))
    clip_limit = settings.get("clahe_clip_limit")
    if clip_limit is not None and float(clip_limit) > 0:
        tile = tuple(map(int, settings.get("clahe_tile_grid", [8, 8])))
        gray = cv2.createCLAHE(float(clip_limit), tile).apply(gray)
    blurred = cv2.GaussianBlur(gray, tuple(settings.get("gaussian_kernel", [5, 5])), 0)
    if settings.get("auto_canny", False):
        region = settings.get("roi_polygon")
        if region:
            roi = np.zeros_like(blurred)
            cv2.fillPoly(roi, [np.asarray(region, dtype=np.int32)], 255)
            values = blurred[roi > 0]
        else:
            values = blurred.reshape(-1)
        if values.size:
            median = float(np.median(values))
            sigma = float(settings.get("canny_sigma", .33))
            low = max(0, int((1.0 - sigma) * median))
            high = min(255, int((1.0 + sigma) * median))
            if high > low:
                return cv2.Canny(blurred, low, high)
    return cv2.Canny(blurred, int(settings.get("canny_low", 50)),
                     int(settings.get("canny_high", 150)))


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
