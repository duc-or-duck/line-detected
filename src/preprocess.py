"""Lane feature extraction and image preprocessing."""

from __future__ import annotations

import cv2
import numpy as np


def lane_mask(image: np.ndarray, params: dict) -> np.ndarray:
    """Extract white and yellow lane markings with HLS thresholds."""
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    source = params.get("data_source")
    # Brighten only the lightness channel for difficult illumination.  Hue and
    # saturation remain untouched, so this does not turn arbitrary coloured
    # objects into yellow paint.
    clip_limit = params.get("lane_clahe_clip_by_source", {}).get(source)
    if clip_limit is not None and float(clip_limit) > 0:
        grid = tuple(map(int, params.get("lane_clahe_tile_grid", [8, 8])))
        hls[:, :, 1] = cv2.createCLAHE(float(clip_limit), grid).apply(hls[:, :, 1])
    yellow_low, yellow_high = params.get("yellow_h_range_by_source", {}).get(
        source, params.get("yellow_h_range", [15, 35]))
    saturation = params.get("yellow_s_min_by_source", {}).get(
        source, params.get("yellow_s_min", 100))
    white = params.get("white_l_min_by_source", {}).get(
        source, params.get("white_l_min", 200))
    yellow = cv2.inRange(hls, np.array([yellow_low, 0, saturation], dtype=np.uint8),
                         np.array([yellow_high, 255, 255], dtype=np.uint8))
    white_mask = cv2.inRange(hls[:, :, 1], int(white), 255)
    return cv2.bitwise_or(yellow, white_mask)


def enhance_gray(gray: np.ndarray, method: str, clip_limit: float = 2.0,
                 tile_grid: tuple[int, int] = (8, 8), gamma: float = 1.5) -> np.ndarray:
    """Apply one of SPEC-permitted illumination enhancement methods to a grayscale image.

    Methods: 'clahe' | 'gamma' | 'log' | 'histeq' | 'none'.
    CLAHE is applied with *clip_limit* and *tile_grid* (Section 4.6 / Bảng 4.5).
    Gamma correction raises the image to power 1/gamma (brightens when gamma > 1).
    """
    if method == "clahe" and clip_limit > 0:
        return cv2.createCLAHE(float(clip_limit), tile_grid).apply(gray)
    if method == "gamma":
        inv = 1.0 / max(float(gamma), 1e-3)
        lut = np.clip(np.power(np.arange(256, dtype=np.float32) / 255.0, inv) * 255.0,
                      0, 255).astype(np.uint8)
        return lut[gray]
    if method == "log":
        lut = np.clip(255.0 * np.log1p(np.arange(256, dtype=np.float32)) / np.log1p(255.0),
                      0, 255).astype(np.uint8)
        return lut[gray]
    if method == "histeq":
        return cv2.equalizeHist(gray)
    return gray  # 'none'


def canny_edges(image: np.ndarray, params: dict) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    settings = dict(params)
    settings.update(params.get("canny_by_source", {}).get(params.get("data_source"), {}))

    # Illumination enhancement (SPEC 3.4.2 / Bảng 4.5)
    enh_method = settings.get("enhancement_method", "clahe")
    clip_limit = settings.get("clahe_clip_limit")
    tile = tuple(map(int, settings.get("clahe_tile_grid", [8, 8])))
    # Legacy behaviour: if clahe_clip_limit is set directly (A.yaml style), apply CLAHE
    if clip_limit is not None and float(clip_limit) > 0:
        gray = enhance_gray(gray, enh_method if enh_method else "clahe",
                            float(clip_limit), tile)

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
            if high <= low:
                low = max(0, int(median * 0.5))
                high = min(255, max(low + 10, int(median * 1.5)))
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
