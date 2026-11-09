"""Image enhancement and lane feature extraction."""

from __future__ import annotations

import cv2
import numpy as np


def enhance(image: np.ndarray, method: str | None, params: dict) -> np.ndarray:
    if not method or method == "none":
        return image.copy()
    if method == "gamma":
        gamma = params.get("gamma")
        if gamma is None:
            return image.copy()
        lut = np.clip(((np.arange(256) / 255.0) ** (1.0 / gamma)) * 255, 0, 255).astype("uint8")
        return cv2.LUT(image, lut)
    if method == "log":
        scale = params.get("log_scale")
        if scale is None:
            return image.copy()
        normalized = image.astype(np.float32) / 255.0
        return np.clip(scale * np.log1p(normalized) / np.log(2), 0, 255).astype(np.uint8)
    if method == "hist_eq":
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    if method == "clahe":
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        clip = params.get("clahe_clip_limit")
        tiles = params.get("clahe_tile_grid_size")
        if clip is None or tiles is None:
            return image.copy()
        hls[:, :, 1] = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=tuple(tiles)).apply(hls[:, :, 1])
        return cv2.cvtColor(hls, cv2.COLOR_HLS2BGR)
    raise ValueError(f"unknown enhancement method: {method}")


def lane_mask(image: np.ndarray, params: dict) -> np.ndarray:
    """Combine HLS lane colors with an L/S x-gradient threshold."""
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    yellow_low, yellow_high = params.get("yellow_h_range", [15, 35])
    saturation = params.get("yellow_s_min", 100)
    white = params.get("white_l_min", 200)
    yellow = cv2.inRange(hls, np.array([yellow_low, 0, saturation], dtype=np.uint8),
                         np.array([yellow_high, 255, 255], dtype=np.uint8))
    white_mask = cv2.inRange(hls[:, :, 1], int(white), 255)
    channel = hls[:, :, 1] if params.get("gradient_channel", "L") == "L" else hls[:, :, 2]
    gx = cv2.Sobel(channel, cv2.CV_64F, 1, 0)
    threshold = params.get("gradient_threshold")
    gradient = np.zeros_like(channel, dtype=np.uint8)
    if threshold is not None:
        gradient[np.abs(gx) >= threshold] = 255
    return cv2.bitwise_or(cv2.bitwise_or(yellow, white_mask), gradient)


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
