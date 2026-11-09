"""Canny, trapezoidal ROI, and Hough line baseline."""

from __future__ import annotations

import cv2
import numpy as np

from .preprocess import canny_edges, roi_mask


def detect_baseline(image: np.ndarray, config: dict) -> dict:
    edges = canny_edges(image, config)
    edges = roi_mask(edges, config.get("roi_polygon"))
    params = config.get("hough", {})
    required = ("threshold", "min_line_length", "max_line_gap")
    if any(params.get(key) is None for key in required):
        return {"left": None, "right": None, "edges": edges, "segments": []}
    segments = cv2.HoughLinesP(edges, 1, np.pi / 180,
                               int(params["threshold"]),
                               minLineLength=int(params["min_line_length"]),
                               maxLineGap=int(params["max_line_gap"]))
    height, width = image.shape[:2]
    groups = {"left": [], "right": []}
    for x1, y1, x2, y2 in ([] if segments is None else segments[:, 0, :]):
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < float(config.get("min_abs_slope", 0)):
            continue
        intercept = y1 - slope * x1
        groups["left" if slope < 0 else "right"].append((slope, intercept))
    fits = {}
    for side, lines in groups.items():
        if not lines:
            fits[side] = None
            continue
        slope, intercept = np.mean(lines, axis=0)
        # x_norm = B * y_norm + C
        fits[side] = np.array([0.0, (1.0 / slope) * height / width,
                               (-intercept / slope) / width], dtype=np.float64)
    return {**fits, "edges": edges,
            "segments": [] if segments is None else segments[:, 0, :].tolist()}
