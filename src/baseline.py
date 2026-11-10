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
        return {"left": None, "right": None, "edges": edges, "segments": [],
                "line_ranges": {"left": None, "right": None}}
    segments = cv2.HoughLinesP(edges, 1, np.pi / 180,
                               int(params["threshold"]),
                               minLineLength=int(params["min_line_length"]),
                               maxLineGap=int(params["max_line_gap"]))
    height, width = image.shape[:2]
    polygon = config.get("roi_polygon")
    if polygon:
        roi_y = np.asarray(polygon, dtype=np.float64)[:, 1]
        roi_top = int(np.clip(np.min(roi_y), 0, height - 1))
        roi_bottom = int(np.clip(np.max(roi_y), roi_top + 1, height - 1))
    else:
        roi_top, roi_bottom = 0, height - 1
    candidates = {"left": [], "right": []}
    min_abs_slope = config.get("min_abs_slope")
    max_abs_slope = config.get("max_abs_slope")
    for x1, y1, x2, y2 in ([] if segments is None else segments[:, 0, :]):
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if slope == 0:
            continue
        if min_abs_slope is not None and abs(slope) < float(min_abs_slope):
            continue
        if max_abs_slope is not None and abs(slope) > float(max_abs_slope):
            continue
        intercept = y1 - slope * x1
        midpoint_x = (x1 + x2) / 2.0
        side = "left" if slope < 0 and midpoint_x < width / 2 else None
        if slope > 0 and midpoint_x > width / 2:
            side = "right"
        if side is not None:
            length = float(np.hypot(x2 - x1, y2 - y1))
            candidates[side].append((slope, intercept, length))
    fits = {"left": None, "right": None}
    line_ranges = {"left": None, "right": None}
    for side, side_candidates in candidates.items():
        if not side_candidates:
            continue
        slopes = np.asarray([item[0] for item in side_candidates], dtype=np.float64)
        intercepts = np.asarray([item[1] for item in side_candidates], dtype=np.float64)
        weights = np.asarray([item[2] for item in side_candidates], dtype=np.float64)
        slope = float(np.average(slopes, weights=weights))
        intercept = float(np.average(intercepts, weights=weights))
        if abs(slope) < 1e-9:
            continue
        x_top = (roi_top - intercept) / slope
        x_bottom = (roi_bottom - intercept) / slope
        # y = slope*x + intercept, converted to normalized x(y).
        slope_xy = 1.0 / slope
        intercept_xy = -intercept / slope
        fits[side] = np.asarray([0.0, slope_xy * (height - 1) / (width - 1),
                                 intercept_xy / (width - 1)], dtype=np.float64)
        line_ranges[side] = (roi_top / (height - 1), roi_bottom / (height - 1))
    if fits["left"] is not None and fits["right"] is not None:
        rows = np.linspace(roi_top, roi_bottom, 10) / (height - 1)
        if np.any(np.polyval(fits["left"], rows) >= np.polyval(fits["right"], rows)):
            fits = {"left": None, "right": None}
            line_ranges = {"left": None, "right": None}
    return {**fits, "edges": edges,
            "segments": [] if segments is None else segments[:, 0, :].tolist(),
            "line_ranges": line_ranges}
