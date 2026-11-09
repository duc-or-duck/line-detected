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
    min_abs_slope = config.get("min_abs_slope")
    for x1, y1, x2, y2 in ([] if segments is None else segments[:, 0, :]):
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if min_abs_slope is not None and abs(slope) < float(min_abs_slope):
            continue
        # Keep the actual segment samples. Averaging slope/intercept values
        # collapses curved lane markings into a straight line and is very
        # sensitive to unrelated Hough segments on bends.
        length = float(np.hypot(x2 - x1, y2 - y1))
        groups["left" if slope < 0 else "right"].append(
            ((x1, y1), (x2, y2), length))
    fits = {}
    for side, line_segments in groups.items():
        if not line_segments:
            fits[side] = None
            continue
        points = np.asarray([point for first, second, _ in line_segments
                             for point in (first, second)], dtype=np.float64)
        x_norm = points[:, 0] / max(width - 1, 1)
        y_norm = points[:, 1] / max(height - 1, 1)
        # Hough segments are local straight approximations. Fit their samples
        # together as x(y) so the output can follow a curved road.
        weights = np.repeat(np.sqrt([length for _, _, length in line_segments]), 2)
        if len(np.unique(y_norm)) < 3:
            fits[side] = None
            continue
        try:
            fit = np.polyfit(y_norm, x_norm, 2, w=weights)
        except np.linalg.LinAlgError:
            fit = None
        fits[side] = (fit if fit is not None and np.isfinite(fit).all()
                      else None)
    return {**fits, "edges": edges,
            "segments": [] if segments is None else segments[:, 0, :].tolist()}
