"""Canny, trapezoidal ROI, and Hough line baseline."""

from __future__ import annotations

import cv2
import numpy as np

from .preprocess import canny_edges, lane_mask, roi_mask
from .lane_fit import constrain_lane_pair


def detect_baseline(image: np.ndarray, config: dict) -> dict:
    """Detect lane markings with Canny/Hough, optionally rejecting non-paint edges."""
    raw_edges = canny_edges(image, config)
    edges = raw_edges
    if config.get("baseline_marking_filter", False):
        # Keep edges near white/yellow paint. This rejects dark tar seams and
        # shadows that otherwise produce long, lane-shaped Hough segments.
        markings = lane_mask(image, config)
        kernel_size = int(config.get("baseline_marking_dilation_by_source", {}).get(
            config.get("data_source"), config.get("baseline_marking_dilation", 5)))
        kernel_size = max(3, kernel_size | 1)
        markings = cv2.dilate(markings, np.ones((kernel_size, kernel_size), dtype=np.uint8), iterations=1)
        edges = cv2.bitwise_and(edges, markings)
    edges = roi_mask(edges, config.get("roi_polygon"))
    result = detect_hough(edges, image.shape[:2], config)
    if result["left"] is None or result["right"] is None:
        raw_edges = roi_mask(raw_edges, config.get("roi_polygon"))
        fallback = detect_hough(raw_edges, image.shape[:2], config)
        for side in ("left", "right"):
            if result[side] is None and fallback[side] is not None:
                result[side] = fallback[side]
                result["line_ranges"][side] = fallback["line_ranges"][side]
        result["edges"] = edges
        result["segments"] = fallback["segments"] or result["segments"]
    return result


def detect_hough(binary: np.ndarray, image_shape: tuple[int, int], config: dict) -> dict:
    """Fit weighted straight lines from an already thresholded binary image."""
    params = dict(config.get("hough", {}))
    params.update(config.get("hough_by_source", {}).get(config.get("data_source"), {}))
    required = ("threshold", "min_line_length", "max_line_gap")
    if any(params.get(key) is None for key in required):
        return {"left": None, "right": None, "edges": binary, "segments": [],
                "line_ranges": {"left": None, "right": None}}

    segments = cv2.HoughLinesP(
        binary, 1, np.pi / 180, int(params["threshold"]),
        minLineLength=int(params["min_line_length"]),
        maxLineGap=int(params["max_line_gap"]))
    height, width = image_shape
    polygon = config.get("roi_polygon")
    if polygon:
        roi_y = np.asarray(polygon, dtype=np.float64)[:, 1]
        roi_top = int(np.clip(np.min(roi_y), 0, height - 1))
        roi_bottom = int(np.clip(np.max(roi_y), roi_top + 1, height - 1))
    else:
        roi_top, roi_bottom = 0, height - 1

    data_source = config.get("data_source")
    min_slope = config.get("min_abs_slope_by_source", {}).get(
        data_source, config.get("min_abs_slope"))
    max_slope = config.get("max_abs_slope_by_source", {}).get(
        data_source, config.get("max_abs_slope"))
    candidates = {"left": [], "right": []}
    if segments is not None:
        for x1, y1, x2, y2 in np.asarray(segments).reshape(-1, 4):
            dx, dy = float(x2 - x1), float(y2 - y1)
            if dx == 0:
                continue
            slope = dy / dx
            if slope == 0 or (min_slope is not None and abs(slope) < float(min_slope)):
                continue
            if max_slope is not None and abs(slope) > float(max_slope):
                continue
            intercept = float(y1) - slope * float(x1)
            length = float(np.hypot(dx, dy))
            side = "left" if slope < 0 else "right"
            x_bottom = (roi_bottom - intercept) / slope
            # Reject opposite-carriageway markings and road edges far outside
            # the ego lane before clustering the surviving Hough fragments.
            side_limit = width / 2.0
            if ((side == "left" and x_bottom > side_limit) or
                    (side == "right" and x_bottom < side_limit) or
                    x_bottom < -0.20 * width or x_bottom > 1.20 * width):
                continue
            candidates[side].append({
                "slope": slope, "intercept": intercept, "length": length,
                "x_bottom": x_bottom, "y_min": min(y1, y2), "y_max": max(y1, y2),
            })

    fits = {"left": None, "right": None}
    line_ranges = {"left": None, "right": None}
    if height > 1 and width > 1:
        cluster_tolerance = float(config.get("hough_cluster_tolerance_px", max(32, width * 0.10)))
        for side, side_candidates in candidates.items():
            if not side_candidates:
                continue
            side_candidates.sort(key=lambda item: item["x_bottom"])
            clusters = []
            for first in range(len(side_candidates)):
                cluster = []
                anchor = side_candidates[first]["x_bottom"]
                for candidate in side_candidates[first:]:
                    if candidate["x_bottom"] - anchor > cluster_tolerance:
                        break
                    cluster.append(candidate)
                clusters.append(cluster)
            support = lambda cluster: sum(item["length"] for item in cluster)
            selected = max(clusters, key=lambda cluster: (support(cluster), len(cluster)))
            weights = np.asarray([item["length"] for item in selected], dtype=np.float64)
            slopes = np.asarray([item["slope"] for item in selected], dtype=np.float64)
            intercepts = np.asarray([item["intercept"] for item in selected], dtype=np.float64)
            slope = float(np.average(slopes, weights=weights))
            intercept = float(np.average(intercepts, weights=weights))
            if abs(slope) < 1e-9:
                continue
            fits[side] = np.asarray([
                0.0,
                (1.0 / slope) * (height - 1) / (width - 1),
                (-intercept / slope) / (width - 1),
            ], dtype=np.float64)
            pad = int(config.get("fit_y_margin_px", max(4, round(height * .015))))
            y_min = max(roi_top, int(min(item["y_min"] for item in selected)) - pad)
            y_max = min(roi_bottom, int(max(item["y_max"] for item in selected)) + pad)
            line_ranges[side] = (y_min / (height - 1), y_max / (height - 1))

    safe_range = constrain_lane_pair(
        fits["left"], fits["right"], line_ranges, width,
        float(config.get("min_lane_width_norm", .05)),
        float(config.get("max_lane_width_norm", .95)))
    if safe_range is None and fits["left"] is not None and fits["right"] is not None:
        fits = {"left": None, "right": None}
        line_ranges = {"left": None, "right": None}
    elif safe_range is not None:
        line_ranges = {"left": safe_range, "right": safe_range}

    return {**fits, "edges": binary,
            "segments": [] if segments is None else np.asarray(segments).reshape(-1, 4).tolist(),
            "line_ranges": line_ranges}
