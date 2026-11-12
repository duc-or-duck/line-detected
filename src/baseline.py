"""Canny, trapezoidal ROI, and Hough line baseline."""

from __future__ import annotations

import cv2
import numpy as np

from .kmeans import kmeans_labels
from .preprocess import canny_edges, roi_mask
from .segment_filter import is_valid_segment, valid_segment_mask


def detect_baseline(image: np.ndarray, config: dict) -> dict:
    edges = canny_edges(image, config)
    edges = roi_mask(edges, config.get("roi_polygon"))
    params = dict(config.get("hough", {}))
    params.update(config.get("hough_by_source", {}).get(config.get("data_source"), {}))
    required = ("threshold", "min_line_length", "max_line_gap")
    if any(params.get(key) is None for key in required):
        return {"left": None, "right": None, "edges": edges, "segments": [],
                "line_ranges": {"left": None, "right": None}}
    segments = cv2.HoughLinesP(edges, 1, np.pi / 180,
                               int(params["threshold"]),
                               minLineLength=int(params["min_line_length"]),
                               maxLineGap=int(params["max_line_gap"]))
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    polygon = config.get("roi_polygon")
    road_median = 0.0
    if np.max(image) > 0:
        if polygon:
            roi_mask_img = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(roi_mask_img, [np.asarray(polygon, dtype=np.int32)], 255)
            roi_vals = gray[roi_mask_img > 0]
            road_median = float(np.median(roi_vals)) if roi_vals.size else float(np.median(gray))
        else:
            road_median = float(np.median(gray))
    if polygon:
        roi_y = np.asarray(polygon, dtype=np.float64)[:, 1]
        roi_top = int(np.clip(np.min(roi_y), 0, height - 1))
        roi_bottom = int(np.clip(np.max(roi_y), roi_top + 1, height - 1))
    else:
        roi_top, roi_bottom = 0, height - 1
    min_abs_slope = config.get("min_abs_slope_by_source", {}).get(
        config.get("data_source"), config.get("min_abs_slope"))
    max_abs_slope = config.get("max_abs_slope_by_source", {}).get(
        config.get("data_source"), config.get("max_abs_slope"))
    candidates, candidate_lines = _collect_candidates(
        segments, gray, image, road_median, polygon, width, height,
        roi_top, roi_bottom, min_abs_slope, max_abs_slope)
    cluster_count = config.get("kmeans", {}).get("clusters")
    fits = {"left": None, "right": None}
    line_ranges = {"left": None, "right": None}
    use_converging_pair = config.get("converging_hough_pair_by_source", {}).get(
        config.get("data_source"), False)
    selected = (_select_converging_pair(candidate_lines, width)
                if use_converging_pair else None)
    if use_converging_pair and selected is not None:
        for side, item in zip(("left", "right"), selected):
            slope, intercept = item["slope"], item["intercept"]
            fits[side] = _normalized_line_fit(slope, intercept, height, width)
            line_ranges[side] = (roi_top / (height - 1), roi_bottom / (height - 1))
    else:
        for side, side_candidates in candidates.items():
            if not side_candidates:
                continue
            clustered = _filter_candidates_by_ego_cluster(
                side_candidates, side, width, roi_bottom, cluster_count=cluster_count)
            inliers = _filter_slope_outliers(clustered)
            slopes = np.asarray([item[0] for item in inliers], dtype=np.float64)
            intercepts = np.asarray([item[1] for item in inliers], dtype=np.float64)
            weights = np.asarray([item[2] for item in inliers], dtype=np.float64)
            slope = float(np.average(slopes, weights=weights))
            intercept = float(np.average(intercepts, weights=weights))
            if abs(slope) < 1e-9:
                continue
            fits[side] = _normalized_line_fit(slope, intercept, height, width)
            line_ranges[side] = (roi_top / (height - 1), roi_bottom / (height - 1))
    valid_pair = False
    if fits["left"] is not None and fits["right"] is not None:
        top_norm, bottom_norm = roi_top / (height - 1), roi_bottom / (height - 1)
        top_gap = float(np.polyval(fits["right"], top_norm) - np.polyval(fits["left"], top_norm))
        bottom_gap = float(np.polyval(fits["right"], bottom_norm) - np.polyval(fits["left"], bottom_norm))
        tolerance_by_source = config.get("vanishing_point_tolerance_by_source", {})
        tolerance = float(tolerance_by_source.get(config.get("data_source"),
                                                   config.get("vanishing_point_tolerance", 0.0)))
        if top_gap <= 0 < bottom_gap and tolerance > 0:
            left_delta = float(fits["right"][1] - fits["left"][1])
            intercept_delta = float(fits["right"][2] - fits["left"][2])
            crossing_y = -intercept_delta / left_delta if abs(left_delta) > 1e-12 else np.inf
            if top_norm < crossing_y <= top_norm + tolerance * (bottom_norm - top_norm):
                top_norm = min(crossing_y + 1.0 / (height - 1), bottom_norm)
                line_ranges = {side: (top_norm, bottom_norm) for side in line_ranges}
                top_gap = float(np.polyval(fits["right"], top_norm) - np.polyval(fits["left"], top_norm))
        if top_gap > 0 and bottom_gap > 0:
            valid_pair = True
    if not valid_pair:
        fallback_selected = _select_converging_pair(candidate_lines, width)
        if fallback_selected is not None:
            for side, item in zip(("left", "right"), fallback_selected):
                slope, intercept = item["slope"], item["intercept"]
                fits[side] = _normalized_line_fit(slope, intercept, height, width)
                line_ranges[side] = (roi_top / (height - 1), roi_bottom / (height - 1))
        else:
            fits = {"left": None, "right": None}
            line_ranges = {"left": None, "right": None}
    return {**fits, "edges": edges,
            "segments": [] if segments is None else segments[:, 0, :].tolist(),
            "line_ranges": line_ranges}


def _collect_candidates(segments: np.ndarray | None, gray: np.ndarray, image: np.ndarray,
                        road_median: float, polygon: list[list[int]] | None,
                        width: int, height: int, roi_top: int, roi_bottom: int,
                        min_abs_slope: float | None,
                        max_abs_slope: float | None
                        ) -> tuple[dict[str, list[tuple[float, float, float]]], list[dict]]:
    """Split Hough fragments into left/right lane candidates in one vectorized pass.

    Every fragment is filtered by slope range, by the road cross-section paint test
    (:func:`valid_segment_mask`) and by which side of the ROI centre it lies on.
    """
    empty = ({"left": [], "right": []}, [])
    if segments is None or len(segments) == 0:
        return empty
    coords = np.asarray(segments, dtype=np.float64).reshape(-1, 4)
    x1, y1, x2, y2 = (coords[:, index] for index in range(4))
    delta_x = x2 - x1
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = (y2 - y1) / delta_x

    usable = (delta_x != 0) & (slope != 0) & np.isfinite(slope)
    if min_abs_slope is not None:
        usable &= np.abs(slope) >= float(min_abs_slope)
    if max_abs_slope is not None:
        usable &= np.abs(slope) <= float(max_abs_slope)
    usable &= valid_segment_mask(coords, gray, image, road_median, width, height)
    keep = np.flatnonzero(usable)
    if keep.size == 0:
        return empty

    slopes = slope[keep]
    intercepts = y1[keep] - slopes * x1[keep]
    lengths = np.hypot(x2[keep] - x1[keep], y2[keep] - y1[keep])
    midpoint_x = (x1[keep] + x2[keep]) / 2.0
    midpoint_y = (y1[keep] + y2[keep]) / 2.0
    center_x = _roi_center_at_y(polygon, midpoint_y, width)
    is_left = (slopes < 0) & (midpoint_x < center_x)
    is_right = (slopes > 0) & (midpoint_x > center_x)

    candidates: dict[str, list[tuple[float, float, float]]] = {"left": [], "right": []}
    candidate_lines: list[dict] = []
    for position in np.flatnonzero(is_left | is_right):
        side = "left" if is_left[position] else "right"
        segment_slope = float(slopes[position])
        intercept = float(intercepts[position])
        length = float(lengths[position])
        candidates[side].append((segment_slope, intercept, length))
        candidate_lines.append({
            "side": side, "slope": segment_slope, "intercept": intercept, "length": length,
            "x_top": (roi_top - intercept) / segment_slope,
            "x_bottom": (roi_bottom - intercept) / segment_slope,
        })
    return candidates, candidate_lines


def _is_valid_lane_segment(x1: int, y1: int, x2: int, y2: int, gray: np.ndarray,
                           image: np.ndarray, road_median: float,
                           width: int, height: int) -> bool:
    """Filter out dark oil stains, tar cracks/seams, water ruts, and red taillight reflections."""
    return is_valid_segment(x1, y1, x2, y2, gray, image, road_median, width, height)


def _filter_candidates_by_ego_cluster(candidates: list[tuple[float, float, float]],
                                     side: str, width: int, roi_bottom: int,
                                     vehicle_x: float | None = None,
                                     cluster_count: int | None = None
                                     ) -> list[tuple[float, float, float]]:
    """Select the spatial cluster of candidate segments corresponding to the ego lane.

    Fragments are grouped by where they cross the ROI bottom row: with
    ``cluster_count`` configured the grouping is the project's self-implemented
    K-means, otherwise a tolerance sweep. The ego-lane selection below is identical
    for either grouping.
    """
    if len(candidates) <= 2 and not cluster_count:
        return candidates
    if vehicle_x is None:
        vehicle_x = width / 2.0
    c_stats = (_kmeans_groups(candidates, roi_bottom, width, int(cluster_count))
               if cluster_count else _tolerance_groups(candidates, roi_bottom, width))
    if side == "left":
        valid = [cs for cs in c_stats if cs["xb"] < vehicle_x]
        if valid:
            max_s = max(cs["support"] for cs in valid)
            good = [cs for cs in valid if cs["support"] >= 0.25 * max_s]
            ego_zone = [cs for cs in good if cs["xb"] <= 0.42 * width]
            target_pool = ego_zone if ego_zone else good
            best = max(target_pool, key=lambda cs: cs["xb"])
            return best["cluster"]
    else:
        valid = [cs for cs in c_stats if cs["xb"] > vehicle_x]
        if valid:
            max_s = max(cs["support"] for cs in valid)
            good = [cs for cs in valid if cs["support"] >= 0.25 * max_s]
            ego_zone = [cs for cs in good if cs["xb"] >= 0.58 * width]
            target_pool = ego_zone if ego_zone else good
            best = min(target_pool, key=lambda cs: cs["xb"])
            return best["cluster"]
    return candidates


def _filter_slope_outliers(candidates: list[tuple[float, float, float]],
                           max_deviation: float = 0.25) -> list[tuple[float, float, float]]:
    """Reject candidate segments whose slope deviates from the weighted median."""
    if len(candidates) <= 2:
        return candidates
    slopes = np.asarray([item[0] for item in candidates], dtype=np.float64)
    weights = np.asarray([item[2] for item in candidates], dtype=np.float64)
    order = np.argsort(slopes)
    cumsum = np.cumsum(weights[order])
    pivot = slopes[order[np.searchsorted(cumsum, cumsum[-1] * 0.5)]]
    inliers = [c for c in candidates if abs(c[0] - pivot) <= max_deviation]
    return inliers if inliers else candidates


def _normalized_line_fit(slope: float, intercept: float, height: int,
                         width: int) -> np.ndarray:
    """Convert y = slope*x + intercept to normalized x(y)."""
    slope_xy = 1.0 / slope
    intercept_xy = -intercept / slope
    return np.asarray([0.0, slope_xy * (height - 1) / (width - 1),
                       intercept_xy / (width - 1)], dtype=np.float64)


def _select_converging_pair(lines: list[dict], width: int) -> tuple[dict, dict] | None:
    """Choose supported Hough lines that converge ahead and bracket the vehicle."""
    vehicle_x = width / 2.0
    pairs = []
    for left in (line for line in lines if line["side"] == "left"):
        if not 0 <= left["x_top"] < width or not 0 <= left["x_bottom"] < vehicle_x:
            continue
        for right in (line for line in lines if line["side"] == "right"):
            if not vehicle_x < right["x_bottom"] < width or not 0 <= right["x_top"] < width:
                continue
            top_gap = right["x_top"] - left["x_top"]
            bottom_gap = right["x_bottom"] - left["x_bottom"]
            if top_gap <= 0 or bottom_gap <= top_gap:
                continue
            support = left["length"] + right["length"]
            pairs.append((bottom_gap, top_gap / max(support, 1e-9), -support, left, right))
    if not pairs:
        return None
    _, _, _, left, right = min(pairs, key=lambda pair: pair[:3])
    return left, right


def _roi_center_at_y(polygon: list[list[int]] | None, y: float | np.ndarray,
                     width: int) -> float | np.ndarray:
    """Horizontal centre of the ROI trapezoid at one row or a whole array of rows."""
    rows = np.asarray(y, dtype=np.float64)
    if not polygon or len(polygon) < 4:
        center = np.full(rows.shape, width / 2.0, dtype=np.float64)
    else:
        points = np.asarray(polygon, dtype=np.float64)
        left_x = _edge_x_at_rows(points[0], points[3], rows)
        right_x = _edge_x_at_rows(points[1], points[2], rows)
        center = (left_x + right_x) / 2.0
    return float(center) if rows.ndim == 0 else center


def _edge_x_at_rows(first: np.ndarray, second: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """x of the trapezoid edge (first -> second) at every row, clamped to the edge span."""
    delta_y = second[1] - first[1]
    if abs(delta_y) < 1e-9:
        return np.full(rows.shape, (first[0] + second[0]) / 2.0, dtype=np.float64)
    ratio = np.clip((rows - first[1]) / delta_y, 0.0, 1.0)
    return first[0] + ratio * (second[0] - first[0])
def _tolerance_groups(candidates: list[tuple[float, float, float]], roi_bottom: int,
                      width: int) -> list[dict]:
    """Group fragments whose bottom-row crossings fall within a fixed tolerance."""
    tolerance = max(20.0, 0.08 * width)
    items = [(candidate, (roi_bottom - candidate[1]) / candidate[0])
             for candidate in candidates]
    items.sort(key=lambda item: item[1])
    clusters = [[items[0]]]
    for item in items[1:]:
        if abs(item[1] - clusters[-1][-1][1]) <= tolerance:
            clusters[-1].append(item)
        else:
            clusters.append([item])
    return [_group_stats([item[0] for item in cluster], [item[1] for item in cluster])
            for cluster in clusters]


def _kmeans_groups(candidates: list[tuple[float, float, float]], roi_bottom: int,
                   width: int, cluster_count: int) -> list[dict]:
    """Group fragments with the self-implemented K-means on bottom-row position.

    Falls back to the tolerance grouping when K-means cannot partition the input
    (too few fragments, empty cluster, or a repeated partition).
    """
    bottom_x = np.asarray([(roi_bottom - candidate[1]) / candidate[0]
                           for candidate in candidates], dtype=np.float64)
    labels = kmeans_labels(bottom_x[:, None] / max(width, 1),
                           min(int(cluster_count), len(candidates)))
    if labels is None:
        return _tolerance_groups(candidates, roi_bottom, width)
    return [_group_stats([candidate for candidate, label in zip(candidates, labels)
                          if label == cluster_id],
                         [position for position, label in zip(bottom_x, labels)
                          if label == cluster_id])
            for cluster_id in np.unique(labels)]


def _group_stats(members: list[tuple[float, float, float]],
                 positions: list[float]) -> dict:
    """Support (total fragment length) and length-weighted position of one group."""
    weights = [candidate[2] for candidate in members]
    return {"cluster": members,
            "support": float(np.sum(weights)),
            "xb": float(np.average(positions, weights=weights))}
