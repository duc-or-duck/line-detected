"""Canny, trapezoidal ROI, and Hough line baseline."""

from __future__ import annotations

import cv2
import numpy as np

from .preprocess import canny_edges, roi_mask


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
    candidates = {"left": [], "right": []}
    candidate_lines = []
    min_abs_slope = config.get("min_abs_slope_by_source", {}).get(
        config.get("data_source"), config.get("min_abs_slope"))
    max_abs_slope = config.get("max_abs_slope_by_source", {}).get(
        config.get("data_source"), config.get("max_abs_slope"))
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
        if not _is_valid_lane_segment(x1, y1, x2, y2, gray, image, road_median, width, height):
            continue
        intercept = y1 - slope * x1
        midpoint_x = (x1 + x2) / 2.0
        midpoint_y = (y1 + y2) / 2.0
        center_x = _roi_center_at_y(polygon, midpoint_y, width)
        side = "left" if slope < 0 and midpoint_x < center_x else None
        if slope > 0 and midpoint_x > center_x:
            side = "right"
        if side is not None:
            length = float(np.hypot(x2 - x1, y2 - y1))
            candidates[side].append((slope, intercept, length))
            candidate_lines.append({
                "side": side, "slope": slope, "intercept": intercept, "length": length,
                "x_top": (roi_top - intercept) / slope,
                "x_bottom": (roi_bottom - intercept) / slope,
            })
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
                side_candidates, side, width, roi_bottom)
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


def _is_valid_lane_segment(x1: int, y1: int, x2: int, y2: int, gray: np.ndarray,
                           image: np.ndarray, road_median: float,
                           width: int, height: int) -> bool:
    """Filter out dark oil stains, tar cracks/seams, water ruts, and red taillight reflections."""
    if road_median <= 5.0:
        return True
    dx, dy = x2 - x1, y2 - y1
    length = float(np.hypot(dx, dy))
    if length < 1e-3:
        return False
    nx, ny = -dy / length, dx / length

    # Reject specular puddle reflection of red taillights on wet asphalt at night
    if image.ndim == 3 and image.shape[2] == 3:
        pts_x = np.linspace(x1, x2, 6).astype(int)
        pts_y = np.linspace(y1, y2, 6).astype(int)
        mr = float(np.mean(image[pts_y, pts_x, 2]))
        mg = float(np.mean(image[pts_y, pts_x, 1]))
        mb = float(np.mean(image[pts_y, pts_x, 0]))
        if mr > 70 and mr > 1.4 * mg and mr > 1.5 * mb:
            return False

    # Sample perpendicular cross-section at multiple points along the segment
    scale = max(1.0, width / 1280.0)
    offsets = np.asarray([-12, -8, -4, -2, 0, 2, 4, 8, 12], dtype=np.float64) * scale

    max_contrasts = []
    min_contrasts = []
    trough_count = 0

    for alpha in (0.25, 0.50, 0.75):
        mx = x1 + alpha * dx
        my = y1 + alpha * dy
        prof = []
        for off in offsets:
            px = int(np.clip(round(mx + off * nx), 0, width - 1))
            py = int(np.clip(round(my + off * ny), 0, height - 1))
            prof.append(float(gray[py, px]))
        p_max = max(prof)
        p_min = min(prof)
        p_mid = prof[4]
        p_left = min(prof[:3])
        p_right = min(prof[6:])
        max_contrasts.append(p_max - road_median)
        min_contrasts.append(road_median - p_min)
        if p_mid < p_left - 6 and p_mid < p_right - 6:
            trough_count += 1

    # Reject dark trough in cross-section (oil leak / water trench where center is darker than flanks)
    if trough_count >= 2:
        return False

    median_max_c = float(np.median(max_contrasts))
    median_min_c = float(np.median(min_contrasts))

    # Adaptive minimum brightness contrast required for a painted marking
    if road_median > 80.0:
        min_bright_margin = 10.0
    elif road_median > 40.0:
        min_bright_margin = 7.0
    else:
        min_bright_margin = 4.0

    # 1. Reject if no part of the cross section is brighter than the road asphalt
    if median_max_c < min_bright_margin:
        return False

    # 2. Reject dark cracks / tar expansion joints / oil streaks:
    # A dark crack dips significantly below asphalt (median_min_c > 20)
    # but lacks strong bright paint contrast (median_max_c < 18).
    if median_min_c > 20.0 and median_max_c < 18.0:
        return False

    # 3. Reject if the depression into dark is much stronger than bright contrast
    if median_min_c > 25.0 and median_min_c > 2.5 * max(median_max_c, 1.0):
        return False

    return True


def _filter_candidates_by_ego_cluster(candidates: list[tuple[float, float, float]],
                                     side: str, width: int, roi_bottom: int,
                                     vehicle_x: float | None = None) -> list[tuple[float, float, float]]:
    """Select the spatial cluster of candidate segments corresponding to the ego lane."""
    if len(candidates) <= 2:
        return candidates
    if vehicle_x is None:
        vehicle_x = width / 2.0
    tol = max(20.0, 0.08 * width)
    items = []
    for c in candidates:
        sl, ic, _ = c
        xb = (roi_bottom - ic) / sl
        items.append((c, xb))
    items.sort(key=lambda it: it[1])
    clusters = []
    curr = [items[0]]
    for it in items[1:]:
        if abs(it[1] - curr[-1][1]) <= tol:
            curr.append(it)
        else:
            clusters.append(curr)
            curr = [it]
    clusters.append(curr)
    c_stats = []
    for c in clusters:
        supp = sum(it[0][2] for it in c)
        avg_xb = float(np.average([it[1] for it in c], weights=[it[0][2] for it in c]))
        c_stats.append({"cluster": [it[0] for it in c], "support": supp, "xb": avg_xb})
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


def _roi_center_at_y(polygon: list[list[int]] | None, y: float, width: int) -> float:
    if not polygon or len(polygon) < 4:
        return width / 2.0
    points = np.asarray(polygon, dtype=np.float64)

    def interpolate_x(first: np.ndarray, second: np.ndarray) -> float:
        if abs(second[1] - first[1]) < 1e-9:
            return float((first[0] + second[0]) / 2.0)
        ratio = np.clip((y - first[1]) / (second[1] - first[1]), 0.0, 1.0)
        return float(first[0] + ratio * (second[0] - first[0]))

    left_x = interpolate_x(points[0], points[3])
    right_x = interpolate_x(points[1], points[2])
    return (left_x + right_x) / 2.0
