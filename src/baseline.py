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
    candidates = {"left": [], "right": []}
    reference_y = float(config.get("kmeans", {}).get("reference_y_fraction", 0.85)) * (height - 1)
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
        bottom_x = (height - 1 - intercept) / slope
        reference_x = (reference_y - intercept) / slope
        if not 0 <= bottom_x < width:
            continue
        if not 0 <= reference_x < width:
            continue
        side = "left" if slope < 0 and bottom_x < width / 2 else None
        if slope > 0 and bottom_x > width / 2:
            side = "right"
        if side is not None:
            length = float(np.hypot(x2 - x1, y2 - y1))
            candidates[side].append((slope, intercept, length, bottom_x,
                                     reference_x, (x1, y1, x2, y2)))

    kmeans_config = config.get("kmeans", {})
    cluster_count = kmeans_config.get("clusters")
    fits = {"left": None, "right": None}
    line_ranges = {"left": None, "right": None}
    labels_by_side = {}
    if cluster_count is not None:
        for side, side_candidates in candidates.items():
            if not side_candidates:
                continue
            side_cluster_count = min(int(cluster_count), len(side_candidates))
            # Cluster where Hough fragments are visible, instead of extrapolating
            # every short dash all the way to the image bottom.
            features = np.asarray([[candidate[4] / max(width, 1)]
                                   for candidate in side_candidates], dtype=np.float64)
            labels = _manual_kmeans(features, side_cluster_count)
            if labels is None:
                continue
            labels_by_side[side] = labels.tolist()
            centers = [float(np.mean([side_candidates[index][4]
                                      for index, label in enumerate(labels)
                                      if label == cluster_id]))
                       for cluster_id in range(side_cluster_count)]
            # Prefer the lane marking nearest the image center on each side.
            selected_cluster = min(range(side_cluster_count),
                                   key=lambda index: abs(centers[index] - width / 2))
            selected = [candidate for index, candidate in enumerate(side_candidates)
                        if labels[index] == selected_cluster]
            points = []
            weights = []
            selected_y = []
            for candidate in selected:
                _, _, length, _, _, (x1, y1, x2, y2) = candidate
                points.extend(((y1, x1), (y2, x2)))
                weights.extend((length, length))
                selected_y.extend((y1, y2))
            design = np.column_stack((np.asarray(points)[:, 0],
                                      np.ones(len(points))))
            weighted_design = design * np.sqrt(weights)[:, None]
            weighted_x = np.asarray(points)[:, 1] * np.sqrt(weights)
            slope_xy, intercept_xy = np.linalg.lstsq(weighted_design, weighted_x, rcond=None)[0]
            # x_norm = B * y_norm + C
            fits[side] = np.array([0.0, slope_xy * height / width,
                                  intercept_xy / width], dtype=np.float64)
            # Keep the observed upper extent, then project the boundary down to
            # the image bottom so both lane edges reach the vehicle area.
            line_ranges[side] = (max(0.0, min(selected_y) / height), 1.0)
    return {**fits, "edges": edges,
            "segments": [] if segments is None else segments[:, 0, :].tolist(),
            "cluster_labels": labels_by_side, "line_ranges": line_ranges}


def _manual_kmeans(features: np.ndarray, cluster_count: int) -> np.ndarray | None:
    """Cluster Hough segments with deterministic Lloyd iterations, no ML library."""
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or cluster_count < 1 or len(values) < cluster_count:
        return None
    if not np.isfinite(values).all():
        return None
    if cluster_count == 1:
        return np.zeros(len(values), dtype=np.int64)
    scale = values.std(axis=0)
    scale[scale == 0] = 1.0
    normalized = (values - values.mean(axis=0)) / scale

    # Farthest-point initialization avoids random starts and uses only input data.
    delta = normalized[:, None, :] - normalized[None, :, :]
    distance_squared = np.sum(delta * delta, axis=2)
    first, second = np.unravel_index(np.argmax(distance_squared), distance_squared.shape)
    if distance_squared[first, second] == 0:
        return None
    centers = [normalized[first], normalized[second]]
    for _ in range(2, cluster_count):
        nearest = np.min(np.sum((normalized[:, None, :] - np.asarray(centers)[None, :, :]) ** 2,
                                axis=2), axis=1)
        centers.append(normalized[int(np.argmax(nearest))])
    centers = np.asarray(centers)

    previous_labels = None
    seen_partitions = set()
    while True:
        distances = np.sum((normalized[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = np.argmin(distances, axis=1)
        if previous_labels is not None and np.array_equal(labels, previous_labels):
            return labels
        partition = labels.tobytes()
        if partition in seen_partitions:
            return None
        seen_partitions.add(partition)
        updated = centers.copy()
        for cluster_id in range(cluster_count):
            members = normalized[labels == cluster_id]
            if not len(members):
                return None
            updated[cluster_id] = members.mean(axis=0)
        previous_labels = labels
        centers = updated
