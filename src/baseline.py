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
    line_features = []
    line_parameters = []
    min_abs_slope = config.get("min_abs_slope")
    for x1, y1, x2, y2 in ([] if segments is None else segments[:, 0, :]):
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if slope == 0:
            continue
        if min_abs_slope is not None and abs(slope) < float(min_abs_slope):
            continue
        intercept = y1 - slope * x1
        midpoint_x = (x1 + x2) / 2.0
        line_features.append((slope, intercept / max(height, 1), midpoint_x / max(width, 1)))
        line_parameters.append((slope, intercept))

    kmeans_config = config.get("kmeans", {})
    cluster_count = kmeans_config.get("clusters")
    labels = None
    if cluster_count is not None and len(line_features) >= int(cluster_count):
        labels = _manual_kmeans(np.asarray(line_features, dtype=np.float64), int(cluster_count))

    fits = {"left": None, "right": None}
    selected_counts = {"left": 0, "right": 0}
    if labels is not None:
        for cluster_id in range(int(cluster_count)):
            members = [line_parameters[index] for index, label in enumerate(labels)
                       if label == cluster_id]
            if not members:
                continue
            slope, intercept = np.mean(members, axis=0)
            side = "left" if slope < 0 else "right"
            # If K-means assigns both groups to one side, keep its larger group.
            member_count = len(members)
            if fits[side] is not None and member_count <= selected_counts[side]:
                continue
            # x_norm = B * y_norm + C
            fits[side] = np.array([0.0, (1.0 / slope) * height / width,
                                  (-intercept / slope) / width], dtype=np.float64)
            selected_counts[side] = member_count
    return {**fits, "edges": edges,
            "segments": [] if segments is None else segments[:, 0, :].tolist(),
            "cluster_labels": None if labels is None else labels.tolist()}


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
