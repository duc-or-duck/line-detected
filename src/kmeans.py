"""Self-implemented K-means (Lloyd iterations) used to group Hough segments.

NumPy only: no scikit-learn, no random restarts. Initialisation is
farthest-point (max-min), so the same segment set always yields the same
labels. This is the algorithm documented for the baseline: Hough fragments
are clustered by where they cross the reference row, then the cluster that
belongs to the ego lane is fitted.
"""

from __future__ import annotations

import numpy as np

MAX_ITERATIONS = 100
MIN_SEGMENT_LENGTH = 1e-3


def kmeans_labels(features: np.ndarray, cluster_count: int) -> np.ndarray | None:
    """Assign each row of *features* to one of *cluster_count* clusters.

    Returns an ``int64`` label per row, or ``None`` when the data cannot be
    clustered (empty input, non-finite values, fewer points than clusters, an
    empty cluster, or a repeated partition instead of convergence).
    """
    values = np.asarray(features, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or len(values) == 0:
        return None
    if cluster_count < 1 or len(values) < cluster_count:
        return None
    if not np.isfinite(values).all():
        return None
    if cluster_count == 1:
        return np.zeros(len(values), dtype=np.int64)

    normalized = _standardize(values)
    centers = _farthest_point_centers(normalized, cluster_count)
    if centers is None:
        return None
    return _lloyd(normalized, centers)


def _standardize(values: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance per feature so every dimension weighs the same."""
    scale = values.std(axis=0)
    scale[scale == 0] = 1.0
    return (values - values.mean(axis=0)) / scale


def _farthest_point_centers(normalized: np.ndarray, cluster_count: int) -> np.ndarray | None:
    """Deterministic max-min initialisation: start from the widest pair, then
    keep adding the point furthest from every centre chosen so far."""
    distances = _squared_distances(normalized, normalized)
    first, second = np.unravel_index(int(np.argmax(distances)), distances.shape)
    if distances[first, second] <= 0:
        return None
    centers = [normalized[first], normalized[second]]
    for _ in range(2, cluster_count):
        nearest = _squared_distances(normalized, np.asarray(centers)).min(axis=1)
        centers.append(normalized[int(np.argmax(nearest))])
    return np.asarray(centers, dtype=np.float64)


def _lloyd(normalized: np.ndarray, centers: np.ndarray) -> np.ndarray | None:
    """Assign / update until the partition stops changing."""
    previous_labels = None
    seen_partitions: set[bytes] = set()
    for _ in range(MAX_ITERATIONS):
        labels = np.argmin(_squared_distances(normalized, centers), axis=1)
        if previous_labels is not None and np.array_equal(labels, previous_labels):
            return labels
        partition = labels.tobytes()
        if partition in seen_partitions:
            return None
        seen_partitions.add(partition)
        updated = centers.copy()
        for cluster_id in range(len(centers)):
            members = normalized[labels == cluster_id]
            if not len(members):
                return None
            updated[cluster_id] = members.mean(axis=0)
        previous_labels, centers = labels, updated
    return None


def _squared_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Pairwise squared Euclidean distance matrix between two point sets."""
    delta = left[:, None, :] - right[None, :, :]
    return np.sum(delta * delta, axis=2)
