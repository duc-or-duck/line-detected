"""Sliding-window lane pixel collection and normalized quadratic fitting."""

from __future__ import annotations

import numpy as np


def sliding_window_fit(binary: np.ndarray, config: dict,
                       previous: tuple[np.ndarray, np.ndarray] | None = None
                       ) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
    image = np.asarray(binary)
    if image.ndim != 2:
        raise ValueError("binary lane image must be 2D")
    height, width = image.shape
    ys, xs = np.nonzero(image)
    if not len(xs):
        return None, None, {"left_count": 0, "right_count": 0}
    normalized_y, normalized_x = ys / max(height - 1, 1), xs / max(width - 1, 1)
    min_points = config.get("min_fit_points")
    margin = config.get("search_margin")
    if previous is not None and margin is not None:
        left_center = np.polyval(previous[0], normalized_y)
        right_center = np.polyval(previous[1], normalized_y)
        left_mask = np.abs(normalized_x - left_center) <= margin / max(width - 1, 1)
        right_mask = np.abs(normalized_x - right_center) <= margin / max(width - 1, 1)
    else:
        histogram = np.sum(image[height // 2:, :] > 0, axis=0)
        midpoint = width // 2
        left_base = int(np.argmax(histogram[:midpoint])) if midpoint else 0
        right_base = int(np.argmax(histogram[midpoint:]) + midpoint) if midpoint < width else width - 1
        scale = config.get("window_count")
        margin = config.get("window_margin")
        if scale is None or margin is None:
            return None, None, {"left_count": 0, "right_count": 0}
        window_count = int(scale)
        left_mask = np.zeros(xs.shape, dtype=bool)
        right_mask = np.zeros(xs.shape, dtype=bool)
        left_current, right_current = left_base, right_base
        for window in range(window_count):
            y_low = height - (window + 1) * height // window_count
            y_high = height - window * height // window_count
            in_window = (ys >= y_low) & (ys < y_high)
            left_here = in_window & (xs >= left_current - margin) & (xs < left_current + margin)
            right_here = in_window & (xs >= right_current - margin) & (xs < right_current + margin)
            left_mask |= left_here
            right_mask |= right_here
            recenter = config.get("recenter_min_pixels")
            if recenter is not None and int(left_here.sum()) >= recenter:
                left_current = int(xs[left_here].mean())
            if recenter is not None and int(right_here.sum()) >= recenter:
                right_current = int(xs[right_here].mean())
    min_points = 0 if min_points is None else int(min_points)
    left_fit = _fit(normalized_y[left_mask], normalized_x[left_mask], min_points)
    right_fit = _fit(normalized_y[right_mask], normalized_x[right_mask], min_points)
    return left_fit, right_fit, {"left_count": int(left_mask.sum()), "right_count": int(right_mask.sum())}


def _fit(y: np.ndarray, x: np.ndarray, min_points: int) -> np.ndarray | None:
    if len(x) < max(3, min_points):
        return None
    fit = np.polyfit(y, x, 2)
    return fit if np.isfinite(fit).all() else None
