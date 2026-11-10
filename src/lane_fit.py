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
    missing_windows = {"left": 0, "right": 0}
    window_boxes = {"left": [], "right": []}
    if previous is not None and margin is not None:
        left_center = np.polyval(previous[0], normalized_y)
        right_center = np.polyval(previous[1], normalized_y)
        left_mask = np.abs(normalized_x - left_center) <= margin / max(width - 1, 1)
        right_mask = np.abs(normalized_x - right_center) <= margin / max(width - 1, 1)
    else:
        histogram = np.sum(image[height // 2:, :] > 0, axis=0)
        midpoint = width // 2
        expected_width = config.get("expected_lane_width_px")
        if expected_width is not None and 0 < float(expected_width) < width:
            left_base = _nearest_histogram_peak(histogram, 0, midpoint,
                                                (width - 1 - float(expected_width)) / 2.0)
            right_base = _nearest_histogram_peak(histogram, midpoint, width,
                                                 (width - 1 + float(expected_width)) / 2.0)
        else:
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
            window_boxes["left"].append((left_current - margin, y_low, left_current + margin, y_high))
            window_boxes["right"].append((right_current - margin, y_low, right_current + margin, y_high))
            minpix = int(config.get("minpix", config.get("recenter_min_pixels") or 0))
            if int(left_here.sum()) < minpix:
                missing_windows["left"] += 1
            if int(right_here.sum()) < minpix:
                missing_windows["right"] += 1
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
    diagnostics = {"left_count": int(left_mask.sum()), "right_count": int(right_mask.sum()),
                   "missing_windows_left": missing_windows.get("left", 0),
                   "missing_windows_right": missing_windows.get("right", 0),
                   "window_boxes": window_boxes}
    for side, mask in (("left", left_mask), ("right", right_mask)):
        if np.any(mask):
            side_y = normalized_y[mask]
            diagnostics[f"y_min_{side}"] = float(side_y.min())
            diagnostics[f"y_max_{side}"] = float(side_y.max())
        else:
            diagnostics[f"y_min_{side}"] = None
            diagnostics[f"y_max_{side}"] = None
    return left_fit, right_fit, diagnostics


def _nearest_histogram_peak(histogram: np.ndarray, start: int, stop: int,
                            target: float) -> int:
    section = histogram[start:stop]
    if not len(section):
        return start
    padded = np.pad(section, (1, 1), mode="edge")
    local = np.flatnonzero((padded[1:-1] >= padded[:-2]) &
                           (padded[1:-1] >= padded[2:]) & (section > 0)) + start
    if not len(local):
        return int(np.argmax(section) + start)
    return int(min(local, key=lambda index: (abs(index - target), -histogram[index])))


def validate_and_recover_lanes(left: np.ndarray | None, right: np.ndarray | None,
                               width: int, height: int, config: dict
                               ) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
    """Validate normalized lane fits and recover one missing side if calibrated."""
    y_top = config.get("valid_y_top_norm")
    y_bottom = config.get("valid_y_bottom_norm")
    y_top = 0.0 if y_top is None else float(y_top)
    y_bottom = 1.0 if y_bottom is None else float(y_bottom)
    rows = np.linspace(y_top, y_bottom, 10)

    def in_bounds(fit):
        if fit is None:
            return False
        values = np.polyval(np.asarray(fit, dtype=np.float64), rows)
        return bool(np.isfinite(values).all() and np.all((values >= 0) & (values <= 1)))

    valid_left, valid_right = in_bounds(left), in_bounds(right)
    left = np.asarray(left, dtype=np.float64) if valid_left else None
    right = np.asarray(right, dtype=np.float64) if valid_right else None
    single_line = False
    expected_px = config.get("expected_lane_width_px")
    if (left is None) != (right is None) and expected_px is not None:
        delta = float(expected_px) / max(width - 1, 1)
        if left is not None:
            right = left.copy()
            right[-1] += delta
        else:
            left = right.copy()
            left[-1] -= delta
        single_line = True
    flags = {"valid": False, "valid_left": valid_left, "valid_right": valid_right,
             "single_line": single_line}
    if left is None or right is None or not in_bounds(left) or not in_bounds(right):
        return None, None, flags
    x_left = np.polyval(left, rows) * max(width - 1, 1)
    x_right = np.polyval(right, rows) * max(width - 1, 1)
    lane_width = x_right - x_left
    valid = (np.isfinite(x_left).all() and np.isfinite(x_right).all() and
             np.all(lane_width > 0))
    bottom_width, top_width = float(lane_width[-1]), float(lane_width[0])
    minimum, maximum = config.get("lane_width_min_px"), config.get("lane_width_max_px")
    if minimum is not None:
        valid = valid and bottom_width >= float(minimum) and top_width >= float(minimum)
    if maximum is not None:
        valid = valid and bottom_width <= float(maximum) and top_width <= float(maximum)
    tolerance = config.get("parallel_tolerance")
    if tolerance is not None:
        valid = valid and abs(top_width - bottom_width) / max(bottom_width, 1e-9) < float(tolerance)
    max_curve = config.get("max_abs_quadratic")
    if max_curve is not None:
        valid = valid and abs(float(left[0])) <= float(max_curve) and abs(float(right[0])) <= float(max_curve)
    flags["valid"] = bool(valid)
    if not valid:
        return None, None, flags
    return left, right, flags


def _fit(y: np.ndarray, x: np.ndarray, min_points: int) -> np.ndarray | None:
    if len(x) < max(3, min_points):
        return None
    fit = np.polyfit(y, x, 2)
    return fit if np.isfinite(fit).all() else None
