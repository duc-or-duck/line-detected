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
        expected_width = config.get("expected_lane_width_px")
        left_base, right_base = _ego_lane_histogram_pair(histogram, width, expected_width)
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


def _ego_lane_histogram_pair(histogram: np.ndarray, width: int,
                             expected_width: float | None) -> tuple[int, int]:
    """Choose supported histogram peaks that surround the camera centre."""
    midpoint = width / 2.0
    peaks = np.flatnonzero((histogram > 0) &
                           (histogram >= np.r_[histogram[0], histogram[:-1]]) &
                           (histogram >= np.r_[histogram[1:], histogram[-1]]))
    lefts, rights = peaks[peaks < midpoint], peaks[peaks > midpoint]
    if not len(lefts) or not len(rights):
        return (int(np.argmax(histogram[:int(midpoint)])) if midpoint else 0,
                int(np.argmax(histogram[int(midpoint):]) + int(midpoint)) if midpoint < width else width - 1)
    target = float(expected_width) if expected_width is not None else .32 * width
    choices = []
    for left in lefts:
        for right in rights:
            lane_width = right - left
            if not .12 * width <= lane_width <= .60 * width:
                continue
            centre_penalty = abs((left + right) / 2 - midpoint) / width
            width_penalty = abs(lane_width - target) / width
            support = histogram[left] + histogram[right]
            choices.append((centre_penalty + width_penalty - .002 * support, int(left), int(right)))
    if not choices:
        return int(lefts[np.argmax(histogram[lefts])]), int(rights[np.argmax(histogram[rights])])
    _, left, right = min(choices)
    return left, right


def validate_and_recover_lanes(left: np.ndarray | None, right: np.ndarray | None,
                               width: int, height: int, config: dict
                               ) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
    """Validate normalized lane fits and recover one missing side if calibrated."""
    y_top = config.get("valid_y_top_norm")
    y_bottom = config.get("valid_y_bottom_norm")
    y_top = 0.0 if y_top is None else float(y_top)
    y_bottom = 1.0 if y_bottom is None else float(y_bottom)
    rows = np.linspace(y_top, y_bottom, 10)
    x_margin = float(config.get("valid_x_margin_norm", 0.0))

    def in_bounds(fit):
        if fit is None:
            return False
        values = np.polyval(np.asarray(fit, dtype=np.float64), rows)
        return bool(np.isfinite(values).all() and
                    np.all((values >= x_margin) & (values <= 1.0 - x_margin)))

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


def fit_support_overlaps(diagnostics: dict, window_count: int,
                         minimum_windows: float) -> bool:
    """Require both fits to be supported over a shared vertical span."""
    left_min, left_max = diagnostics.get("y_min_left"), diagnostics.get("y_max_left")
    right_min, right_max = diagnostics.get("y_min_right"), diagnostics.get("y_max_right")
    if any(value is None for value in (left_min, left_max, right_min, right_max)):
        return False
    shared_span = min(float(left_max), float(right_max)) - max(float(left_min), float(right_min))
    return shared_span >= float(minimum_windows) / max(int(window_count), 1)


def constrain_lane_pair(left: np.ndarray | None, right: np.ndarray | None,
                        line_ranges: dict | None, width: int,
                        min_width_norm: float = 0.05,
                        max_width_norm: float = 0.95) -> tuple[float, float] | None:
    """Return the shared supported y-span where lane boundaries stay ordered."""
    if left is None or right is None:
        return None
    ranges = line_ranges or {}
    left_range = ranges.get("left") or (0.0, 1.0)
    right_range = ranges.get("right") or (0.0, 1.0)
    low = max(float(left_range[0]), float(right_range[0]))
    high = min(float(left_range[1]), float(right_range[1]))
    if high <= low:
        return None
    ys = np.linspace(low, high, 256)
    gap = (np.polyval(right, ys) - np.polyval(left, ys))
    valid = np.isfinite(gap) & (gap >= min_width_norm) & (gap <= max_width_norm)
    indices = np.flatnonzero(valid)
    if not len(indices):
        return None
    runs = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    longest = max(runs, key=len)
    if len(longest) < 5:
        return None
    return float(ys[longest[0]]), float(ys[longest[-1]])


def output_geometry_is_valid(left: np.ndarray | None, right: np.ndarray | None,
                             bounds: dict | None, width: int, config: dict) -> bool:
    """Reject extrapolated hooks and implausible lane pairs after unwarping."""
    minimum = config.get("min_lane_width_norm")
    maximum = config.get("max_lane_width_norm")
    safe = constrain_lane_pair(left, right, bounds, width,
                               .05 if minimum is None else float(minimum),
                               .95 if maximum is None else float(maximum))
    if safe is None:
        return False
    low, high = safe
    if high - low < float(config.get("minimum_support_norm", .15)):
        return False
    ys = np.linspace(low, high, 64)
    lx, rx = np.polyval(left, ys), np.polyval(right, ys)
    lane_width = rx - lx
    if not (np.isfinite(lx).all() and np.isfinite(rx).all() and np.all(lane_width > 0)):
        return False
    if config.get("require_ego_containment", False):
        # At the closest observed row the ego lane must surround the optical
        # centre.  A small margin permits ordinary lateral offset while
        # rejecting a pair made from adjacent centre-line strokes.
        margin = float(config.get("ego_containment_margin", .10)) * lane_width[-1]
        if not lx[-1] - margin <= .5 <= rx[-1] + margin:
            return False
    chord = lambda values: np.linspace(values[0], values[-1], len(values))
    bend = max(float(np.max(np.abs(lx - chord(lx)))), float(np.max(np.abs(rx - chord(rx)))))
    limit = float(config.get("max_curve_deviation_lane_fraction", .25)) * float(np.median(lane_width))
    # Perspective naturally makes the apparent width change in the original
    # camera image. Width consistency is checked in BEV before unwarping;
    # rejecting it here would discard every valid perspective lane.
    return bend <= limit


def _fit(y: np.ndarray, x: np.ndarray, min_points: int) -> np.ndarray | None:
    if len(x) < max(3, min_points):
        return None
    # A quadratic fit is unstable when sliding windows found pixels in only
    # one or two rows (common with glare, road seams, and sparse night markings).
    # Reject near-horizontal support; use a line until pixels span enough of
    # the image to justify curvature.
    y_span = float(np.ptp(y))
    if np.unique(y).size < 2 or y_span < 0.015:
        return None
    if y_span < 0.25 or np.unique(y).size < 3:
        degree = 1
    else:
        degree = 2
    inliers = np.ones(len(x), dtype=bool)
    fit = None
    for _ in range(3):
        if int(inliers.sum()) < max(3, min_points):
            return None
        coefficients = np.polyfit(y[inliers], x[inliers], degree)
        fit = coefficients if degree == 2 else np.asarray([0.0, *coefficients])
        residuals = np.abs(x - np.polyval(fit, y))
        center = float(np.median(residuals[inliers]))
        mad = float(np.median(np.abs(residuals[inliers] - center)))
        cutoff = max(center + 3.5 * 1.4826 * mad, 0.008)
        updated = residuals <= cutoff
        if np.array_equal(updated, inliers):
            break
        inliers = updated
    return fit if np.isfinite(fit).all() else None
