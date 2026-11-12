"""Per-segment paint test: sample a perpendicular cross-section of the road.

A Hough fragment can follow a tar seam, an oil streak, a water rut or the
specular reflection of a red taillight instead of a painted marking. Each
fragment is therefore accepted only when the sampled profile is brighter than
the surrounding asphalt (paint) and is not a dark trough (oil or water trench).

The test is evaluated for every fragment at once: the arithmetic is the same as
a per-fragment loop, but one call samples all cross-sections instead of running
27 scalar NumPy calls per fragment.
"""

from __future__ import annotations

import numpy as np

# Sample positions across the marking, in pixels at the 1280-px reference width.
CROSS_SECTION_OFFSETS_PX = (-12.0, -8.0, -4.0, -2.0, 0.0, 2.0, 4.0, 8.0, 12.0)
REFERENCE_WIDTH_PX = 1280.0

# Fractions along the fragment where a cross-section is sampled.
CROSS_SECTION_ALPHAS = (0.25, 0.5, 0.75)
CENTER_SAMPLE_INDEX = 4
LEFT_SAMPLE_SLICE = slice(0, 3)
RIGHT_SAMPLE_SLICE = slice(6, 9)

# Red taillight reflection probe.
RED_SAMPLE_COUNT = 6
RED_MIN_LEVEL = 70.0
RED_GREEN_RATIO = 1.4
RED_BLUE_RATIO = 1.5

# Dark trough probe (paint sits above the asphalt; oil and water sit below it).
TROUGH_DEPTH = 6.0
MIN_TROUGH_SAMPLES = 2

# Contrast required of a painted marking, per asphalt brightness band.
BRIGHT_MARGIN_BRIGHT_ROAD = 10.0
BRIGHT_MARGIN_MID_ROAD = 7.0
BRIGHT_MARGIN_DARK_ROAD = 4.0
BRIGHT_ROAD_MEDIAN = 80.0
MID_ROAD_MEDIAN = 40.0

# A dark dip this deep with no bright paint is a crack, not a marking.
CRACK_MIN_DIP = 20.0
CRACK_MAX_BRIGHTNESS = 18.0
DEPRESSION_MIN_DIP = 25.0
DEPRESSION_BRIGHTNESS_RATIO = 2.5

MIN_SEGMENT_LENGTH = 1e-3
UNMEASURABLE_ROAD_MEDIAN = 5.0


def valid_segment_mask(segments: np.ndarray, gray: np.ndarray, image: np.ndarray,
                       road_median: float, width: int, height: int) -> np.ndarray:
    """Return a boolean mask marking the fragments that look like painted lane lines."""
    coords = np.asarray(segments, dtype=np.float64).reshape(-1, 4)
    count = len(coords)
    if count == 0:
        return np.zeros(0, dtype=bool)
    if road_median <= UNMEASURABLE_ROAD_MEDIAN:
        return np.ones(count, dtype=bool)

    x1, y1, x2, y2 = (coords[:, index] for index in range(4))
    dx, dy = x2 - x1, y2 - y1
    length = np.hypot(dx, dy)
    with np.errstate(divide="ignore", invalid="ignore"):
        safe_length = np.where(length < MIN_SEGMENT_LENGTH, 1.0, length)
        normal_x, normal_y = -dy / safe_length, dx / safe_length

    profile = _sample_profile(gray, x1, y1, dx, dy, normal_x, normal_y, width, height)
    keep = length >= MIN_SEGMENT_LENGTH
    keep = keep & ~_red_reflection_mask(coords, image, width, height)
    keep = keep & ~_dark_trough_mask(profile)
    keep = keep & _paint_contrast_mask(profile, road_median)
    return keep


def is_valid_segment(x1: float, y1: float, x2: float, y2: float, gray: np.ndarray,
                     image: np.ndarray, road_median: float,
                     width: int, height: int) -> bool:
    """Single-fragment form of :func:`valid_segment_mask`."""
    segment = np.asarray([[x1, y1, x2, y2]], dtype=np.float64)
    return bool(valid_segment_mask(segment, gray, image, road_median, width, height)[0])


def _red_reflection_mask(coords: np.ndarray, image: np.ndarray,
                         width: int, height: int) -> np.ndarray:
    """Fragments whose colour is dominated by red (wet-road taillight glare)."""
    count = len(coords)
    if image.ndim != 3 or image.shape[2] != 3:
        return np.zeros(count, dtype=bool)
    x1, y1, x2, y2 = (coords[:, index] for index in range(4))
    steps = RED_SAMPLE_COUNT - 1
    column = x1[:, None] + np.arange(RED_SAMPLE_COUNT)[None, :] * ((x2 - x1) / steps)[:, None]
    row = y1[:, None] + np.arange(RED_SAMPLE_COUNT)[None, :] * ((y2 - y1) / steps)[:, None]
    column[:, -1] = x2
    row[:, -1] = y2
    columns = np.clip(column.astype(np.int64), 0, width - 1)
    rows = np.clip(row.astype(np.int64), 0, height - 1)
    red = image[rows, columns, 2].mean(axis=1)
    green = image[rows, columns, 1].mean(axis=1)
    blue = image[rows, columns, 0].mean(axis=1)
    return (red > RED_MIN_LEVEL) & (red > RED_GREEN_RATIO * green) & (red > RED_BLUE_RATIO * blue)


def _sample_profile(gray: np.ndarray, x1: np.ndarray, y1: np.ndarray, dx: np.ndarray,
                    dy: np.ndarray, normal_x: np.ndarray, normal_y: np.ndarray,
                    width: int, height: int) -> np.ndarray:
    """Brightness profile of every fragment's cross-section, shape (n, alphas, offsets)."""
    scale = max(1.0, width / REFERENCE_WIDTH_PX)
    offsets = np.asarray(CROSS_SECTION_OFFSETS_PX, dtype=np.float64) * scale
    alphas = np.asarray(CROSS_SECTION_ALPHAS, dtype=np.float64)

    center_x = x1[:, None] + alphas[None, :] * dx[:, None]
    center_y = y1[:, None] + alphas[None, :] * dy[:, None]
    sample_x = center_x[:, :, None] + offsets[None, None, :] * normal_x[:, None, None]
    sample_y = center_y[:, :, None] + offsets[None, None, :] * normal_y[:, None, None]

    columns = np.clip(np.rint(sample_x).astype(np.int64), 0, width - 1)
    rows = np.clip(np.rint(sample_y).astype(np.int64), 0, height - 1)
    return gray[rows, columns].astype(np.float64)


def _dark_trough_mask(profile: np.ndarray) -> np.ndarray:
    """Fragments whose centre is darker than both flanks (oil leak, water trench)."""
    center = profile[:, :, CENTER_SAMPLE_INDEX]
    left = profile[:, :, LEFT_SAMPLE_SLICE].min(axis=2)
    right = profile[:, :, RIGHT_SAMPLE_SLICE].min(axis=2)
    trough = (center < left - TROUGH_DEPTH) & (center < right - TROUGH_DEPTH)
    return trough.sum(axis=1) >= MIN_TROUGH_SAMPLES


def _paint_contrast_mask(profile: np.ndarray, road_median: float) -> np.ndarray:
    """Fragments that rise above the asphalt like paint instead of dipping below it."""
    contrast_up = np.median(profile.max(axis=2) - road_median, axis=1)
    contrast_down = np.median(road_median - profile.min(axis=2), axis=1)

    if road_median > BRIGHT_ROAD_MEDIAN:
        margin = BRIGHT_MARGIN_BRIGHT_ROAD
    elif road_median > MID_ROAD_MEDIAN:
        margin = BRIGHT_MARGIN_MID_ROAD
    else:
        margin = BRIGHT_MARGIN_DARK_ROAD

    keep = contrast_up >= margin
    keep = keep & ~((contrast_down > CRACK_MIN_DIP) & (contrast_up < CRACK_MAX_BRIGHTNESS))
    keep = keep & ~((contrast_down > DEPRESSION_MIN_DIP) &
                    (contrast_down > DEPRESSION_BRIGHTNESS_RATIO * np.maximum(contrast_up, 1.0)))
    return keep
