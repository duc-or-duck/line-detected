"""Unit tests for optimized baseline (A_opt) enhancements:
- Synthetic dark frame with noise, faint markings, and distractor glare/blobs.
- Anti-jump gate rejecting 5-frame outlier injections.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.pipeline import LanePipeline, load_config


def create_synthetic_dark_frame(height: int = 720, width: int = 1280,
                                add_distractors: bool = True) -> tuple[np.ndarray, dict]:
    """Create a dark synthetic road frame with faint markings and distractors."""
    # Near-black asphalt background with slight texture noise
    rng = np.random.default_rng(42)
    image = rng.integers(15, 25, (height, width, 3), dtype=np.uint8)

    # True lane lines:
    # Left: from (250, 719) to (560, 432), slope = (432-719)/(560-250) = -287/310 = -0.925
    # Right: from (1030, 719) to (720, 432), slope = (432-719)/(720-1030) = -287/-310 = +0.925
    true_left_pts = ((250, 719), (560, 432))
    true_right_pts = ((1030, 719), (720, 432))

    # Draw faint white markings (gray ~ 75, low amplitude)
    cv2.line(image, true_left_pts[0], true_left_pts[1], (75, 75, 75), 4)
    cv2.line(image, true_right_pts[0], true_right_pts[1], (75, 75, 75), 4)

    if add_distractors:
        # 1. Bright near-vertical streak (glare / pole / light streak, slope ~ -10)
        cv2.line(image, (600, 719), (630, 420), (220, 220, 220), 5)
        # 2. Bright light blob (oncoming headlight reflection)
        cv2.circle(image, (640, 500), 25, (230, 230, 230), -1)

    true_sl_l = (true_left_pts[1][1] - true_left_pts[0][1]) / (true_left_pts[1][0] - true_left_pts[0][0])
    true_sl_r = (true_right_pts[1][1] - true_right_pts[0][1]) / (true_right_pts[1][0] - true_right_pts[0][0])
    return image, {
        "true_slope_left": true_sl_l,
        "true_slope_right": true_sl_r,
        "true_bottom_left_x": 250.0,
        "true_bottom_right_x": 1030.0,
    }


def test_synthetic_dark_frame_a_opt_outperforms_or_equals_a():
    """A_opt stays within tolerance of true lines and error <= A error."""
    frame, ground_truth = create_synthetic_dark_frame(720, 1280, add_distractors=True)

    config_a = load_config("configs/A.yaml")
    config_opt = load_config("configs/A_opt.yaml")

    pipe_a = LanePipeline(config_a, "night")
    pipe_opt = LanePipeline(config_opt, "night")

    res_a = pipe_a.process(frame.copy(), method="baseline")
    res_opt = pipe_opt.process(frame.copy(), method="baseline")

    # A_opt must detect valid lanes
    assert res_opt["lane_measurement_available"] is True, "A_opt failed to detect lanes in dark frame"
    assert res_opt["left"] is not None and res_opt["right"] is not None

    # Compute bottom x for A_opt
    h, w = frame.shape[:2]
    opt_bot_l = np.polyval(res_opt["left"], 1.0) * (w - 1)
    opt_bot_r = np.polyval(res_opt["right"], 1.0) * (w - 1)

    error_opt_l = abs(opt_bot_l - ground_truth["true_bottom_left_x"])
    error_opt_r = abs(opt_bot_r - ground_truth["true_bottom_right_x"])

    # A_opt lines stay within tolerance (<= 45 px) of true lines
    assert error_opt_l <= 45.0, f"A_opt left bottom error too high: {error_opt_l:.1f}px"
    assert error_opt_r <= 45.0, f"A_opt right bottom error too high: {error_opt_r:.1f}px"

    # If A detected something, compare errors (A often latches onto the vertical streak or fails)
    if res_a["lane_measurement_available"]:
        a_bot_l = np.polyval(res_a["left"], 1.0) * (w - 1)
        a_bot_r = np.polyval(res_a["right"], 1.0) * (w - 1)
        total_err_a = abs(a_bot_l - ground_truth["true_bottom_left_x"]) + abs(a_bot_r - ground_truth["true_bottom_right_x"])
        total_err_opt = error_opt_l + error_opt_r
        assert total_err_opt <= total_err_a + 5.0, f"A_opt error ({total_err_opt:.1f}) should be <= A error ({total_err_a:.1f})"


def test_jump_gate_rejects_outlier_injection_and_recovers():
    """Injected outlier lines for 5 frames do not move output more than jump_px; recovers after."""
    config_opt = load_config("configs/A_opt.yaml")
    jump_px = float(config_opt.get("preprocess", {}).get("jump_px", 80))
    pipe = LanePipeline(config_opt, "day")

    h, w = 720, 1280

    def make_frame(offset_px: float = 0.0) -> np.ndarray:
        img = np.full((h, w, 3), 120, dtype=np.uint8)
        # Normal left: (200 + offset, 719) -> (500 + offset, 380)
        # Normal right: (1080 + offset, 719) -> (780 + offset, 380)
        cv2.line(img, (int(200 + offset_px), 719), (int(500 + offset_px), 380), (240, 240, 240), 6)
        cv2.line(img, (int(1080 + offset_px), 719), (int(780 + offset_px), 380), (240, 240, 240), 6)
        return img

    # Step 1: Run 5 normal frames to establish smoothed baseline history
    for _ in range(5):
        res = pipe.process(make_frame(0.0), method="baseline")
        assert res["lane_measurement_available"] is True

    prev_bot_l = np.polyval(res["left"], 1.0) * (w - 1)

    # Step 2: Inject 5 outlier frames shifted by 180 px (> jump_px=80)
    for _ in range(5):
        outlier_frame = make_frame(180.0)
        res_outlier = pipe.process(outlier_frame, method="baseline")
        curr_bot_l = np.polyval(res_outlier["left"], 1.0) * (w - 1) if res_outlier["left"] is not None else prev_bot_l
        # The output must NOT jump by more than jump_px (it should hold the previous line!)
        assert abs(curr_bot_l - prev_bot_l) <= jump_px, (
            f"Jump gate failed to hold! Line moved by {abs(curr_bot_l - prev_bot_l):.1f} px"
        )

    # Step 3: Return to normal frames; verify recovery
    for _ in range(5):
        res_recovered = pipe.process(make_frame(0.0), method="baseline")
    assert res_recovered["lane_measurement_available"] is True
    rec_bot_l = np.polyval(res_recovered["left"], 1.0) * (w - 1)
    assert abs(rec_bot_l - prev_bot_l) <= 15.0, "Pipeline failed to recover to original position"
