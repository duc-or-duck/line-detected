"""Synthetic BEV lanes warped into a camera frame with known ground truth."""
from __future__ import annotations
import copy
from pathlib import Path
import cv2
import numpy as np
from src.pipeline import LanePipeline, load_config
from src.warp import unwarp_points


def scene(kind="straight", brightness=1.0, dashed=False, size=(720, 1280)):
    h, w = size; bev = np.full((h, w, 3), 90, np.uint8)
    ys = np.arange(70, h - 10, 2, dtype=np.float32)
    bend = (ys / h - .55) ** 2 * .08 * w if kind == "curve" else np.zeros_like(ys)
    left, right = .30 * w + bend, .70 * w + bend
    for xs in (left, right):
        for start in range(0, len(ys), 20 if dashed else len(ys)):
            end = min(start + (12 if dashed else len(ys)), len(ys))
            cv2.polylines(bev, [np.int32(np.column_stack((xs[start:end], ys[start:end])))], False, (230, 230, 230), 10)
    src = np.float32([[.43*w,.38*h],[.57*w,.38*h],[.12*w,.96*h],[.88*w,.96*h]])
    dst = np.float32([[.30*w,.10*h],[.70*w,.10*h],[.30*w,.95*h],[.70*w,.95*h]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    frame = cv2.warpPerspective(bev, np.linalg.inv(matrix), (w, h))
    frame = np.clip(frame.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
    truth = unwarp_points(np.column_stack((np.r_[left, right], np.r_[ys, ys])), matrix)
    return frame, src.tolist(), dst.tolist(), truth


def run(kind="straight", brightness=1.0, dashed=False):
    frame, src, dst, truth = scene(kind, brightness, dashed)
    cfg = copy.deepcopy(load_config(Path("configs/C.yaml")))
    cfg["homography"]["day"] = {"source": src, "destination": dst}
    cfg["preprocess"]["roi_polygon_by_source"] = {"day": [[0, 0], [1279, 0], [1279, 719], [0, 719]]}
    cfg["preprocess"]["white_l_min"] = 120 if brightness < .2 else 180
    cfg["fit"].update({"min_lane_width_norm": .2, "max_lane_width_norm": .6,
                       "parallel_tolerance": .25, "max_abs_quadratic": .12,
                       "fit_y_margin_norm": .02})
    result = LanePipeline(cfg, "day").process(frame, "proposed")
    return result, truth
