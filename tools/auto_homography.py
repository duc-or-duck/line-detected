"""Calibrate a per-video BEV homography from confident baseline lane pairs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.baseline import detect_baseline
from src.pipeline import load_config


def _line(fit, y, width, height):
    return float(np.polyval(fit, y / (height - 1)) * (width - 1))


def _quality(source, destination, candidates, y_top, y_bottom):
    destination = np.asarray(destination, dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(np.float32(source), np.float32(destination))
    angles, widths, centers = [], [], []
    for left, right, h, w in candidates:
        points = np.float32([[_line(left, y_top, w, h), y_top], [_line(right, y_top, w, h), y_top],
                             [_line(left, y_bottom, w, h), y_bottom], [_line(right, y_bottom, w, h), y_bottom]])
        warped = cv2.perspectiveTransform(points.reshape(-1, 1, 2), matrix).reshape(-1, 2)
        angles += [abs(np.degrees(np.arctan2(warped[2, 0] - warped[0, 0], warped[2, 1] - warped[0, 1]))),
                   abs(np.degrees(np.arctan2(warped[3, 0] - warped[1, 0], warped[3, 1] - warped[1, 1])))]
        near, far = warped[3, 0] - warped[2, 0], warped[1, 0] - warped[0, 0]
        widths.append(100 * abs(near - far) / max(abs(near), 1))
        centers.append(100 * abs((warped[:, 0].mean() - destination[:, 0].mean()) /
                                 (destination[1, 0] - destination[0, 0])))
    return matrix, tuple(float(np.median(values)) for values in (angles, widths, centers))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--data-source", default="day", choices=("day", "night", "rain"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "A.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "configs" / "homography_by_video.yaml")
    args = parser.parse_args()
    cap = cv2.VideoCapture(str(args.video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened() or total < 2:
        raise SystemExit("Cannot open video")
    cfg = load_config(args.config)
    prep = cfg.get("preprocess", {})
    roi = prep.get("roi_polygon_by_source", {}).get(args.data_source, prep.get("roi_polygon"))
    ref = prep.get("roi_reference_size", [1280, 720])
    candidates = []
    for index in np.linspace(0, total - 1, min(total, 60), dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index)); ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        scaled = [[round(x * w / ref[0]), round(y * h / ref[1])] for x, y in roi] if roi else None
        found = detect_baseline(frame, prep | {"roi_polygon": scaled, "data_source": args.data_source})
        if found["left"] is not None and found["right"] is not None:
            candidates.append((found["left"], found["right"], h, w))
    cap.release()
    if len(candidates) < 8:
        print(f"{args.video.name}: NEEDS_MANUAL_CLICK (only {len(candidates)} confident lane pairs)")
        return
    h, w = candidates[0][2:]
    left, right = (np.median([row[i] for row in candidates], axis=0) for i in (0, 1))
    vp_den = left[1] - right[1]
    if abs(vp_den) < 1e-6:
        print(f"{args.video.name}: NEEDS_MANUAL_CLICK (parallel fits have no stable vanishing point)")
        return
    vp_y = float((right[2] - left[2]) / vp_den * (h - 1))
    y_bottom = max(point[1] for point in roi) * h / ref[1] if roi else h - 1
    destination = [[.30 * w, .10 * h], [.70 * w, .10 * h], [.30 * w, .95 * h], [.70 * w, .95 * h]]
    choices = []
    for k in (.5, .6, .7):
        y_top = vp_y + k * (y_bottom - vp_y)
        if not (0 <= y_top < y_bottom < h):
            continue
        source = [[_line(left, y_top, w, h), y_top], [_line(right, y_top, w, h), y_top],
                  [_line(left, y_bottom, w, h), y_bottom], [_line(right, y_bottom, w, h), y_bottom]]
        matrix, metrics = _quality(source, destination, candidates, y_top, y_bottom)
        if np.isfinite(np.asarray(source)).all():
            choices.append((sum(metrics), k, source, metrics))
    if not choices:
        print(f"{args.video.name}: NEEDS_MANUAL_CLICK (invalid trapezoid)")
        return
    _, k, source, (angle, width_error, center_error) = min(choices)
    if angle >= 2 or width_error >= 5 or center_error >= 10:
        print(f"{args.video.name}: NEEDS_MANUAL_CLICK (angle={angle:.2f}, width={width_error:.2f}%, center={center_error:.2f}%)")
        return
    data = yaml.safe_load(args.output.read_text(encoding="utf-8")) if args.output.exists() else {}
    coverage = (source[2][1] - source[0][1]) / h
    if coverage < .25:
        print(f"{args.video.name}: NEEDS_MANUAL_CLICK (source vertical coverage={coverage:.2f})")
        return
    data[args.video.name] = {"accepted": True, "source": [[round(x, 2), round(y, 2)] for x, y in source], "destination": destination,
                             "bev_size": [w, h], "k": k, "y_vp": round(vp_y, 2),
                             "vertical_angle_deg": round(angle, 3), "lane_width_error_pct": round(width_error, 3),
                             "center_error_pct": round(center_error, 3)}
    args.output.write_text(yaml.safe_dump(data, sort_keys=True), encoding="utf-8")
    print(f"{args.video.name}: CALIBRATED k={k}, angle={angle:.2f}, width={width_error:.2f}%, center={center_error:.2f}%")


if __name__ == "__main__":
    main()
