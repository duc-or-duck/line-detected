"""Per-frame visual debug dump for baseline lane detection.

Saves:
- 1_enhanced_gray.png: Grayscale image after illumination enhancement (CLAHE/gamma/log/histeq).
- 2_canny_roi.png: Canny edges with trapezoidal ROI outline.
- 3_hough_segments.png: Raw Hough segments colored by accepted (green/orange) and rejected (gray).
- 4_final_lines.png: Final lane lines, departure warning banner/arrow, and offset HUD.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import LanePipeline, load_config
from src.preprocess import canny_edges, enhance_gray, roi_mask


def dump_frame(pipeline: LanePipeline, frame: np.ndarray,
               output_dir: Path, prefix: str = "frame",
               fps: float = 30.0) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    h, w = frame.shape[:2]
    config = pipeline.config
    prep = config.get("preprocess", {})
    source = pipeline.data_source or "day"

    # 1. Enhancement
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()
    enh_method = prep.get("enhancement_by_source", {}).get(source, prep.get("enhancement_method", "clahe"))
    clip_limit = prep.get("canny_by_source", {}).get(source, {}).get(
        "clahe_clip_limit", prep.get("clahe_clip_limit", 2.0))
    tile = tuple(map(int, prep.get("clahe_tile_grid", [8, 8])))
    enhanced = enhance_gray(gray, enh_method, float(clip_limit), tile)
    cv2.imwrite(str(output_dir / f"{prefix}_1_enhanced_gray.png"), enhanced)

    # 2. Canny edges with ROI outline
    roi_poly = prep.get("roi_polygon_by_source", {}).get(source, prep.get("roi_polygon"))
    ref_size = prep.get("roi_reference_size_by_source", {}).get(source, prep.get("roi_reference_size"))
    if roi_poly and ref_size:
        rw, rh = map(float, ref_size)
        scaled_roi = [[round(p[0] * w / rw), round(p[1] * h / rh)] for p in roi_poly]
    else:
        scaled_roi = roi_poly

    source_cfg = prep | {"roi_polygon": scaled_roi, "data_source": source}
    edges = canny_edges(frame, source_cfg)
    edges_masked = roi_mask(edges, scaled_roi)

    edges_vis = cv2.cvtColor(edges_masked, cv2.COLOR_GRAY2BGR)
    if scaled_roi:
        pts = np.asarray(scaled_roi, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(edges_vis, [pts], True, (0, 255, 255), 2)
    cv2.imwrite(str(output_dir / f"{prefix}_2_canny_roi.png"), edges_vis)

    # 3. Hough segments accepted vs rejected
    hparams = dict(prep.get("hough", {}))
    hparams.update(prep.get("hough_by_source", {}).get(source, {}))
    segments = None
    if all(hparams.get(k) is not None for k in ("threshold", "min_line_length", "max_line_gap")):
        segments = cv2.HoughLinesP(
            edges_masked, 1, np.pi / 180,
            int(hparams["threshold"]),
            minLineLength=int(hparams["min_line_length"]),
            maxLineGap=int(hparams["max_line_gap"]))

    segs_vis = frame.copy()
    min_sl = float(prep.get("min_abs_slope_by_source", {}).get(source, prep.get("min_abs_slope", 0.15)))
    max_sl = float(prep.get("max_abs_slope_by_source", {}).get(source, prep.get("max_abs_slope", 2.5)))
    cx = w / 2.0

    accepted_cnt = 0
    rejected_cnt = 0
    if segments is not None:
        for x1, y1, x2, y2 in segments[:, 0, :]:
            if x2 == x1:
                cv2.line(segs_vis, (x1, y1), (x2, y2), (100, 100, 100), 1)
                rejected_cnt += 1
                continue
            sl = (y2 - y1) / (x2 - x1)
            mx = (x1 + x2) / 2.0
            if abs(sl) < min_sl or abs(sl) > max_sl:
                cv2.line(segs_vis, (x1, y1), (x2, y2), (80, 80, 80), 1)
                rejected_cnt += 1
            elif sl < 0 and mx < cx:
                cv2.line(segs_vis, (x1, y1), (x2, y2), (255, 180, 0), 2)  # Left accepted
                accepted_cnt += 1
            elif sl > 0 and mx > cx:
                cv2.line(segs_vis, (x1, y1), (x2, y2), (0, 200, 255), 2)  # Right accepted
                accepted_cnt += 1
            else:
                cv2.line(segs_vis, (x1, y1), (x2, y2), (0, 0, 180), 1)  # Cross / wrong side
                rejected_cnt += 1

    if scaled_roi:
        pts = np.asarray(scaled_roi, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(segs_vis, [pts], True, (255, 255, 0), 1)

    cv2.putText(segs_vis, f"Hough: accepted={accepted_cnt}, rejected={rejected_cnt}",
                (24, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(output_dir / f"{prefix}_3_hough_segments.png"), segs_vis)

    # 4. Final detection with lines and departure HUD
    res = pipeline.process(frame, method="baseline", fps=fps)
    cv2.imwrite(str(output_dir / f"{prefix}_4_final_lines.png"), res["frame"])

    return {
        "accepted_segments": accepted_cnt,
        "rejected_segments": rejected_cnt,
        "lane_measurement_available": res["lane_measurement_available"],
        "offset_m": res.get("offset_m"),
        "warning": res.get("warning", {}).get("warning"),
        "state": res.get("warning", {}).get("state"),
        "tlc_s": res.get("warning", {}).get("tlc_s"),
        "lost": res.get("lost"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-frame visual debug dump")
    parser.add_argument("--config", default="configs/A_opt.yaml")
    parser.add_argument("--input", help="Video or image file path")
    parser.add_argument("--frame-idx", type=int, default=0, help="Frame index if input is video")
    parser.add_argument("--data-source", choices=("day", "night", "rain", "culane"))
    parser.add_argument("--output-dir", default="results/debug_frame")
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    output_dir = Path(args.output_dir)
    pipeline = LanePipeline(config, args.data_source)

    if args.input:
        in_path = Path(args.input)
        if in_path.suffix.lower() in (".png", ".jpg", ".jpeg"):
            frame = cv2.imread(str(in_path))
            if frame is None:
                raise SystemExit(f"Cannot load image: {in_path}")
            info = dump_frame(pipeline, frame, output_dir, prefix=in_path.stem)
            print(f"Dumped debug images for {in_path.name} to {output_dir}: {info}")
        else:
            cap = cv2.VideoCapture(str(in_path))
            if not cap.isOpened():
                raise SystemExit(f"Cannot open video: {in_path}")
            cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame_idx)
            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None:
                raise SystemExit(f"Cannot read frame {args.frame_idx} from {in_path}")
            prefix = f"{in_path.stem}_frame{args.frame_idx}"
            info = dump_frame(pipeline, frame, output_dir, prefix=prefix)
            print(f"Dumped debug images for {prefix} to {output_dir}: {info}")
    else:
        # Default: run on 1 night frame and 2 rain frames as specified in Requirement 8
        night_vid = Path("test_video/YTSave.to_YouTube_Media_MefjPQfHBJA_Tesla-Full-Self-Driving-prevents-night-time-deer-strike-front-camera-FSD-fullsel_003_1080p.mp4")
        rain_vid1 = Path("test_video/YTSave.to_YouTube_Media_2CIxM7x-Clc_Lane-detect-test-data_001_720p.mp4")
        rain_vid2 = Path("test_video/Screen Recording 2026-11-10 070340.mp4")

        runs = [
            (night_vid, 50, "night", "night_frame50"),
            (rain_vid1, 50, "rain", "rain1_frame50"),
            (rain_vid2, 200, "rain", "rain2_frame200"),
        ]
        for vid_path, f_idx, src, pref in runs:
            if not vid_path.is_file():
                continue
            cap = cv2.VideoCapture(str(vid_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                p = LanePipeline(config, src)
                info = dump_frame(p, frame, output_dir, prefix=pref)
                print(f"Dumped {pref} ({src}): {info}")


if __name__ == "__main__":
    main()
