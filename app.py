"""Command-line video demo for baseline or proposed lane detection."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import time

import cv2
import numpy as np

from src.pipeline import LanePipeline, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo phat hien lan va canh bao chech lan")
    parser.add_argument("--input", required=True, help="Duong dan video dashcam dau vao")
    parser.add_argument("--method", choices=("baseline", "proposed"), required=True)
    parser.add_argument("--config", help="Tep YAML; mac dinh A cho baseline, E cho proposed")
    parser.add_argument("--data-source", choices=("tusimple", "culane", "day", "night", "rain"))
    parser.add_argument("--csv-out", help="Duong dan CSV ket qua canh bao chech lan")
    parser.add_argument("--no-display", action="store_true", help="Chay khong hien thi cua so OpenCV")
    parser.add_argument("--max-frames", type=int, default=None, help="So luong khung hinh toi da can xu ly")
    args = parser.parse_args()
    config_path = Path(args.config or ("configs/A.yaml" if args.method == "baseline" else "configs/E.yaml"))
    config = load_config(config_path)
    capture = cv2.VideoCapture(args.input)
    if not capture.isOpened():
        raise SystemExit(f"Cannot open input video: {args.input}")
    pipeline = LanePipeline(config, args.data_source)
    processed_frames = 0
    processing_seconds = 0.0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        started = time.perf_counter()
        result = pipeline.process(frame, args.method)
        processing_seconds += time.perf_counter() - started
        processed_frames += 1
        if args.max_frames and processed_frames >= args.max_frames:
            break
        fps = processed_frames / processing_seconds if processing_seconds else 0.0
        cv2.putText(result["frame"], f"FPS: {fps:.1f}", (24, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2, cv2.LINE_AA)
        if result["lane_measurement_available"] and result["offset_m"] is not None:
            offset_text = f"DO LECH: {result['offset_m']:+.3f} m ({result['offset_px']:+.1f} px)"
            cv2.putText(result["frame"], offset_text, (24, 76),
                        cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2, cv2.LINE_AA)
        elif result["lane_measurement_available"] and result["offset_px"] is not None:
            cv2.putText(result["frame"], f"DO LECH: {result['offset_px']:+.1f} px (chua hieu chuan)",
                        (24, 76), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2, cv2.LINE_AA)
        if not args.no_display:
            cv2.imshow("Phat hien lan duong", result["frame"])
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    capture.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    # Per-frame CSV in results/: frame, d_m, tlc_s, warning, lost, x_left, x_right
    csv_target = args.csv_out
    if not csv_target and ("A_opt" in str(config_path) or pipeline.csv_records):
        csv_target = f"results/{Path(args.input).stem}_departure.csv"
    if csv_target and pipeline.csv_records:
        csv_path = Path(csv_target)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["frame", "d_m", "tlc_s", "warning", "lost", "x_left", "x_right"]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in pipeline.csv_records:
                row = {
                    "frame": r["frame"],
                    "d_m": f"{r['d_m']:.4f}" if r["d_m"] is not None else "",
                    "tlc_s": f"{r['tlc_s']:.2f}" if r["tlc_s"] is not None and np.isfinite(r["tlc_s"]) else "",
                    "warning": int(r["warning"]) if r["warning"] is not None else "",
                    "lost": int(r["lost"]),
                    "x_left": f"{r['x_left']:.1f}" if r["x_left"] is not None else "",
                    "x_right": f"{r['x_right']:.1f}" if r["x_right"] is not None else "",
                }
                writer.writerow(row)
        print(f"Da ghi CSV ket qua canh bao chech lan: {csv_path}")


if __name__ == "__main__":
    main()
