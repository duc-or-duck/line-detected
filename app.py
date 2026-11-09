"""Command-line video demo for baseline or proposed lane detection."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import cv2

from src.pipeline import LanePipeline, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo phat hien lan va canh bao chech lan")
    parser.add_argument("--input", required=True, help="Duong dan video dashcam dau vao")
    parser.add_argument("--method", choices=("baseline", "proposed"), required=True)
    parser.add_argument("--config", help="Tep YAML; mac dinh A cho baseline, E cho proposed")
    parser.add_argument("--data-source", choices=("tusimple", "culane", "day", "night", "rain"))
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
        cv2.imshow("Phat hien lan duong", result["frame"])
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
