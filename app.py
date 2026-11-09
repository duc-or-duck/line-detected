"""Command-line video demo for baseline or proposed lane detection."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from src.pipeline import LanePipeline, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo phat hien lan va canh bao chech lan")
    parser.add_argument("--input", required=True, help="Duong dan video dashcam dau vao")
    parser.add_argument("--method", choices=("baseline", "proposed"), required=True)
    parser.add_argument("--config", help="Tep YAML; mac dinh A cho baseline, E cho proposed")
    parser.add_argument("--data-source", choices=("tusimple", "day", "night", "rain"))
    args = parser.parse_args()
    config_path = Path(args.config or ("configs/A.yaml" if args.method == "baseline" else "configs/E.yaml"))
    config = load_config(config_path)
    capture = cv2.VideoCapture(args.input)
    if not capture.isOpened():
        raise SystemExit(f"Cannot open input video: {args.input}")
    pipeline = LanePipeline(config, args.data_source)
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        result = pipeline.process(frame, args.method)
        detected = (result["left"] is not None and result["right"] is not None and
                    result["offset_m"] is not None)
        status = "DA PHAT HIEN 2 LAN" if detected else "CHUA NHAN DIEN DU 2 LAN"
        color = (0, 220, 0) if detected else (0, 165, 255)
        cv2.putText(result["frame"], status, (24, 42), cv2.FONT_HERSHEY_SIMPLEX,
                    .7, color, 2, cv2.LINE_AA)
        offset = result["offset_m"]
        if offset is not None:
            cv2.putText(result["frame"], f"DO LECH: {offset:.3f} m", (24, 76),
                        cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Phat hien lan duong", result["frame"])
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
