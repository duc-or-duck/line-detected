"""Play every video in test_video in order; press n to skip and q to quit."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import LanePipeline, load_config


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}
WINDOW_NAME = "Phat hien lan duong - click VIDEO TIEP / n: next, q: quit"


def _draw_next_button(frame: np.ndarray, button: dict) -> np.ndarray:
    """Draw a clickable control on a display copy, never on the saved video."""
    display = frame.copy()
    height, width = display.shape[:2]
    button_width, button_height, margin = min(190, width - 24), 46, 12
    x1, y1 = width - button_width - margin, margin
    x2, y2 = x1 + button_width, y1 + button_height
    button["rect"] = (x1, y1, x2, y2)
    cv2.rectangle(display, (x1, y1), (x2, y2), (30, 115, 35), -1)
    cv2.rectangle(display, (x1, y1), (x2, y2), (245, 245, 245), 2)
    cv2.putText(display, "VIDEO TIEP  >", (x1 + 15, y1 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2, cv2.LINE_AA)
    return display


def _guess_source(capture: cv2.VideoCapture, path: Path) -> str:
    stem = path.stem.lower()
    if any(tag in stem for tag in ("rain", "raining", "wet", "mua")):
        return "rain"
    if any(tag in stem for tag in ("night", "dark", "dem", "toi")):
        return "night"
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    brightness = []
    for fraction in (.15, .50, .85):
        capture.set(cv2.CAP_PROP_POS_FRAMES, round(max(frame_count - 1, 0) * fraction))
        ok, frame = capture.read()
        if ok:
            value = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[int(frame.shape[0] * .45):, :, 2]
            brightness.append(float(np.mean(value)))
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return "night" if brightness and float(np.median(brightness)) < 75 else "day"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=Path("test_video"))
    parser.add_argument("--method", choices=("baseline", "proposed"), default="proposed")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data-source", choices=("auto", "tusimple", "culane", "day", "night", "rain"),
                        default="auto", help="auto selects night for dark clips and recognizes night/rain in filenames")
    parser.add_argument("--output-dir", type=Path, help="Also save processed MP4 files here")
    args = parser.parse_args()

    videos = sorted(p for p in args.folder.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES)
    if not videos:
        raise SystemExit(f"No video files found in: {args.folder}")
    config_path = args.config or Path("configs/A.yaml" if args.method == "baseline" else "configs/E.yaml")
    config = load_config(config_path)
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{len(videos)} videos | method={args.method} | focus the video window; n: next, q: quit", flush=True)
    button = {"rect": (0, 0, 0, 0), "next": False}
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    def on_mouse(event, x, y, _flags, state):
        if event == cv2.EVENT_LBUTTONUP:
            x1, y1, x2, y2 = state["rect"]
            if x1 <= x <= x2 and y1 <= y <= y2:
                state["next"] = True

    cv2.setMouseCallback(WINDOW_NAME, on_mouse, button)
    for video_index, video_path in enumerate(videos, start=1):
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            print(f"Skip (cannot open): {video_path.name}")
            continue
        data_source = args.data_source
        if data_source == "auto":
            data_source = _guess_source(capture, video_path)
        pipeline = LanePipeline(config, data_source, video_path.name)
        fps_in = capture.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps_in) or fps_in <= 0:
            fps_in = 30.0
        writer = None
        frames = 0
        elapsed = 0.0
        key = 0
        print(f"[{video_index}/{len(videos)}] [{data_source}] {video_path.name}", flush=True)
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            started = time.perf_counter()
            result = pipeline.process(frame, args.method)
            elapsed += time.perf_counter() - started
            frames += 1
            fps = frames / elapsed if elapsed else 0.0
            cv2.putText(result["frame"], f"FPS: {fps:.1f}", (24, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2, cv2.LINE_AA)
            if writer is None and args.output_dir:
                h, w = result["frame"].shape[:2]
                target = args.output_dir / f"{video_path.stem}_{args.method}.mp4"
                writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))
                if not writer.isOpened():
                    capture.release()
                    raise SystemExit(f"Cannot write output video: {target}")
            if writer is not None:
                writer.write(result["frame"])
            display_frame = _draw_next_button(result["frame"], button)
            cv2.imshow(WINDOW_NAME, display_frame)
            # Pace display to the source clip so the user has time to press n/q.
            key = cv2.waitKey(max(1, round(1000.0 / fps_in))) & 0xFF
            if button["next"]:
                button["next"] = False
                key = ord("n")
            if key in (ord("n"), ord("q")):
                break
        capture.release()
        if writer is not None:
            writer.release()
        if key == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        print("\nStopped by user.")
