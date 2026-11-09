"""Measure per-block processing time; video I/O remains outside timed sections."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.common import NA, write_csv
from src.pipeline import LanePipeline, load_config


def benchmark(video: Path, config: dict, data_source: str | None = None) -> list[list[object]]:
    runs = config.get("evaluation", {}).get("fps_runs")
    if runs is None:
        return [[block, NA, NA, NA, NA, "Thiếu số lần lặp N trong YAML"] for block in _blocks()]
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        return [[block, NA, NA, NA, NA, f"Không mở được video: {video}"] for block in _blocks()]
    pipeline = LanePipeline(config, data_source)
    timings: dict[str, list[float]] = {}
    count = 0
    processing_seconds = 0.0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        for _ in range(int(runs)):
            started = time.perf_counter()
            result = pipeline.process(frame, "proposed")
            processing_seconds += time.perf_counter() - started
            for key, value in result["timings_ms"].items():
                timings.setdefault(key, []).append(value)
        count += 1
    capture.release()
    if count == 0:
        return [[block, NA, NA, NA, 0, "Video không có khung đọc được"] for block in _blocks()]
    medians = {key: float(np.median(values)) for key, values in timings.items()}
    total = sum(medians.values())
    average_fps = count * int(runs) / processing_seconds if processing_seconds else None
    names = {"preprocess_detection": "Tiền xử lý và tách vạch", "perspective": "Bird's-eye view",
             "lane_fit": "Cửa sổ trượt và đa thức", "kalman": "Kalman và gating",
             "offset_warning": "Độ lệch và cảnh báo"}
    rows = []
    for key, label in names.items():
        value = medians.get(key)
        rows.append([label, NA if value is None else value,
                     NA if value is None or total == 0 else 100 * value / total,
                     NA, count, ""])
    rows.append(["Tổng", total, 100 if total else NA,
                 NA if average_fps is None else average_fps, count,
                 "FPS theo thời gian xử lý; không tính đọc video"])
    return rows


def _blocks():
    return ["Tiền xử lý và tách vạch", "Bird's-eye view", "Cửa sổ trượt và đa thức",
            "Kalman và gating", "Độ lệch và cảnh báo", "Tổng"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--config", default="configs/E.yaml")
    parser.add_argument("--data-source", choices=("day", "night", "rain", "tusimple", "culane"))
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    rows = benchmark(Path(args.video), config, args.data_source)
    write_csv("Bảng_4.6_runtime.csv", ["Khối", "Thời gian (ms)", "Tỷ lệ (%)",
                                      "FPS trung bình", "Số khung", "Ghi chú"], rows)
    print("Đã ghi results/Bảng_4.6_runtime.csv")


if __name__ == "__main__":
    main()
