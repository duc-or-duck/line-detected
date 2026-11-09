"""Run or initialize all requested result tables for ablation configs A–F."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.common import NA, placeholder_png, write_csv
from eval.benchmark_fps import benchmark
from src.departure import (DepartureWarning, false_alarms_per_minute,
                           offset_only_warning)
from eval.tusimple_eval import evaluate
from src.pipeline import load_config


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--tusimple-root", default="data/tusimple")
    parser.add_argument("--test-labels", nargs="*", default=[])
    parser.add_argument("--dashcam-labels", help="CSV with condition, video, frame_index, baseline_correct, proposed_correct")
    parser.add_argument("--warning-labels", help="CSV with offset_m, offset_velocity_mps, true_departure")
    args = parser.parse_args()
    root = Path(args.tusimple_root)
    labels = [Path(path) for path in args.test_labels if Path(path).is_file()]
    rows = []
    pixel_rows = []
    reasons = []
    for name in "ABCDEF":
        config = load_config(ROOT / "configs" / f"{name}.yaml")
        score = evaluate(config, root, labels, "test")
        reason = score.get("reason", "") or "Không có kết quả"
        reasons.append(reason)
        accuracy = score.get("accuracy")
        rows.append([f"{name}. " + _label(name), _value(100 * accuracy if accuracy is not None else None),
                     _value(score.get("fp")), _value(score.get("fn")),
                     _value(score.get("pixel_error")), score.get("frames", 0), reason])
        per_height = score.get("errors_by_sample", [])
        from src.dataset import H_SAMPLES
        pixel_rows.extend([[name, height,
                            per_height[index] if index < len(per_height) and per_height[index] is not None else NA]
                           for index, height in enumerate(H_SAMPLES)])
    write_csv("Bảng_4.3_ablation_tusimple.csv",
              ["Cấu hình", "Accuracy (%)", "FP", "FN", "Sai số (px)", "Số khung", "Ghi chú"], rows)
    write_csv("Bảng_4.3_pixel_error_h_samples.csv",
              ["Cấu hình", "h_samples (px)", "Sai số trung bình (px)"], pixel_rows)

    dashcam_rows = _dashcam_rows(args.dashcam_labels)
    write_csv("Bảng_4.4_dashcam.csv",
              ["Điều kiện", "Số khung gán nhãn", "Baseline (A) (%)", "Đề xuất (F) (%)", "fps", "Ghi chú"],
              dashcam_rows)
    timing_rows = [[block, NA, NA, NA, NA, "Chưa chạy benchmark với số lần lặp N đã cấu hình"]
                   for block in ("Tiền xử lý và tách vạch", "Bird's-eye view",
                                 "Cửa sổ trượt và đa thức", "Kalman và gating",
                                 "Độ lệch và cảnh báo", "Tổng")]
    write_csv("Bảng_4.6_runtime.csv", ["Khối", "Thời gian (ms)", "Tỷ lệ (%)",
                                      "FPS trung bình", "Số khung", "Ghi chú"], timing_rows)
    write_csv("Hình_4.3_departure_warning.csv",
              ["Logic cảnh báo", "Cảnh báo sai mỗi phút", "Ghi chú"],
              _warning_rows(args.warning_labels))

    placeholder_png("Hình_4.1_hough_sensitivity.png", "Hình 4.1. Độ nhạy tham số Hough",
                    "Thiếu dữ liệu TuSimple và các giá trị tham số Hough.")
    placeholder_png("Hình_4.2_dashcam_comparison.png", "Hình 4.2. So sánh trên video dashcam",
                    "Thiếu video dashcam cho ba điều kiện.")
    placeholder_png("Hình_4.3_departure_warning.png", "Hình 4.3. Độ lệch và trạng thái cảnh báo",
                    "Thiếu video và ngưỡng cảnh báo trong YAML.")
    placeholder_png("Hình_4.4_error_examples.png", "Hình 4.4. Khung hình lỗi ban đêm và trời mưa",
                    "Thiếu video và khung hình có nhãn.")
    print("Đã ghi bảng và hình vào results/. Các ô n/a kèm lý do do thiếu dữ liệu hoặc giá trị cấu hình.")


def _read_rows(path: str | None) -> list[dict[str, str]]:
    if not path or not Path(path).is_file():
        return []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _truth(value: str | None) -> bool | None:
    if value is None or not value.strip():
        return None
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    return None


def _dashcam_rows(label_path: str | None) -> list[list[object]]:
    annotations = _read_rows(label_path)
    labels = {"day": "Ban ngày", "night": "Ban đêm", "rain": "Trời mưa"}
    if not annotations:
        return [[label, NA, NA, NA, NA, "Thiếu video và khung gán nhãn"] for label in labels.values()]
    config = load_config(ROOT / "configs/F.yaml")
    result = []
    for condition, display in labels.items():
        rows = [row for row in annotations if row.get("condition", "").lower() == condition]
        baseline = [_truth(row.get("baseline_correct")) for row in rows]
        proposed = [_truth(row.get("proposed_correct")) for row in rows]
        baseline = [value for value in baseline if value is not None]
        proposed = [value for value in proposed if value is not None]
        unique_videos = sorted({row.get("video", "") for row in rows if row.get("video")})
        fps_values = []
        for video in unique_videos:
            timing = benchmark(Path(video), config, condition)
            total_row = next((item for item in timing if item[0] == "Tổng"), None)
            if total_row and isinstance(total_row[3], (int, float)) and total_row[3] > 0:
                fps_values.append(float(total_row[3]))
        fps = sum(fps_values) / len(fps_values) if fps_values else None
        notes = []
        if not baseline or not proposed:
            notes.append("Thiếu nhãn correctness cho một hoặc cả hai phương pháp")
        if fps is None:
            notes.append("Thiếu video mở được hoặc số lần lặp N trong YAML")
        result.append([display, len(rows), _rate(baseline), _rate(proposed), _value(fps), "; ".join(notes)])
    return result


def _warning_rows(label_path: str | None, config: dict | None = None) -> list[list[object]]:
    annotations = _read_rows(label_path)
    config = config or load_config(ROOT / "configs/F.yaml")
    departure = config.get("departure", {})
    fps = config.get("dataset_metadata", {}).get("dashcam_fps")
    threshold = departure.get("offset_only_threshold_m")
    required = (fps, threshold, departure.get("on_offset_m"), departure.get("off_offset_m"),
                departure.get("tlc_threshold_s"), departure.get("consecutive_frames"))
    if not annotations:
        reason = "Thiếu video lái bình thường có nhãn"
        return [["Ngưỡng độ lệch đơn thuần", NA, reason], ["TLC + hysteresis", NA, reason]]
    if any(value is None for value in required):
        reason = "Thiếu ngưỡng cảnh báo hoặc fps trong YAML"
        return [["Ngưỡng độ lệch đơn thuần", NA, reason], ["TLC + hysteresis", NA, reason]]
    baseline, combined, truth = [], [], []
    state = DepartureWarning(departure)
    for row in annotations:
        offset = float(row["offset_m"])
        velocity = float(row["offset_velocity_mps"]) if row.get("offset_velocity_mps", "").strip() else None
        label = _truth(row.get("true_departure"))
        if label is None:
            reason = "Thiếu nhãn trạng thái chệch làn"
            return [["Ngưỡng độ lệch đơn thuần", NA, reason], ["TLC + hysteresis", NA, reason]]
        base_warning = offset_only_warning(offset, float(threshold))
        combined_warning = state.update(offset, velocity, departure.get("lane_width_m", 3.7))["warning"]
        if combined_warning is None:
            reason = "Thiếu ngưỡng cảnh báo hoặc fps trong YAML"
            return [["Ngưỡng độ lệch đơn thuần", NA, reason], ["TLC + hysteresis", NA, reason]]
        baseline.append(bool(base_warning))
        combined.append(bool(combined_warning))
        truth.append(label)
    base_rate = false_alarms_per_minute(baseline, truth, float(fps))
    combined_rate = false_alarms_per_minute(combined, truth, float(fps))
    return [["Ngưỡng độ lệch đơn thuần", base_rate, ""], ["TLC + hysteresis", combined_rate, ""]]


def _rate(values: list[bool]) -> object:
    return 100.0 * sum(values) / len(values) if values else NA


def _value(value):
    return NA if value is None else value


def _label(name: str) -> str:
    return {"A": "Canny + ROI + Hough", "B": "HLS + ROI + Hough",
            "C": "HLS + BEV + cửa sổ trượt + đa thức",
            "D": "C + tìm quanh đa thức trước", "E": "D + Kalman",
            "F": "E + tăng cường ảnh"}[name]


if __name__ == "__main__":
    main()
