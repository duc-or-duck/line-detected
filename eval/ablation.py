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
    config_names = ["A", "A+", "B", "C", "D", "E", "F"]
    for name in config_names:
        cfg_file = "A_opt.yaml" if name == "A+" else f"{name}.yaml"
        cfg_path = ROOT / "configs" / cfg_file
        if not cfg_path.is_file():
            row_label = "A+ (baseline tối ưu)" if name == "A+" else f"{name}. " + _label(name)
            rows.append([row_label, NA, NA, NA, NA, 0, "Thiếu tệp cấu hình"])
            continue
        config = load_config(cfg_path)
        score = evaluate(config, root, labels, "test")
        reason = score.get("reason", "") or "Không có kết quả"
        reasons.append(reason)
        accuracy = score.get("accuracy")
        row_label = "A+ (baseline tối ưu)" if name == "A+" else f"{name}. " + _label(name)
        rows.append([row_label, _value(100 * accuracy if accuracy is not None else None),
                     _value(score.get("fp")), _value(score.get("fn")),
                     _value(score.get("pixel_error")), score.get("frames", 0), reason])
        per_height = score.get("errors_by_sample", [])
        from src.dataset import H_SAMPLES
        pixel_rows.extend([[name, height,
                            per_height[index] if index < len(per_height) and per_height[index] is not None else NA]
                           for index, height in enumerate(H_SAMPLES)])
    ablation_headers = ["Cấu hình", "Accuracy (%)", "FP", "FN", "Sai số (px)", "Số khung", "Ghi chú"]
    write_csv("Bảng_4.3_ablation_tusimple.csv", ablation_headers, rows)
    write_csv("Bảng_4.3_pixel_error_h_samples.csv",
              ["Cấu hình", "h_samples (px)", "Sai số trung bình (px)"], pixel_rows)

    dashcam_headers = ["Điều kiện", "Số khung gán nhãn", "Baseline (A) (%)", "Đề xuất (F) (%)", "fps", "Ghi chú"]
    dashcam_rows = _dashcam_rows(args.dashcam_labels)
    write_csv("Bảng_4.4_dashcam.csv", dashcam_headers, dashcam_rows)
    timing_rows = [[block, NA, NA, NA, NA, "Chưa chạy benchmark với số lần lặp N đã cấu hình"]
                   for block in ("Tiền xử lý và tách vạch", "Bird's-eye view",
                                 "Cửa sổ trượt và đa thức", "Kalman và gating",
                                 "Độ lệch và cảnh báo", "Tổng")]
    write_csv("Bảng_4.6_runtime.csv", ["Khối", "Thời gian (ms)", "Tỷ lệ (%)",
                                      "FPS trung bình", "Số khung", "Ghi chú"], timing_rows)
    warning_headers = ["Logic cảnh báo", "Cảnh báo sai mỗi phút", "Ghi chú"]
    warning_rows = _warning_rows(args.warning_labels)
    write_csv("Hình_4.3_departure_warning.csv", warning_headers, warning_rows)

    # Bảng Jitter & Frame-to-frame stability (A vs A+)
    jitter_headers = ["Điều kiện", "Jitter A (std px)", "Jumps A (>80px)", "Jitter A+ (std px)", "Jumps A+ (>80px)", "Kết luận"]
    jitter_rows = _compute_jitter_table()
    write_csv("Bảng_4.8_jitter.csv", jitter_headers, jitter_rows)

    placeholder_png("Hình_4.1_hough_sensitivity.png", "Hình 4.1. Độ nhạy tham số Hough",
                    "Thiếu dữ liệu TuSimple và các giá trị tham số Hough.")
    placeholder_png("Hình_4.2_dashcam_comparison.png", "Hình 4.2. So sánh trên video dashcam",
                    "Thiếu video dashcam cho ba điều kiện.")
    placeholder_png("Hình_4.3_departure_warning.png", "Hình 4.3. Độ lệch và trạng thái cảnh báo",
                    "Thiếu video và ngưỡng cảnh báo trong YAML.")
    placeholder_png("Hình_4.4_error_examples.png", "Hình 4.4. Khung hình lỗi ban đêm và trời mưa",
                    "Thiếu video và khung hình có nhãn.")

    # In ra man hinh cac bang ket qua theo yeu cau
    print("\n=== BẢNG 4.3: ABLATION TUSIMPLE ===")
    _print_table(ablation_headers, rows)

    print("\n=== BẢNG 4.4: DASHCAM PERFORMANCE ===")
    _print_table(dashcam_headers, dashcam_rows)

    print("\n=== BẢNG ĐỘ ỔN ĐỊNH VÀ CHỐNG NHẢY KHUNG HÌNH (JITTER: A vs A+) ===")
    _print_table(jitter_headers, jitter_rows)

    print("\n=== BẢNG CẢNH BÁO CHỆCH LÀN (DEPARTURE WARNING: FALSE ALARMS) ===")
    _print_table(warning_headers, warning_rows)

    print("\nĐã ghi bảng và hình vào results/. Các ô n/a kèm lý do do thiếu dữ liệu hoặc giá trị cấu hình.")


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
        res = [[label, NA, NA, NA, NA, "Thiếu video và khung gán nhãn"] for label in labels.values()]
        res.append(["A+ (baseline tối ưu)", NA, NA, NA, NA,
                    "Tối ưu hóa Canny, ROI, và chống nhảy khung hình cho mưa và đêm"])
        return res
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
    result.append(["A+ (baseline tối ưu)", NA, NA, NA, NA,
                   "Tối ưu hóa Canny, ROI, và chống nhảy khung hình cho mưa và đêm"])
    return result


def _compute_jitter_table() -> list[list[object]]:
    """Compute frame-to-frame jitter (std of delta x at bottom) and count of jumps > 80px (SPEC 3.6)."""
    import cv2
    import numpy as np
    from src.pipeline import LanePipeline

    cfg_a_path = ROOT / "configs/A.yaml"
    cfg_opt_path = ROOT / "configs/A_opt.yaml"
    if not cfg_a_path.is_file() or not cfg_opt_path.is_file():
        return [["Ban ngày", NA, NA, NA, NA, "Thiếu tệp cấu hình"],
                ["Ban đêm", NA, NA, NA, NA, "Thiếu tệp cấu hình"],
                ["Trời mưa", NA, NA, NA, NA, "Thiếu tệp cấu hình"]]

    cfg_a = load_config(cfg_a_path)
    cfg_opt = load_config(cfg_opt_path)

    vids = {
        "Trời mưa": (ROOT / "test_video/YTSave.to_YouTube_Media_2CIxM7x-Clc_Lane-detect-test-data_001_720p.mp4", "rain"),
        "Ban đêm": (ROOT / "test_video/YTSave.to_YouTube_Media_MefjPQfHBJA_Tesla-Full-Self-Driving-prevents-night-time-deer-strike-front-camera-FSD-fullsel_003_1080p.mp4", "night"),
        "Ban ngày": (ROOT / "test_video/YTSave.to_YouTube_Media_6q5_A5wOwDM_Lane-Detection-Test-Video-01_001_720p.mp4", "day"),
    }

    jump_px = float(cfg_opt.get("preprocess", {}).get("jump_px", 80))
    table_rows = []

    for display, (vpath, src) in vids.items():
        if not vpath.is_file():
            table_rows.append([display, NA, NA, NA, NA, "Thiếu video kiểm thử"])
            continue
        cap = cv2.VideoCapture(str(vpath))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        pipe_a = LanePipeline(cfg_a, src)
        pipe_opt = LanePipeline(cfg_opt, src)
        xs_a, xs_opt = [], []
        for _ in range(50):
            ret, frame = cap.read()
            if not ret:
                break
            ra = pipe_a.process(frame.copy(), method="baseline")
            ro = pipe_opt.process(frame.copy(), method="baseline")
            xa = np.polyval(ra["left"], 1.0) * (w - 1) if ra.get("left") is not None else None
            xo = np.polyval(ro["left"], 1.0) * (w - 1) if ro.get("left") is not None else None
            xs_a.append(xa)
            xs_opt.append(xo)
        cap.release()

        def stats(xs):
            deltas = [abs(xs[i] - xs[i-1]) for i in range(1, len(xs)) if xs[i] is not None and xs[i-1] is not None]
            if not deltas:
                return (NA, 0, 0)
            return (round(float(np.std(deltas)), 2), sum(1 for d in deltas if d > jump_px), len(deltas))

        sa, ja, na = stats(xs_a)
        so, jo, no = stats(xs_opt)
        if isinstance(sa, (int, float)) and isinstance(so, (int, float)):
            conclusion = f"jitter(A_opt) < jitter(A)" if so < sa else "jitter(A_opt) >= jitter(A)"
        else:
            conclusion = NA
        table_rows.append([display, sa, ja, so, jo, conclusion])

    return table_rows


def _print_table(headers: list[str], rows: list[list[object]]) -> None:
    """Format and print Markdown table to stdout."""
    col_w = [len(str(h)) for h in headers]
    str_rows = []
    for r in rows:
        sr = [str(c) for c in r]
        str_rows.append(sr)
        for i, val in enumerate(sr):
            col_w[i] = max(col_w[i], len(val))
    header_str = " | ".join(h.ljust(col_w[i]) for i, h in enumerate(headers))
    sep_str = "-+-".join("-" * col_w[i] for i in range(len(headers)))
    print(header_str)
    print(sep_str)
    for sr in str_rows:
        print(" | ".join(val.ljust(col_w[i]) for i, val in enumerate(sr)))


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
    return {"A": "Canny + ROI + Hough",
            "A+": "baseline tối ưu",
            "B": "HLS + ROI + Hough",
            "C": "HLS + BEV + cửa sổ trượt + đa thức",
            "D": "C + tìm quanh đa thức trước",
            "E": "D + Kalman",
            "F": "E + tăng cường ảnh"}[name]


if __name__ == "__main__":
    main()
