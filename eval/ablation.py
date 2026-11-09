"""Run or initialize all requested result tables for ablation configs A–F."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.common import NA, placeholder_png, write_csv
from eval.tusimple_eval import evaluate
from src.pipeline import load_config


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--tusimple-root", default="data/tusimple")
    parser.add_argument("--test-labels", nargs="*", default=[])
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
        rows.append([f"{name}. " + _label(name), _value(score.get("accuracy")),
                     _value(score.get("fp")), _value(score.get("fn")),
                     _value(score.get("pixel_error")), reason])
        per_height = score.get("errors_by_sample", [])
        from src.dataset import H_SAMPLES
        pixel_rows.extend([[name, height,
                            per_height[index] if index < len(per_height) and per_height[index] is not None else NA]
                           for index, height in enumerate(H_SAMPLES)])
    write_csv("Bảng_4.3_ablation_tusimple.csv",
              ["Cấu hình", "Accuracy (%)", "FP", "FN", "Sai số (px)", "Ghi chú"], rows)
    write_csv("Bảng_4.3_pixel_error_h_samples.csv",
              ["Cấu hình", "h_samples (px)", "Sai số trung bình (px)"], pixel_rows)

    dashcam_rows = [[condition, NA, NA, NA, "Thiếu video và khung gán nhãn"]
                    for condition in ("Ban ngày", "Ban đêm", "Trời mưa")]
    write_csv("Bảng_4.4_dashcam.csv",
              ["Điều kiện", "Số khung gán nhãn", "Baseline (A) (%)", "Đề xuất (F) (%)", "Ghi chú"],
              dashcam_rows)
    enhancement_rows = [[label, NA, NA, "Thiếu tập khung gán nhãn / cấu hình phép tăng cường"]
                        for label in ("Không tăng cường", "Gamma", "Log", "Cân bằng histogram", "CLAHE")]
    write_csv("Bảng_4.5_enhancement.csv",
              ["Phép tăng cường", "Ban đêm (%)", "Trời mưa (%)", "Ghi chú"], enhancement_rows)
    timing_rows = [[block, NA, NA, "Chưa chạy benchmark với số lần lặp N đã cấu hình"]
                   for block in ("Tiền xử lý và tách vạch", "Bird's-eye view",
                                 "Cửa sổ trượt và đa thức", "Kalman và gating",
                                 "Độ lệch và cảnh báo", "Tổng")]
    write_csv("Bảng_4.6_runtime.csv", ["Khối", "Thời gian (ms)", "Tỷ lệ (%)", "Ghi chú"], timing_rows)
    write_csv("Hình_4.3_departure_warning.csv",
              ["Logic cảnh báo", "Cảnh báo sai mỗi phút", "Ghi chú"],
              [["Ngưỡng độ lệch đơn thuần", NA, "Thiếu video lái bình thường có nhãn"],
               ["TLC + hysteresis", NA, "Thiếu video lái bình thường có nhãn"]])

    placeholder_png("Hình_4.1_hough_sensitivity.png", "Hình 4.1. Độ nhạy tham số Hough",
                    "Thiếu dữ liệu TuSimple và các giá trị tham số Hough.")
    placeholder_png("Hình_4.2_dashcam_comparison.png", "Hình 4.2. So sánh trên video dashcam",
                    "Thiếu video dashcam cho ba điều kiện.")
    placeholder_png("Hình_4.3_departure_warning.png", "Hình 4.3. Độ lệch và trạng thái cảnh báo",
                    "Thiếu video và ngưỡng cảnh báo trong YAML.")
    placeholder_png("Hình_4.4_error_examples.png", "Hình 4.4. Khung hình lỗi ban đêm và trời mưa",
                    "Thiếu video và khung hình có nhãn.")
    print("Đã ghi bảng và hình vào results/. Các ô n/a kèm lý do do thiếu dữ liệu hoặc giá trị cấu hình.")


def _value(value):
    return NA if value is None else value


def _label(name: str) -> str:
    return {"A": "Canny + ROI + Hough", "B": "HLS + ROI + Hough",
            "C": "HLS + BEV + cửa sổ trượt + đa thức",
            "D": "C + tìm quanh đa thức trước", "E": "D + Kalman",
            "F": "E + tăng cường ảnh"}[name]


if __name__ == "__main__":
    main()
