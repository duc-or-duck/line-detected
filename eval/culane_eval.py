"""Pixel-mask evaluation for CULane lane annotations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.common import NA, write_csv
from src.pipeline import LanePipeline, load_config
from eval.tusimple_eval import score_lanes


def score_masks(predicted: np.ndarray, ground_truth: np.ndarray,
                tolerance_px: float | None) -> dict:
    if tolerance_px is None:
        return {"precision": None, "recall": None, "f1": None, "position_error_px": None}
    prediction = np.asarray(predicted) > 0
    target = np.asarray(ground_truth) > 0
    if prediction.shape != target.shape:
        raise ValueError("prediction and CULane mask dimensions must match")
    if not prediction.any() or not target.any():
        precision = 0.0 if prediction.any() else None
        recall = 0.0 if target.any() else None
        f1 = 0.0 if precision == 0.0 or recall == 0.0 else None
        return {"precision": precision, "recall": recall, "f1": f1,
                "position_error_px": None}
    target_distance = cv2.distanceTransform((~target).astype(np.uint8), cv2.DIST_L2, 3)
    prediction_distance = cv2.distanceTransform((~prediction).astype(np.uint8), cv2.DIST_L2, 3)
    prediction_errors = target_distance[prediction]
    target_errors = prediction_distance[target]
    precision = float(np.mean(prediction_errors <= tolerance_px))
    recall = float(np.mean(target_errors <= tolerance_px))
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "position_error_px": float(np.mean(prediction_errors))}


def read_culane_lines(path: Path) -> list[list[tuple[float, float]]]:
    lanes = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            values = [float(value) for value in line.split()]
            if len(values) < 4 or len(values) % 2:
                continue
            lanes.append(list(zip(values[0::2], values[1::2])))
    return lanes


def score_line_annotations(predicted_fits: tuple[np.ndarray | None, np.ndarray | None],
                           annotations: list[list[tuple[float, float]]], height: int,
                           width: int, point_threshold_px: float | None,
                           lane_threshold: float | None) -> dict:
    if point_threshold_px is None or lane_threshold is None:
        return {"accuracy": None, "fp": None, "fn": None, "pixel_error": None}
    ground_truth = []
    for lane in annotations:
        points = sorted(lane, key=lambda point: point[1])
        ys = np.arange(max(0, int(np.ceil(points[0][1]))),
                       min(height, int(np.floor(points[-1][1])) + 1))
        if len(ys) == 0:
            continue
        line = np.full(height, -2, dtype=np.int32)
        values = np.interp(ys, [point[1] for point in points],
                           [point[0] for point in points])
        line[ys] = np.rint(values).astype(np.int32)
        ground_truth.append(line.tolist())
    predicted = []
    y_norm = np.arange(height, dtype=np.float64) / max(height - 1, 1)
    for fit in predicted_fits:
        if fit is None:
            predicted.append([-2] * height)
        else:
            xs = np.rint(np.polyval(fit, y_norm) * (width - 1)).astype(np.int32)
            predicted.append([int(x) if 0 <= x < width else -2 for x in xs])
    return score_lanes(predicted, ground_truth, point_threshold_px, lane_threshold)


def evaluate(config: dict, root: Path, list_file: Path,
             label_dir: Path | None = None) -> dict:
    evaluation = config.get("evaluation", {})
    tolerance = evaluation.get("pixel_threshold_px")
    lane_threshold = evaluation.get("lane_accuracy_threshold")
    label_root = label_dir or root / "laneseg_label_w16"
    with list_file.open("r", encoding="utf-8") as stream:
        image_paths = [line.split()[0] for line in stream if line.strip()]
    pipeline = LanePipeline(config, "culane")
    scores = []
    for listed_path in image_paths:
        relative = Path(listed_path.lstrip("/\\"))
        image_path = root / relative
        mask_path = label_root / relative.with_suffix(".png")
        annotation_candidates = (root / relative.with_suffix(".lines.txt"),
                                root / relative.with_suffix(".txt"))
        annotation_path = next((path for path in annotation_candidates if path.is_file()), None)
        frame = cv2.imread(str(image_path))
        target = None if annotation_path else cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if frame is None or (annotation_path is None and target is None):
            missing = image_path if frame is None else mask_path
            return {"precision": None, "recall": None, "f1": None,
                    "position_error_px": None, "frames": len(scores),
                    "reason": f"Thiếu ảnh hoặc mask CULane: {missing}"}
        if target is not None and frame.shape[:2] != target.shape:
            return {"precision": None, "recall": None, "f1": None,
                    "position_error_px": None, "frames": len(scores),
                    "reason": f"Kích thước ảnh/mask không khớp: {relative}"}
        result = pipeline.process(frame, config.get("method", "proposed"))
        if annotation_path:
            if tolerance is None or lane_threshold is None:
                return {"precision": None, "recall": None, "f1": None,
                        "position_error_px": None, "frames": len(scores),
                        "reason": "Thiếu ngưỡng point/lane accuracy trong YAML"}
            line_score = score_line_annotations((result["left"], result["right"]),
                                                read_culane_lines(annotation_path),
                                                frame.shape[0], frame.shape[1],
                                                tolerance, lane_threshold)
            scores.append(line_score | {"precision": None, "recall": None, "f1": None})
            continue
        if tolerance is None:
            return {"precision": None, "recall": None, "f1": None,
                    "position_error_px": None, "frames": len(scores),
                    "reason": "Thiếu evaluation.pixel_threshold_px trong YAML"}
        prediction = np.zeros(target.shape, dtype=np.uint8)
        height, width = target.shape
        y = np.arange(height, dtype=np.float64) / max(height - 1, 1)
        for fit in (result["left"], result["right"]):
            if fit is None:
                continue
            x = np.rint(np.polyval(fit, y) * (width - 1)).astype(np.int32)
            points = np.column_stack((x, np.arange(height))).reshape(-1, 1, 2)
            cv2.polylines(prediction, [points], False, 255, 1)
        scores.append(score_masks(prediction, target, float(tolerance)))
    keys = ("accuracy", "fp", "fn", "precision", "recall", "f1", "pixel_error",
            "position_error_px")
    return {key: (float(np.mean([row[key] for row in scores if row.get(key) is not None]))
                  if any(row.get(key) is not None for row in scores) else None) for key in keys} | {
                      "frames": len(scores), "reason": "" if scores else "Danh sách CULane không có ảnh"}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Đánh giá mask làn CULane")
    parser.add_argument("--root", required=True, help="Thư mục gốc ảnh CULane")
    parser.add_argument("--list-file", required=True, help="Danh sách đường dẫn ảnh")
    parser.add_argument("--label-dir", help="Thư mục mask, mặc định <root>/laneseg_label_w16")
    parser.add_argument("--config", default="configs/E.yaml")
    args = parser.parse_args()
    result = evaluate(load_config(ROOT / args.config), Path(args.root), Path(args.list_file),
                      Path(args.label_dir) if args.label_dir else None)
    headers = ["Số khung", "Accuracy/Precision (%)", "Recall (%)", "F1 (%)", "FP", "FN",
               "Sai số vị trí (px)", "Ghi chú"]
    accuracy = result.get("accuracy") if result.get("accuracy") is not None else result.get("precision")
    values = [result["frames"], *(100 * value if value is not None else NA for value in
                                  (accuracy, result.get("recall"), result.get("f1"))),
              result.get("fp") if result.get("fp") is not None else NA,
              result.get("fn") if result.get("fn") is not None else NA,
              result.get("pixel_error") if result.get("pixel_error") is not None else
              (result.get("position_error_px") if result.get("position_error_px") is not None else NA),
              result["reason"]]
    write_csv("Bảng_4.7_culane.csv", headers, [values])
    print(result["reason"] or "Đã ghi results/Bảng_4.7_culane.csv")


if __name__ == "__main__":
    main()
