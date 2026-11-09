"""TuSimple metrics using the benchmark lane-accuracy/FP/FN conventions."""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.common import NA, write_csv
from src.dataset import H_SAMPLES, read_annotations
from src.pipeline import LanePipeline, load_config


def score_lanes(predicted: list[list[int]], ground_truth: list[list[int]],
                point_threshold_px: float | None, lane_accuracy_threshold: float | None) -> dict:
    if point_threshold_px is None or lane_accuracy_threshold is None:
        return {"accuracy": None, "fp": None, "fn": None, "pixel_error": None, "errors_by_sample": []}
    gt_lanes = [lane for lane in ground_truth if any(x != -2 for x in lane)]
    pred_lanes = list(predicted)
    if not gt_lanes:
        return {"accuracy": None, "fp": None, "fn": None, "pixel_error": None, "errors_by_sample": []}
    lane_scores, errors = [], []
    errors_by_sample = [[] for _ in range(len(ground_truth[0]))]
    predicted_count = sum(any(x != -2 for x in lane) for lane in pred_lanes)
    pred_lanes = [lane for lane in pred_lanes if any(x != -2 for x in lane)]
    pair_scores = {}
    for gt_index, gt in enumerate(gt_lanes):
        for pred_index, pred in enumerate(pred_lanes):
            valid_count = sum(g != -2 for g in gt)
            pair_scores[(gt_index, pred_index)] = (
                sum(p != -2 and abs(p - g) < point_threshold_px for p, g in zip(pred, gt)) /
                max(valid_count, 1))
    match_count = min(len(gt_lanes), len(pred_lanes))
    best_matches: tuple[tuple[int, int], ...] = ()
    best_total = -1.0
    for gt_indices in itertools.combinations(range(len(gt_lanes)), match_count):
        for pred_indices in itertools.permutations(range(len(pred_lanes)), match_count):
            matches = tuple(zip(gt_indices, pred_indices))
            total = sum(pair_scores[pair] for pair in matches)
            if total > best_total:
                best_total, best_matches = total, matches
    matched_scores = [pair_scores[pair] for pair in best_matches]
    for gt_index, pred_index in best_matches:
        gt, pred = gt_lanes[gt_index], pred_lanes[pred_index]
        for sample_index, (prediction_x, target_x) in enumerate(zip(pred, gt)):
            if target_x != -2 and prediction_x != -2:
                error = abs(prediction_x - target_x)
                errors.append(error)
                errors_by_sample[sample_index].append(error)
    correct_count = sum(score > lane_accuracy_threshold for score in matched_scores)
    fp = max(0, predicted_count - correct_count) / max(predicted_count, 1)
    fn = max(0, len(gt_lanes) - correct_count) / len(gt_lanes)
    return {"accuracy": correct_count / len(gt_lanes),
            "fp": fp, "fn": fn, "pixel_error": float(np.mean(errors)) if errors else None,
            "errors_by_sample": errors_by_sample}


def predict_frame(pipeline: LanePipeline, frame: np.ndarray) -> list[list[int]]:
    result = pipeline.process(frame, pipeline.config.get("method", "proposed"))
    height, width = frame.shape[:2]
    lanes = []
    for fit in (result["left"], result["right"]):
        if fit is None:
            lanes.append([-2] * len(H_SAMPLES))
        else:
            lanes.append([int(round(float(np.polyval(fit, y / max(height - 1, 1))) * (width - 1)))
                          for y in H_SAMPLES])
    return lanes


def evaluate(config: dict, root: Path, labels: list[Path], split: str) -> dict:
    evaluation = config.get("evaluation", {})
    threshold = evaluation.get("pixel_threshold_px")
    lane_threshold = evaluation.get("lane_accuracy_threshold")
    if not labels:
        return {"accuracy": None, "fp": None, "fn": None, "pixel_error": None, "errors_by_sample": [],
                "reason": "Thiếu tệp nhãn TuSimple"}
    annotations = read_annotations(labels)
    if split == "test" and annotations and any(
            (root / record["raw_file"]).parent.resolve() == root.resolve()
            for record in annotations):
        return {"accuracy": None, "fp": None, "fn": None, "pixel_error": None,
                "errors_by_sample": [],
                "reason": "Nguồn chỉ chứa ảnh khung cuối dạng phẳng, thiếu chuỗi 20 khung cho mỗi clip"}
    if threshold is None or lane_threshold is None:
        return {"accuracy": None, "fp": None, "fn": None, "pixel_error": None, "errors_by_sample": [],
                "reason": "Thiếu ngưỡng đánh giá trong YAML"}
    if split == "train" and evaluation.get("tune_samples") is None:
        return {"accuracy": None, "fp": None, "fn": None, "pixel_error": None, "errors_by_sample": [],
                "reason": "Thiếu cỡ mẫu tuning trong YAML"}
    if split == "train":
        rng = np.random.default_rng(evaluation.get("seed", 42))
        sample_size = min(int(evaluation["tune_samples"]), len(annotations))
        annotations = [annotations[i] for i in sorted(rng.choice(len(annotations), sample_size, replace=False))]
    scores = []
    for record in annotations:
        pipeline = LanePipeline(config, "tusimple")
        image_path = root / record["raw_file"]
        candidates = sorted(image_path.parent.glob("*.jpg"))
        clip = candidates[-20:]
        if len(clip) != 20:
            return {"accuracy": None, "fp": None, "fn": None, "pixel_error": None, "errors_by_sample": [],
                    "reason": f"Clip không có đủ 20 khung: {image_path.parent}"}
        prediction = None
        for frame_path in clip:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                return {"accuracy": None, "fp": None, "fn": None, "pixel_error": None, "errors_by_sample": [],
                        "reason": f"Không đọc được ảnh: {frame_path}"}
            prediction = predict_frame(pipeline, frame)
        score = score_lanes(prediction, record["lanes"], threshold, lane_threshold)
        scores.append(score)
    keys = ("accuracy", "fp", "fn", "pixel_error")
    averaged = {key: float(np.mean([row[key] for row in scores if row[key] is not None])) if any(row[key] is not None for row in scores) else None
                for key in keys}
    averaged["errors_by_sample"] = [float(np.mean(values)) if values else None
                                    for values in zip(*(row["errors_by_sample"] for row in scores))]
    return averaged | {"reason": ""}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/E.yaml")
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--root", default="data/tusimple")
    parser.add_argument("--labels", nargs="*")
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    labels = [Path(path) for path in (args.labels or []) if Path(path).is_file()]
    result = evaluate(config, Path(args.root), labels, args.split)
    table = [[args.split, *(result[key] if result.get(key) is not None else NA
                            for key in ("accuracy", "fp", "fn", "pixel_error")), result["reason"]]]
    write_csv("Bảng_4.3_tusimple.csv",
              ["Tập", "Accuracy (%)", "FP", "FN", "Sai số vị trí (px)", "Ghi chú"], table)
    errors = result.get("errors_by_sample", [])
    error_rows = [[height, errors[index] if index < len(errors) and errors[index] is not None else NA]
                  for index, height in enumerate(H_SAMPLES)]
    write_csv("Bảng_4.3_pixel_error_h_samples.csv",
              ["h_samples (px)", "Sai số trung bình (px)"], error_rows)
    print(result["reason"] or "Đã ghi results/Bảng_4.3_tusimple.csv")


if __name__ == "__main__":
    main()
