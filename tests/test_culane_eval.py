import numpy as np
import cv2

from pathlib import Path

from eval.culane_eval import (evaluate, read_culane_lines, score_line_annotations,
                              score_masks)
from src.pipeline import load_config


def test_culane_mask_metrics_score_aligned_lane_pixels():
    target = np.zeros((32, 32), dtype=np.uint8)
    prediction = np.zeros_like(target)
    target[4:28, 10] = 255
    prediction[4:28, 11] = 255

    result = score_masks(prediction, target, 2)

    assert result["precision"] == 1
    assert result["recall"] == 1
    assert result["f1"] == 1
    assert result["position_error_px"] < 2


def test_culane_metrics_require_configured_tolerance():
    result = score_masks(np.zeros((8, 8)), np.zeros((8, 8)), None)
    assert result["f1"] is None


def test_culane_text_annotations_are_read_and_scored(tmp_path: Path):
    labels = tmp_path / "sample.txt"
    labels.write_text("25 10 25 20 25 30\n75 10 75 20 75 30\n", encoding="utf-8")
    lanes = read_culane_lines(labels)
    fits = (np.array([0, 0, .25]), np.array([0, 0, .75]))

    result = score_line_annotations(fits, lanes, 40, 101, 2, .8)

    assert result["accuracy"] == 1
    assert result["fp"] == 0
    assert result["fn"] == 0
    assert result["pixel_error"] == 0


def test_culane_evaluator_reads_list_image_and_text_annotation(tmp_path: Path):
    cv2.imwrite(str(tmp_path / "frame.jpg"), np.zeros((80, 120, 3), dtype=np.uint8))
    (tmp_path / "frame.txt").write_text(
        "30 30 30 60 30 75\n90 30 90 60 90 75\n", encoding="utf-8")
    image_list = tmp_path / "test.txt"
    image_list.write_text("frame.jpg\n", encoding="utf-8")
    config = load_config("configs/A.yaml")
    config["evaluation"]["pixel_threshold_px"] = 2
    config["evaluation"]["lane_accuracy_threshold"] = .8

    result = evaluate(config, tmp_path, image_list)

    assert result["frames"] == 1
    assert result["accuracy"] == 0
    assert result["fn"] == 1
