import json

from eval.tusimple_eval import evaluate, score_lanes


def test_tusimple_lane_metrics_and_pixel_error():
    score = score_lanes([[10, 13, -2]], [[10, 12, -2]], 5, .85)
    assert score["accuracy"] == 1
    assert score["fp"] == 0
    assert score["fn"] == 0
    assert score["pixel_error"] == 0.5


def test_unknown_tusimple_thresholds_yield_unavailable_metrics():
    score = score_lanes([[10]], [[10]], None, None)
    assert score["accuracy"] is None
    assert score["fp"] is None


def test_tusimple_matching_is_independent_of_lane_list_order():
    score = score_lanes([[80, 80], [20, 20]], [[20, 20], [80, 80]], 5, .85)
    assert score["accuracy"] == 1
    assert score["pixel_error"] == 0


def test_flat_tusimple_train_frames_are_not_misused_as_a_20_frame_clip(tmp_path):
    labels = tmp_path / "labels.json"
    record = {"raw_file": "frame.jpg", "h_samples": [160], "lanes": [[20]]}
    labels.write_text(json.dumps(record) + "\n", encoding="utf-8")
    config = {"evaluation": {"pixel_threshold_px": 20,
                             "lane_accuracy_threshold": .85, "tune_samples": 1}}

    result = evaluate(config, tmp_path, [labels], "train")

    assert result["accuracy"] is None
    assert result["frames"] == 0
    assert "20 khung" in result["reason"]
