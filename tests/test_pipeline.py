import numpy as np

from src.pipeline import LanePipeline, load_config


def test_all_ablation_configs_load_and_keep_unknowns_null():
    for name in "ABCDEF":
        config = load_config(f"configs/{name}.yaml")
        assert config["seed"] == 42
        assert config["evaluation"]["tune_samples"] is None
        assert set(config["homography"]) == {"tusimple", "culane", "day", "night", "rain"}


def test_proposed_day_demo_uses_its_configured_roi_by_default():
    pipeline = LanePipeline(load_config("configs/E.yaml"))
    assert pipeline.data_source == "day"


def test_baseline_runs_with_unknown_hough_values_without_guessing():
    config = load_config("configs/A.yaml")
    result = LanePipeline(config).process(np.zeros((80, 120, 3), dtype=np.uint8), "baseline")
    assert result["frame"].shape == (80, 120, 3)
    assert result["left"] is None and result["right"] is None
    assert result["warning"]["warning"] is False
    assert "preprocess_detection" in result["timings_ms"]


def test_proposed_pipeline_runs_with_unknown_geometry_unavailable():
    config = load_config("configs/C.yaml")
    result = LanePipeline(config, "day").process(np.zeros((80, 120, 3), dtype=np.uint8))
    assert result["left"] is None and result["right"] is None
    assert result["offset_m"] is None
    assert result["warning"]["available"] is False


def test_lane_overlay_stops_at_configured_roi_bottom():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    roi = [[0, 0], [99, 0], [99, 60], [0, 60]]
    ranges = {"left": (.2, .6), "right": (.2, .6)}
    LanePipeline._draw(frame, np.array([0.0, 0.0, .25]),
                       np.array([0.0, 0.0, .75]), False, roi, ranges)
    assert tuple(frame[55, 25]) == (0, 255, 0)
    assert tuple(frame[55, 74]) == (0, 255, 0)
    assert tuple(frame[95, 25]) == (0, 0, 0)
