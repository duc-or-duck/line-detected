import numpy as np

from src.pipeline import LanePipeline, load_config


def test_all_ablation_configs_load_and_keep_unknowns_null():
    for name in "ABCDEF":
        config = load_config(f"configs/{name}.yaml")
        assert config["seed"] == 42
        assert config["evaluation"]["tune_samples"] is None
        assert set(config["homography"]) == {"tusimple", "culane", "day", "night", "rain"}


def test_baseline_runs_with_unknown_hough_values_without_guessing():
    config = load_config("configs/A.yaml")
    result = LanePipeline(config).process(np.zeros((80, 120, 3), dtype=np.uint8), "baseline")
    assert result["frame"].shape == (80, 120, 3)
    assert result["left"] is None and result["right"] is None
    assert result["warning"]["warning"] is None
    assert "preprocess_detection" in result["timings_ms"]


def test_proposed_pipeline_runs_with_unknown_geometry_unavailable():
    config = load_config("configs/C.yaml")
    result = LanePipeline(config, "day").process(np.zeros((80, 120, 3), dtype=np.uint8))
    assert result["left"] is None and result["right"] is None
    assert result["offset_m"] is None
    assert result["warning"]["available"] is False


def test_baseline_lane_overlay_reaches_vehicle_even_below_roi():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    roi = [[40, 0], [60, 0], [60, 60], [40, 60]]
    ranges = {"left": (.2, 1.0), "right": (.2, 1.0)}
    LanePipeline._draw(frame, np.array([0.0, 0.0, .25]),
                       np.array([0.0, 0.0, .75]), False, roi, ranges)
    assert tuple(frame[95, 25]) == (0, 255, 0)
    assert tuple(frame[95, 74]) == (0, 255, 0)
