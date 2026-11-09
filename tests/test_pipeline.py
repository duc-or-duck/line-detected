import numpy as np

from src.pipeline import LanePipeline, load_config


def test_all_ablation_configs_load_and_keep_unknowns_null():
    for name in "ABCDEF":
        config = load_config(f"configs/{name}.yaml")
        assert config["seed"] == 42
        assert config["evaluation"]["tune_samples"] is None
        assert set(config["homography"]) == {"tusimple", "day", "night", "rain"}


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
