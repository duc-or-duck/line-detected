from pathlib import Path

import numpy as np

from src.pipeline import LanePipeline


def test_draw_matches_golden():
    golden_path = Path(__file__).with_name("golden.npz")
    if not golden_path.exists():
        import pytest

        pytest.skip("golden output has not been generated")
    with np.load(golden_path) as golden:
        frame = golden["frame"].copy()
        LanePipeline._draw(
            frame,
            golden["left"],
            golden["right"],
            True,
            roi_polygon=golden["roi"].tolist(),
            offset_px=35.0,
            warning_state="DANGER",
            offset_m=0.42,
            tlc_s=2.2,
        )
        np.testing.assert_array_equal(frame, golden["expected"])
