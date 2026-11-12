"""Temporary probe: A vs A_opt on night/rain frames, dump overlays."""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import LanePipeline, load_config

out = ROOT / "results" / "_probe"
vids = {
    "night": (ROOT / "test_video/YTSave.to_YouTube_Media_MefjPQfHBJA_Tesla-Full-Self-Driving-prevents-night-time-deer-strike-front-camera-FSD-fullsel_003_1080p.mp4", 50),
    "rain1": (ROOT / "test_video/YTSave.to_YouTube_Media_2CIxM7x-Clc_Lane-detect-test-data_001_720p.mp4", 50),
    "rain2": (ROOT / "test_video/Screen Recording 2026-11-10 070340.mp4", 200),
    "rain3": (ROOT / "test_video/Screen Recording 2026-11-10 070635.mp4", 200),
}
for name, (path, idx) in vids.items():
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        continue
    for cfg_name in ("A", "A_opt"):
        cfg = load_config(ROOT / f"configs/{cfg_name}.yaml")
        pipe = LanePipeline(cfg, name.split("1")[0].split("2")[0].split("3")[0])
        res = pipe.process(frame.copy(), "baseline")
        cv2.imwrite(str(out / f"{name}_{cfg_name}.png"), res["frame"])
        print(name, cfg_name, res["lane_measurement_available"], res["lost"], res["offset_m"])
