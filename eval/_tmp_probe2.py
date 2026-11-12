"""Temporary probe: condition stats + A_opt detection on sample frames."""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import LanePipeline, load_config

vids = {
    "night": ROOT / "test_video/YTSave.to_YouTube_Media_MefjPQfHBJA_Tesla-Full-Self-Driving-prevents-night-time-deer-strike-front-camera-FSD-fullsel_003_1080p.mp4",
    "rain1": ROOT / "test_video/YTSave.to_YouTube_Media_2CIxM7x-Clc_Lane-detect-test-data_001_720p.mp4",
    "rain2": ROOT / "test_video/Screen Recording 2026-11-10 070340.mp4",
    "rain3": ROOT / "test_video/Screen Recording 2026-11-10 070635.mp4",
    "day": ROOT / "test_video/YTSave.to_YouTube_Media_6q5_A5wOwDM_Lane-Detection-Test-Video-01_001_720p.mp4",
}
cfg = load_config(ROOT / "configs/A_opt.yaml")
for name, path in vids.items():
    if not path.is_file():
        print(name, "missing")
        continue
    cap = cv2.VideoCapture(str(path))
    for idx in (0, 50, 200):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        poly = cfg["preprocess"]["roi_polygon_by_source"]["night"]
        ref = cfg["preprocess"]["roi_reference_size"]
        poly = [[round(p[0] * w / ref[0]), round(p[1] * h / ref[1])] for p in poly]
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [np.asarray(poly, np.int32)], 255)
        vals = gray[mask > 0]
        pipe = LanePipeline(cfg, name.split("1")[0].split("2")[0].split("3")[0])
        res = pipe.process(frame.copy(), "baseline")
        print(name, idx, w, h, "med=%.1f std=%.1f" % (np.median(vals), np.std(vals)),
              "avail=", res["lane_measurement_available"], "lost=", res["lost"],
              "d=", res["offset_m"])
    cap.release()
