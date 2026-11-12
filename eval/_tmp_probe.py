"""Temporary probe: dump frames from test videos for visual inspection."""
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / "results" / "_probe"
out.mkdir(parents=True, exist_ok=True)
vids = sorted((ROOT / "test_video").glob("*.mp4"))
for v in vids:
    cap = cv2.VideoCapture(str(v))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(v.name, n, w, h, fps)
    for idx in (0, 50, 200, 400):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            cv2.imwrite(str(out / f"{v.stem[:40]}_{idx}.jpg"), frame)
    cap.release()
