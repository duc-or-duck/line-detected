"""Save a compact G2 diagnostic montage for one labelled or reviewed frame."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src.pipeline import LanePipeline, load_config
from src.preprocess import lane_mask, roi_mask

def main():
    p = argparse.ArgumentParser(); p.add_argument("video", type=Path); p.add_argument("--frame", type=int, default=0)
    p.add_argument("--source", choices=("day", "night", "rain"), required=True); p.add_argument("--config", default="configs/F.yaml")
    p.add_argument("--output", type=Path, default=Path("results/debug")); a = p.parse_args()
    cap=cv2.VideoCapture(str(a.video)); cap.set(cv2.CAP_PROP_POS_FRAMES,a.frame); ok,frame=cap.read(); cap.release()
    if not ok: raise SystemExit("Cannot read frame")
    cfg=load_config(a.config); prep=cfg["preprocess"]; h,w=frame.shape[:2]; ref=prep.get("roi_reference_size",[1280,720])
    roi=prep.get("roi_polygon_by_source",{}).get(a.source,prep.get("roi_polygon")); roi=[[round(x*w/ref[0]),round(y*h/ref[1])] for x,y in roi] if roi else None
    mask=lane_mask(frame,prep|{"data_source":a.source}); clean=roi_mask(mask,roi)
    pipe=LanePipeline(cfg,a.source,a.video.name); result=pipe.process(frame,"proposed")
    bev=cv2.warpPerspective(clean, cv2.getPerspectiveTransform(np.float32(pipe._homography_for_frame()["source"]),np.float32(pipe._homography_for_frame()["destination"])), (w,h)) if pipe._homography_for_frame() else clean
    panels=[frame,cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR),cv2.cvtColor(clean,cv2.COLOR_GRAY2BGR),cv2.cvtColor(bev,cv2.COLOR_GRAY2BGR),result["frame"]]
    panels=[cv2.resize(x,(400,225)) for x in panels]; montage=np.hstack(panels); a.output.mkdir(parents=True,exist_ok=True)
    path=a.output/f"{a.video.stem}_{a.frame}_{a.source}.png"; cv2.imwrite(str(path),montage)
    ratio=float(np.count_nonzero(clean))/max(np.count_nonzero(roi_mask(np.full_like(clean,255),roi)),1)
    count=cv2.connectedComponents(clean)[0]-1
    print(f"{path}: mask_ratio={ratio:.4f}, components={count}, calibrated={result['calibration_status']}, valid={result['lane_measurement_available']}")
if __name__ == "__main__": main()
