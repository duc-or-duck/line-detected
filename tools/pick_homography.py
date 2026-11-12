"""Click source trapezoid corners: left-top, right-top, left-bottom, right-bottom."""
import argparse
from pathlib import Path
import cv2
import yaml

parser = argparse.ArgumentParser(); parser.add_argument("image", type=Path); parser.add_argument("--output", type=Path, default=Path("configs/homography_by_video.yaml"))
args = parser.parse_args(); image = cv2.imread(str(args.image)); points = []
if image is None: raise SystemExit("Cannot open image")
cv2.namedWindow("Click LT RT LB RB; q saves")
cv2.setMouseCallback("Click LT RT LB RB; q saves", lambda e,x,y,f,p: points.append([x,y]) if e == cv2.EVENT_LBUTTONDOWN and len(points) < 4 else None)
while True:
    shown=image.copy()
    for p in points: cv2.circle(shown, tuple(p), 5, (0,255,0), -1)
    cv2.imshow("Click LT RT LB RB; q saves", shown)
    if cv2.waitKey(30) & 255 == ord("q"): break
cv2.destroyAllWindows()
if len(points) != 4: raise SystemExit("Need exactly 4 points")
h,w=image.shape[:2]; data=yaml.safe_load(args.output.read_text(encoding="utf-8")) if args.output.exists() else {}
data[args.image.name]={"source":points,"destination":[[.30*w,.10*h],[.70*w,.10*h],[.30*w,.95*h],[.70*w,.95*h]],"bev_size":[w,h]}
args.output.write_text(yaml.safe_dump(data,sort_keys=True),encoding="utf-8")
print(f"Saved {args.image.name}")
