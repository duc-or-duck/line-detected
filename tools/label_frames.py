"""Label lane-marking points on evenly spaced dashcam frames."""
import argparse
import json
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--condition", choices=("day", "night", "rain"), required=True)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("dashcam_labels.json"))
    parser.add_argument("--split-output", type=Path, default=Path("tools/split.json"))
    args = parser.parse_args()
    cap = cv2.VideoCapture(str(args.video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or total < 1 or args.count < 1:
        raise SystemExit("Cannot open video or invalid frame count")
    indices = sorted(set(round(i * (total - 1) / max(args.count - 1, 1)) for i in range(args.count)))
    rows = json.loads(args.output.read_text(encoding="utf-8")) if args.output.exists() else []
    split = json.loads(args.split_output.read_text(encoding="utf-8")) if args.split_output.exists() else {"videos": {}}
    split["videos"][args.video.name] = {"tune": indices[::2], "valid": indices[1::2]}
    win, points = "Lane labels: click 4 points per marking", []
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, lambda event, x, y, flags, data: points.append([x, y]) if event == cv2.EVENT_LBUTTONDOWN and len(points) < 8 else None)
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            continue
        points.clear()
        while True:
            view = frame.copy()
            for x, y in points:
                cv2.circle(view, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(view, f"{args.condition} frame {index}: click 4 left + 4 right; n skip, q save", (12, 28), 0, .65, (0, 255, 255), 2)
            cv2.imshow(win, view)
            key = cv2.waitKey(30) & 0xFF
            if len(points) == 8 or key in (ord("n"), ord("q")):
                break
        if len(points) == 8:
            rows.append({"video": args.video.name, "condition": args.condition, "frame": index, "fps": fps,
                         "left_points": points[:4], "right_points": points[4:],
                         "baseline_correct": "", "proposed_correct": ""})
        if key == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    args.split_output.parent.mkdir(parents=True, exist_ok=True)
    args.split_output.write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(rows)} labeled frames to {args.output}")


if __name__ == "__main__":
    main()
