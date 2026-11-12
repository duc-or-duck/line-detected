"""Sequentially play test videos on screen using baseline lane detection."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import LanePipeline, load_config


def get_data_source(video_name: str) -> str:
    """Phân loại điều kiện môi trường dựa trên nội dung thực tế của từng video.

    Mapping xác nhận bằng mắt (mean_gray và ảnh thumbnail):
      night: 20261110-0654... (mean~17, đêm tối hẻm vắng),
             Screen Recording 070340 (mean~7, cao tốc đêm),
             Tesla_MefjPQfHBJA (mean~40, Tesla FSD ban đêm)
      rain : Screen Recording 070635 (gạt nước Tokyo mưa, mean~99 vì đèn đường),
             2CIxM7x (mưa ban ngày, mean~86)
      day  : 6q5_A5wOwDM, B6WK5qZLkbQ, cpGd2cIDY5s (nắng ban ngày)
    """
    name_lower = video_name.lower()
    # Videos ban đêm
    if ("20261110-0654" in name_lower
            or "070340" in name_lower
            or "tesla" in name_lower
            or "mefjpqfhbja" in name_lower):
        return "night"
    # Videos mưa (ban ngày Tokyo + dashcam mưa)
    if ("070635" in name_lower
            or "rain" in name_lower
            or "2cixm7x" in name_lower):
        return "rain"
    # Còn lại: ban ngày, trời khô
    return "day"


def main() -> None:
    parser = argparse.ArgumentParser(description="Chay lan luot test videos bang baseline")
    parser.add_argument("--video-dir", default="test_video", help="Thu muc chua video")
    parser.add_argument("--method", default="baseline", choices=("baseline", "proposed"))
    parser.add_argument("--config", default="configs/A.yaml")
    parser.add_argument("--scale", type=float, default=1.0, help="Ti le thu phong cua so")
    parser.add_argument("--max-frames-per-video", type=int, default=0,
                        help="Gioi han so khung moi video (0 = chay het video)")
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    videos = sorted(list(video_dir.glob("*.mp4")))
    if not videos:
        print(f"Khong tim thay video mp4 nao trong {video_dir}")
        return

    config = load_config(ROOT / args.config)
    print(f"=== TIM THAY {len(videos)} VIDEO TRONG '{video_dir}' ===")
    print("Dieu khien khi xem:")
    print("  - [Q] hoac [ESC] : Thoat chuong trinh")
    print("  - [N] hoac [Space]: Chuyen sang video tiep theo")
    print("  - [P]            : Tam dung / Tiep tuc")
    print("====================================================\n")

    window_name = "Lane Detection - Baseline Method"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    for idx, video_path in enumerate(videos, 1):
        data_source = get_data_source(video_path.name)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[{idx}/{len(videos)}] Khong the mo: {video_path.name}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_delay = max(1, int(1000 / video_fps))

        pipeline = LanePipeline(config, data_source=data_source)
        print(f"[{idx}/{len(videos)}] Dang chay: {video_path.name}")
        print(f"       Dieu kien: {data_source.upper()} | Do phan giai: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} | Khung hinh: {total_frames} | FPS: {video_fps:.1f}")

        frame_count = 0
        detected_count = 0
        processing_times = []
        skip_video = False
        paused = False

        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if args.max_frames_per_video > 0 and frame_count > args.max_frames_per_video:
                    break

                t0 = time.perf_counter()
                result = pipeline.process(frame, method=args.method, fps=video_fps)
                dt_ms = (time.perf_counter() - t0) * 1000
                processing_times.append(dt_ms)

                is_detected = result["left"] is not None and result["right"] is not None
                if is_detected:
                    detected_count += 1

                disp = result["frame"]
                H, W = disp.shape[:2]

                # Ve panel thong tin tren cung
                cv2.rectangle(disp, (0, 0), (W, 46), (20, 20, 20), -1)
                cv2.line(disp, (0, 46), (W, 46), (60, 60, 60), 1)

                info_left = f"[{idx}/{len(videos)}] {video_path.name[:32]}... | Mode: {args.method.upper()} ({data_source.upper()})"
                cv2.putText(disp, info_left, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

                status_text = "LANE: DETECTED" if is_detected else "LANE: SEARCHING"
                status_color = (0, 255, 0) if is_detected else (0, 0, 255)
                cv2.putText(disp, status_text, (W - 420, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2, cv2.LINE_AA)

                avg_fps = 1000.0 / np.mean(processing_times[-30:]) if processing_times else 0.0
                frame_info = f"F: {frame_count}/{total_frames} | {avg_fps:.0f} FPS"
                cv2.putText(disp, frame_info, (W - 190, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (200, 200, 200), 1, cv2.LINE_AA)

                # Thanh huong dan o day
                guide_text = "[Q]: Thoat | [N/Space]: Video tiep theo | [P]: Tam dung"
                cv2.putText(disp, guide_text, (12, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1, cv2.LINE_AA)

                # Tinh toan thoi gian nghi de phat dung toc do video goc (1.0x)
                wait_time = max(1, int(1000 / video_fps) - int(dt_ms))
                cv2.imshow(window_name, disp)

            key = cv2.waitKey(wait_time if not paused else 50) & 0xFF
            if key == ord("q") or key == 27:  # Q or ESC
                print("\nNguoi dung yeu cau thoat chuong trinh.")
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key == ord("n") or key == 32:  # N or SPACE
                print(f"  -> Bo qua {video_path.name}")
                skip_video = True
                break
            elif key == ord("p"):
                paused = not paused
                print("  -> Tam dung" if paused else "  -> Tiep tuc")

        cap.release()
        det_pct = (detected_count / frame_count * 100) if frame_count else 0.0
        print(f"  Ket qua: {detected_count}/{frame_count} khung ({det_pct:.1f}% nhan dien vach lan)\n")

    cv2.destroyAllWindows()
    print("=== DA HOAN TAT CHAY TAT CA CAC VIDEO TRONG test_video ===")


if __name__ == "__main__":
    main()
