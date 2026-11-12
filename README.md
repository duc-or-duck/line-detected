# Phát hiện làn đường và cảnh báo chệch làn

Dự án dùng baseline Canny + ROI + Hough và pipeline đề xuất HLS, homography (bird's-eye view), sliding window, khớp đa thức bậc hai, cùng Kalman tự cài đặt (`src/kalman.py`). Hệ thống ước lượng tâm làn/xe, độ lệch pixel và độ lệch mét khi đã hiệu chuẩn; cảnh báo có hysteresis.

## Cài đặt

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Dữ liệu

- TuSimple: [benchmark chính thức](https://github.com/TuSimple/tusimple-benchmark/tree/master/doc/lane_detection); bản Kaggle đã dùng: [TuSimple Road](https://www.kaggle.com/datasets/touhidhaider/tusimple-road). Dữ liệu hiện có trong `dataset/train/images/`, `dataset/train/labels.json`, `dataset/test/images/`, `dataset/test/labels.json`.
- Dashcam: đặt video ngày, đêm và mưa tại `data/dashcam/day/`, `data/dashcam/night/`, `data/dashcam/rain/`.
  - Link tải video: [điền link]
  - Nguồn/giấy phép: [điền nguồn và giấy phép]

Không có dữ liệu nào được tải tự động. TuSimple hiện lưu ảnh khung cuối dạng phẳng, không có chuỗi 20 khung/clip; evaluator sẽ báo `n/a` thay vì chấm nhầm các ảnh như một clip.

## Chạy demo

OpenCV `VideoCapture` nhận MP4 và AVI.

```powershell
python app.py --input "test_video/YTSave.to_YouTube_Media_6q5_A5wOwDM_Lane-Detection-Test-Video-01_001_720p.mp4" --method baseline --data-source day
python app.py --input "test_video/YTSave.to_YouTube_Media_6q5_A5wOwDM_Lane-Detection-Test-Video-01_001_720p.mp4" --method proposed --config configs/E.yaml --data-source day
```

Nhấn `q` để đóng cửa sổ. Demo vẽ biên làn và tâm làn/xe, hiển thị độ lệch, trạng thái cảnh báo và FPS xử lý trung bình. Homography cần bốn điểm nguồn/đích đúng camera trong YAML; nếu để `null`, xử lý tiếp tục trên ảnh gốc.

## Đánh giá

```powershell
pytest -q
python eval/tusimple_eval.py --config configs/E.yaml --split train --root dataset/train/images --labels dataset/train/labels.json
python eval/tusimple_eval.py --config configs/E.yaml --split test --root dataset/test/images --labels dataset/test/labels.json
python eval/benchmark_fps.py --video "<đường-dẫn-video>" --config configs/E.yaml --data-source day
python eval/ablation.py
```

TuSimple chấm accuracy, FP/FN và sai số pixel; mỗi clip cần đủ 20 khung và chỉ chấm khung cuối. Video dashcam ngày/đêm/mưa được đánh giá theo tỷ lệ khung phát hiện đúng và FPS. `evaluation.pixel_threshold_px`, `evaluation.lane_accuracy_threshold` và `evaluation.fps_runs` phải được cấu hình trước; nếu thiếu ngưỡng hoặc dữ liệu, kết quả là `n/a`.

Benchmark ghi số khung, FPS trung bình không tính thời gian đọc video, và median thời gian từng khối vào `results/Bảng_4.6_runtime.csv`. Các kết quả khác được ghi ở `results/` dưới dạng CSV/PNG; không có số đo giả.

## Hiệu chuẩn và giới hạn

- Điền `homography.{tusimple,day,night,rain}.source/destination` theo camera.
- Điền `departure.lane_width_m` khi biết bề rộng làn; nếu chưa hiệu chuẩn, chỉ dùng độ lệch pixel.
- Điền ngưỡng warning/danger và tham số Kalman trong YAML trước khi đánh giá cảnh báo. Nếu thiếu biên làn hoặc ngưỡng cần thiết, trạng thái là `UNAVAILABLE`.
- Video dashcam phải được tải và đặt đúng đường dẫn thủ công.
