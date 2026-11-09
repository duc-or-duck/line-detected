# Phát hiện làn đường và cảnh báo chệch làn

Mã nguồn hiện thực phương pháp cơ sở (baseline) Canny + vùng quan tâm (region of interest, ROI) + Hough + K-means tự cài đặt để gom các đoạn thẳng thành hai biên làn. Pipeline đề xuất dùng ngưỡng màu Hue-Lightness-Saturation (HLS), bird's-eye view (BEV), cửa sổ trượt, đa thức bậc hai, Kalman tự cài đặt và hệ thống cảnh báo chệch làn (lane departure warning system, LDWS). Các khối được bật/tắt bằng YAML; cấu hình A–F dùng chung mã nguồn.

## Cài đặt

Python 3.9 trở lên:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Seed tái lập là `42` trong YAML và các phép lấy mẫu dùng `numpy.random.default_rng(42)`.

## Dữ liệu

TuSimple Road đã tải từ [Kaggle](https://www.kaggle.com/datasets/touhidhaider/tusimple-road) vào `dataset/TuSimple-Road/`. Bản nguồn có 3.626 nhãn train và 819 nhãn test; 97 ảnh được tham chiếu trong nhãn train không có trong archive. Các split tiện dùng nằm ở `dataset/train/{images,labels.json}` và `dataset/test/{images,labels.json}`. Split train giữ 3.529 ảnh có sẵn; test có đủ 819 ảnh. Archive chỉ có ảnh khung cuối dạng phẳng, không có chuỗi 20 khung/clip theo yêu cầu đánh giá; do đó TuSimple test được ghi `n/a` kèm lý do. Ảnh được hard-link khi cùng ổ đĩa để không nhân đôi dung lượng; dữ liệu lớn nằm trong `.gitignore`.

Đặt video dashcam người dùng tự quay hoặc có giấy phép tại:

- `data/dashcam/day/`
- `data/dashcam/night/`
- `data/dashcam/rain/`

SPEC chưa cung cấp tệp video, số video, độ phân giải, tốc độ khung hình, thiết bị, nguồn/giấy phép hoặc số khung gán nhãn mỗi điều kiện. Các giá trị được để `null` trong `dataset_metadata` của YAML. Gán nhãn 20–30 khung cho mỗi điều kiện theo yêu cầu SPEC. Không đưa CULane vào kết quả mặc định vì đây là tập tùy chọn trong SPEC.

## Chạy

Kiểm tra nhãn và tạo ảnh xem trước:

```powershell
python -m src.dataset tusimple --root dataset/train/images --labels dataset/train/labels.json --preview results/gt_preview_train
python -m src.dataset tusimple --root dataset/test/images --labels dataset/test/labels.json --preview results/gt_preview_test
```

Demo tương tác:

```powershell
python app.py --input data/dashcam/day/<video>.mp4 --method proposed
python app.py --input data/dashcam/day/<video>.mp4 --method baseline
```

Chạy thử clip được cung cấp trong `test_video/`:

```powershell
python app.py --input "test_video/YTSave.to_YouTube_Media_6q5_A5wOwDM_Lane-Detection-Test-Video-01_001_720p.mp4" --method proposed --config configs/E.yaml --data-source day
python app.py --input "test_video/YTSave.to_YouTube_Media_6q5_A5wOwDM_Lane-Detection-Test-Video-01_001_720p.mp4" --method baseline --config configs/A.yaml --data-source day
```

ROI và fit riêng cho `day` là preset demo cho clip này; chúng không được dùng khi đánh giá nguồn `tusimple`. Hiệu chỉnh này chỉ để kiểm tra hình ảnh demo, không thay cho đánh giá định lượng.

Chọn cấu hình hoặc khai báo nguồn hình học homography:

```powershell
python app.py --input data/dashcam/night/<video>.mp4 --method proposed --config configs/F.yaml --data-source night
```

Đánh giá TuSimple. Dùng tập train để chọn tham số; chỉ chạy test một lần để báo cáo:

```powershell
python eval/tusimple_eval.py --config configs/E.yaml --split train --root dataset/train/images --labels dataset/train/labels.json
python eval/tusimple_eval.py --config configs/E.yaml --split test --root dataset/test/images --labels dataset/test/labels.json
```

Chạy ablation và benchmark:

```powershell
python eval/ablation.py --tusimple-root data/tusimple/test_set --test-labels data/tusimple/test_label.json
python eval/benchmark_fps.py --video data/dashcam/night/<video>.mp4 --config configs/E.yaml --data-source night
```

Để điền bảng dashcam, cung cấp CSV gán nhãn với các cột `condition,video,frame_index,baseline_correct,proposed_correct`; `condition` là `day`, `night` hoặc `rain`, và hai cột correctness là kết quả kiểm tra thủ công xem cả hai biên có được phát hiện đúng không. Bảng tăng cường ảnh nhận CSV `condition,enhancement,correct` với phép tăng cường `none`, `gamma`, `log`, `hist_eq` hoặc `clahe`. Sau đó chạy:

```powershell
python eval/ablation.py --dashcam-labels data/dashcam_labels.csv --enhancement-labels data/enhancement_labels.csv --warning-labels data/normal_driving_offsets.csv
```

CSV cảnh báo dùng các cột `offset_m,offset_velocity_mps,true_departure`; `true_departure` đánh dấu trạng thái chệch làn theo nhãn thủ công. Điền `dataset_metadata.dashcam_fps` để tính cảnh báo sai mỗi phút.

Kiểm thử:

```powershell
pytest -q
```

Khi dữ liệu hoặc tham số bắt buộc còn thiếu, script ghi `n/a` kèm lý do; chúng không tạo số đo thay thế. `eval/tusimple_eval.py` hiện thực lại các phép đo Accuracy/FP/FN và pixel error theo thông lệ benchmark TuSimple; cần xác minh lại với `lane.py` chính thức. Kết quả chạy nằm trong `results/` dưới dạng CSV và PNG theo tên Bảng/Hình trong SPEC.

## Tham số cần cung cấp

Các giá trị `null` dưới đây tương ứng với chỗ trống hoặc tham số chưa chốt trong SPEC. Hãy điền giá trị đã hiệu chỉnh vào YAML trước khi dùng khối tương ứng; không dùng kết quả test để hiệu chỉnh.

| Khối | Khóa YAML cần điền |
|---|---|
| ROI, Hough và K-means | `preprocess.roi_polygon`, `preprocess.hough.threshold`, `preprocess.hough.min_line_length`, `preprocess.hough.max_line_gap`, `preprocess.kmeans.clusters`, `preprocess.min_abs_slope` |
| Gradient và tăng cường ảnh | `preprocess.gradient_threshold`, `pipeline.enhancement` (chọn `gamma`, `log`, `hist_eq`, `clahe` hoặc `none`), `enhancement.gamma`, `enhancement.log_scale`, `enhancement.clahe_clip_limit`, `enhancement.clahe_tile_grid_size` |
| Homography | `homography.{tusimple,day,night,rain}.source` và `.destination` (mỗi nguồn bốn điểm) |
| Cửa sổ và kiểm tra đa thức | `fit.window_count`, `fit.window_margin`, `fit.recenter_min_pixels`, `fit.min_fit_points`, `fit.search_margin`, `fit.min_lane_width_norm`, `fit.max_lane_width_norm` |
| Kalman | `kalman.dt`, `kalman.process_noise`, `kalman.measurement_noise`, `kalman.initial_covariance` |
| Cảnh báo | `departure.on_offset_m`, `departure.off_offset_m`, `departure.tlc_threshold_s`, `departure.consecutive_frames`, `departure.offset_only_threshold_m` |
| Đánh giá | `evaluation.tune_samples`, `evaluation.fps_runs`, `evaluation.pixel_threshold_px`, `evaluation.lane_accuracy_threshold` |
| Thông tin bộ dữ liệu | `dataset_metadata.tusimple_train_samples`, `dataset_metadata.tusimple_test_samples`, các khóa `dataset_metadata.dashcam_*`, `dataset_metadata.culane_sample_count` |
| Đối chiếu đề cương | Các khóa `course_lessons.*` |

`seed=42`, Canny `50/150`, dải HLS vàng `15–35`, `S>100`, trắng `L>200`, gating `11.34`, và chiều rộng làn mặc định `3.7 m` đã được đưa từ SPEC vào mã/cấu hình. Gating dùng đơn vị Mahalanobis và hệ số đa thức được chuẩn hóa theo kích thước ảnh.

## Kết quả hiện có

Chưa có dữ liệu TuSimple hoặc dashcam trong kho nên hiện chưa có số đo thực nghiệm. Chạy `python eval/ablation.py` để tạo CSV với `n/a` và PNG giải thích dữ liệu còn thiếu. Không có bảng hoặc hình kết quả được điền bằng số giả.

## Phân công công việc

| Thành viên | Phần việc | Commit |
|---|---|---|
| Thành viên 1 | Dataset, gán nhãn và hiệu chỉnh tham số trên train | Bổ sung sau |
| Thành viên 2 | Baseline, tiền xử lý và homography | Bổ sung sau |
| Thành viên 3 | Fit làn, Kalman và kiểm chứng với OpenCV | Bổ sung sau |
| Thành viên 4 | Cảnh báo, đánh giá, ablation và demo | Bổ sung sau |

Điều chỉnh phân công theo danh sách thành viên thực tế; kho hiện tại không có dữ liệu thành viên hoặc mã học viên.

## Đối chiếu yêu cầu môn học

| Kỹ thuật | Vai trò | Bài học |
|---|---|---|
| Canny + ROI | Baseline phát hiện biên | Bài [ ] |
| Biến đổi Hough + K-means | Baseline gom đoạn Hough thủ công thành hai biên làn | Bài [ ] |
| Ngưỡng màu HLS + CLAHE/gamma | Tách làn và tăng cường ảnh | Bài [ ] |
| Homography | Bird's-eye view | Bài [ ] |
| Cửa sổ trượt + đa thức bậc hai | Khớp làn cong | Bài [ ] |
| Kalman tự cài đặt | Theo dõi làn theo thời gian | Bài [ ] |

Kỹ thuật, baseline, cải tiến, tự cài đặt Kalman có kiểm chứng thư viện, đánh giá định lượng, ablation và demo đã có trong mã. Điền số bài học từ đề cương môn vào các ô `Bài [ ]` trước khi nộp.
