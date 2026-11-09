# Phát hiện làn đường và cảnh báo chệch làn

Mã nguồn hiện thực phương pháp cơ sở (baseline) Canny + vùng quan tâm (region of interest, ROI) + Hough và pipeline đề xuất dùng ngưỡng màu Hue-Lightness-Saturation (HLS), bird's-eye view (BEV), cửa sổ trượt, đa thức bậc hai, Kalman tự cài đặt và hệ thống cảnh báo chệch làn (lane departure warning system, LDWS). Các khối được bật/tắt bằng YAML; cấu hình A–F dùng chung mã nguồn.

## Cài đặt

Python 3.9 trở lên:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Seed tái lập là `42` trong YAML và các phép lấy mẫu dùng `numpy.random.default_rng(42)`.

## Dữ liệu

Không có dữ liệu được tải tự động. Tải TuSimple từ trang benchmark chính thức: [TuSimple lane detection benchmark](https://github.com/TuSimple/tusimple-benchmark). Đặt ảnh và tệp nhãn dưới `data/tusimple/`; truyền chính xác các tệp nhãn bằng `--labels`.

Đặt video dashcam người dùng tự quay hoặc có giấy phép tại:

- `data/dashcam/day/`
- `data/dashcam/night/`
- `data/dashcam/rain/`

SPEC chưa cung cấp tệp video, số video, độ phân giải, tốc độ khung hình, thiết bị, nguồn/giấy phép hoặc số khung gán nhãn mỗi điều kiện. Các giá trị được để `null` trong `dataset_metadata` của YAML. Gán nhãn 20–30 khung cho mỗi điều kiện theo yêu cầu SPEC. Không đưa CULane vào kết quả mặc định vì đây là tập tùy chọn trong SPEC.

## Chạy

Kiểm tra nhãn và tạo ảnh xem trước:

```powershell
python -m src.dataset tusimple --root data/tusimple/train_set --labels data/tusimple/train_set/label_data.json --preview results/gt_preview
```

Demo tương tác:

```powershell
python app.py --input data/dashcam/day/<video>.mp4 --method proposed
python app.py --input data/dashcam/day/<video>.mp4 --method baseline
```

Chọn cấu hình hoặc khai báo nguồn hình học homography:

```powershell
python app.py --input data/dashcam/night/<video>.mp4 --method proposed --config configs/F.yaml --data-source night
```

Đánh giá TuSimple. Dùng tập train để chọn tham số; chỉ chạy test một lần để báo cáo:

```powershell
python eval/tusimple_eval.py --config configs/E.yaml --split train --root data/tusimple/train_set --labels data/tusimple/train_set/label_data.json
python eval/tusimple_eval.py --config configs/E.yaml --split test --root data/tusimple/test_set --labels data/tusimple/test_label.json
```

Chạy ablation và benchmark:

```powershell
python eval/ablation.py --tusimple-root data/tusimple/test_set --test-labels data/tusimple/test_label.json
python eval/benchmark_fps.py --video data/dashcam/night/<video>.mp4 --config configs/E.yaml
```

Kiểm thử:

```powershell
pytest -q
```

Khi dữ liệu hoặc tham số bắt buộc còn thiếu, script ghi `n/a` kèm lý do; chúng không tạo số đo thay thế. `eval/tusimple_eval.py` hiện thực lại các phép đo Accuracy/FP/FN và pixel error theo thông lệ benchmark TuSimple; cần xác minh lại với `lane.py` chính thức. Kết quả chạy nằm trong `results/` dưới dạng CSV và PNG theo tên Bảng/Hình trong SPEC.

## Tham số cần cung cấp

Các giá trị `null` dưới đây tương ứng với chỗ trống hoặc tham số chưa chốt trong SPEC. Hãy điền giá trị đã hiệu chỉnh vào YAML trước khi dùng khối tương ứng; không dùng kết quả test để hiệu chỉnh.

| Khối | Khóa YAML cần điền |
|---|---|
| ROI và Hough | `preprocess.roi_polygon`, `preprocess.hough.threshold`, `preprocess.hough.min_line_length`, `preprocess.hough.max_line_gap`, `preprocess.min_abs_slope` |
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
| Biến đổi Hough | Baseline khớp đoạn thẳng | Bài [ ] |
| Ngưỡng màu HLS + CLAHE/gamma | Tách làn và tăng cường ảnh | Bài [ ] |
| Homography | Bird's-eye view | Bài [ ] |
| Cửa sổ trượt + đa thức bậc hai | Khớp làn cong | Bài [ ] |
| Kalman tự cài đặt | Theo dõi làn theo thời gian | Bài [ ] |

Kỹ thuật, baseline, cải tiến, tự cài đặt Kalman có kiểm chứng thư viện, đánh giá định lượng, ablation và demo đã có trong mã. Điền số bài học từ đề cương môn vào các ô `Bài [ ]` trước khi nộp.
