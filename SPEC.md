# SPEC (nguồn chuẩn, trích nguyên văn từ 2 file docx; KHÔNG sửa, KHÔNG suy diễn)

## PART A — Lane_Detection_Project_Outline.docx (bỏ mục lục/danh mục bảng-hình)

Lecturer:

1. Dr. Nguyen Ngoc Truong Minh

© Copyright by [Họ và tên học viên] [Năm]

Lane Detection and Lane Departure Warning from Dashcam Video Using Classical Image Processing and a From-Scratch Kalman Filter

Abstract

The report presents a lane detection and lane departure warning system from dashcam video using classical image processing. The baseline method consists of Canny, region of interest (ROI), and the Hough transform. The proposed method adds HLS color thresholding, perspective transformation to bird’s-eye view, sliding window, quadratic polynomial fitting, and a custom-implemented Kalman filter, validated against cv2.KalmanFilter. The vehicle’s deviation from the lane center is converted to meters for warning, with a hysteresis mechanism. The system is evaluated on TuSimple (accuracy, FP, FN, lane position error) and daytime, nighttime, and rainy dashcam videos (correct-frame rate, fps), along with per-block ablation. Results: accuracy increases from [A]% to [E]%, [F] fps on [CPU]; errors are concentrated in nighttime scenes and wet roads.

1.  Introduction

Context and motivation

Hệ thống cảnh báo chệch làn (Lane Departure Warning System – LDWS) là một mô-đun của hệ thống hỗ trợ lái xe. LDWS không điều khiển xe mà phát tín hiệu âm thanh, rung hoặc hình ảnh khi xe sắp hoặc đã lệch khỏi làn trong lúc người lái không bật xi-nhan [1]. Một LDWS thường gồm bốn bước: phát hiện vạch làn, khớp đường làn, quyết định chệch làn và phát cảnh báo [1].

Độ tin cậy của LDWS dựa trên thị giác phụ thuộc chủ yếu vào chất lượng phát hiện làn, nhất là khi thời tiết, ánh sáng thay đổi hoặc có bóng đổ [1]. Báo cáo này đặc biệt quan tâm đến cảnh ban đêm và trời mưa, là hai điều kiện làm vạch làn khó nhận dạng.

[Bổ sung 1–2 câu số liệu về tai nạn liên quan đến chệch làn, trích từ nguồn gốc (ví dụ báo cáo của WHO hoặc cơ quan an toàn giao thông), không trích lại số liệu thứ cấp từ bài review.]

Problem statement

Đầu vào là chuỗi khung hình màu I_t từ camera hành trình gắn phía trước xe. Đầu ra gồm:

(i) Đường làn trái và phải dưới dạng đa thức x = f(y);

(ii) Độ lệch d_t của xe so với tâm làn (mét) và độ cong của làn;

(iii) Trạng thái cảnh báo w_t ∈ {0, 1}. Với TuSimple, đầu ra được biểu diễn bằng toạ độ x tại các hàng ảnh h_samples, và giá trị −2 tại hàng không có làn.

Các giả định của báo cáo:

- Camera đặt ở giữa xe, hướng thẳng về phía trước.

- Chiều rộng làn cố định, mặc định 3,7 m (có thể cấu hình), dùng để quy đổi pixel sang mét.

- Dữ liệu dashcam không có tín hiệu xi-nhan và tốc độ, trong khi LDWS thương mại thường không cảnh báo khi bật xi-nhan và chỉ hoạt động ở tốc độ cao [1]. Đây là một hạn chế được ghi nhận.

- Đường có vạch kẻ; không xét đường không vạch hoặc nhiều làn song song.

Objectives and research questions

Mục tiêu là xây dựng pipeline xử lý ảnh cổ điển có thể bật/tắt từng khối, đo đóng góp của từng khối và xác định giới hạn ở điều kiện đêm và mưa. Bốn câu hỏi nghiên cứu được đặt ra:

- RQ1. Mỗi khối xử lý (ngưỡng HLS, bird's-eye view, cửa sổ trượt + đa thức bậc 2, Kalman) cải thiện accuracy, FP, FN, sai số vị trí làn và fps bao nhiêu so với baseline?

- RQ2. Bộ lọc Kalman tự cài đặt có giảm nhiễu và giữ được làn khi vạch bị mất hoặc đứt nét không, và chi phí về fps là bao nhiêu?

- RQ3. Ở điều kiện đêm và mưa, pipeline lỗi ở đâu, và phép tăng cường ảnh nào giảm được lỗi?

- RQ4. Cảnh báo kết hợp độ lệch, thời gian tới vạch và hysteresis có ít báo sai hơn so với ngưỡng độ lệch đơn thuần không?

Contributions and scope

- Pipeline mô-đun hóa: mỗi khối được bật/tắt bằng tệp cấu hình, nhờ đó ablation chỉ là chạy cùng một mã với các cấu hình khác nhau.

- Bộ lọc Kalman tự cài đặt bằng NumPy, kiểm chứng với cv2.KalmanFilter trên dữ liệu tổng hợp và dữ liệu thật.

- Ablation theo từng khối trên TuSimple và đánh giá trên video dashcam ban ngày, ban đêm, trời mưa, kèm phân tích lỗi.

- Logic cảnh báo chệch làn dựa trên độ lệch, thời gian tới vạch (TLC) và hysteresis, đo bằng số cảnh báo sai mỗi phút.

Bảng 1.1. Kỹ thuật sử dụng và bài học tương ứng

---

Kỹ thuật Vai trò trong dự án Bài học

---

Canny + ROI Baseline: phát hiện biên Bài [ ]

Biến đổi Hough Baseline: khớp đoạn thẳng Bài [ ]

Ngưỡng màu HLS, CLAHE/gamma Tách vạch làn, tăng cường cảnh tối Bài [ ]

Homography (bird's-eye view) Biến đổi hình học, làm làn song song Bài [ ]

Cửa sổ trượt + đa thức bậc 2 Khớp đường làn Bài [ ]

Bộ lọc Kalman (tự cài đặt) Làm trơn và bám làn theo thời gian Bài [ ]

---

Yêu cầu của môn học là ít nhất 4 kỹ thuật thuộc ít nhất 3 bài học khác nhau. Cột “Bài học” cần được đối chiếu với đề cương môn học.

Report organization

Chương 2 tổng quan các phương pháp liên quan. Chương 3 mô tả dữ liệu, baseline, pipeline đề xuất và quy trình đánh giá. Chương 4 trình bày kết quả, ablation và phân tích lỗi. Chương 5 kết luận, nêu hạn chế và hướng phát triển.

2.  Literature Review

Lane departure warning systems

Theo [1], LDWS trên xe được chia thành hệ thống nhìn bên và hệ thống nhìn trước. Hệ thống nhìn trước khai thác được nhiều thông tin về đường hơn nhưng dễ bị nhiễu bởi xe khác, người đi bộ và vật cản. Một số hệ thống thương mại như DSS và AWS ước lượng thời gian còn lại trước khi xe cắt vạch (time-to-line-crossing, TLC) từ vị trí và vận tốc ngang, rồi so với ngưỡng đặt trước [1].

[1] cũng nhận xét rằng nhiều thuật toán cảnh báo chỉ xét vị trí tương đối của xe so với vạch mà bỏ qua xu hướng chuyển động, dẫn đến cảnh báo sai hoặc bỏ sót và làm người dùng giảm tin cậy vào hệ thống. Nhận xét này là cơ sở cho thiết kế cảnh báo ở Mục 3.4.6.

Traditional lane detection

Quy trình truyền thống gồm năm bước: thu ảnh, chọn vùng quan tâm hợp lệ, tăng cường ảnh, trích đặc trưng và mô hình hóa làn [1]. Trích đặc trưng chia thành nhóm dựa trên độ tương đồng (ngưỡng, màu) và nhóm dựa trên độ gián đoạn (toán tử biên như Canny, Sobel) [1]. Toán tử biên có tốc độ cao nhưng nhạy với ánh sáng và vật cản. Ngưỡng xám toàn cục thường thất bại khi có bóng cây, mưa hoặc ban đêm vì lược đồ xám không có cấu trúc hai đỉnh tương ứng với vạch làn [1].

Ở bước mô hình hóa, biến đổi Hough tìm các đoạn thẳng nhưng kết quả phụ thuộc mạnh vào tham số, và mô hình đường thẳng không phù hợp với làn cong [1]. Các mô hình đường cong như parabol hoặc spline mô tả tốt hơn nhưng cần ràng buộc ở hai đầu đường cong và nhạy với điểm nhiễu [1], [2].

Image enhancement and inverse perspective mapping

[1] so sánh năm phép tăng cường cho ảnh mờ (biến đổi hàm mũ, log, cân bằng histogram, dark channel prior và retinex) và kết luận phép phù hợp phụ thuộc loại ảnh như đêm, sương hay bóng đổ. Báo cáo này thử các phép tương ứng gồm gamma, log, cân bằng histogram và CLAHE (Mục 4.6).

Biến đổi phối cảnh ngược (IPM, bird's-eye view) làm các vạch làn gần song song và giảm khối lượng tính toán nhưng mất một phần thông tin ảnh [1], [4]. Khi ma trận biến đổi chỉ tính một lần và dùng chung cho mọi ảnh, sai số xuất hiện khi độ dốc mặt đường thay đổi [1].

Temporal filtering

Bộ lọc Kalman [10] được dùng để dự đoán và bám làn qua các khung hình, thường kết hợp với bước khớp đường làn [1], [3]. [1] mô tả một hướng làm gồm khớp bình phương tối thiểu và dùng Kalman để dự đoán, theo dõi làn hiện tại khi vạch bị che hoặc biến mất.

Deep-learning approaches

Các phương pháp phân đoạn ngữ nghĩa và học sâu như LaneNet [5] và SCNN [6] cho độ chính xác cao nhưng cần dữ liệu huấn luyện lớn và tài nguyên tính toán [1]. Bảng so sánh trong [1] (từ Li, 2020) cho thấy độ chính xác của SCNN giảm từ 93,2% ở tập làn thẳng xuống 68,1% ở môi trường ban đêm, nghĩa là ban đêm vẫn là điều kiện khó ngay cả với học sâu.

Báo cáo này không huấn luyện mô hình học sâu. Các con số trên chỉ dùng làm bối cảnh cho phần phân tích lỗi đêm và mưa.

Positioning of this work

Báo cáo này tập trung vào việc tách riêng đóng góp của từng khối xử lý cổ điển (màu, bird's-eye view, đa thức, Kalman) trên cùng dữ liệu và cùng độ đo, bổ sung bộ lọc Kalman tự cài đặt có kiểm chứng và phân tích lỗi trong điều kiện đêm và mưa.

[Lưu ý: [2]–[7] được biết qua danh mục tài liệu tham khảo của bài review [1]. Cần tự tra cứu và đọc phần liên quan của từng tài liệu trước khi giữ lại trích dẫn; chỉ giữ những tài liệu đã mở ra đọc. Tổng quan cần ít nhất 5 tài liệu.]

3.  Methodology (Dataset + Algorithms)

Datasets

TuSimple

TuSimple consists of 1280×720 images per clip, each clip has 20 frames, and only the last frame has labels. The labels are the x-coordinate of each lane at the h_samples rows (160 to 710, step 10); a value of −2 means there is no lane at that row. The data is read by the module src/dataset.py, which includes validation of label format, missing images, and label preview.
To avoid bias, parameters are tuned on a random subset of the training set (n = [200], seed = 42); the test set is run only once for reporting. For the Kalman filter, all 20 frames of the clip are run through the pipeline so that the filter converges before scoring on the final frame.

Dashcam video

Dashcam videos were [self-recorded / obtained from a licensed source (specify source)] under three conditions: daytime, nighttime, and rain, comprising [N] videos, resolution [ ], [ ] fps, device [ ]. The data is organized into condition folders (day/night/rain). License plates and faces are blurred before being included in the report, slides, and demo video.
For each condition, [20–30] frames are manually labeled according to correct/incorrect criteria (both boundaries of the current lane are detected at the correct positions) to compute the correct-frame rate. Homography points are stored separately for each data source in the configs folder. [Optional: additionally use a nighttime subset of CULane [6] to obtain quantitative results for the nighttime condition.]

Bảng 3.1. Tóm tắt các tập dữ liệu

---

Dataset Purpose Scale Labels

---

TuSimple (training subset) Parameter tuning [ ] samples Yes (x-coordinates)

TuSimple (test) Quantitative evaluation, single run [ ] samples Yes (x-coordinates)

Daytime/nighttime/rainy dashcam footage Qualitative evaluation, frame accuracy rate, FPS [N] videos Manually annotated small subset

CULane – nighttime (optional) Supplementary nighttime metrics [ ] Yes

---

System overview

The pipeline consists of the following modules: preprocessing (ROI and image enhancement), lane line detection (Canny or HLS thresholding), perspective transformation, lane fitting, Kalman filtering, offset calculation, and warning generation. Each module can be enabled or disabled through a YAML configuration file in pipeline.py, allowing different ablation configurations to share the same codebase.

[Chèn sơ đồ khối pipeline]

Hình 3.1. Sơ đồ khối của pipeline và các điểm bật/tắt cho ablation

Baseline: Canny, ROI and Hough transform

The baseline consists of grayscale conversion, Gaussian filtering, Canny edge detection (initial thresholds of 50/150), a trapezoidal ROI mask, and HoughLinesP. The detected line segments are classified as left or right lane lines based on the sign of their slopes. The average slope and intercept are then calculated, and the two lane lines are extrapolated accordingly. The Hough transform parameters (threshold, minLineLength, and maxLineGap) are systematically varied to evaluate parameter sensitivity [1].

The expected failure cases of the baseline include curved lanes, dashed lane markings, shadows, and false lane markings caused by reflections. Selected failure examples are retained to motivate the proposed method.

Proposed pipeline

HLS color thresholding

The image is converted into the HLS color space. Yellow lane markings are selected using an H range of 15–35 (on OpenCV's 0–179 scale) and S > 100, while white lane markings are selected using L > 200. This threshold is lowered or adaptively adjusted in dark scenes. The color masks are combined with the Sobel x-gradient computed on the L or S channel. This approach replaces global grayscale thresholding, which is less effective in scenes involving shadows, rain, and nighttime conditions [1].

Illumination enhancement

For nighttime and rainy images, four enhancement techniques are compared before the thresholding step: gamma correction, logarithmic transformation, histogram equalization, and CLAHE applied to the L channel. The selected technique is determined based on the results presented in Section 4.6.

Perspective transform (bird's-eye view)

Bốn điểm nguồn (hình thang trên ảnh gốc) và bốn điểm đích (hình chữ nhật) xác định ma trận homography qua cv2.getPerspectiveTransform. Mỗi nguồn dữ liệu có bộ điểm riêng. Phép biến đổi được kiểm tra bằng ảnh đường thẳng: sau khi warp, hai vạch phải song song. Hạn chế: một ma trận cố định gây sai số khi độ dốc mặt đường thay đổi [1].

[Chèn ảnh trước/sau warp]

Hình 3.2. Ảnh gốc và ảnh bird's-eye view sau khi warp

Sliding window and quadratic fitting

A histogram of the lower half of the binary image is used to determine the initial positions of the two lane lines. Then, [9] sliding windows (margin ≈ [100] px, minpix ≈ [50]) are used to collect pixel coordinates, and a second-degree polynomial x = Ay² + By + C

is fitted using least squares. Starting from the second frame, [optionally] pixels are searched within a region around the previous frame's polynomial fit instead of rerunning the sliding window search.

Since polynomial fitting can become unstable when there are insufficient points or excessive noise [1], each fitted result is validated based on the following criteria: lane width must remain within an acceptable range, the two lane lines must be approximately parallel, and curvature must not exceed a specified threshold. If a fit is invalid, the system reuses the result from the previous frame.

[Chèn ảnh cửa sổ trượt]

Hình 3.3. Minh họa cửa sổ trượt và đa thức bậc hai trên ảnh bird's-eye view

Kalman filter (self-implemented)

Each lane line is tracked using an independent Kalman filter based on a constant-velocity model, implemented with NumPy (src/kalman.py). The coordinates are normalized to the range [0, 1] before fitting so that the coefficients A, B, and C have similar orders of magnitude, allowing the Q and R matrices to be properly balanced.

Trạng thái: x = [A, B, C, A', B', C']^T; đo lường: z = [A, B, C]^T

F = [I3, I3], [0, I3]; H = [I3, 0]

Dự đoán: x(k|k−1) = F·x(k−1); P(k|k−1) = F·P(k−1)·F^T + Q

Cập nhật: S = H·P·H^T + R; K = P·H^T·S^(−1); x = x + K·(z − H·x); P = (I − K·H)·P

Outlier measurements are rejected using Mahalanobis distance gating:

d² = (z−H⋅x)^(T) ⋅ S⁻¹ ⋅ (z−H⋅x) < γ

Where  γ = 11.34, corresponding to the 99% confidence level of a chi-square distribution with 3 degrees of freedom. When a measurement is rejected, the filter performs only the prediction step.

Validation: The same measurement sequence is processed through cv2.KalmanFilter, and the resulting states are compared (with a tolerance of approximately 1e-4 due to OpenCV's use of float32). Additionally, synthetic data (a curve with added Gaussian noise) is used to demonstrate that the Kalman filter reduces estimation error.

Lane departure estimation and warning

The vehicle's lateral offset from the lane center is calculated in pixels and then converted to meters using the meters-per-pixel scale derived from the assumed lane width.

Lateral offset: $d = \left( x_{\text{vehicle}} - \frac{x_{\text{left}} + x_{\text{right}}}{2} \right) \cdot x_{m}\mspace{6mu}\left\lbrack \text{m} \right\rbrack$ , where x\_(vehicle) is the center of the bird's-eye view image (assuming the camera is mounted at the center of the vehicle).

Curvature radius:

$$R = \frac{\left( 1 + (2Ay + B)^{2} \right)^{3/2}}{|2A|}$$

(converted to meters using the x*(m) and y*(m) scaling factors).

The time to lane crossing is estimated using the following formula:

$$TLC = \frac{\text{distance~to~the~nearest~lane~marking}}{\text{lateral~velocity}}$$

where lateral velocity is derived from the velocity components in the Kalman filter state.

The warning is triggered when |d| > [0.3] m continuously for [N] consecutive frames or when TLC < [] s. It is deactivated when |d| < [0.2] m (hysteresis) to prevent flickering.

Implementation

The source code is written in Python [≥ 3.9] using OpenCV [9] and NumPy. The repository structure consists of src/ (dataset.py, baseline.py, preprocess.py, warp.py, lane_fit.py, kalman.py, departure.py, pipeline.py), configs/, eval/, tests/, and results/. The random seed and library versions are fixed in requirements.txt to ensure reproducibility.

Hardware: AMD Ryzen 5000 Series CPU, 20 GB RAM, and NVIDIA GeForce GTX 1650 GPU.

Evaluation protocol and metrics

- TuSimple: Accuracy, FP, and FN are calculated using the official lane.py evaluation script [8], together with lane position error |x*(predicted)−x*(ground truth)| at the h_samples, measured in pixels).

- Dashcam: Frame-level detection accuracy on the manually annotated dataset, evaluated separately under daytime, nighttime, and rainy conditions.

- Processing speed: FPS and per-module processing time (ms/frame) are measured using time.perf_counter, excluding video reading/writing time. The median is calculated over [N] runs.

- Warnings: Number of false alarms per minute on normal-driving sequences (any warning triggered during correct lane-keeping is counted as a false alarm).

- Kalman: Maximum deviation compared with cv2.KalmanFilter and RMSE reduction on synthetic data.

Bảng 3.2. Các cấu hình ablation

---

Cấu hình Nội dung

---

A Canny + ROI + Hough (baseline)

B Ngưỡng HLS (+ Sobel) + ROI + Hough

C B + bird's-eye view + cửa sổ trượt + đa thức bậc 2 + kiểm tra hợp lệ

D (tùy chọn) C + tìm điểm ảnh quanh đa thức của khung trước

E C hoặc D + Kalman tự cài đặt + gating

F E + tăng cường ảnh (CLAHE/gamma), đánh giá trên đêm/mưa

---

4.  Simulation Results

Experimental setup

[Mô tả phần cứng, phiên bản phần mềm, và các tham số cuối cùng (đã chỉnh trên tập con train) vào Bảng 4.1.]

Bảng 4.1. Tham số chính của pipeline

---

Tham số Giá trị Cách chọn

---

Canny (thấp/cao) [ ] Tập con train

Hough (threshold, minLineLength, maxLineGap) [ ] Quét tham số

Ngưỡng HLS (H, S, L) [ ] Tập con train

Cửa sổ trượt (số cửa sổ, margin, minpix) [ ] Tập con train

Kalman (Q, R, γ) [ ] Dữ liệu tổng hợp và train

Cảnh báo (ngưỡng bật/tắt, N, TLC) [ ] Đoạn lái bình thường

---

Kalman filter verification

[Mô tả kịch bản kiểm chứng và nhận xét kết quả sau khi có số liệu.]

Bảng 4.2. Kiểm chứng bộ lọc Kalman tự cài đặt với cv2.KalmanFilter

---

Kịch bản Chênh lệch lớn nhất so với cv2 RMSE đo thô RMSE sau lọc

---

Dữ liệu tổng hợp (nhiễu thấp) … … …

Dữ liệu tổng hợp (nhiễu cao, mất đo) … … …

Chuỗi hệ số thật từ clip TuSimple … … …

---

Quantitative results on TuSimple

[Phân tích sau khi có số liệu: khối nào đóng góp nhiều nhất cho accuracy, FP, FN; khối nào làm giảm fps; khối nào không cải thiện (vẫn đưa vào báo cáo kèm giải thích).]

Bảng 4.3. Ablation trên tập test TuSimple (chạy một lần)

---

Cấu hình Acc (%) FP FN Sai số (px) fps

---

A. Canny + ROI + Hough … … … … …

B. HLS + ROI + Hough … … … … …

C. HLS + BEV + cửa sổ trượt + đa thức … … … … …

D. C + tìm quanh đa thức trước … … … … …

E. D + Kalman … … … … …

---

[Chèn biểu đồ độ nhạy tham số Hough]

Hình 4.1. Độ nhạy của baseline theo tham số Hough (threshold, minLineLength)

Results on dashcam video

Bảng 4.4. Tỉ lệ khung phát hiện đúng và fps trên video dashcam

---

Điều kiện Số khung gán nhãn Baseline (A) Đề xuất (E/F) fps

---

Ban ngày [ ] … … …

Ban đêm [ ] … … …

Trời mưa [ ] … … …

---

[Chèn ảnh so sánh baseline và đề xuất (ngày/đêm/mưa)]

Hình 4.2. So sánh định tính giữa baseline và phương pháp đề xuất ở ba điều kiện

Lane departure warning evaluation

[Báo cáo số cảnh báo sai mỗi phút trên đoạn lái bình thường, so sánh ngưỡng độ lệch đơn thuần với logic có TLC và hysteresis; minh họa bằng đồ thị độ lệch theo thời gian.]

[Chèn đồ thị độ lệch và cảnh báo]

Hình 4.3. Độ lệch theo thời gian và trạng thái cảnh báo trên một đoạn video

Effect of image enhancement at night and in rain

Bảng 4.5. Tỉ lệ khung phát hiện đúng theo phép tăng cường ảnh

---

Phép tăng cường Ban đêm (%) Trời mưa (%)

---

Không tăng cường … …

Gamma … …

Log … …

Cân bằng histogram … …

CLAHE … …

---

Runtime analysis

Bảng 4.6. Thời gian xử lý từng khối (trung vị, ms/khung)

---

Khối Thời gian (ms) Tỉ lệ (%)

---

Tiền xử lý và tách vạch … …

Bird's-eye view … …

Cửa sổ trượt + đa thức … …

Kalman + gating … …

Độ lệch + cảnh báo … …

Tổng … 100

---

Error analysis

Các khung hình lỗi được gom và phân loại theo nguyên nhân. Mỗi nhóm cần có ví dụ minh họa, giải pháp đã thử và kết quả trước/sau. Giả thuyết ban đầu, dựa trên [1] và cần được kiểm chứng bằng dữ liệu của báo cáo:

- Ban đêm: tương phản thấp; lược đồ xám bị chi phối bởi vùng sáng và vùng tối, nên vạch làn lẫn vào vùng sáng mạnh và khó tách bằng ngưỡng; chói đèn pha xe ngược chiều. [Hiện tượng quan sát được; giải pháp đã thử; kết quả.]

- Trời mưa: mặt đường ướt phản chiếu đèn tạo vạch giả, nước và nhiễu làm lược đồ mất đặc trưng; gạt mưa và giọt nước trên kính che tầm nhìn. [Hiện tượng quan sát được; giải pháp đã thử; kết quả.]

- Chung: vạch đứt nét hoặc bị xóa mờ, bóng cây, xe (đặc biệt xe màu trắng) đi sát vạch bị nhận nhầm là vạch làn, xe máy che vạch [1]. [Hiện tượng quan sát được; giải pháp đã thử (Kalman, gating, kiểm tra hợp lệ); kết quả.]

[Chèn ảnh các khung lỗi đêm/mưa]

Hình 4.4. Ví dụ các khung hình lỗi ban đêm và trời mưa kèm nguyên nhân

5.  Conclusions

Summary of findings

[Trả lời lần lượt RQ1–RQ4 bằng số liệu từ Chương 4: đóng góp của từng khối, hiệu quả và chi phí của Kalman, nguyên nhân lỗi đêm/mưa và phép tăng cường hiệu quả, hiệu quả của logic cảnh báo.]

Limitations

- Ma trận homography cố định không bù được thay đổi độ dốc mặt đường [1].

- Hough nhạy với tham số, và mô hình đường thẳng không mô tả được làn cong [1].

- Không có tín hiệu xi-nhan và tốc độ nên không loại được trường hợp chủ động chuyển làn; giả định camera ở giữa xe và chiều rộng làn cố định.

- TuSimple chủ yếu là đường cao tốc; tập dashcam đêm và mưa có quy mô nhỏ và tự gán nhãn. [Bổ sung hạn chế khác sau khi có kết quả.]

Future work

- Vùng quan tâm và homography thích nghi theo điều kiện đường [1].

- So sánh với một mô hình học sâu có sẵn (ví dụ SCNN [6] hoặc Ultra-Fast-Lane-Detection) và mở rộng sang nhiều làn.

- Tối ưu cho thiết bị nhúng và kết hợp tín hiệu xi-nhan, tốc độ xe để giảm cảnh báo sai.

References

[1] W. Chen, W. Wang, K. Wang, Z. Li, H. Li, and S. Liu, “Lane departure warning systems and lane line detection methods based on image processing and semantic segmentation: A review,” J. Traffic Transp. Eng. (Engl. Ed.), vol. 7, no. 6, pp. 748–774, 2020, doi: 10.1016/j.jtte.2020.10.002.

[2] C. R. Jung and C. R. Kelber, “Lane following and lane departure using a linear-parabolic model,” Image Vis. Comput., vol. 23, no. 13, pp. 1192–1202, 2005.

[3] Z. W. Kim, “Robust lane detection and tracking in challenging scenarios,” IEEE Trans. Intell. Transp. Syst., vol. 9, no. 1, pp. 16–26, 2008.

[4] M. Bertozzi and A. Broggi, “GOLD: A parallel real-time stereo vision system for generic obstacle and lane detection,” IEEE Trans. Image Process., vol. 7, no. 1, pp. 62–81, 1998.

[5] D. Neven et al., “Towards end-to-end lane detection: An instance segmentation approach,” in Proc. IEEE Intell. Veh. Symp. (IV), 2018.

[6] X. Pan et al., “Spatial as deep: Spatial CNN for traffic scene understanding,” arXiv:1712.06080, 2018.

[7] S. P. Narote et al., “A review of recent advances in lane detection and departure warning system,” Pattern Recognit., vol. 73, pp. 216–234, 2018.

[8] TuSimple, “TuSimple lane detection benchmark,” [Online]. Available: https://github.com/TuSimple/tusimple-benchmark [kiểm tra lại liên kết và ngày truy cập]

[9] G. Bradski, “The OpenCV Library,” Dr. Dobb’s Journal of Software Tools, 2000.

[10] R. E. Kalman, “A new approach to linear filtering and prediction problems,” J. Basic Eng., vol. 82, no. 1, pp. 35–45, 1960.

Appendices

Appendix A. Task allocation

Dự án được thực hiện [cá nhân – cần giảng viên xác nhận / theo nhóm].

Bảng A.1. Phân công và đóng góp

---

Thành viên MSHV Công việc Số commit

---

[Họ và tên] [ ] [Toàn bộ các hạng mục / liệt kê theo thành viên] [ ]

---

Appendix B. Reproduction guide

Các lệnh dưới đây là kế hoạch; tên script trong eval/ sẽ được hoàn thiện cùng mã nguồn.

pip install -r requirements.txt

python -m src.dataset tusimple --root data/tusimple/train_set --labels <label files> --preview results/gt_preview

python eval/tusimple_eval.py --config configs/E.yaml --split test

python eval/ablation.py

python eval/benchmark_fps.py --video data/dashcam/night/<file>.mp4

python app.py --input data/dashcam/day/<file>.mp4 --method proposed

Appendix C. AI usage declaration

Theo quy định của môn học, các công cụ AI được dùng phải được khai báo gồm công cụ, mục đích và phạm vi.

Bảng C.1. Khai báo sử dụng AI

---

Công cụ Mục đích Phạm vi Xác nhận

---

Claude (Anthropic) Lập kế hoạch và soạn đề cương báo cáo Cấu trúc báo cáo và bản nháp nội dung chưa có số liệu [Tác giả đã rà soát và chỉnh sửa]

Claude (Anthropic) Viết module đọc dữ liệu và test src/dataset.py, tests/test_dataset.py [Tác giả đã chạy và hiểu mã]

[Công cụ khác, nếu có] [Debug / diễn đạt / khác] [ ] [ ]

Không dùng AI Cài đặt bộ lọc Kalman (src/kalman.py) Tự viết, kiểm chứng với cv2.KalmanFilter [Xác nhận sau khi hoàn thành]

---

## PART B — Hướng dẫn Dự án nhóm.docx (chỉ các mục liên quan)

2. Tổ chức và quy định chung

+----------------+-------------------------------------------------------------------------------------+
| Nội dung | Quy định |
+================+=====================================================================================+
| Quy mô nhóm | 4–5 học viên; nhóm tự đăng ký trên website môn học, giảng viên cân đối |
+----------------+-------------------------------------------------------------------------------------+
| Thời gian | 3 tuần (21 ngày) kể từ ngày công bố |
+----------------+-------------------------------------------------------------------------------------+
| Đề tài | Chọn 1 trong 10 đề tài gợi ý (mục 4) hoặc tự đề xuất (duyệt trước ngày 4) |
+----------------+-------------------------------------------------------------------------------------+
| Giới hạn | Mỗi đề tài tối đa 2 nhóm/lớp, hai nhóm phải khác hướng tiếp cận |
+----------------+-------------------------------------------------------------------------------------+
| Công cụ | Python ≥ 3.9, OpenCV, scikit-image, NumPy/SciPy, scikit-learn; PyTorch (tùy chọn) |
+----------------+-------------------------------------------------------------------------------------+
| Mã nguồn | Repository Git riêng (GitHub/GitLab), mời giảng viên; mọi thành viên có commit |
+----------------+-------------------------------------------------------------------------------------+
| Nộp muộn | Trừ 10% điểm của mốc tương ứng cho mỗi ngày trễ |
+----------------+-------------------------------------------------------------------------------------+

3. Yêu cầu kỹ thuật tối thiểu

• Sử dụng ít nhất 4 kỹ thuật thuộc ít nhất 3 bài học khác nhau của môn học. • Có phương pháp cơ sở (baseline) và ít nhất một cải tiến hoặc phương án so sánh. • Tự cài đặt ít nhất một thuật toán cốt lõi (ví dụ: Otsu, Canny, Hough, Harris, phân thủy, k-means, block matching, Kalman, DCT/Huffman…), kiểm chứng với hàm thư viện.

• Đánh giá định lượng trên dữ liệu có nhãn/ground truth, với độ đo phù hợp (bảng mục 3.1). • Ablation study: phân tích ảnh hưởng của từng khối xử lý hoặc tham số chính. • Demo chạy được: ứng dụng Streamlit/Gradio hoặc script dòng lệnh, kèm video minh họa.

3.1. Độ đo gợi ý
Đề tài 4: Phát hiện làn đường và cảnh báo chệch làn (Lane Detection)

• Mô tả: Từ video camera hành trình: phát hiện hai vạch làn, ước lượng vị trí xe trong làn, cảnh báo chệch làn

• Kỹ thuật: Kỹ thuật: ngưỡng màu HLS, Canny, ROI, biến đổi Hough / cửa sổ trượt, khớp đa thức, bird's-eye view (homography), lọc Kalman theo thời gian

• Dữ liệu / yêu cầu tối thiểu: Dữ liệu: TuSimple, CULane, video dashcam tự quay (ban ngày/đêm/mưa)

• Đánh giá / mở rộng: Đánh giá: tỉ lệ khung phát hiện đúng, sai số vị trí làn, tốc độ xử lý (fps) Đề tài 5: Giám sát thông minh - phát hiện, theo dõi, đếm (Smart Surveillance)
4.1. Đề tài tự đề xuất

Học viên được khuyến khích đề xuất đề tài gắn với lĩnh vực công tác. Đề xuất 1 trang gồm: bài toán, dữ liệu, kỹ thuật dự kiến (≥ 3 bài học), độ đo, rủi ro và phương án dự phòng. Tiêu chí duyệt: khả thi trong 3 tuần, có dữ liệu đánh giá, đủ hàm lượng xử lý ảnh/video. 7. Quy định học thuật và sử dụng AI

• Được dùng thư viện, mã nguồn mở nhưng phải ghi rõ nguồn; phần tự cài đặt phải do nhóm viết. • Được dùng công cụ AI hỗ trợ (viết mã, sửa lỗi, diễn đạt) nhưng phải khai báo trong phụ lục báo cáo: công cụ, mục đích, phạm vi.

• Mọi thành viên phải giải thích được mã nguồn và kết quả khi bảo vệ.

• Sao chép không trích dẫn → 0 điểm dự án và xử lý theo quy chế.

• Dữ liệu có thông tin cá nhân (khuôn mặt, biển số) phải được ẩn danh khi công bố. 9. Cấu trúc mã nguồn gợi ý

+----------------------------------------------------------------------------------------------------------------------+
| # project/ |
| |
| # ├── data/ # dữ liệu (không đẩy file lớn lên Git; ghi link tải trong README) # ├── src/ |
| |
| # │ ├── preprocess.py # tiền xử lý |
| |
| # │ ├── method.py # phương pháp chính + baseline |
| |
| # │ └── evaluate.py # độ đo, bảng kết quả |
| |
| # ├── app.py # demo (Streamlit / Gradio / OpenCV GUI) |
| |
| # ├── notebooks/ # thử nghiệm |
| |
| # ├── results/ # hình, bảng, video kết quả |
| |
| # ├── requirements.txt |
| |
| # └── README.md # cài đặt, cách chạy, phân công, kết quả chính |
Phụ lục B. Mẫu báo cáo cuối kỳ

+----------------------+---------------------------------------------------------+-------------+
| Phần | Nội dung | Độ dài |
+======================+=========================================================+=============+
| Tóm tắt | 150–250 từ: bài toán, phương pháp, kết quả chính | ½ trang |
+----------------------+---------------------------------------------------------+-------------+
| 1. Giới thiệu | Bài toán, động lực, mục tiêu, đóng góp | 1–2 trang |
+----------------------+---------------------------------------------------------+-------------+
| 2. Tổng quan | Các phương pháp liên quan (≥ 5 tài liệu) | 1–2 trang |
+----------------------+---------------------------------------------------------+-------------+
| 3. Phương pháp | Sơ đồ khối, công thức, thuật toán, lý do lựa chọn | 3–5 trang |
+----------------------+---------------------------------------------------------+-------------+
| 4. Thực nghiệm | Dữ liệu, thiết lập, độ đo, kết quả, so sánh, ablation | 4–6 trang |
+----------------------+---------------------------------------------------------+-------------+
| 5. Thảo luận | Phân tích lỗi, hạn chế, hướng phát triển | 1–2 trang |
+----------------------+---------------------------------------------------------+-------------+
| 6. Kết luận | Tóm tắt kết quả | ½ trang |
+----------------------+---------------------------------------------------------+-------------+
| Tài liệu tham khảo | Định dạng IEEE | - |
+----------------------+---------------------------------------------------------+-------------+
| Phụ lục | Phân công, hướng dẫn chạy, khai báo sử dụng AI | - |
+----------------------+---------------------------------------------------------+-------------+
