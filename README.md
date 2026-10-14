# DT049 — Phát Hiện Bất Thường Chuỗi Thời Gian Đa Biến Trên Bộ Dữ Liệu HAI 21.03

Phát hiện bất thường chuỗi thời gian đa biến (Multivariate Time-Series Anomaly Detection) trên bộ dữ liệu chuẩn công nghiệp Hardware-in-the-Loop Security (HAI) 21.03, sử dụng các mô hình dự đoán nền tảng kiểm soát rò rỉ dữ liệu (Persistence, Ridge) và mô hình mạng nơ-ron hồi quy Gated Recurrent Unit (GRU) gọn nhẹ dưới một giao thức đánh giá chính thức đã đóng băng.

---

## 1. Tổng Quan Dự Án (Project Overview)

DT049 là đồ án nghiên cứu thực nghiệm môn học, thực hiện nghiên cứu tái lập có chọn lọc và điều chỉnh (selective and adapted reproduction) lấy cảm hứng từ kiến trúc mạng nơ-ron của bài báo khoa học năm 2025 trên tạp chí *Computers & Security*, được thích ứng và triển khai trên bộ dữ liệu hệ thống điều khiển công nghiệp (ICS) công khai **HAI 21.03**:

* **Bài báo tham chiếu gốc:** Karel Kuchař & Radek Fujdiak (2025), *"Analyzing anomalies in industrial networks: A data-driven approach to enhance security in manufacturing processes"*, *Computers & Security*, Vol. 153, 104395. DOI: [`10.1016/j.cose.2025.104395`](https://doi.org/10.1016/j.cose.2025.104395) (Open Access).
* **Bộ dữ liệu thực nghiệm:** **HAI Security Dataset 21.03** (công bố bởi ETRI Hàn Quốc, gồm 79 kênh tín hiệu cảm biến và cơ cấu chấp hành ở tần số lấy mẫu 1 Hz).

Dự án thiết lập một khung thực nghiệm kiểm soát chặt chẽ rò rỉ dữ liệu (leakage-controlled experimental framework) với hai nhánh thực nghiệm có ranh giới rõ ràng:
1. **Thực nghiệm V1 (Dự Báo Chuỗi Thời Gian Không Giám Sát — Unsupervised Time-Series Forecasting):**
   - **Baseline 0 (Persistence):** Mô hình quán tính giữ nguyên trạng thái một bước ($\hat{\mathbf{x}}_t = \mathbf{x}_{t-1}$).
   - **Baseline 1 (Ridge Regression):** Bộ dự đoán tuyến tính đa đầu ra có điều chuẩn L2 ($\alpha = 300.0$).
   - **Mô Hình Chính — Primary Model (GRU):** Mạng nơ-ron hồi quy Gated Recurrent Unit 1 tầng gọn nhẹ ($W=30$, Hidden=64, $32.975$ tham số).
   - Huấn luyện hoàn toàn trên dữ liệu bình thường (`train1 + train2`, $0$ cuộc tấn công), hiệu chuẩn ngưỡng $Q_{99,9}$ trên `train3` normal, và đánh giá trên toàn bộ 50 cuộc tấn công của `test1`–`test5`.
2. **Thực nghiệm V2 (Tái Lập Kiến Trúc Có Điều Chỉnh — Selective Adapted DNN Reproduction):**
   - **Feedforward DNN V2:** Mạng truyền thẳng 6 tầng ẩn Dense `[100, 70, 100, 80, 60, 10]` với chuỗi kích hoạt `ReLU/Tanh` và 2 lớp `Dropout(0.3)` xác minh trực tiếp từ **Section 5.5 / Listing 1** của bài báo Kuchař & Fujdiak (2025), gồm **$35.731$ tham số**.
   - Thích ứng sang bài toán phân loại nhị phân `Normal/Attack` có giám sát trên 79 kênh tín hiệu HAI 21.03 (huấn luyện trên `test1 + test2` gồm 25 cuộc tấn công, kiểm định chọn ngưỡng trên `test3` gồm 8 cuộc tấn công, và đánh giá giữ ngoài trên `test4 + test5` gồm 17 cuộc tấn công).

Quá trình đánh giá tập trung vào các thước đo vận hành thực tế trong công nghiệp: Precision, Recall và F1 ở cấp độ sự kiện tấn công liên tục (contiguous attack event-level 1-to-1 matching), tần suất báo động giả trong quá trình vận hành bình thường kéo dài (`Normal FA/h`), và độ trễ phát hiện thực nghiệm (detection delay).

> [!NOTE]
> Nghiên cứu này **không** tuyên bố tái lập nguyên bản 100% các con số kết quả hay độ chính xác 4 lớp ($99,72\%$) của bài báo gốc. Bài báo gốc của Kuchař & Fujdiak (2025) đánh giá bài toán phân loại 4 trạng thái (`Normal`, `Security`, `Operational`, `Service`) trên một hệ thống testbed mạng công nghiệp riêng; đồng thời bài báo có sự không nhất quán nội bộ giữa phần mô tả thực nghiệm (đề cập 87 đặc trưng) và Listing 1 (khai báo cứng `input_dim = 10`). DT049 kiểm chứng cấu trúc mạng 6 tầng ẩn của Listing 1 khi được thích ứng có hệ thống sang bộ dữ liệu chuẩn HAI 21.03 (79 biến quá trình vật lý liên tục) dưới bài toán phân loại nhị phân `Normal/Attack` thì thể hiện hiệu năng như thế nào và có sự đánh đổi gì so với biến thể dự báo chuỗi thời gian không giám sát GRU.

---

## 2. Các Câu Hỏi Nghiên Cứu Chính (Main Research Questions)

1. **Câu hỏi nghiên cứu V1:** *Liệu một bộ dự đoán chuỗi thời gian gọn nhẹ như GRU ($W=30$) có thể cải thiện khả năng phát hiện sự kiện tấn công và giảm độ trễ phát hiện so với các mô hình nền tảng đơn giản (Persistence và Ridge) trên HAI 21.03 dưới một giao thức đánh giá đóng băng không rò rỉ dữ liệu hay không?*
2. **Câu hỏi nghiên cứu V2 (Tái lập có điều chỉnh):** *Khi tái lập kiến trúc mạng truyền thẳng Feedforward DNN 6 tầng ẩn từ Listing 1 của Kuchař & Fujdiak (2025) và điều chỉnh sang phân loại nhị phân có giám sát trên 79 đặc trưng của HAI 21.03, mô hình đạt hiệu năng phát hiện như thế nào trên tập giữ ngoài `test4 + test5` (17 sự kiện), và sự đánh đổi về mặt vận hành (độ trễ, độ nhạy ngưỡng, báo động giả) giữa DNN có giám sát và GRU không giám sát là gì?*
3. **Câu hỏi khảo sát hậu nghiệm (Dynamic57 Ablation):** *Các kênh tín hiệu có phương sai bằng 0 trong quá trình huấn luyện có đóng góp ý nghĩa vào điểm số bất thường và quyết định báo động trên tập kiểm thử chính thức hay không?*.

---

## 3. Bộ Dữ Liệu (Dataset)

Dự án sử dụng bộ dữ liệu **HAI (Hardware-in-the-loop Augmented ICS) Security Dataset 21.03**, được công bố bởi Viện Nghiên cứu Điện tử và Viễn thông Hàn Quốc (ETRI).
- **Kho lưu trữ gốc (Upstream Repository)**: [https://github.com/icsdataset/hai](https://github.com/icsdataset/hai)
- **Upstream Git Commit**: `2a814cebc9a66b06c9e5cd545e2d72e65d383737`
- **Tổng số tệp**: 8 tệp CSV nén (`train1`–`train3`, `test1`–`test5`)
- **Số kênh tín hiệu (Telemetry Channels)**: 79 biến quá trình liên tục trải rộng trên 4 phân hệ vật lý liên kết chặt chẽ (P1: Lò hơi - Boiler, P2: Tua-bin - Turbine, P3: Xử lý nước - Water Treatment, P4: Mô phỏng HIL - HIL Simulator)
- **Số sự kiện tấn công trên tập Test**: 50 sự kiện tấn công vật lý đa cảm biến liên tục

| Tên Phân Tập | Tên Tệp | Vai Trò | Số Dòng (Giây) | Thời Lượng | Số Giây Tấn Công | Số Sự Kiện Tấn Công |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| Development Train 1 | `train1.csv.gz` | Huấn luyện mô hình (Normal) | 216,001 | 60.00 h | 0 | 0 |
| Development Train 2 | `train2.csv.gz` | Huấn luyện mô hình (Normal) | 226,801 | 63.00 h | 0 | 0 |
| Validation Normal | `train3.csv.gz` | Chọn mô hình & Hiệu chuẩn ngưỡng | 478,801 | 133.00 h | 0 | 0 |
| Official Test 1 | `test1.csv.gz` | Phiên kiểm thử chính thức 1 | 43,201 | 12.00 h | 629 | 5 |
| Official Test 2 | `test2.csv.gz` | Phiên kiểm thử chính thức 2 | 118,801 | 33.00 h | 3,449 | 20 |
| Official Test 3 | `test3.csv.gz` | Phiên kiểm thử chính thức 3 | 108,001 | 30.00 h | 1,535 | 8 |
| Official Test 4 | `test4.csv.gz` | Phiên kiểm thử chính thức 4 | 39,601 | 11.00 h | 1,157 | 5 |
| Official Test 5 | `test5.csv.gz` | Phiên kiểm thử chính thức 5 | 92,401 | 25.67 h | 2,177 | 12 |
| **Tổng Tập Test** | `test1`–`test5` | **Kiểm Thử Chính Thức Đóng Băng** | **402,005** | **111.67 h** | **8,947** | **50** |

*Lưu ý: Các tệp dữ liệu thô được cấu hình trong `.gitignore` và không đẩy lên Git. Toàn bộ mã băm toàn vẹn SHA-256 của dữ liệu gốc được lưu trữ cố định tại `results/audit/`.*

### Sự Sai Lệch Tài Liệu Trong Tệp `train3.csv.gz`
Tài liệu README gốc của ETRI ghi nhận `train3.csv.gz` chứa 229 giờ dữ liệu bình thường. Tuy nhiên, đối tượng Git blob thực tế trên kho lưu trữ upstream tại commit `2a814cebc9a66b06c9e5cd545e2d72e65d383737` (Git blob `207e4b9f4c157505d5cb4a66e8ebe84d03575dab`, dung lượng nén 67,961,579 bytes) chứa chính xác 478,801 bản ghi liên tục ở tần số 1 Hz, tương ứng 133.00 giờ (thiếu hụt 96 giờ so với mô tả trong tài liệu). Kiểm tra đối chiếu SHA-256 và cây Git xác nhận đây là sai lệch trong tài liệu gốc của ETRI, hoàn toàn không phải lỗi tải dữ liệu cục bộ.

---

## 4. Thiết Kế Thực Nghiệm Kiểm Soát Rò Rỉ Dữ Liệu (Leakage-Controlled Design)

Dự án áp dụng ranh giới phân tách dữ liệu và trình tự thời gian nghiêm ngặt nhằm ngăn chặn hiện tượng rò rỉ dữ liệu (data leakage):
- **Huấn luyện phát triển (`train1` + `train2`):** 442,802 giây bình thường (123.0 giờ). Chỉ sử dụng để fit bộ chuẩn hóa `StandardScaler`, xác định phương sai kênh tín hiệu, huấn luyện mô hình hồi quy Ridge và tối ưu hóa trọng số mạng GRU.
- **Kiểm định bình thường (`train3`):** 478,801 giây bình thường (133.0 giờ). Chỉ sử dụng để chọn checkpoint dừng sớm (early stopping) và hiệu chuẩn ngưỡng báo động bất thường (a priori threshold calibration).
- **Kiểm thử chính thức (`test1`–`test5`):** 402,005 giây (111.67 giờ) chứa 50 cuộc tấn công vật lý. Được giữ kín hoàn toàn cho đến khi tất cả mô hình, kích thước cửa sổ trượt, siêu tham số và ngưỡng báo động đã được đóng băng vĩnh viễn.

> [!IMPORTANT]
> Bước kiểm toán mô tả dữ liệu ban đầu (descriptive dataset audit) có đọc qua tệp kiểm thử và nhãn để kiểm tra tính toàn vẹn cũng như cấu trúc sự kiện. Tuyệt đối không có bất kỳ thông tin nào từ tập kiểm thử được sử dụng cho việc huấn luyện mô hình, lựa chọn đặc trưng, chọn siêu tham số/cửa sổ thời gian, chọn mô hình hay hiệu chuẩn ngưỡng báo động.

---

## 5. Các Mô Hình (Models)

Tất cả các mô hình đều thực hiện dự đoán vector tín hiệu đã chuẩn hóa ở giây tiếp theo $\hat{\mathbf{x}}_t \in \mathbb{R}^{79}$ dựa trên các quan sát trước đó:

1. **Baseline 0 — Persistence:**
   $$\hat{\mathbf{x}}_t = \mathbf{x}_{t-1}$$
   Mô hình nền tảng không tham số, nắm bắt tính tự tương quan thời gian bậc một.
2. **Baseline 1 — Ridge Regression:**
   $$\hat{\mathbf{x}}_t = \mathbf{x}_{t-1} \mathbf{W} + \mathbf{b}$$
   Bộ dự đoán tuyến tính đa đầu ra với điều chuẩn L2 ($\alpha = 300.0$), được giải nghiệm đóng (closed-form) trên các bước chuyển trạng thái huấn luyện đã chuẩn hóa.
3. **Mô Hình Chính — GRU (Gated Recurrent Unit):**
   - Số chiều đầu vào: 79 kênh
   - Kích thước cửa sổ trượt (Window size): $W = 30$ giây (cửa sổ quá khứ $[\mathbf{x}_{t-W}, \dots, \mathbf{x}_{t-1}]$)
   - Số đơn vị ẩn (Hidden units): 64
   - Số tầng hồi quy: 1 (kiến trúc gọn nhẹ với **32,975 tham số** huấn luyện)
   - Số chiều đầu ra: 79 kênh
   - Các hạt giống ngẫu nhiên (Seeds): 42, 43, 44 (Seed 42 được chỉ định làm mô hình đại diện chuẩn — Canonical; Seed 43 và 44 đóng vai trò bản sao kiểm tra độ ổn định)
   - Hàm mất mát: Mean Squared Error (MSE), tối ưu bằng Adam ($lr = 10^{-3}$) kết hợp lịch giảm tốc Cosine Annealing
   - Thực thi: Chạy hoàn toàn trên **CPU** (CPU-only execution, không phụ thuộc GPU).

---

## 6. Điểm Số Bất Thường & Hiệu Chuẩn Ngưỡng (Anomaly Score & Threshold Calibration)

Đối với mọi bộ phát hiện, điểm số bất thường tại giây thứ $t$ là Căn bậc hai Sai số Toàn phương Trung bình (RMSE / chuẩn L2 chuẩn hóa của phần dư dự đoán) trên toàn bộ 79 kênh tín hiệu đã chuẩn hóa:
$$\text{score}(t) = \sqrt{\frac{1}{79} \sum_{j=1}^{79} \left( x_{t, j} - \hat{x}_{t, j} \right)^2}$$

### Hiệu Chuẩn Ngưỡng (Threshold Calibration)
Các ngưỡng vận hành được hiệu chuẩn cố định trên phân tập kiểm định bình thường (`train3`) trước khi đánh giá trên dữ liệu kiểm thử:
- **Điểm Vận Hành Chính ($Q_{99.9}$):** Phân vị thực nghiệm thứ 99.9 của phần dư kiểm định bình thường (mục tiêu tương ứng $1 \text{ cảnh báo trên mỗi } 1,000 \text{ giây bình thường}$).
  - Persistence: $\tau = 1.680523$
  - Ridge: $\tau = 1.400928$
  - GRU Seed 42: $\tau = 1.058582$ (Seed 43: $\tau = 1.195869$, Seed 44: $\tau = 1.076832$)
- **Các Phân Vị Độ Nhạy (Sensitivity Quantiles):** $Q_{99.5}$ (cận dưới) và $Q_{99.99}$ (cận trên) chỉ được đánh giá để phân tích độ nhạy.

---

## 7. Giao Thức Kiểm Thử Chính Thức Đóng Băng (Frozen Final-Test Protocol)

Quá trình đánh giá kiểm thử cuối cùng được thực hiện dưới một giao thức đóng băng nghiêm ngặt ([docs/final_test_protocol.md](docs/final_test_protocol.md)):
- **Trục Thời Gian So Sánh Chung:** Đánh giá trên toàn bộ các bước chuyển trạng thái $t \ge 30$ nhằm đảm bảo tính tương đương ghép cặp 1-1 tuyệt đối giữa các mô hình nền tảng $W=1$ và mô hình GRU $W=30$. Trục này bao gồm chính xác 401,855 bước chuyển trạng thái được chấm điểm và giữ nguyên 100% (50/50) sự kiện tấn công thực tế.
- **Tính Độc Lập Giữa Các Phiên:** Mỗi tệp kiểm thử (`test1`–`test5`) được xử lý độc lập; không có cửa sổ trượt nào vắt qua ranh giới giữa hai tệp.
- **Định Nghĩa Sự Kiện:** 
  - Sự kiện tấn công thực tế (Ground-truth attack event): Chuỗi liên tục dài nhất thỏa mãn $\text{attack} == 1$.
  - Sự kiện báo động dự đoán (Predicted alarm event): Chuỗi liên tục dài nhất thỏa mãn $\text{score}(t) > \tau$.
- **Quy Tắc Khớp Sự Kiện Theo Trình Tự Thời Gian (Strict Chronological Matching):** Một sự kiện báo động dự đoán được tính là khớp với một sự kiện tấn công nếu chúng có giao thoa thời gian. Mỗi cuộc tấn công chỉ được tính phát hiện tối đa một lần (báo động giao thoa đầu tiên được tính là TP; các báo động phát sinh thêm trong cùng một cuộc tấn công được coi là dư thừa và tính vào FP cấp sự kiện). Các báo động không khớp nằm trong thời gian bình thường cấu thành báo động giả vận hành (normal false alarms).
- **Các Thước Đo Đánh Giá:**
  - $\text{Event Recall} = \text{TP} / \text{Tổng số cuộc tấn công đánh giá}$
  - $\text{Event Precision} = \text{TP} / \text{Tổng số sự kiện báo động dự đoán}$
  - $\text{Event F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$
  - $\text{Detection Delay} = t_{\text{báo động đầu tiên}} - t_{\text{bắt đầu tấn công}}$ (giây)
  - $\text{Normal False Alarm Rate} = \frac{\text{Số báo động không khớp trong thời gian bình thường}}{\text{Số giờ vận hành bình thường}}$ ($\text{sự kiện/giờ}$)
  - $\text{False Positive Alarm Time} = \text{Tổng số giây báo động giả / Số giờ vận hành bình thường}$ ($\text{giây/giờ}$)
  - $\text{Point FPR} = \text{Số giây báo động giả} / \text{Tổng số giây bình thường}$

> [!NOTE]
> Giao thức phân biệt rõ ràng giữa **báo động dự đoán không khớp cấp sự kiện** (bao gồm cả các chuông báo lặp lại dư thừa bên trong một cuộc tấn công) và **báo động giả vận hành lúc bình thường** (các sự kiện làm gián đoạn người vận hành khi hệ thống thực sự không có tấn công).

---

## 8. Kết Quả Chính Thức Đã Đóng Băng (Main Frozen Results)

Kết quả kiểm thử chính thức đã đóng băng trên cả 5 phiên kiểm thử (401,855 bước chuyển trạng thái, 109.1411 giờ vận hành bình thường) tại ngưỡng chính $Q_{99.9}$:

| Mô Hình | Phát Hiện (TP) | Bỏ Lỡ (FN) | Event Recall | Event Precision | Event F1 | Báo Động Giả (ev/h) | Thời Gian FA (s/h) | Độ Trễ Trung Vị |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline 0 (Persistence)** | 22 / 50 | 28 | 0.4400 | 0.0487 | 0.0876 | 3.7016 | **3.92 s/h** | 40.5 s |
| **Baseline 1 (Ridge $\alpha=300$)** | 35 / 50 | 15 | 0.7000 | 0.0636 | 0.1167 | 4.4804 | 6.52 s/h | **13.0 s** |
| **Primary GRU (Seed 42, Canonical)** | **40 / 50** | **10** | **0.8000** | **0.1067** | **0.1882** | **2.8220** | 5.89 s/h | **22.0 s** |
| *Primary GRU (Seed 43, Replicate 1)* | 39 / 50 | 11 | 0.7800 | 0.1096 | 0.1921 | 2.7212 | 5.54 s/h | 21.0 s |
| *Primary GRU (Seed 44, Replicate 2)* | 41 / 50 | 9 | 0.8200 | 0.1297 | 0.2240 | 2.2906 | 5.31 s/h | 24.0 s |
| **Trung Bình 3-Seed GRU** | **40.0 ± 0.8** | **10.0 ± 0.8** | **0.8000 ± 0.0163** | **0.1153 ± 0.0103** | **0.2015 ± 0.0160** | **2.6113 ± 0.2305** | 5.58 ± 0.24 s/h | **22.33 ± 1.25 s** |

### Các Phát Hiện Khoa Học & Sự Đánh Đổi Cốt Lõi (Key Findings & Tradeoffs)
1. **Độ Nhạy Phát Hiện (Detection Sensitivity):** GRU vượt trội rõ rệt so với các mô hình nền tảng, phát hiện thành công **$80.0\%$** các cuộc tấn công thực tế ($40/50$), so với $44.0\%$ của Persistence và $70.0\%$ của Ridge.
2. **Giảm Thiểu Báo Động Giả (False-Alarm Mitigation):** GRU giảm tần suất các sự kiện báo động giả trong điều kiện vận hành bình thường kéo dài:
   - **Tính trên trung bình 3-seed GRU ($2.6113\text{ ev/h}$):** Giảm **$29.5\%$** so với Persistence ($3.7016\text{ ev/h}$) và giảm **$41.7\%$** so với Ridge ($4.4804\text{ ev/h}$).
   - **Tính riêng cho mô hình Canonical Seed 42 ($2.8220\text{ ev/h}$):** Giảm **$23.8\%$** so với Persistence và giảm **$37.0\%$** so với Ridge.
   - *Bối cảnh vận hành:* Mức $2.61\text{ ev/h}$ (hay $2.82\text{ ev/h}$ của Seed 42) tương đương khoảng **$20.6$ đến $22.6$ lần cảnh báo trong một ca trực 8 giờ** của kỹ sư vận hành. Đây là bằng chứng cho thấy báo động giả vẫn là một thách thức lớn khi triển khai thực tế trên chuỗi thời gian công nghiệp dài, đòi hỏi các cơ chế quản lý cảnh báo (như hysteresis hoặc alarm persistence filter) trong các nghiên cứu mở rộng tiếp theo.
3. **Sự Đánh Đổi Về Thời Gian Chiếm Dụng Chuông Báo (The Alarm Time Tradeoff):** Mặc dù GRU tạo ra ít sự kiện báo động giả rời rạc nhất, Persistence lại đạt tổng thời lượng chuông báo giả thấp nhất ($3.92\text{ giây/giờ}$ so với $5.89\text{ giây/giờ}$ của GRU seed 42). Nguyên nhân là sai số của Persistence lập tức quay về 0 sau 1–2 giây khi nhiễu tức thời đi qua, trong khi trạng thái ẩn của GRU cần vài giây để dao động tắt dần.
4. **Bối Cảnh Của Event Precision:** Giá trị Event Precision khiêm tốn (~$0.11$) là đặc tính tất yếu khi đánh giá liên tục suốt 109.14 giờ ($392,908$ giây bình thường) dưới quy tắc khớp sự kiện 1-1 khắt khe mà không dùng thủ thuật gộp chuông (alarm merging) hay trễ nhân tạo (hysteresis). Con số này **không** phản ánh sự yếu kém của bộ phát hiện.

---

## 9. Phân Tích Độ Nhạy Ngưỡng (Sensitivity Analysis)

Bảng khảo sát độ nhạy trên các phân vị kiểm định được định trước nhằm đánh giá tính bền vững của mô hình:

| Điểm Vận Hành | Mô Hình | Recall | Precision | Event F1 | Tần Suất FA Bình Thường | Độ Trễ Trung Vị | Trạng Thái |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **$Q_{99.5}$** | GRU Seed 42 | 0.9800 | 0.0269 | 0.0524 | 14.8707 ev/h | 15.0 s | Cận dưới độ nhạy |
| **$Q_{99.9}$** | **GRU Seed 42** | **0.8000** | **0.1067** | **0.1882** | **2.8220 ev/h** | **22.0 s** | **ĐIỂM VẬN HÀNH CHÍNH ĐÓNG BĂNG** |
| **$Q_{99.99}$** | GRU Seed 42 | 0.7200 | 0.4000 | 0.5143 | 0.3390 ev/h | 24.5 s | Cận trên độ nhạy |

> [!WARNING]
> Mặc dù $Q_{99.99}$ đạt chỉ số Event F1 trên tập test cao hơn ($0.5143$), **$Q_{99.9}$ vẫn giữ nguyên là điểm vận hành chính thức duy nhất**. Trong thực nghiệm khoa học kiểm soát rò rỉ nghiêm ngặt, ngưỡng đã được chốt trên tập kiểm định không được phép thay thế hậu nghiệm (post-hoc) bằng ngưỡng tối ưu nhìn thấy trên tập kiểm thử.

---

## 10. Nghiên Cứu Cắt Giảm Đặc Trưng Động (Dynamic57 Feature Ablation)

Trong quá trình huấn luyện phát triển, 22 trên tổng số 79 kênh tín hiệu có phương sai bằng 0 trong điều kiện vận hành bình thường, trong khi 57 kênh có phương sai động khác 0. Nghiên cứu khảo sát hậu nghiệm **Dynamic57 Ablation** kiểm tra xem việc chỉ tính điểm bất thường trên 57 kênh biến thiên có làm thay đổi hiệu năng phát hiện hay không (mà không huấn luyện lại mô hình).

### Chứng Minh Lý Thuyết & Hệ Số Tỷ Lệ (Theoretical Derivation & Scaling Factor)
Trên dữ liệu kiểm định bình thường (`train3`), 22 kênh bị loại trừ có phần dư bằng 0 ($e_j(t) = 0$). Dưới điều kiện này, điểm số trên 57 đặc trưng là một phép co giãn tỷ lệ hằng số chính xác của điểm số 79 đặc trưng:
$$\kappa = \sqrt{\frac{79}{57}} \approx 1.177270$$
$$\text{RMSE}_{57}(t) = \sqrt{\frac{79}{57}} \cdot \text{RMSE}_{79}(t) = \kappa \cdot \text{RMSE}_{79}(t)$$
*(Lưu ý: Tỷ số chiều không gian đặc trưng là $\frac{79}{57} \approx 1.385965$, do đó hệ số tỷ lệ căn bậc hai là $\kappa = \sqrt{\frac{79}{57}} \approx 1.177270$.)*

Bởi vì cả điểm số lẫn ngưỡng phân vị đều nhân với cùng hằng số $\kappa$, mọi quyết định báo động là bất biến về mặt toán học trên tập kiểm định bình thường.

### Kết Luận Trên Tập Kiểm Thử Chính Thức
Trên tập kiểm thử chính thức, một số kênh trong 22 kênh hằng số có phát sinh dao động, do đó tính chất co giãn hằng số tuyệt đối không còn giữ nguyên. Tuy nhiên, xét về mặt thực nghiệm, đóng góp của chúng không đủ lớn để làm thay đổi bất kỳ quyết định báo động nào tại ngưỡng $Q_{99.9}$:
> **"22 kênh không đổi lúc huấn luyện không làm thay đổi bất kỳ quyết định báo động nào của bộ phát hiện dưới phép thử hậu nghiệm Q99.9 trên HAI 21.03."**

---

## 11. Thí Nghiệm V2 — Tái Lập Có Điều Chỉnh Feedforward DNN (Kuchař & Fujdiak, 2025) trên HAI 21.03

Thí nghiệm V2 (`DT049_V2_SUPERVISED_DNN`) bổ sung mô hình **Feedforward DNN** làm nhánh tái lập có điều chỉnh tham chiếu bài báo gốc Kuchař & Fujdiak (2025), DOI [`10.1016/j.cose.2025.104395`](https://doi.org/10.1016/j.cose.2025.104395), và đối chiếu đối đầu với biến thể dự báo chuỗi thời gian không giám sát **GRU V1** trên cùng tập kiểm tra giữ ngoài `test4 + test5` ($t \ge 30$, 17 sự kiện tấn công).

### 11.1. Kiến trúc Đã Xác minh từ Section 5.5 / Listing 1
* **Kiến trúc mạng:** `Input(79)` $\rightarrow$ `Dense(100, relu)` $\rightarrow$ `Dense(70, tanh)` $\rightarrow$ `Dense(100, relu)` $\rightarrow$ `Dense(80, tanh)` $\rightarrow$ `Dense(60, tanh)` $\rightarrow$ `Dropout(0.3)` $\rightarrow$ `Dense(10, tanh)` $\rightarrow$ `Dropout(0.3)` $\rightarrow$ `Linear(1 logit)`.
* **Tổng số tham số:** Đúng **35.731 tham số** khả huấn luyện.
* **Hàm mất mát & Suy luận:** `BCEWithLogitsLoss` kết hợp trọng số phạt lớp dương $\text{pos\_weight\_sqrt} = \sqrt{157.924 / 4.078} \approx 6.223009$ (tính nghiêm ngặt từ tập Train `test1 + test2` gồm $157.924$ mẫu Normal và $4.078$ mẫu Attack); áp dụng `Sigmoid` ($\sigma(z_t) \in (0, 1)$) khi suy luận.
* **Huấn luyện & Dừng sớm (Early Stopping):** Huấn luyện với Adam ($\text{lr}=10^{-3}$, $\text{batch\_size}=256$), gradient clipping ($1.0$), theo dõi `val_pr_auc` trên `test3` với `patience = 15`. Cả 3 seed đạt metric validation cao nhất ngay tại các epoch đầu (`best_epoch`: Seed 42 = Epoch 2, Seed 43 = Epoch 1, Seed 44 = Epoch 2) và dừng huấn luyện tại `stop_epoch` = 16–17 khi hiện tượng quá khớp sớm (early overfitting) xuất hiện trên tập Train.

### 11.2. Giao thức Dữ liệu V2 & Phân Định Mô Thức Học
* **Scaler Fit Only:** `train1 + train2` (442.802 bản ghi normal — tái sử dụng `scaler.joblib` đóng băng từ V1).
* **DNN Supervised Train:** `test1 + test2` (162.002 mẫu, gồm 157.924 mẫu Normal và 4.078 mẫu Attack, 25 sự kiện tấn công).
* **DNN Validation:** `test3` (108.001 mẫu, gồm 106.466 mẫu Normal và 1.535 mẫu Attack, 8 sự kiện tấn công — early stopping và hiệu chuẩn ngưỡng $\tau^*$).
* **Held-out Evaluation V2:** `test4 + test5` tại $t \ge 30$ (131.942 mẫu chấm điểm, 35.7244 giờ normal, 17 sự kiện tấn công).

> [!IMPORTANT]
> **Tuyên bố về Khác biệt Mô thức Học:** DNN V2 là mô hình **phân loại có giám sát** (học trực tiếp từ nhãn tấn công thật trên `test1 + test2` và hiệu chuẩn ngưỡng trên `test3` có nhãn), trong khi Persistence, Ridge và GRU V1 là các mô hình **dự báo chuỗi thời gian không giám sát** (chỉ học từ dữ liệu normal `train1 + train2` và chọn ngưỡng $Q_{99.9}$ trên `train3` normal). So sánh trên `test4 + test5` là đối chiếu giữa hai mô thức học và chế độ giám sát khác nhau, không phải so sánh cô lập riêng kiến trúc.
> Ngoài ra, `test4` và `test5` đã từng được phân tích trong V1 nên **không phải là tập kiểm tra mù hoàn toàn ở cấp độ toàn dự án**.

### 11.3. Bảng Kết quả So sánh trên 17 Sự kiện Chung (`test4 + test5`, $t \ge 30$)
Nguồn dữ liệu: [`results/v2_dnn/v2_heldout_test4_test5_summary.csv`](results/v2_dnn/v2_heldout_test4_test5_summary.csv) và [`results/v2_dnn/v2_heldout_multiseed_stats.csv`](results/v2_dnn/v2_heldout_multiseed_stats.csv).

| Mô Hình / Nhóm Mô Hình | Chế Độ Giám Sát | Ngưỡng ($\tau$) | TP / 17 | Event Recall | Event Precision | Event F1 | Normal FA / h | Median Delay (s) | Point F1 | Point PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Persistence (V1)** | Không giám sát | $1.680523$ | 11 / 17 | 64.71% | 7.53% | 0.1350 | 3.2751 | 45.0 s | 0.0263 | 0.0712 |
| **Ridge $\alpha=300$ (V1)** | Không giám sát | $1.400928$ | 13 / 17 | 76.47% | 7.18% | 0.1313 | 4.2548 | **8.0 s** | 0.3061 | 0.3322 |
| **GRU Seed 42 (V1, Canonical)** | Không giám sát | $1.058582$ | 16 / 17 | 94.12% | 11.68% | 0.2078 | 2.8832 | 13.0 s | 0.4667 | 0.4869 |
| **GRU 3-Seed Mean ± Std (V1)** | Không giám sát | $Q_{99.9}$ | $16.33 \pm 0.58$ | **96.08% ± 3.40%** | $13.04\% \pm 1.43\%$ | $0.2295 \pm 0.0231$ | **2.5753 ± 0.2841** | **16.50 ± 4.09 s** | **0.4609 ± 0.0135** | 0.4810 ± 0.0052 |
| **DNN Seed 42 (V2 $\tau^*$, Canonical)** | Có giám sát | $0.953203$ | **17 / 17** | **100.00%** | 11.81% | **0.2112** | **2.3513** | 22.0 s | **0.4758** | 0.4574 |
| **DNN Seed 43 (V2 $\tau^*$)** | Có giám sát | $0.958436$ | 16 / 17 | 94.12% | 25.40% | **0.4000** | **0.1680** | 22.5 s | 0.4598 | **0.6182** |
| **DNN Seed 44 (V2 $\tau^*$)** | Có giám sát | $0.958462$ | 8 / 17 | 47.06% | **42.11%** | **0.4444** | **0.0840** | 40.0 s | 0.0301 | 0.5298 |
| **DNN 3-Seed Mean ± Std (V2 $\tau^*$)** | Có giám sát | $\tau^*$ | $13.67 \pm 4.93$ | $80.39\% \pm 29.02\%$ | **26.44% ± 15.18%** | **0.3519 ± 0.1239** | **0.8678 ± 1.2855** | $28.17 \pm 10.25\text{ s}$ | $0.3219 \pm 0.2528$ | **0.5351 ± 0.0805** |
| **DNN 3-Seed Mean ± Std (V2 $\tau=0.50$)**| Có giám sát | $0.500000$ | **17.00 ± 0.00** | **100.00% ± 0.00%** | $1.29\% \pm 0.57\%$ | $0.0255 \pm 0.0112$ | $42.8651 \pm 26.0051$ | **4.33 ± 2.31 s** | $0.3360 \pm 0.1361$ | **0.5351 ± 0.0805** |

*Ghi chú:* Seed 44 đạt Event F1 cao hơn do số báo động giả cực thấp (Normal FA chỉ 0.0840 ev/h), nhưng Event Recall giảm xuống 47.06%. Dự án cố định **Seed 42 làm Canonical** trước khi đánh giá held-out, không chọn lọc hậu nghiệm seed có F1 cao nhất.

---

## 12. Cấu Trúc Kho Lưu Trữ (Repository Structure)

```text
DT049/
├── README.md                            # Tài liệu tổng quan dự án và hướng dẫn tái lập
├── requirements-lock.txt                # Tệp khóa phiên bản thư viện pip chính xác
├── configs/                             # Các tệp cấu hình YAML
│   ├── final_evaluation.yaml            # Tham số giao thức kiểm thử đóng băng V1
│   ├── primary_gru.yaml                 # Cấu hình huấn luyện chuẩn của mô hình GRU V1
│   └── dnn_v2_protocol.yaml             # Cấu hình kiến trúc paper Kuchař & Fujdiak (2025) và giao thức V2
├── data/                                # Thư mục dữ liệu (dữ liệu thô được gitignore)
│   ├── processed/hai_21_03/             # Các mảng dữ liệu đã xử lý tối ưu bộ nhớ
│   └── source/                          # Nơi chứa dữ liệu tải về từ kho gốc
├── demo/                                # Ứng dụng Dashboard tương tác Streamlit
│   ├── app.py                           # Giao diện Dashboard đa trang (bao gồm trang DNN V2)
│   ├── ui_utils.py                      # Các hàm tiện ích nạp cache và hiển thị
│   ├── run_dashboard.sh                 # Script khởi chạy nhanh Dashboard cục bộ
│   ├── stop_dashboard.sh                # Script dừng tiến trình Dashboard an toàn
│   └── check_product.py                 # Script tự động kiểm tra sức khỏe sản phẩm
├── docs/                                # Hệ thống tài liệu kỹ thuật chi tiết
│   ├── final_test_protocol.md           # Giao thức đánh giá chính thức V1
│   ├── final_error_analysis.md          # Phân tích lỗi và độ trễ hậu kiểm thử V1
│   ├── ablation_dynamic57.md            # Báo cáo nghiên cứu cắt giảm Dynamic57
│   ├── experiment_v2_dnn.md             # Báo cáo thiết kế và đánh giá thí nghiệm V2 DNN
│   └── environment.md                   # Thông số môi trường tính toán & kiểm toán RAM
├── models/                              # Các checkpoint mô hình đã đóng băng
│   ├── preprocessing/scaler.joblib      # Bộ StandardScaler đóng băng (fit trên train1+train2)
│   ├── baselines/ridge.joblib           # Mô hình hồi quy Ridge đóng băng
│   ├── primary/gru_w30_seed*.pt         # Các checkpoint PyTorch GRU đóng băng (seeds 42, 43, 44)
│   └── v2_dnn/                          # Checkpoint DNN V2 đóng băng (seeds 42, 43, 44) & metadata
├── results/                             # Đầu ra đánh giá và nhật ký kiểm toán
│   ├── audit/                           # Các bảng kê SHA-256 và nhật ký thực thi V1
│   ├── metrics/                         # Các bảng kết quả CSV tổng hợp và chi tiết từng sự kiện V1
│   ├── figures/                         # Các biểu đồ phân tích chất lượng xuất bản V1
│   └── v2_dnn/                          # Bảng kết quả CSV, ma trận nhầm lẫn, biểu đồ và audit SHA-256 V2
└── src/                                 # Các mô-đun mã nguồn Python thực thi
    ├── preprocess.py                    # Chuẩn hóa và fit scaler trên dữ liệu huấn luyện
    ├── preprocess_test.py               # Biến đổi dữ liệu kiểm thử bằng scaler đóng băng
    ├── baseline_persistence.py          # Pipeline thực thi Baseline 0 (Persistence)
    ├── baseline_ridge.py                # Pipeline thực thi Baseline 1 (Ridge)
    ├── train_gru.py                     # Huấn luyện mô hình chính GRU đa seed
    ├── evaluate_final_test.py           # Engine đánh giá kiểm thử chính thức đóng băng V1
    ├── analyze_final_errors.py          # Phân tích chuyên sâu lỗi hậu kiểm thử V1
    ├── run_ablation_dynamic57.py        # Engine thực thi nghiên cứu cắt giảm Dynamic57
    ├── dnn_dataset_v2.py                # Pipeline nạp dữ liệu V2 theo giao thức không rò rỉ
    ├── dnn_model.py                     # Mô hình PyTorch DNN 6 tầng Dense theo Listing 1
    ├── train_dnn_v2.py                  # Huấn luyện DNN V2 đa seed (BCEWithLogitsLoss, pos_weight)
    ├── evaluate_v2_heldout.py           # Đánh giá held-out test4+test5 đối đầu V1 vs DNN V2
    ├── test_dnn_dataset_v2.py           # Bộ kiểm thử đơn vị cho pipeline dữ liệu V2 (7/7 pass)
    ├── test_dnn_model_v2.py             # Bộ kiểm thử đơn vị cho kiến trúc mô hình V2 (8/8 pass)
    └── test_evaluate_v2_heldout.py      # Bộ kiểm thử đơn vị cho engine đánh giá V2 (6/6 pass)
```

---

## 13. Tính Tái Lập & Nhật Ký Kiểm Toán (Reproducibility & Audit Trail)

Toàn bộ lịch sử phát triển và đánh giá khoa học được lưu vết minh bạch trong Git:

| Mốc Kiểm Soát (Checkpoint) | Mã Commit / Git Tag | Mô Tả |
| :--- | :--- | :--- |
| **Upstream HAI Commit** | `2a814cebc9a66b06c9e5cd545e2d72e65d383737` | Trạng thái kho dữ liệu gốc ETRI đã xác thực |
| **Development Freeze** | `6ce11fb8bcb682c10e125219d0b964f394b8a5d2` | Đóng băng vĩnh viễn mã nguồn, mô hình và ngưỡng |
| **Evaluation Protocol Freeze** | `6a0613b87a0a9d89c89b6ae848350b44c1aa1740` | Khóa giao thức đánh giá (`dt049-pretest-freeze`) |
| **Official Final-Test Tag** | `dt049-final-test-v1` | Nhãn Git vĩnh viễn cho kết quả kiểm thử chính thức |
| **Final-Test Execution** | `331cc07cb72b8b3f30a64bbe7af52248bd0cd9ba` | Ghi nhận toàn bộ bảng số liệu kiểm thử chính thức |
| **Post-Test Analysis** | `4104576d1c4ad69edd87cc97ea6463974becf460` | Bổ sung phân tích lỗi và nghiên cứu Dynamic57 |
| **Dashboard Interface** | `a9f03c940b066bd3bea46007a21f372915f45db9` | Tích hợp giao diện tương tác Streamlit |
| **Product Release v1.0** | `dt049-v1.0` (`f8d1b25`) | Hoàn thiện sản phẩm trình diễn cục bộ v1.0 |

Toàn bộ mảng dữ liệu tiền xử lý, trọng số mô hình và tệp CSV kết quả đều được bảo vệ bằng mã băm SHA-256 lưu tại `results/audit/` (cho V1) và `results/v2_dnn/` (cho V2).

---

## 14. Môi Trường & Phiên Bản Phần Mềm (Environment & Software Versions)

Toàn bộ thực nghiệm được thực thi hoàn toàn trên máy ảo Kali Linux:
- **Hệ điều hành:** Kali GNU/Linux Rolling 2026.2 (Linux 6.19.14+kali-amd64)
- **Phiên bản Python:** `3.14.7`
- **Thiết bị tính toán:** CPU-only (bản dựng PyTorch CPU, hoàn toàn không phụ thuộc GPU)
- **Bộ nhớ máy ảo (RAM):** 4.0 GB RAM vật lý

### Các Phiên Bản Thư Viện Đã Xác Thực
```text
numpy==2.5.3
pandas==3.0.6
scipy==1.18.1
scikit-learn==1.9.1
matplotlib==3.11.2
PyYAML==6.0.3
torch==2.14.0+cpu
streamlit==1.65.0
joblib==1.6.0
tqdm==4.70.1
psutil==7.2.2
```

---

## 15. Hướng Dẫn Chạy Dashboard Tương Tác (Running the Interactive Dashboard)

Ứng dụng Dashboard tương tác cung cấp 8 trang phân tích toàn diện: bảng so sánh mô hình, dòng thời gian tấn công, chẩn đoán lỗi, khảo sát độ nhạy ngưỡng, kết quả cắt giảm đặc trưng Dynamic57, kiểm tra sức khỏe hệ thống và trang mới **DNN Replication V2** trực quan hóa kết quả đối đầu V1 vs DNN V2 trên 17 sự kiện held-out (`test4 + test5`), hiện tượng bão hòa xác suất $[0.953, 0.959]$, ma trận nhầm lẫn và bảng kê kiểm toán SHA-256:

```bash
# Cách 1: Khởi chạy tự động bằng script đóng gói sẵn
cd /home/kali/DT049
./demo/run_dashboard.sh

# Cách 2: Khởi chạy thủ công qua môi trường ảo .venv
cd /home/kali/DT049
source .venv/bin/activate
streamlit run demo/app.py --server.address 127.0.0.1 --server.port 8501
```

Sau khi khởi chạy, mở trình duyệt bên trong Kali tại địa chỉ:
**[http://127.0.0.1:8501](http://127.0.0.1:8501)**

*(Để kiểm tra sức khỏe sản phẩm độc lập: `./demo/check_product.py` | Để dừng server: `./demo/stop_dashboard.sh`)*

> [!NOTE]
> Dashboard hoạt động ở chế độ chỉ đọc (read-only) sử dụng các tệp kết quả đã đóng băng trên đĩa; ứng dụng hoàn toàn không chạy lại suy luận mô hình, không tính lại tập test và chỉ tiêu thụ ~61.4 MB RAM.

---

## 16. Hướng Dẫn Tái Lập Toàn Bộ Thực Nghiệm (Reproducing Experiments)

Tất cả các giai đoạn trong pipeline thực nghiệm đều có thể chạy lại thông qua các script Python trong thư mục `src/`:

### Bước 1: Tiền xử lý dữ liệu
```bash
# Fit StandardScaler trên train1+train2 và biến đổi train1, train2, train3
python src/preprocess.py

# Biến đổi tập kiểm thử chính thức test1..test5 bằng scaler đã đóng băng (không fit lại)
python src/preprocess_test.py
```

### Bước 2: Các mô hình nền tảng (Baselines)
```bash
# Đánh giá Baseline 0 (Persistence) và hiệu chuẩn ngưỡng trên train3
python src/baseline_persistence.py

# Huấn luyện Baseline 1 (Ridge alpha=300) và hiệu chuẩn ngưỡng trên train3
python src/baseline_ridge.py
```

### Bước 3: Huấn luyện mô hình chính GRU
```bash
# Huấn luyện GRU trên 3 seeds 42, 43, 44 với dừng sớm (early stopping) trên train3
python src/train_gru.py
```

### Bước 4: Đánh giá trên tập kiểm thử chính thức
```bash
# Đánh giá các mô hình đóng băng trên test1..test5 tại ngưỡng Q99.9 và các phân vị độ nhạy
python src/evaluate_final_test.py
```

### Bước 5: Phân tích lỗi hậu kiểm thử & Nghiên cứu cắt giảm Dynamic57
```bash
# Chạy phân tích lỗi theo phân hệ và độ trễ ghép cặp trên tập phát hiện chung
python src/analyze_final_errors.py

# Chạy nghiên cứu cắt giảm hậu nghiệm trên 57 đặc trưng động
python src/run_ablation_dynamic57.py
```

### Bước 6: Kiểm thử và Đánh giá DNN V2 (Replication Kuchař & Fujdiak, 2025)
```bash
# Chạy bộ 3 kiểm thử đơn vị tự động (21/21 tests pass)
python src/test_dnn_dataset_v2.py
python src/test_dnn_model_v2.py
python src/test_evaluate_v2_heldout.py

# Huấn luyện DNN V2 trên 3 seeds 42, 43, 44 (nếu cần huấn luyện lại từ đầu)
python src/train_dnn_v2.py

# Chạy đánh giá held-out test4 + test5 đối đầu V1 vs DNN V2 và sinh toàn bộ artifact
python src/evaluate_v2_heldout.py
```

---

## 17. Hạn Chế Của Nghiên Cứu (Limitations)

1. **Tính Đặc Thù Của Testbed:** Các kết luận được xác lập trên hệ thống HAI 21.03 và có thể không khái quát hóa trực tiếp sang các bộ dữ liệu ICS khác (như SWaT, WADI) vốn có cấu trúc vật lý hoặc bề mặt tấn công mạng khác biệt.
2. **Hiệu Chuẩn Chỉ Dựa Trên Dữ Liệu Bình Thường:** Việc chọn ngưỡng dựa hoàn toàn vào tập kiểm định bình thường (`train3`). Trong thực tế vận hành nhà máy dài hạn, hiện tượng trôi cảm biến theo mùa (sensor drift) có thể đòi hỏi cơ chế hiệu chuẩn thích nghi.
3. **Tái Sử Dụng Tập Kiểm Định `train3`:** Tệp `train3` được dùng chung cho cả bước chọn checkpoint dừng sớm của GRU và bước tính ngưỡng phân vị; dù hợp lệ đối với bài toán không giám sát nhưng tạo ra sự phụ thuộc nhất định vào phân phối của `train3`.
4. **Event Precision Khiêm Tốn:** Chỉ số Event Precision đạt mức vừa phải (~$0.11$) do quy tắc khớp sự kiện 1-1 khắt khe trên suốt 109.14 giờ vận hành liên tục mà không áp dụng bộ lọc trễ hay gộp chuông nhân tạo.
5. **Tính Hai Mặt Của Thước Đo Báo Động Giả:** Tần suất sự kiện báo động giả và tổng thời gian chiếm dụng chuông phản ánh hai khía cạnh vận hành khác nhau: GRU tối ưu hóa việc giảm số lần làm phiền người vận hành ($2.61\text{ ev/h}$), trong khi Persistence có tổng số giây báo động thấp nhất ($3.92\text{ s/h}$).
6. **Phạm Vi Kiến Trúc:** Dự án tập trung đánh giá sâu một cấu hình GRU gọn nhẹ ($W=30$, Hidden=64, 1 tầng) phù hợp giới hạn phần cứng CPU; các kiến trúc sâu hơn hoặc Transformer nằm ngoài phạm vi đồ án.
7. **Tính Chất Khảo Sát Của Dynamic57:** Nghiên cứu cắt giảm Dynamic57 được thực hiện hậu nghiệm sau khi đã có kết quả kiểm thử chính thức, đóng vai trò minh chứng giải thích bổ sung chứ không thay thế kết quả chính.
8. **Khác Biệt & Sự Không Nhất Quán Của Bài Báo Gốc:** Bài báo năm 2025 trên tạp chí *Computers & Security* sử dụng bộ dữ liệu tùy chỉnh riêng và có sự không nhất quán giữa phần mô tả thực nghiệm (đề cập 87 đặc trưng) với Listing 1 (ghi cứng `input_dim = 10`), do đó không thể so sánh ngang hàng về mặt con số tuyệt đối.
9. **Khác biệt về Bề mặt Bài toán giữa Paper gốc và V2:** Paper gốc (Kuchař & Fujdiak, 2025) giải bài toán phân loại 4 trạng thái (Normal, Security, Operational, Service) với output sigmoid 4 node, trong khi DNN V2 chuyển dịch sang bài toán nhị phân Normal/Attack với 1 logit trên HAI 21.03 (79 đặc trưng).
10. **Không Thể So Sánh Tuyệt Đối với Accuracy 99.72% Của Paper Gốc:** Sự khác biệt về testbed, số chiều đặc trưng và định nghĩa nhãn khiến việc so sánh trực tiếp chỉ số thô là vô nghĩa về mặt khoa học.
11. **Hiện Tượng Bão Hòa Xác Suất Của DNN V2 Là Giả Thuyết Cơ Chế:** Tầng nút cổ chai `Dense(10, tanh)` kết hợp trọng số lớp dương $\text{pos\_weight} \approx 6.223$ dẫn đến hiện tượng xác suất dự báo bị nén dồn vào dải hẹp $[0.953, 0.959]$ trên tập validation, khiến mô hình cực kỳ nhạy cảm với việc dịch chuyển ngưỡng (như thể hiện ở sự sụt giảm recall của Seed 44 tại $\tau^*$). Tuy nhiên, đây là một giả thuyết cơ chế toán học hợp lý, chưa được kiểm chứng bằng thí nghiệm cô lập nhân quả (causal ablation), nên không được kết luận là nguyên nhân duy nhất.
12. **Tính Chất Bán Mù Của Held-out `test4 + test5`:** Dù DNN V2 chưa từng thấy `test4` và `test5` trong quá trình huấn luyện và chọn ngưỡng, hai tệp này đã từng được khảo sát trong nghiên cứu V1, do đó không phải là tập kiểm thử mù tuyệt đối ở cấp độ toàn dự án.
13. **Đối Chiếu Giữa Hai Mô Thức Giám Sát Khác Nhau:** GRU/Ridge/Persistence là mô hình dự báo không giám sát (học từ normal), còn DNN là mô hình phân loại có giám sát (học trực tiếp từ nhãn tấn công). Sự so sánh phản ánh sự đánh đổi giữa hai triết lý tiếp cận chứ không đơn thuần là so sánh cấu trúc mạng.
14. **Giới Hạn Quy Tắc Khớp Sự Kiện 1-1 và FP:** Thước đo `event_level_fp` đo đếm các khoảng báo động dư thừa, bao gồm cả rung chuông trong khoảng tấn công và phân mảnh biên, cần được phân biệt rõ với `normal_fa_events` (báo động giả thuần túy trong giờ bình thường).
15. **Giới Hạn Tuyên Bố Zero-Day:** Dự án không tuyên bố GRU đã chứng minh phát hiện tấn công zero-day; đây chỉ là mô hình dự báo không giám sát có tiềm năng lý thuyết phát hiện các sai lệch chưa từng biết (unseen deviations), cần được kiểm chứng thêm qua các kịch bản chuyên biệt.
16. **Chưa Sẵn Sàng Triển Khai Trong Môi Trường Sản Xuất:** Tần suất báo động giả hiện tại ($2.61\text{ ev/h}$ của GRU, tương đương $\sim 20.6$ lần trong ca trực 8 giờ) cho thấy hệ thống vẫn cần các cơ chế lọc hậu xử lý (alarm filtering) và thích nghi ngưỡng; kết quả hiện tại phục vụ mục đích nghiên cứu học thuật giữa kỳ, chưa đủ điều kiện triển khai tự động trong nhà máy công nghiệp thực tế.

---

## 18. Giao Diện Trình Diễn (Dashboard)

Khởi chạy Streamlit Dashboard cục bộ để khám phá trực quan toàn bộ kết quả:
```text
http://127.0.0.1:8501
```
*(Mở trình duyệt trong máy ảo Kali Linux để tương tác trực tiếp với biểu đồ, dòng thời gian tấn công và bảng kiểm tra sức khỏe hệ thống.)*
