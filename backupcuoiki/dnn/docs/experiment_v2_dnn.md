# Thí nghiệm V2: Giao thức, Pipeline Dữ liệu, Kiến trúc & Kết quả Huấn luyện Feedforward DNN trên HAI 21.03

## 1. Mục tiêu và Bối cảnh Học thuật

Tài liệu này mô tả giao thức dữ liệu, kiến trúc mạng truyền thẳng (Feedforward DNN), các điều chỉnh có chọn lọc so với bài báo gốc và kết quả huấn luyện/hiệu chuẩn trên tập Validation cho **Thí nghiệm V2 (`DT049_V2_SUPERVISED_DNN`)** trên bộ dữ liệu **HAI Security Dataset 21.03**, tham chiếu bài báo gốc:

* **Tác giả & Năm:** Karel Kuchař & Radek Fujdiak (2025).
* **Tên bài báo:** *"Analyzing anomalies in industrial networks: A data-driven approach to enhance security in manufacturing processes"*, *Computers & Security*, Vol. 153, 104395.
* **DOI:** `10.1016/j.cose.2025.104395` (Open Access).

### 1.1. Kiến trúc Đã Xác minh từ Section 5.5 / Listing 1
Từ đối chiếu trực tiếp **Section 5.5 / Listing 1** của bài báo gốc, thứ tự 6 tầng ẩn `Dense`, 6 hàm kích hoạt (`ReLU` / `Tanh`), 2 lớp `Dropout(0.3)` và tầng đầu ra của bài báo như sau:

1. `Dense(100, activation="relu")`
2. `Dense(70, activation="tanh")`
3. `Dense(100, activation="relu")`
4. `Dense(80, activation="tanh")`
5. `Dense(60, activation="tanh")` $\rightarrow$ `Dropout(0.3)`
6. `Dense(10, activation="tanh")` $\rightarrow$ `Dropout(0.3)`
7. Đầu ra gốc của bài báo: `Dense(4, activation="sigmoid")`.

### 1.2. Tuyên bố Minh bạch về Phạm vi Tái lập Có Điều chỉnh (Adaptation Disclaimer)
Mô hình `FeedforwardDNNV2` trong dự án DT049 là **bản tái lập có chọn lọc và điều chỉnh (selective/adapted reproduction)**, **KHÔNG tuyên bố tái lập nguyên bản 100% bài báo gốc** vì 5 khác biệt bắt buộc sau:
1. **Khác bộ dữ liệu (Dataset):** Thực hiện trên **HAI Security Dataset 21.03** (79 tín hiệu cảm biến/cơ cấu chấp hành lấy mẫu 1 Hz), không phải bộ dữ liệu mạng công nghiệp nội bộ của bài báo.
2. **Khác hệ thống nhãn (Label Taxonomy):** HAI 21.03 chỉ cung cấp nhãn nhị phân `attack` (`0 = Normal`, `1 = Attack`) và các cờ phân hệ vật lý `attack_P1..P3` (vốn giao thoa nhau và không đại diện cho `Operational` hay `Service`), do đó không thể phân loại 4 lớp `Normal / Security / Operational / Service` hay tái lập con số Accuracy 4 lớp ($99,72\%$) của bài báo.
3. **Khác số chiều đầu vào (`input_dim`) & Sự không nhất quán của paper gốc:** Trong phần mô tả thực nghiệm (Section 4/5), bài báo Kuchař & Fujdiak (2025) đề cập đến 87 đặc trưng mạng/quá trình, tuy nhiên trong Listing 1 (mã nguồn Keras định nghĩa kiến trúc) lại khai báo cứng `input_dim = 10`. Dự án DT049 ghi nhận sự không nhất quán nội bộ này của bài báo gốc, và khi thích ứng sang bộ dữ liệu HAI 21.03, mô hình DNN V2 sử dụng toàn bộ **`input_dim = 79` đặc trưng** vật lý liên tục của hệ thống.
4. **Khác lớp chuẩn hóa đầu vào (`norm_layer` vs. `StandardScaler`):** Đầu Listing 1 có dòng `model.add(norm_layer)` nhưng không có đủ thông tin xác minh định nghĩa toán học/tham số cụ thể của `norm_layer`. Do đó, DNN V2 không tuyên bố tái lập `norm_layer` mà tái sử dụng bộ chuẩn hóa đóng băng **`StandardScaler`** ([models/preprocessing/scaler.joblib](file:///home/kali/DT049/models/preprocessing/scaler.joblib)) đã fit trên `train1 + train2` ($442.802$ bản ghi normal) để đảm bảo cùng không gian đầu vào với GRU/Ridge/Persistence (V1).
5. **Khác tầng đầu ra (`Output Layer`):** Chuyển từ `Dense(4, activation="sigmoid")` sang **`Linear(10, 1)` (1 logit nhị phân)** kết hợp hàm mất mát **`BCEWithLogitsLoss`** khi huấn luyện và áp dụng **`Sigmoid`** ($\sigma(z_t) \in (0, 1)$) khi suy luận.

---

## 2. Thiết kế Pipeline Dữ liệu V2 (`src/dnn_dataset_v2.py`)

### 2.1. Phân chia Tập dữ liệu V2 & Phân bố Nhãn Chính xác

| Tập dữ liệu (Split V2) | Các phiên ghi (Sessions) | Tổng số mẫu ($N$) | Số mẫu Normal (`attack = 0`) | Số mẫu Attack (`attack = 1`) | Tỷ lệ Attack (%) | Số sự kiện tấn công | Vai trò trong Giao thức V2 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **DNN Train** | `test1` + `test2` | **$162.002$**<br>(`43.201` + `118.801`) | **$157.924$**<br>(`42.572` + `115.352`) | **$4.078$**<br>(`629` + `3.449`) | **$2,5173\%$** | **25**<br>(`5` + `20`) | Tối ưu trọng số mạng DNN và tính trọng số lớp (`class_weights`). |
| **DNN Validation** | `test3` | **$108.001$** | **$106.466$** | **$1.535$** | **$1,4213\%$** | **8** | Chọn epoch (early stopping), chọn siêu tham số/trọng số lớp, và hiệu chuẩn ngưỡng cảnh báo $\tau^*$. |
| **DNN Evaluation V2** *(Toàn phiên, $t \ge 0$)* | `test4` + `test5` | **$132.002$**<br>(`39.601` + `92.401`) | **$128.668$**<br>(`38.444` + `90.224`) | **$3.334$**<br>(`1.157` + `2.177`) | **$2,5257\%$** | **17**<br>(`5` + `12`) | Tập giữ ngoài (held-out) đối với quá trình huấn luyện và tinh chỉnh DNN V2. |
| **DNN Evaluation V2** *(Căn chỉnh V1, $t \ge 30$)* | `test4` + `test5` | **$131.942$**<br>(`39.571` + `92.371`) | **$128.608$** | **$3.334$** | **$2,5269\%$** | **17**<br>(`5` + `12`) | Vùng chỉ số chung $t \ge 30$ để đối chiếu trực tiếp với GRU, Ridge và Persistence. |

### 2.2. Trọng số Lớp (`Class Weights`) chỉ tính từ DNN Train (`test1 + test2`)
* Tổng số mẫu Train: $N_{\text{train}} = 162.002$ ($N_0 = 157.924$, $N_1 = 4.078$).
* `pos_weight_sqrt` ($\sqrt{N_0 / N_1}$): **`6,223009`** (cấu hình mặc định sử dụng trong `BCEWithLogitsLoss`).
* `pos_weight_balanced` ($N_0 / N_1$): **`38,725846`**.

---

## 3. Kiến trúc Mô hình (`src/dnn_model.py`) & Phân rã 35.731 Tham số

Kiến trúc trong [src/dnn_model.py](file:///home/kali/DT049/src/dnn_model.py) giữ nguyên 6 tầng ẩn và 2 lớp `Dropout(0.3)` của Listing 1, hoàn toàn không thêm `BatchNorm` hay `LayerNorm`:

| Thứ tự tầng | Tên module trong PyTorch | Kích thước `(in → out)` | Hàm kích hoạt & Regularization | Số tham số `weight` | Số tham số `bias` | Tổng tham số tầng |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | `fc1` + `act1` | `79 → 100` | `ReLU` | $7.900$ | $100$ | **$8.000$** |
| 2 | `fc2` + `act2` | `100 → 70` | `Tanh` | $7.000$ | $70$ | **$7.070$** |
| 3 | `fc3` + `act3` | `70 → 100` | `ReLU` | $7.000$ | $100$ | **$7.100$** |
| 4 | `fc4` + `act4` | `100 → 80` | `Tanh` | $8.000$ | $80$ | **$8.080$** |
| 5 | `fc5` + `act5` + `drop1` | `80 → 60` | `Tanh` + `Dropout(0.3)` | $4.800$ | $60$ | **$4.860$** |
| 6 | `fc6` + `act6` + `drop2` | `60 → 10` | `Tanh` + `Dropout(0.3)` | $600$ | $10$ | **$610$** |
| 7 | `out` | `10 → 1` | Logit (`Sigmoid` khi suy luận) | $10$ | $1$ | **$11$** |
| **Tổng** | **`FeedforwardDNNV2`** | **`79 → ... → 1`** | `["relu","tanh","relu","tanh","tanh","tanh"]` | **$35.310$** | **$421$** | **$35.731$** |

---

## 4. Kết quả Huấn luyện & Hiệu chuẩn Ngưỡng trên Validation (`test3`) cho 3 Seeds (`42, 43, 44`)

### 4.1. Bảng Tổng hợp Kết quả Huấn luyện & Hiệu chuẩn trên `test3`
Nguồn dữ liệu: [results/v2_dnn/v2_dnn_validation_summary.csv](file:///home/kali/DT049/results/v2_dnn/v2_dnn_validation_summary.csv) và [models/v2_dnn/dnn_v2_metadata.json](file:///home/kali/DT049/models/v2_dnn/dnn_v2_metadata.json).

| Seed | Best Epoch / Stop Epoch | Train BCE Loss (@Best) | Val BCE Loss (@Best) | Val PR-AUC (@Best) | Val ROC-AUC (@Best) | Ngưỡng hiệu chuẩn $\tau^*$ (`test3`) | TP / 8 (`test3`) | Val Event Recall | Val Event Precision | Val Event F1 ($\tau^*$) | Val Normal FA/h ($\tau^*$) | Val Point F1 ($\tau^*$) | Thời gian (s) / Peak RSS (MB) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **42** | `2 / 17` | $0,371984$ | $0,241895$ | **$0,512897$** | $0,891617$ | **`0,953203`** | `7 / 8` | $87,50\%$ | $14,00\%$ | **$0,241379$** | $1,3529$ | $0,532110$ | $427,50$ s / $593,15$ MB |
| **43** | `1 / 16` | $0,341254$ | $0,209642$ | **$0,639830$** | $0,895439$ | **`0,958436`** | `7 / 8` | $87,50\%$ | $35,00\%$ | **$0,500000$** | $0,1691$ | $0,643333$ | $351,20$ s / $601,62$ MB |
| **44** | `2 / 17` | $0,334494$ | $0,436152$ | **$0,562134$** | $0,901920$ | **`0,958462`** | `6 / 8` | $75,00\%$ | $33,33\%$ | **$0,461538$** | $0,0676$ | $0,103258$ | $617,26$ s / $602,93$ MB |
| **TB ± ĐLC** | — | $0,349244$ | $0,295896 \pm 0,100046$ | **$0,571620 \pm 0,052253$** | $0,896325$ | $0,956700$ | — | $83,33\%$ | $27,44\%$ | **$0,400972 \pm 0,113937$** | $0,5299$ | $0,426234$ | — |

### 4.2. Phát hiện Kỹ thuật Quan trọng khi Huấn luyện và Hiệu chuẩn trên `test3`
1. **Hiện tượng quá khớp nhanh (Early Overfitting) sau Epoch 1–2:**
   * Trên cả 3 seeds (`42, 43, 44`), `train_bce_loss` giảm đều từ $\sim 0,48$ ở Epoch 1 xuống $\sim 0,03 - 0,05$ ở Epoch 16–17, nhưng **`val_pr_auc` trên `test3` đạt đỉnh ngay tại Epoch 1 (Seed 43) hoặc Epoch 2 (Seeds 42, 44)** rồi giảm dần khi mạng bắt đầu ghi nhớ đặc thù của 25 cuộc tấn công đơn trình trong `test1 + test2`. Cơ chế Early Stopping (`patience = 15`) đã dừng đúng quy định tại Epoch 16–17 và khôi phục chính xác trọng số tại Epoch 1–2.
2. **Hiện tượng nén xác suất biên trên do nút cổ chai `Dense(10, activation="tanh")`:**
   * Vì tầng ẩn thứ 6 dùng hàm kích hoạt `Tanh` với 10 nơ-ron ($h_{6,k} \in (-1, 1)$), logit đầu ra $z_t = \mathbf{w}_{\text{out}}^\top \mathbf{h}_6 + b_{\text{out}}$ bị chặn hai đầu tại các epoch đầu. Cụ thể trên `test3` (Seed 42), xác suất dự đoán $p_t = \sigma(z_t)$ nằm trong đoạn $[0,052788;\; 0,953874]$.
   * Hệ quả 1: Tại ngưỡng thấp $\tau = 0,05 < \min(p_t)$, $100\%$ thời gian của phiên `test3` đều bật cảnh báo (`point_fpr = 1,0`), tạo thành 1 cảnh báo liên tục dài 30 giờ làm méo chỉ số `normal_fa_events_per_hour`.
   * Hệ quả 2: Toàn bộ phần đuôi phân phối phân biệt tấn công nằm sát biên trên $[0,951;\; 0,959]$.
   * **Xử lý trong phạm vi hiệu chuẩn Validation `test3`:** Bổ sung điều kiện chặn ngưỡng suy biến `max_point_fpr_cap = 0.05` (loại bỏ mọi ngưỡng có tỷ lệ báo động giả theo giây $> 5\%$), sàn `min_event_recall_floor = 0.75`, và bổ sung các phân vị xác suất thực nghiệm của `test3` (`validation_quantiles: [0.90 .. 0.999]`, làm tròn 6 chữ số thập phân) vào lưới ứng viên trên `test3`. Nhờ đó, bộ hiệu chuẩn chọn được ngưỡng $\tau^* \in [0,953203;\; 0,958462]$ đạt Recall $75\% - 87,5\%$ và Normal FA/h chỉ $0,07 - 1,35$ sự kiện/giờ trên `test3`.

---

## 5. Đóng băng Trước Đánh giá, Kết quả Held-out (`test4 + test5`, $t \ge 30$) & So sánh với Mô hình V1

### 5.1. Đóng băng Trước Đánh giá (`results/v2_dnn/v2_pre_evaluation_freeze_manifest.json`)
Trước khi đọc và chấm điểm tập Held-out (`test4 + test5`), toàn bộ 3 checkpoint DNN V2 (`seed 42, 43, 44`), mã băm SHA-256 và các ngưỡng hiệu chuẩn trên Validation (`test3`) đã được xác minh và đóng băng tại [results/v2_dnn/v2_pre_evaluation_freeze_manifest.json](file:///home/kali/DT049/results/v2_dnn/v2_pre_evaluation_freeze_manifest.json):
* **Seed 42 (Canonical):** SHA-256 `3f00a57398251f4519cf54e8ef6605e82b86dc9ecd1bfb33dd9be8dd47c3f8eb`, $\tau^* = 0,953203$ (đã ghi nhận rõ việc cập nhật trường `calibrated_threshold_test3` sau khi bổ sung điều kiện hiệu chuẩn hậu nghiệm trên `test3`, không thay đổi trọng số mô hình).
* **Seed 43:** SHA-256 `569741efeee64d3ecf1d017f6a42e171004266a3af88d7035a97f3590aace684`, $\tau^* = 0,958436$.
* **Seed 44:** SHA-256 `498bb0346a292ba9e9c7257797bb23c8a6e35e489fcde2ba73f3d332ff6806a3`, $\tau^* = 0,958462$.
* **Ghi chú minh bạch về hiệu chuẩn hậu nghiệm trên `test3`:** Việc bổ sung `max_point_fpr_cap = 0.05`, `min_event_recall_floor = 0.75` và lưới phân vị thực nghiệm `validation_quantiles: [0.90 .. 0.999]` là điều chỉnh hậu nghiệm dựa trên tập Validation `test3` (sau khi phát hiện hiện tượng bão hòa `Tanh(10)` trên Seed 42), **không được mô tả là quy tắc chọn ngưỡng tiền định hoàn toàn trước khi chạy `test3`**.

### 5.2. Tuyên bố Bắt buộc về Khác biệt Mô thức Học và Tính Không Mù Tuyệt đối của `test4 + test5`
1. **Khác biệt mô thức học (Paradigm Disclaimer):**
   * **DNN V2 (Có giám sát — Supervised Binary Classification):** Được huấn luyện trực tiếp với nhãn tấn công thật (`attack = 0/1`) trên `test1 + test2` ($25$ cuộc tấn công) và hiệu chuẩn ngưỡng $\tau^*$ có giám sát trên `test3` ($8$ cuộc tấn công).
   * **GRU / Ridge / Persistence V1 (Không giám sát — Unsupervised Forecasting):** Chỉ được huấn luyện trên dữ liệu vận hành bình thường (`train1 + train2`, $0$ cuộc tấn công) và chọn ngưỡng phân vị $Q_{99,9}$ hoàn toàn không giám sát trên `train3` ($0$ cuộc tấn công).
   * Do đó, bảng đối chiếu trên `test4 + test5` ($t \ge 30$, $17$ cuộc tấn công) là **so sánh giữa hai nhánh phương pháp khác nhau về mức độ giám sát nhãn**, không phải phép so sánh tách biệt hoàn toàn ảnh hưởng của kiến trúc mạng.
2. **Tuyên bố về tính không mù hoàn toàn (Non-Blindness Disclaimer):**
   * Dù `test4` và `test5` được giữ ngoài nghiêm ngặt đối với toàn bộ quá trình huấn luyện, chọn mô hình và hiệu chuẩn ngưỡng của DNN V2, hai phiên này đã từng được phân tích trong báo cáo đánh giá cuối cùng của V1 nên **không còn là tập kiểm tra mù tuyệt đối (completely blind test) của toàn bộ dự án DT049**.

### 5.3. Bảng Kết quả Gộp (Pooled `test4 + test5`, $t \ge 30$, $131.942$ mẫu, $35,7244$ giờ normal, $17$ sự kiện tấn công)
Nguồn dữ liệu: [results/v2_dnn/v2_heldout_test4_test5_summary.csv](file:///home/kali/DT049/results/v2_dnn/v2_heldout_test4_test5_summary.csv) và [results/v2_dnn/v2_heldout_multiseed_stats.csv](file:///home/kali/DT049/results/v2_dnn/v2_heldout_multiseed_stats.csv). Mọi chỉ số gộp (`Event Precision/Recall/F1`, `Normal FA/h`, `Median Delay`) đều được tính trực tiếp từ tổng `TP, FP, FN` gộp, tổng số giờ normal gộp và vector độ trễ gộp trên `test4 + test5`.

| Mô hình & Chế độ vận hành | Mô thức học | Ngưỡng ($\tau$) | TP / 17 | FN | Event FP | Event Recall | Event Precision | Event F1 | Normal FA Events | Normal FA / h | FP Sec / h | Median Delay (s) | Point Precision | Point Recall | Point F1 | Point ROC-AUC | Point PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Persistence (V1)** | Không giám sát | `1,680523` | `11 / 17` | $6$ | $135$ | $64,71\%$ | $7,53\%$ | $0,134969$ | $117$ | $3,2751$ | $3,4430$ | $45,0$ | $27,22\%$ | $1,38\%$ | $0,026263$ | $0,645364$ | $0,071233$ |
| **Ridge $\alpha=300$ (V1)** | Không giám sát | `1,400928` | `13 / 17` | $4$ | $168$ | $76,47\%$ | $7,18\%$ | $0,131313$ | $152$ | $4,2548$ | $5,5144$ | $8,0$ | $76,41\%$ | $19,14\%$ | $0,306069$ | $0,785240$ | $0,332210$ |
| **GRU W=30 Seed 42 (V1, Canonical)** | Không giám sát | `1,058582` | `16 / 17` | $1$ | $121$ | $94,12\%$ | $11,68\%$ | $0,207792$ | $103$ | $2,8832$ | $5,1785$ | $13,0$ | $85,27\%$ | $32,12\%$ | $0,466667$ | $0,846204$ | $0,486911$ |
| **GRU W=30 Seed 43 (V1)** | Không giám sát | `1,195869` | `16 / 17` | $1$ | $108$ | $94,12\%$ | $12,90\%$ | $0,226950$ | $90$ | $2,5193$ | $4,3668$ | $15,5$ | $86,51\%$ | $29,99\%$ | $0,445434$ | $0,846895$ | $0,477006$ |
| **GRU W=30 Seed 44 (V1)** | Không giám sát | `1,076832` | `17 / 17` | $0$ | $100$ | $100,00\%$ | $14,53\%$ | $0,253731$ | $83$ | $2,3233$ | $4,5067$ | $21,0$ | $86,97\%$ | $32,24\%$ | $0,470460$ | $0,846990$ | $0,479088$ |
| **GRU V1 (Mean ± Std, 3 seeds)** | Không giám sát | — | $16,33 \pm 0,58$ | — | — | $96,08\% \pm 3,40\%$ | $13,04\% \pm 1,43\%$ | **$0,229491 \pm 0,023075$** | — | **$2,5753 \pm 0,2841$** | $4,6840 \pm 0,4339$ | $16,5 \pm 4,09$ | $86,25\% \pm 0,88\%$ | $31,45\% \pm 1,27\%$ | $0,460854 \pm 0,013488$ | $0,846696 \pm 0,000429$ | $0,481002 \pm 0,005222$ |
| **DNN V2 Seed 42 (Primary $\tau^*$, Canonical)** | Có giám sát | `0,953203` | **`17 / 17`** | $0$ | $127$ | **$100,00\%$** | $11,81\%$ | **$0,211180$** | $84$ | **$2,3513$** | $17,3271$ | $22,0$ | $66,59\%$ | $37,01\%$ | $0,475805$ | $0,847786$ | $0,457378$ |
| **DNN V2 Seed 43 (Primary $\tau^*$)** | Có giám sát | `0,958436` | **`16 / 17`** | $1$ | $47$ | $94,12\%$ | $25,40\%$ | **$0,400000$** | $6$ | **$0,1680$** | $1,5676$ | $22,5$ | $94,76\%$ | $30,35\%$ | $0,459791$ | $0,847953$ | $0,618167$ |
| **DNN V2 Seed 44 (Primary $\tau^*$)** | Có giám sát | `0,958462` | `8 / 17` | $9$ | $11$ | $47,06\%$ | $42,11\%$ | **$0,444444$** | $3$ | **$0,0840$** | $0,1680$ | $40,0$ | $89,47\%$ | $1,53\%$ | $0,030080$ | $0,845722$ | $0,529797$ |
| **DNN V2 Primary $\tau^*$ (Mean ± Std, 3 seeds)** | Có giám sát | — | $13,67 \pm 4,93$ | — | — | $80,39\% \pm 29,02\%$ | $26,44\% \pm 15,18\%$ | **$0,351875 \pm 0,123855$** | — | **$0,8678 \pm 1,2855$** | $6,3542 \pm 9,5285$ | $28,17 \pm 10,25$ | $83,61\% \pm 14,97\%$ | $22,97\% \pm 18,86\%$ | $0,321892 \pm 0,252843$ | $0,847154 \pm 0,001243$ | $0,535114 \pm 0,080526$ |
| **DNN V2 Seed 42 (Secondary $\tau=0,50$)** | Có giám sát | `0,500000` | `17 / 17` | $0$ | $1.087$ | $100,00\%$ | $1,54\%$ | $0,030330$ | $1.037$ | $29,0277$ | $272,6424$ | $7,0$ | $17,42\%$ | $61,64\%$ | $0,271664$ | $0,847786$ | $0,457378$ |
| **DNN V2 Seed 43 (Secondary $\tau=0,50$)** | Có giám sát | `0,500000` | `17 / 17` | $0$ | $983$ | $100,00\%$ | $1,70\%$ | $0,033432$ | $954$ | $26,7044$ | $104,2703$ | $3,0$ | $38,23\%$ | $69,14\%$ | $0,492311$ | $0,847953$ | $0,618167$ |
| **DNN V2 Seed 44 (Secondary $\tau=0,50$)** | Có giám sát | `0,500000` | `17 / 17` | $0$ | $2.634$ | $100,00\%$ | $0,64\%$ | $0,012744$ | $2.603$ | $72,8633$ | $389,3972$ | $3,0$ | $14,70\%$ | $71,90\%$ | $0,244069$ | $0,845722$ | $0,529797$ |

### 5.4. Biểu đồ Chẩn đoán Đã Sinh trong `results/v2_dnn/figures/`
1. [results/v2_dnn/figures/v2_event_metrics_comparison.png](file:///home/kali/DT049/results/v2_dnn/figures/v2_event_metrics_comparison.png): So sánh Event Recall, Event Precision, Event F1 và Normal FA/h trên `test4 + test5` giữa V1 và V2.
2. [results/v2_dnn/figures/v2_confusion_matrices_seed42.png](file:///home/kali/DT049/results/v2_dnn/figures/v2_confusion_matrices_seed42.png): Ma trận nhầm lẫn mức thời điểm (Point-level Confusion Matrix, $N=131.942$) của GRU Seed 42, DNN Seed 42 ($\tau^*$) và DNN Seed 42 ($\tau=0,50$).
3. [results/v2_dnn/figures/v2_attack_event_heatmap_17.png](file:///home/kali/DT049/results/v2_dnn/figures/v2_attack_event_heatmap_17.png): Bản đồ nhiệt độ trễ phát hiện (giây) trên toàn bộ 17 cuộc tấn công của `test4 + test5`.
4. [results/v2_dnn/figures/v2_probability_saturation_analysis.png](file:///home/kali/DT049/results/v2_dnn/figures/v2_probability_saturation_analysis.png): Phân phối xác suất dự đoán của DNN V2 Seed 42 trên Validation (`test3`) và Held-out (`test4 + test5`), minh họa trực quan hiện tượng nén xác suất sát biên trên do tầng `Dense(10, activation="tanh")`.

---

## 6. Phân tích Khoa học Chi tiết trên 17 Sự kiện Tấn công (`test4 + test5`) và Phân hệ Vật lý (`P1/P2/P3`)

### 6.1. Đối chiếu Từng Sự kiện trong 17 Cuộc tấn công (`results/v2_dnn/v2_heldout_17_attack_events.csv`)
Nguồn dữ liệu: [results/v2_dnn/v2_heldout_17_attack_events.csv](file:///home/kali/DT049/results/v2_dnn/v2_heldout_17_attack_events.csv).

* **DNN V2 Seed 42 (Primary $\tau^* = 0,953203$, Canonical):** Phát hiện **17 / 17** cuộc tấn công ($\text{Event Recall} = 100,00\%$), bao gồm cả cuộc tấn công đơn trình khó nhất `test5_atk_09` (`P1`, độ trễ $91,0\text{ s}$, đỉnh xác suất $0,953711 > 0,953203$) mà **GRU V1 Seed 42 (Canonical)** bỏ sót (`peak_score = 1,019327 < 1,058582`).
* **DNN V2 Seed 43 (Primary $\tau^* = 0,958436$):** Phát hiện **16 / 17** cuộc tấn công ($\text{Event Recall} = 94,12\%$), chỉ bỏ sót duy nhất `test5_atk_09` (`peak_score = 0,957471 < 0,958436$), nhưng giảm báo động giả bình thường xuống mức cực thấp (**$6$ sự kiện / $35,7244$ giờ $\rightarrow 0,1680\text{ FA/h}$**, trong đó `test4` đạt **$0\text{ FA/h}$** với Precision điểm $100\%$), đưa $\text{Event F1}$ gộp lên **$0,400000$** (cao gấp $\sim 1,92$ lần GRU V1 Seed 42).
* **DNN V2 Seed 44 (Primary $\tau^* = 0,958462$):** Phát hiện **8 / 17** cuộc tấn công ($\text{Event Recall} = 47,06\%$), bỏ sót 9 sự kiện dù tại $\tau = 0,50$ nó phát hiện đủ $17/17$. Nguyên nhân trực tiếp: toàn bộ 9 cuộc tấn công bị bỏ sót đều có đỉnh xác suất nằm trong dải rất hẹp $[0,956466;\; 0,958458]$, chỉ thấp hơn ngưỡng $\tau^* = 0,958462$ từ **$0,000004$ đến $0,001996$** (tỷ số `peak / threshold` từ $0,997917$ đến $0,999995$). Dù vậy, do số báo động giả cực thấp ($\text{Event FP} = 11$, $\text{Normal FA} = 3 \rightarrow 0,0840\text{ FA/h}$), $\text{Event Precision}$ đạt $42,11\%$ và $\text{Event F1}$ đạt $0,444444$.

### 6.2. Phân tích Theo Phân hệ Vật lý (`P1, P2, P3`, Đơn trình vs Đa trình)
Nguồn dữ liệu: [results/v2_dnn/v2_heldout_subsystem_breakdown.csv](file:///home/kali/DT049/results/v2_dnn/v2_heldout_subsystem_breakdown.csv).

| Nhóm phân hệ tấn công | Số sự kiện | Persistence (V1) | Ridge (V1) | GRU s42 (V1) | GRU s43 (V1) | GRU s44 (V1) | DNN s42 ($\tau^*$) | DNN s43 ($\tau^*$) | DNN s44 ($\tau^*$) | DNN s42/43/44 ($\tau=0,50$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tổng cộng (`test4 + test5`)** | **17** | 11 | 13 | 16 | 16 | 17 | **17** | **16** | 8 | **17 / 17 / 17** |
| **Đơn trình (Single-Process)** | **6** | 2 | 3 | 5 | 5 | 6 | **6** | **5** | 2 | **6 / 6 / 6** |
| **Đa trình (Multi-Process)** | **11** | 9 | 10 | 11 | 11 | 11 | **11** | **11** | 6 | **11 / 11 / 11** |
| **Liên quan P1 (Boiler)** | **13** | 7 | 9 | 12 | 12 | 13 | **13** | **12** | 6 | **13 / 13 / 13** |
| **Liên quan P2 (Turbine)** | **9** | 8 | 9 | 9 | 9 | 9 | **9** | **9** | 5 | **9 / 9 / 9** |
| **Liên quan P3 (Water-Treatment)** | **6** | 5 | 5 | 6 | 6 | 6 | **6** | **6** | 3 | **6 / 6 / 6** |

* **Nhận xét:** Các cuộc tấn công đa trình ($11$ sự kiện) và liên quan đến cụm Turbine `P2` ($9$ sự kiện) tạo biến động chéo rõ rệt trên nhiều cảm biến đồng thời nên cả GRU V1 và DNN V2 (Seeds 42, 43) đều phát hiện $100\%$. Ngược lại, các cuộc tấn công **đơn trình trên cụm Lò hơi `P1`** (`test4_atk_01`, `test5_atk_08`, `test5_atk_09`, `test5_atk_10`, `test5_atk_12`) có biên độ dị thường nhỏ: Persistence bỏ sót $4/5$ sự kiện này, Ridge bỏ sót $3/5$, GRU Seed 42 bỏ sót `test5_atk_09`, GRU Seed 43 bỏ sót `test5_atk_08`, trong khi DNN Seed 42 ($\tau^*$) phát hiện cả $5/5$ sự kiện đơn trình `P1`.

### 6.3. Phân tích Quan sát Thực nghiệm, Giả thuyết Cơ chế và Hạn chế Phương pháp luận
1. **Phân biệt Quan sát Trực tiếp và Giả thuyết Cơ chế về Độ nhạy Ngưỡng $\tau^*$:**
   * **Quan sát trực tiếp từ dữ liệu:** Trên cả `test3` và `test4 + test5`, phân phối xác suất dự đoán $p_t = \sigma(z_t)$ của DNN V2 nén chặt sát biên trên $\approx [0,953;\; 0,959]$. Một dịch chuyển ngưỡng nhỏ giữa Seed 43 ($\tau^* = 0,958436$) và Seed 44 ($\tau^* = 0,958462$) làm Event Recall trên `test4 + test5` giảm từ $94,12\%$ xuống $47,06\%$ (trong đó 8/9 sự kiện bị bỏ sót của Seed 44 có đỉnh xác suất đạt $0,958428 \div 0,958458$, tức đạt $99,996\% \div 99,999\%$ ngưỡng).
   * **Giả thuyết cơ chế (chưa kiểm chứng tách biệt bằng ablation):** Hiện tượng nén xác suất sát trần và dao động lớn giữa các seed có khả năng liên quan đồng thời đến: (a) tầng ẩn thứ 6 `Dense(10, activation="tanh")` chặn đầu vào lớp logit cuối trong $(-1, 1)$, (b) việc dừng sớm tại Epoch 1–2 khi biên độ trọng số lớp đầu ra còn hạn chế, (c) trọng số phạt lớp dương `pos_weight = 6,223009`, và (d) quy mô sự kiện nhỏ của tập Validation `test3` (chỉ có 8 cuộc tấn công). Vì dự án giữ nguyên kiến trúc Listing 1 và không chạy ablation thay đổi hàm kích hoạt, cơ chế `Tanh(10)` được ghi nhận là giả thuyết toán học hợp lý chứ không tuyên bố là nguyên nhân duy nhất.
2. **Cách đếm `Event FP` (`event_level_fp`) và `Normal FA Events` (`normal_fa_events`) trong hàm đánh giá:**
   * Trong [src/evaluate_v2_heldout.py](file:///home/kali/DT049/src/evaluate_v2_heldout.py), `event_level_fp` được tính bằng tổng số cụm cảnh báo liên tiếp (`score > τ`) trên toàn miền $t \ge 30$ trừ đi số sự kiện tấn công được ghép cặp 1–1 (`total_alarm_events - tp_events`). Ngược lại, `normal_fa_events` đếm số cụm cảnh báo liên tiếp sau khi đã áp mặt nạ loại bỏ toàn bộ thời điểm tấn công (`score > τ AND attack == 0`).
   * Do hai chỉ số được trích xuất từ hai định nghĩa khoảng thời gian khác nhau (một bên không che vùng tấn công rồi trừ số TP ghép cặp 1–1; một bên cắt đứt chuỗi tại mọi ranh giới `attack == 1`), hiệu số `event_level_fp - normal_fa_events` phản ánh sự khác biệt tổng hợp giữa hai quy tắc trích xuất sự kiện (bao gồm các cụm cảnh báo dư thừa nằm trọn trong thời gian tấn công và hiệu ứng biên khi cảnh báo vắt qua ranh giới normal/attack), không tự động đồng nhất hoàn toàn $100\%$ hiệu số này là alarm chatter nếu không phân rã từng khoảng cảnh báo.
