<!-- Academic text retained from the original DNN Replication V2 dashboard. -->

# DNN Replication V2 — Tái lập Có Điều chỉnh Kuchař & Fujdiak (2025) trên HAI 21.03

Thí nghiệm V2 (`DT049_V2_SUPERVISED_DNN`): Phân loại nhị phân Normal/Attack có giám sát bằng mạng Feedforward DNN 6 tầng ẩn và đối chiếu với biến thể dự báo chuỗi thời gian không giám sát GRU V1 trên tập Held-out `test4 + test5` (17 sự kiện, t >= 30).

## 1. RESEARCH BACKGROUND — Bối cảnh Nghiên cứu & Phạm vi Tái lập Có Điều chỉnh

- **Bài báo tham chiếu gốc:** Kuchař & Fujdiak (2025), *"Analyzing anomalies in industrial networks: A data-driven approach to enhance security in manufacturing processes"*, *Computers & Security*, DOI `10.1016/j.cose.2025.104395`.
- **Bài toán gốc của bài báo:** Sử dụng mạng truyền thẳng (Feedforward DNN) để phân loại **4 trạng thái**: `Normal`, `Security`, `Operational`, `Service`.
- **Đặc thù bộ dữ liệu HAI 21.03:** Không có 4 trạng thái tương đương với bộ dữ liệu nội bộ của bài báo; HAI 21.03 chỉ cung cấp nhãn nhị phân `attack` (`0 = Normal`, `1 = Attack`) và các nhãn phân hệ vật lý `attack_P1`, `attack_P2`, `attack_P3`.
- **Kiến trúc DNN V2 đã xác minh theo Listing 1:** Giữ nguyên **6 tầng ẩn Dense `[100, 70, 100, 80, 60, 10]`**, chuỗi kích hoạt **`relu → tanh → relu → tanh → tanh → tanh`**, và **2 lớp `Dropout(0.3)`** đặt sau `Dense(60)` và `Dense(10)` (tổng cộng **`35,731` tham số**).
- **Điều chỉnh sang HAI 21.03:** Đầu vào gồm **79 đặc trưng** cảm biến/cơ cấu chấp hành (sử dụng bộ chuẩn hóa `StandardScaler` đã đóng băng từ V1) và đầu ra nhị phân **1 logit (`Normal` / `Attack`)** kết hợp `BCEWithLogitsLoss` / `Sigmoid`.
- **Vai trò của GRU V1:** Mô hình **GRU $W=30$ (`32,975` tham số)** được giữ nguyên làm **biến thể dự báo chuỗi thời gian không giám sát** (Unsupervised Time-Series Forecasting Variant).

📌 **Tuyên bố Minh bạch Học thuật (Adaptation Disclaimer):**

- **DNN V2 là bản tái lập kiến trúc có điều chỉnh sang HAI 21.03**, **KHÔNG tuyên bố tái lập nguyên bản 100% bài báo gốc** và **KHÔNG tuyên bố đạt độ chính xác 4 lớp `99,72%`** của bài báo (do khác tập dữ liệu, khác hệ thống nhãn, khác số chiều đầu vào `79 đặc trưng HAI` so với sự không nhất quán giữa 87 đặc trưng mô tả trong bài báo và `input_dim=10` trong Listing 1, và chưa có định nghĩa chi tiết của `norm_layer` trong Listing 1).
- Mục tiêu của trang này là giúp sinh viên trình bày rõ ràng sự khác biệt giữa **Phân loại mẫu đơn có giám sát (DNN V2)** và **Dự báo chuỗi thời gian không giám sát (GRU V1)** trên cùng một tập kiểm tra giữ ngoài.

---

## 2. DATA SPLIT — Giao thức Phân chia Dữ liệu Thí nghiệm V2

⚠️ **Ghi chú Minh bạch về Tính Độc lập của Tập Đánh giá (`test4 + test5`):**
`test4` và `test5` hoàn toàn không tham gia vào quá trình huấn luyện, chọn epoch hay hiệu chuẩn ngưỡng của **DNN V2**. Tuy nhiên, vì hai phiên này đã từng được đánh giá và phân tích lỗi trong báo cáo **V1**, chúng **không phải là tập dữ liệu mù hoàn toàn (not completely blind) ở cấp độ toàn bộ dự án DT049**.

---

## 3. CANONICAL PERFORMANCE — Hiệu năng Chính tắc (Seed 42 trên `test4 + test5`, $t \ge 30$)

**A. Mô hình Tái lập Có Điều chỉnh: `Feedforward DNN V2 — Seed 42 (Canonical)`** *(Ngưỡng đóng băng $\tau^* = 0.953203$ chọn trên `test3`)*

**B. Mô hình Biến thể Chuỗi thời gian: `Primary GRU W=30 — Seed 42 (V1, Canonical)` trên chính `test4 + test5` ($t \ge 30$)** *(Ngưỡng đóng băng $Q_{99.9} = 1.058582$ chọn trên `train3`)*

---

## 4. MODEL COMPARISON — COMMON 17 ATTACKS (`test4 + test5`, $t \ge 30$)

Bảng đối chiếu trực tiếp giữa các mô hình **V1 (`Persistence`, `Ridge`, `GRU seeds 42/43/44`)** và **V2 (`DNN seeds 42/43/44` tại ngưỡng đóng băng $\tau^*$)** trên đúng cùng tập `test4 + test5` ($131.942$ mẫu tại $t \ge 30$, $17$ sự kiện tấn công). **Tuyệt đối không trộn lẫn với bảng 50 sự kiện của toàn bộ `test1–test5` trong V1.**

⚖️ **Chú thích Bắt buộc về Chế độ Giám sát (Supervision Regime Disclaimer):**
- **DNN V2 (Có giám sát):** Huấn luyện trực tiếp với nhãn tấn công `attack = 0/1` trên `test1 + test2` (25 cuộc tấn công) và hiệu chuẩn ngưỡng $\tau^*$ có giám sát trên `test3` (8 cuộc tấn công).
- **Persistence, Ridge, GRU V1 (Không giám sát):** Chỉ huấn luyện trên dữ liệu vận hành bình thường `train1 + train2` (0 cuộc tấn công) và chọn ngưỡng phân vị $Q_{99.9}$ hoàn toàn không dùng nhãn tấn công trên `train3`.
- Do đó, sự khác biệt kết quả phản ánh **sự đánh đổi giữa hai mô thức học và chế độ giám sát nhãn**, không phải phép so sánh cô lập riêng kiến trúc mạng.

### 📂 Xem bảng kết quả tách riêng từng phiên (`test4` gồm 5 sự kiện vs `test5` gồm 12 sự kiện)

---

## 5. MULTISEED STABILITY — Độ Ổn định Đa Seed (`Seeds 42, 43, 44`)

##### 🔒 Seed 42 (Canonical — $\tau^*$)

Normal FA/h: `2.3513` | Median Delay: `22.0 s` | Point PR-AUC: `0.457378`

##### 🧪 Seed 43 (Replicate 1 — $\tau^*$)

Normal FA/h: `0.1680` | Median Delay: `22.5 s` | Point PR-AUC: `0.618167`

##### 🧪 Seed 44 (Replicate 2 — $\tau^*$)

Normal FA/h: `0.0840` | Median Delay: `40.0 s` | Point PR-AUC: `0.529797`

#### Thống kê Tổng hợp 3 Seeds (`Mean ± Std` từ `v2_heldout_multiseed_stats.csv`)

📌 **Giải thích về Seed 44 và Nguyên tắc Giữ Cố định Seed 42 làm Canonical:**
- **Tại sao Seed 44 có `Event F1 = 0.444444` cao hơn nhưng `Event Recall` giảm mạnh xuống `47.06%` (`8/17`)?** Do ngưỡng đóng băng của Seed 44 rất sát biên trên ($\tau^* = 0.958462$), mô hình chỉ phát sinh tổng cộng `11` cảnh báo dư mức sự kiện (`Normal FA = 3` sự kiện, tức `0.0840 FA/h`), làm `Event Precision` tăng lên `42.11%` (`8/19`) và kéo trung bình điều hòa `Event F1` lên cao dù bỏ sót tới `9/17` cuộc tấn công (và `Point Recall` chỉ còn `1.53%`).
- **Không chọn lọc hậu nghiệm (No Post-hoc Cherry-Picking):** `Seed 42` được cố định làm mô hình **Canonical** trước khi mở tập đánh giá `test4 + test5`. Dự án tuyệt đối không lấy `Seed 43` hay `Seed 44` có `Event F1` held-out cao hơn để thay thế kết quả đại diện chính.

---

## 6. ATTACK EVENT EXPLORER V2 — Khám phá 17 Sự kiện Tấn công (`test4 + test5`)

Đang hiển thị **17 / 17** sự kiện tấn công trên `test4 + test5` (Mô hình đang chọn: **DNN Seed 42 (Primary τ*, Canonical)**).

### 📊 Xem bảng tổng hợp số sự kiện phát hiện theo từng Phân hệ Vật lý (`P1, P2, P3`, Đơn trình vs Đa trình)

---

## 7. ERROR ANALYSIS & THRESHOLD SENSITIVITY — Phân tích Lỗi & Độ nhạy Ngưỡng

⚠️ **Phân biệt Chế độ Ngưỡng Chính ($\tau^*$) và Phân tích Độ nhạy Phụ ($\tau = 0.50$):**
- **Ngưỡng chính $\tau^*$ (`0.953203` cho Seed 42, `0.958436` cho Seed 43, `0.958462` cho Seed 44):** Được chọn trên tập Validation `test3` và đóng băng trước khi đánh giá `test4 + test5`.
- **Ngưỡng $\tau = 0.50$:** Chỉ là phân tích độ nhạy phụ để minh họa hành vi xác suất chưa hiệu chuẩn (tại $\tau = 0.50$, tỷ lệ báo động giả vọt lên `26.70 – 72.86 FA/h` và `Event F1` giảm còn `0.0127 – 0.0334`). **Tuyệt đối không dùng $\tau = 0.50$ để thay thế kết quả benchmark chính.**

### 🔢 Xem bảng Ma trận Nhầm lẫn Mức điểm (Point-Level Confusion Matrices) & Chỉ số Độ nhạy Đầy đủ

🔬 **Phân tích Khoa học: Tách biệt Quan sát Thực nghiệm Trực tiếp và Giả thuyết Cơ chế**

**1. Quan sát trực tiếp từ dữ liệu thực nghiệm (Direct Empirical Observations):**
- **Sự tập trung xác suất sát biên trên:** Từ dữ liệu điểm và *Hình V2-4*, xác suất dự đoán $p_t = \sigma(z_t)$ của DNN V2 nén dày đặc trong dải hẹp sát biên trên $\approx [0.953;\; 0.959]$.
- **Độ nhạy ngưỡng cực lớn giữa các seed:** Trên `test4 + test5`, cả 3 seed của DNN V2 có khả năng xếp hạng liên tục rất gần nhau (`Point ROC-AUC` $= 0.8472 \pm 0.0012$). Tuy nhiên, tại Seed 44 ($\tau^* = 0.958462$), **8 trong 9 sự kiện bị bỏ sót có đỉnh xác suất đạt từ `0.958428` đến `0.958458`** (`Peak/Threshold Ratio` $= 0.999964 \div 0.999995$), tức chỉ thấp hơn ngưỡng đóng băng từ $4 \times 10^{-6}$ đến $3.4 \times 10^{-5}$.
- **Cách đếm `Event FP` (`event_level_fp`) và `Normal FA Events` (`normal_fa_events`):** Trong mã đánh giá (`src/evaluate_v2_heldout.py`), `Event FP` đếm tổng số cụm cảnh báo (`score > τ`) trên toàn miền $t \ge 30$ không ghép cặp 1–1 với sự kiện tấn công mới (`total_alarm_events - tp_events`), trong khi `Normal FA Events` đếm các cụm cảnh báo sau khi đã che toàn bộ thời điểm tấn công (`score > τ AND attack == 0`). Do hai chỉ số được trích xuất theo hai định nghĩa khoảng thời gian khác nhau, **không thể tự động đồng nhất toàn bộ hiệu số `Event FP - Normal FA Events` là alarm chatter trong vùng tấn công nếu không phân rã từng khoảng cảnh báo**.

**2. Giả thuyết cơ chế (Mechanistic Hypotheses — Chưa kiểm chứng tách biệt bằng Ablation riêng):**
- Hiện tượng xác suất bị nén sát biên trên và dao động ngưỡng giữa 3 seeds có thể liên quan đến sự kết hợp của nhiều yếu tố: (a) tầng ẩn thứ 6 `Dense(10, activation='tanh')` giới hạn đầu vào lớp logit cuối trong $(-1, 1)$, (b) việc dừng sớm (Early Stopping) tại Epoch 1–2 khi biên độ trọng số lớp đầu ra còn hạn chế, (c) hệ số phạt lớp dương `pos_weight = 6.223009`, và (d) số lượng sự kiện tấn công ít trên tập Validation `test3` (chỉ có 8 sự kiện). **Do giữ nguyên kiến trúc Listing 1 và không thực hiện thí nghiệm ablation thay đổi hàm kích hoạt, cơ chế `Tanh(10)` được trình bày như một giả thuyết giải thích hợp lý, không tuyên bố là nguyên nhân duy nhất.**

---

## 8. RESEARCH LIMITATIONS — Các Giới hạn Nghiên cứu & Lưu ý Khi Thuyết trình

1. **Không tái lập toàn bộ mô hình và bài toán gốc của bài báo:** DNN V2 chỉ tái lập có điều chỉnh cấu trúc 6 tầng ẩn Dense từ Listing 1 của Kuchař & Fujdiak (2025) sang bài toán nhị phân `Normal/Attack` với 79 đặc trưng trên HAI 21.03; không tái lập 4 trạng thái `Normal/Security/Operational/Service`, không tái lập lớp `norm_layer` chưa rõ định nghĩa, và không tuyên bố đạt độ chính xác `99,72%` của bài báo.
2. **Hai mô hình DNN V2 và GRU V1 được huấn luyện dưới hai chế độ giám sát khác nhau:** DNN V2 sử dụng nhãn tấn công thật trong quá trình huấn luyện (`test1 + test2`) và chọn ngưỡng (`test3`), trong khi GRU/Ridge/Persistence V1 hoàn toàn không sử dụng nhãn tấn công khi huấn luyện (`train1 + train2`) và chọn ngưỡng $Q_{99.9}$ (`train3`).
3. **Dao động lớn giữa các seeds của DNN V2 tại ngưỡng đóng băng $\tau^*$:** Dù diện tích dưới đường cong `Point ROC-AUC` và `Point PR-AUC` ổn định, `Event Recall` tại ngưỡng cứng $\tau^*$ biến thiên mạnh từ `100%` (Seed 42) và `94.12%` (Seed 43) xuống `47.06%` (Seed 44).
4. **Ngưỡng V2 được hiệu chỉnh hậu nghiệm dựa trên tập Validation `test3`:** Việc bổ sung trần `max_point_fpr_cap = 0.05`, sàn `min_event_recall_floor = 0.75` và lưới phân vị thực nghiệm là bước điều chỉnh hậu nghiệm sau khi quan sát hiện tượng cảnh báo luôn bật ở ngưỡng thấp trên `test3`, không phải quy tắc tiền định hoàn toàn từ trước.
5. **Tập Held-out V2 (`test4 + test5`) không hoàn toàn mù (not completely blind) ở cấp độ toàn bộ dự án:** Mặc dù `test4 + test5` không tham gia vào quá trình huấn luyện hay chọn ngưỡng của DNN V2, hai phiên này đã được phân tích trong thí nghiệm V1 trước đó.
6. **Kết quả thực nghiệm không chứng minh sẵn sàng triển khai trên hệ thống ICS thực tế:** Mức báo động giả và độ nhạy ngưỡng trên tập dữ liệu học thuật HAI 21.03 cho thấy hệ thống hiện tại phục vụ mục tiêu nghiên cứu và giảng dạy, chưa đủ điều kiện vận hành tự động trong môi trường công nghiệp thực tế.
