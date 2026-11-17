# DT049 — GRU Terminal

Nhánh `main` dùng cho demo giữa kỳ trên HAI 21.03.
[Nhánh `full-project`](https://github.com/tuvyyy/DT049_NguyenNgocTuVy_1150070050_11TMDT_ATBMHTTT/tree/full-project)
giữ bản đầy đủ: UI, DNN, tài liệu và các phân tích bổ sung.

## Chạy trên Kali

Trong thư mục dự án, với môi trường `.venv` và run đã chuẩn bị:

```bash
./show samples
./show log
./show result
./show event
```

- `samples`: dự đoán lại 3 cửa sổ train1 bằng checkpoint đã lưu; hiển thị một
  tín hiệu đã chuẩn hóa và RMSE trên 79 tín hiệu.
- `log`: đọc log huấn luyện đã lưu.
- `result`: đọc kết quả đánh giá đã lưu.
- `event`: tính lại RMSE và đối chiếu nhãn cho `test1_atk_02`.

Các lệnh không huấn luyện lại hay ghi đè kết quả. Có thể thêm `--detail`,
`--run runs/gru/<id>`; `event` hỗ trợ `--event test1_atk_01`.

## File cần đọc

| File | Vai trò |
| --- | --- |
| `src/gru_dataset.py` | Cửa sổ tín hiệu đầu vào và giá trị thật kế tiếp. |
| `src/gru_model.py` | Cấu trúc GRU, dự đoán 79 giá trị tín hiệu. |
| `src/train_gru.py` | Tính lỗi MSE, cập nhật trọng số và lưu mô hình. |
| `demo/show_event.py` | Các lệnh Terminal, suy luận và cảnh báo theo RMSE. |

`src/preprocess.py`, `src/preprocess_test.py` và `src/evaluate_final_test.py`
giữ quy trình tiền xử lý/đánh giá. `models/`, `results/`, `configs/` chứa
artifact và cấu hình nghiên cứu; không chỉnh chúng để khớp số demo.

## Dữ liệu và môi trường

Launcher dùng `.venv/bin/python`, mặc định run `runs/gru/20261018T114814_2e1846de`.
Clone GitHub không kèm dữ liệu HAI hoặc run demo. Giữ các file của run và các mảng
`.npy` mà `data_manifest.json` trỏ tới trên Kali. Thiếu file hoặc sai SHA-256 thì dừng.

Một run có `settings.json`, `training.json`, `gru.pt`, `data_manifest.json`,
`train.log`, `training_history.csv`, `evaluation.json`, `attack_events.csv`,
`per_session.csv`. Các mảng tín hiệu có dạng `(N, 79)`, float32 đã chuẩn hóa;
thời gian cách nhau một giây. Nhãn tấn công chỉ dùng đối chiếu, không đưa vào GRU.

Nguồn HAI: [icsdataset/hai](https://github.com/icsdataset/hai), phiên bản 21.03.
Train1+train2 dùng học; train3 chọn mô hình/ngưỡng; test1–test5 dùng đánh giá.
Demo nhanh và benchmark là hai lần chạy khác nhau, không trộn số liệu.

Giữ `.venv` đang hoạt động. Nếu cài mới:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-lock.txt
```

Kiểm tra code: `.venv/bin/python -B -m unittest discover -s tests -p 'test_*.py' -v`.
Test tự động không thay thế việc xác minh với dữ liệu HAI thật trên Kali.

## Chạy lại GRU và đối chiếu kết quả

Script [`DT049_replay_GRU_readonly.py`](DT049_replay_GRU_readonly.py) nạp checkpoint GRU seed 42 đã lưu, suy luận lại trên HAI 21.03 và so sánh từng chỉ số với các bảng CSV trong `results/metrics/`. Script **chỉ đọc**, không huấn luyện lại, không điều chỉnh ngưỡng, không ghi đè checkpoint hoặc kết quả.

Trên Kali, tại thư mục gốc dự án, khi đã có dữ liệu HAI và môi trường `.venv`:

```bash
# Kiểm tra kỹ thuật mạng bằng dữ liệu GIẢ LẬP; không phải chạy HAI
./.venv/bin/python DT049_replay_GRU_readonly.py --smoke

# Chạy lại riêng test1 trước
./.venv/bin/python DT049_replay_GRU_readonly.py --repo . --session test1

# Chạy lại đủ test1–test5 và đối chiếu benchmark
./.venv/bin/python DT049_replay_GRU_readonly.py --repo .
```

Script sử dụng cấu hình cố định của benchmark: 30 thời điểm đầu vào, 79 tín hiệu, checkpoint `models/primary/gru_w30_seed42.pt` và ngưỡng Q99.9 = `1.0585819482803345`. Dữ liệu được nạp từ các mảng `.npy` có sẵn (đối chiếu SHA-256 theo manifest), hoặc từ CSV HAI gốc kết hợp scaler đã khóa. Nếu thiếu file, sai hash hoặc sai cấu trúc dữ liệu, script dừng; **clone GitHub không kèm dữ liệu HAI**.

Đầu ra in `replay`, `saved`, `delta` và `MATCH/DIFF` cho TP, FN, cảnh báo không khớp, Event Recall, Event F1, cảnh báo sai/giờ và độ trễ phát hiện trung bình. Kết quả tham chiếu đã lưu cho seed 42 là **40/50 sự kiện phát hiện**, Event Recall **80%**, Event F1 **18,82%**, **2,822** cảnh báo sai/giờ bình thường. Các số này chỉ được xác nhận *chạy lại* khi lệnh trên Kali hoàn tất và in `MATCH`.

Bài báo Kuchař & Fujdiak (2025) báo cáo Accuracy phân loại bốn trạng thái trên một bộ dữ liệu robot khác; không lấy các chỉ số GRU trên HAI để tính chênh lệch Accuracy trực tiếp với bài báo.

## Đổi nhánh

Kali đang ở `main`: `git pull --ff-only origin main`.
Mở bản đầy đủ: `git fetch origin` rồi `git switch full-project`.
Quay về bản gọn: `git switch main`. Đổi nhánh không đồng bộ dữ liệu/run bị Git bỏ qua.
