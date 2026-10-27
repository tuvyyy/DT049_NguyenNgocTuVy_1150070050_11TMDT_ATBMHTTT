# DT049 — Phát hiện sai số bất thường trên HAI 21.03 bằng GRU

Đề tài giữa kỳ dùng GRU để dự đoán mẫu tín hiệu tiếp theo, so sai số RMSE với
ngưỡng chọn trên dữ liệu bình thường và đối chiếu cảnh báo với nhãn HAI.
PyTorch cung cấp `nn.GRU`, `nn.MSELoss` và `torch.optim.Adam`; dự án xây dựng
quy trình dữ liệu, cửa sổ thời gian, huấn luyện, chọn ngưỡng và đánh giá trên HAI.

## Hai nhánh của dự án

- `main`: bản gọn cho GRU Terminal giữa kỳ; giữ nguyên code GRU, tiền xử lý,
  đánh giá và các model/kết quả nghiên cứu chính thức.
- [`full-project`](https://github.com/tuvyyy/DT049_NguyenNgocTuVy_1150070050_11TMDT_ATBMHTTT/tree/full-project):
  bản đầy đủ có Streamlit, DNN trong `backupcuoiki/`, tài liệu và các phân tích bổ sung.

Kali đang dùng `main` thì chỉ cần cập nhật code; giữ nguyên `.venv`, dữ liệu HAI
và thư mục run đã có:

```bash
git pull --ff-only origin main
```

Khi cần mở lại bản đầy đủ:

```bash
git fetch origin
git switch full-project
# Quay về bản GRU Terminal:
git switch main
```

Đổi nhánh sẽ đổi các file do Git quản lý. Dữ liệu HAI, run và môi trường Python
bị Git bỏ qua nên không được đồng bộ giữa hai máy bằng các lệnh này.

Các file chính cần đọc trên `main`:

| File | Vai trò |
| --- | --- |
| `src/gru_dataset.py` | Tạo cửa sổ dữ liệu đầu vào. |
| `src/gru_model.py` | Cấu trúc GRU và dự đoán tín hiệu kế tiếp. |
| `src/train_gru.py` | Huấn luyện, chọn checkpoint và ngưỡng benchmark. |
| `demo/show_event.py` | Các lệnh Terminal đọc kết quả và tính lại dự đoán. |

`src/preprocess.py`, `src/preprocess_test.py` và `src/evaluate_final_test.py`
được giữ để đối chiếu quy trình dữ liệu và đánh giá. `models/`, `results/`
và `configs/` giữ nguyên nội dung nghiên cứu.

## Demo terminal khoảng 60 giây

Trong thư mục repo trên Kali, với môi trường `.venv` và run đã chuẩn bị:

```bash
./show samples
./show log
./show result
./show event
```

| Lệnh | Thực sự làm gì? |
| --- | --- |
| `./show samples` | Nạp checkpoint của run, dự đoán lại 3 cửa sổ đầu tiên của train1, dài 30 mẫu và bước nhảy 5 mẫu. Hiện một tín hiệu có tên theo metadata và RMSE trên cả 79 tín hiệu; giá trị đã chuẩn hóa, không học lại. |
| `./show log` | Đọc `train.log`, đối chiếu lịch sử CSV/JSON và SHA-256 checkpoint; không học lại. |
| `./show result` | Đọc `evaluation.json`, đối chiếu checkpoint, ngưỡng, bảng sự kiện và bảng theo phiên; không chạy suy luận. |
| `./show event` | Nạp `gru.pt`, dùng 79 tín hiệu HAI để tính lại RMSE quanh `test1_atk_02`, rồi đối chiếu nhãn và kết quả đã lưu. Không chạy thời gian thực. |

`show` dùng `.venv/bin/python`. Run mẫu được chỉ định rõ trong launcher là
`runs/gru/20261018T114814_2e1846de`; không tự chọn run mới nhất hay đổi sang benchmark.
Nếu thiếu file hoặc các nguồn không khớp, lệnh dừng với thông báo lỗi.

Chọn một run khác bằng `--run`, hoặc đặt `DT049_RUN`. Các tùy chọn được chuyển
nguyên vẹn tới Python. Ví dụ sau cần thay `<id>` bằng tên thư mục thực tế:

```bash
./show event --run runs/gru/<id> --detail
```

`./show samples --run runs/gru/<id>` dùng train1 và giới hạn dòng trong manifest/cấu hình
của run được chọn. Lệnh cần `models/preprocessing/scaler_metadata.json`, cấu hình
`configs/primary_gru.yaml` và PyYAML để đọc bước nhảy huấn luyện. Các giá trị được
tính khi chạy; không dùng `demo_gru_60s.txt`, bảng kết quả hay nhãn tấn công.
Ba mẫu dùng cùng mô hình đã lưu, không minh họa ba lần cập nhật trọng số trong quá khứ.

`--detail` hiện thêm thông tin run/checkpoint; với lệnh đánh giá còn có metric,
ngưỡng, đỉnh RMSE và thông tin đối chiếu.
Lời thoại và checklist nằm trong [hướng dẫn giữa kỳ](docs/HUONG_DAN_GIUA_KY.md).

## Điều kiện để chạy

**Clone GitHub không cung cấp sẵn dữ liệu HAI hoặc run demo.** `.gitignore` loại
`data/source/`, `data/raw/`, `data/processed/`, `results/logs/` và `runs/gru/`.
Cần giữ lại run đã chạy trên Kali và dữ liệu mà `data_manifest.json` trỏ tới.
`demo_gru_60s.txt` chỉ là bản đầu ra đã lưu, không thay thế các file đó.

Nếu chưa có môi trường, tạo `.venv` và cài các phiên bản trong lock file:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-lock.txt
```

Nếu `.venv` đã chạy được thì giữ nguyên. Demo terminal cần NumPy, PyTorch và PyYAML;
các lệnh `./show` không mở Streamlit và không tự huấn luyện.
Lock file trên `main` đã bỏ các thư viện chỉ phục vụ UI; phiên bản các thư viện
GRU/xử lý dữ liệu giữ theo bản đầy đủ trên `full-project`.

## Dữ liệu HAI 21.03

Nguồn công khai: [icsdataset/hai](https://github.com/icsdataset/hai), thư mục
[`hai-21.03`](https://github.com/icsdataset/hai/tree/master/hai-21.03).
Metadata dự án ghi checkout nguồn tại commit `2a814cebc9a66b06c9e5cd545e2d72e65d383737`.
Tải đúng phiên bản, giữ các file CSV nén gốc:

```text
data/source/hai_repo/hai-21.03/
  train1.csv.gz  train2.csv.gz  train3.csv.gz
  test1.csv.gz   test2.csv.gz   test3.csv.gz   test4.csv.gz   test5.csv.gz
data/processed/hai_21_03/
  train{1,2,3}_features.npy      train{1,2,3}_timestamps.npy
  final_test/test{1..5}_{features,timestamps,attack,attack_P1,attack_P2,attack_P3}.npy
```

Đây là ký hiệu gộp tên file. Features có dạng `(N, 79)`, `float32`; thời gian
liên tục từng giây. Nhãn `attack` không được đưa vào đầu vào GRU.
Scaler được học bằng train1+train2; train3 dùng kiểm định/chọn ngưỡng;
test1–test5 chỉ biến đổi bằng scaler đã lưu, không fit lại.

Mã chuẩn bị dữ liệu nằm ở `src/preprocess.py` và `src/preprocess_test.py`.
Phần tạo run qua UI nằm trong `demo/be/gru_runtime.py` trên nhánh `full-project`.
`preprocess.py` có ghi lại scaler, nên không chạy nó
chỉ để xem demo hoặc trên bộ artifact đang cần bảo toàn. Chuẩn bị dữ liệu từ
đầu là công việc riêng; tải CSV không tự tạo ra checkpoint hay run mẫu.

Một run cần có `settings.json`, `training.json`, `training_history.csv`,
`train.log`, `gru.pt`, `evaluation.json`, `per_session.csv`, `attack_events.csv`
và `data_manifest.json`. Manifest chứa đường dẫn `.npy` thực tế; nếu chuyển máy
phải chuẩn bị đúng các đường dẫn đó. Nếu có `scores_test1.npz`, script còn đối
chiếu RMSE tính lại với scores đã lưu; không dùng file này thay cho suy luận.

## Demo nhanh và benchmark là hai thí nghiệm khác nhau

| | Run demo mẫu | Benchmark chính thức |
| --- | --- | --- |
| Dữ liệu học/kiểm định | Tối đa 10.000 dòng mỗi phiên | Cấu hình nghiên cứu đã lưu |
| Huấn luyện | 3 epoch | Cấu hình trong `models/primary/gru_metadata.json` |
| Kiểm thử | test1 đầy đủ | test1–test5, 50 sự kiện |
| Checkpoint | `runs/gru/<id>/gru.pt` | `models/primary/gru_w30_seed42.pt` |
| Ngưỡng seed 42 | Khoảng 1.605896 trong run mẫu | Khoảng 1.058582 |
| Cảnh báo khớp nhãn | 5/5 trong bản đầu ra lưu | 40/50 trong CSV chính thức |

Số demo ở đây là tham chiếu từ `demo_gru_60s.txt`, không phải hằng số trả về của
script. Run khác có thể cho kết quả khác. Các kiểm tra dùng artifact của chính
run được chọn; không chỉnh số cho giống mẫu. Metric nghiên cứu đầy đủ giữ ở
`results/metrics/final_test_summary.csv` và các CSV liên quan.
5/5 sự kiện khớp nhãn không có nghĩa tất cả cảnh báo đều đúng.

DNN **đã được triển khai** như thí nghiệm bổ sung có điều chỉnh; hiện được lưu
ở [`full-project/backupcuoiki/`](https://github.com/tuvyyy/DT049_NguyenNgocTuVy_1150070050_11TMDT_ATBMHTTT/tree/full-project/backupcuoiki),
không trình diễn giữa kỳ.
Bài báo Kuchař & Fujdiak (2025) dùng DNN, không phải GRU; không gọi phần GRU
của dự án là tái lập nguyên bản bài báo.

## Kiểm tra

```bash
.venv/bin/python -B -m unittest discover -s tests -p 'test_*.py' -v
```

Kiểm thử terminal dùng dữ liệu tổng hợp và trọng số kiểm thử để kiểm tra lỗi,
phép tính RMSE và tính chỉ đọc. Chúng không chứng minh thực nghiệm HAI thật
đã được xác minh. Ở đợt rà soát này, môi trường Codex không có run demo/dữ liệu
HAI cục bộ; cần chạy lại bốn lệnh và đối chiếu trực tiếp trên Kali.

Kiểm tra Dashboard bằng `demo/check_product.py` và chạy UI bằng
`bash demo/run_dashboard.sh` sau khi chuyển sang nhánh `full-project`.
