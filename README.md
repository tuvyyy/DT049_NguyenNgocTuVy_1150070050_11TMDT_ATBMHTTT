# DT049 — GRU phát hiện bất thường trên HAI 21.03

Bản giữa kỳ tập trung vào **GRU**. Phần DNN và tài liệu dài đã chuyển vào
[`backupcuoiki/`](backupcuoiki/README.md), giữ lại cho cuối kỳ.

## Chạy trên Kali

Tại thư mục dự án, dùng môi trường Python đã cài thư viện:

```sh
source .venv/bin/activate
bash demo/run_dashboard.sh
```

Mở **http://127.0.0.1:8501** → **Overview → Thực hiện GRU**.

1. Chọn thư mục dữ liệu → **Chuẩn bị dữ liệu**.
2. Chọn seed, số epoch → **Huấn luyện GRU** → xem log.
3. Chọn phiên test → **Chạy đánh giá**.
4. Chọn sự kiện → kiểm tra kết quả và đường sai số.

Mỗi lần chạy thật lưu vào `runs/gru/<id>/`, không ghi đè kết quả chính thức.
Chế độ nhanh dùng một phần dữ liệu học; số liệu mới có thể khác benchmark.

## Đọc gì để chuẩn bị giữa kỳ?

Chỉ bắt đầu bằng [hướng dẫn giữa kỳ](docs/HUONG_DAN_GIUA_KY.md):
cách thao tác UI, giải thích các ô và những đoạn code GRU cần nắm.

| Thư mục | Nội dung |
| --- | --- |
| `demo/app.py` | File khởi động Streamlit, không chứa nội dung UI |
| `demo/fe/` | FE: các màn hình, thành phần giao diện và CSS |
| `demo/be/` | BE: đọc artifact, điều phối huấn luyện và đánh giá GRU |
| `tests/` | Kiểm thử UI, chạy tác vụ và kiểm tra backup |
| `src/` | Mô hình GRU, dữ liệu cửa sổ, huấn luyện, đánh giá và baseline |
| `configs/` | Cấu hình GRU và xử lý dữ liệu |
| `models/` | Scaler và checkpoint chính thức của GRU/baseline |
| `results/` | Kết quả nghiên cứu chính thức và hình phân tích GRU |
| `runs/gru/` | Các lần chạy thử trên UI; không đưa vào Git |
| `docs/` | Một hướng dẫn cho giữa kỳ |
| `backupcuoiki/` | DNN và tài liệu tra cứu, chưa cần cho buổi show |

## Kết quả chính thức để đối chiếu

Trên `test1–test5`, GRU seed 42 phát hiện **40/50 sự kiện**, Event F1 **0.188235**,
Normal FA/h **2.8220**, Median Delay **22 giây**. Các số này nằm trong
`results/metrics/final_test_summary.csv`; không lấy số chạy nhanh thay vào báo cáo.

Bài báo tham khảo Kuchař & Fujdiak (2025) dùng DNN; GRU là hướng thực nghiệm
chính của bản giữa kỳ, không gọi là tái lập nguyên bản bài báo. Toàn bộ diễn giải
học thuật trước đây vẫn còn trong backup.

## Kiểm tra

```sh
python -m unittest discover -s tests -p 'test_*.py' -v
python demo/check_product.py
```
