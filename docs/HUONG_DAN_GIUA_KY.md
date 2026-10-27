# Giữa kỳ: demo terminal một sự kiện GRU

Nhánh `main` dùng cho GRU Terminal. UI, DNN và các phân tích bổ sung được giữ
nguyên trên nhánh `full-project`; Kali đang ở `main` chỉ cần `git pull --ff-only origin main`.

## Chuẩn bị trước giờ bảo vệ

- Vào đúng thư mục repo trên Kali, kiểm tra `.venv/bin/python` và `show` có sẵn.
  Launcher dùng đúng Python của `.venv`, không phụ thuộc lệnh `python` toàn hệ thống.
- Giữ run `runs/gru/20261018T114814_2e1846de` đã tạo trước đó. Dữ liệu, log cục bộ
  và `runs/gru/` bị `.gitignore` loại khỏi Git; clone mới không có đủ bộ demo.
- Nếu dùng run khác, truyền `--run` sau tên lệnh, hoặc đặt biến `DT049_RUN`.
  Không tự chọn run mới nhất vì có thể là lần thử hoặc chưa đánh giá.
- Dữ liệu `.npy` phải tồn tại đúng đường dẫn trong `data_manifest.json`.
  Nguồn tải và cấu trúc thư mục được mô tả trong [README](../README.md#dữ-liệu-hai-2103).
- Phóng to terminal, chọn font khoảng 20–24 và chạy thử cả ba lệnh trước.
  Gặp `DỪNG` thì xử lý lỗi dữ liệu/artifact; không chạy train để che lỗi.
- Nếu cần một bản dự phòng mới, chạy `./show all --detail > demo_gru_local.txt`
  và đọc kiểm tra sau khi lệnh thành công. File đó là output lưu, không phải
  bằng chứng đang thực thi lại khi mở nó sau này.

## Đúng ba lệnh trình diễn

```bash
./show log
./show result
./show event
```

| Lệnh | Đọc hoặc tính gì? | Không nên nói |
| --- | --- | --- |
| `log` | Trích log huấn luyện thật đã lưu, kiểm tra CSV/JSON và hash checkpoint. | “Em đang train.” |
| `result` | Đọc kết quả kiểm thử đã lưu, đối chiếu các bảng và checkpoint. | “Em vừa chạy lại toàn bộ test1.” |
| `event` | Nạp trọng số GRU, tính lại dự báo/RMSE quanh một sự kiện, đối chiếu nhãn HAI. | “Hệ thống đang giám sát ICS trực tiếp.” |

Mặc định là `test1_atk_02`. Dùng `./show event --detail` khi được hỏi về ngưỡng,
đỉnh RMSE, giờ kết thúc và việc đối chiếu. Dùng `./show result --detail` để xem
Event F1, Normal FA/h, Median Delay. Đây vẫn là các số đọc từ artifact.
`./show timeline` là biểu đồ chữ tùy chọn, không nằm trong ba bước chính.

## Lời thoại khoảng 45–60 giây

Mẫu này dựa vào run đã lưu trong `demo_gru_60s.txt`. Chỉ đọc các con số này nếu
đã chạy thử thành công trên Kali và output thực tế khớp. Nếu chọn run khác,
nói theo output thật, không sửa số để khớp mẫu.

**Sau `./show log` (khoảng 15 giây):**

“Đây là log huấn luyện em đã lưu trước buổi báo cáo, không phải đang train lại.
Lần chạy nhanh dùng ba epoch; loss từ khoảng 0,499 giảm còn 0,207 và đã lưu checkpoint.”

**Sau `./show result` (khoảng 10–15 giây):**

“Đây là kết quả kiểm thử đã lưu trên test1 đầy đủ: cảnh báo khớp 5/5 sự kiện nhãn.
Lần chạy nhanh này khác benchmark 40/50 trên năm phiên ở slide, nên số liệu khác nhau.”

**Sau `./show event` (khoảng 20–25 giây):**

“Ở bước này em nạp checkpoint và tính lại sai số dự đoán cho một sự kiện.
Nhãn bắt đầu lúc 17:28:11; cảnh báo đầu trong vùng nhãn là 17:28:29, trễ 18 giây.
GRU chỉ nhìn tín hiệu; nhãn tấn công chỉ dùng đối chiếu. Đây là sai số bất thường
trùng nhãn HAI, chưa kết luận nguyên nhân vật lý.”

Khi mở chi tiết, mốc kết thúc theo nhãn là 17:29:48. Ba mốc cần phân biệt:
**17:28:11 → 17:28:29 → 17:29:48**. Không suy ra nguyên nhân vật lý chỉ từ việc
sai số tăng hoặc giảm; nếu chưa kiểm chứng thì nói chưa kết luận.

## Nắm code đủ để giải thích

| File | Ý chính |
| --- | --- |
| `src/gru_model.py` | Dùng `torch.nn.GRU` và `Linear`: nhận 30 mẫu × 79 tín hiệu, dự đoán mẫu tiếp theo. |
| `src/gru_dataset.py` | Cắt cửa sổ trong từng phiên, không ghép qua ranh giới hai file. |
| `src/train_gru.py` | Dùng `torch.nn.MSELoss` và `torch.optim.Adam`; lưu checkpoint tốt nhất theo validation. |
| `demo/show_event.py` | Đọc artifact hoặc suy luận một sự kiện, không gọi hàm train và không ghi file kết quả. |
| `src/evaluate_final_test.py` | Phép tính RMSE và quy tắc ghép sự kiện của benchmark. |

PyTorch cung cấp các khối GRU, loss và optimizer; phần của dự án là ứng dụng
chúng vào quy trình HAI: tiền xử lý, cửa sổ, lựa chọn ngưỡng và đánh giá.
Train1+train2 dùng học; train3 kiểm định/chọn ngưỡng; test1–test5 dùng đánh giá.
Scaler chỉ fit trên train1+train2. Nhãn `attack` không nằm trong 79 đầu vào GRU.

`event` tính lại một khoảng dữ liệu, không thay thế việc đánh giá toàn phiên.
Kết quả ghép 1–1 đã lưu phải phù hợp với suy luận. Nếu có điểm vượt ngưỡng nhưng
CSV không ghi sự kiện khớp, script dừng để kiểm tra quy tắc ghép trên toàn phiên,
không tự chuyển trạng thái hoặc sửa metric.

## Khi được hỏi thêm

- `./show seeds`: in ba seed từ `results/metrics/final_test_summary.csv`.
- `./show benchmark-log`: đọc 12 dòng cuối `results/logs/train_gru.log` nếu file
  còn trên máy. Thiếu file thì báo thiếu, không tạo log thay thế.
- `./show benchmark-threshold`: đọc ngưỡng benchmark seed 42, khoảng 1.058582.
  Không dùng ngưỡng này thay cho khoảng 1.605896 của run demo mẫu.
- **5/5 có phải chính xác 100%?** Không. Vẫn có cảnh báo không khớp nhãn;
  xem Event F1 và Normal FA/h bằng chế độ chi tiết.
- **DNN ở đâu?** Đã triển khai thí nghiệm bổ sung và lưu trong `backupcuoiki/`
  trên nhánh `full-project`.
  Giữa kỳ chỉ trình diễn GRU. Bài báo Kuchař & Fujdiak (2025) dùng DNN;
  không gán GRU cho các tác giả, không nói đã tái lập nguyên bản 100%.

## Bằng chứng và phần cần kiểm tra lại

`demo_gru_60s.txt` là bản đầu ra cũ đã lưu trong Git, có ghi nguồn ở đầu file.
Nó không phải log mới và không chứng minh checkpoint/dữ liệu trên máy khác còn nguyên.
Các test tự động dùng dữ liệu tổng hợp, không phải HAI thật. Môi trường Codex
không có run mẫu và dữ liệu HAI cục bộ; người trình bày cần chạy lại ba lệnh
trên Kali, thêm `--detail` để đối chiếu ngưỡng, RMSE và thời gian trước giờ bảo vệ.
