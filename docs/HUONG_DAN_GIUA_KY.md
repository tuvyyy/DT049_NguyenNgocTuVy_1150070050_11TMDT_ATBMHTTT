# Giữa kỳ: thao tác GRU và phần code cần nắm

## Chạy UI

Vào **Overview → Thực hiện GRU**. Các trang phân tích khác để tra cứu khi được hỏi.

| Bước | Bạn làm gì? | Kết quả nhìn thấy |
| --- | --- | --- |
| Dữ liệu | Chọn `.npy`, giữ `data/processed/hai_21_03`, bấm **Chuẩn bị dữ liệu** | Bảng kiểm tra các file |
| Huấn luyện | Giữ seed 42, thử 3 epoch, learning rate 0.001; bấm **Huấn luyện GRU** | Log chạy thật và checkpoint mới |
| Đánh giá | Chọn test1 để demo nhanh, bấm **Chạy đánh giá** | Các chỉ số vừa tính trên test1 |
| Sự kiện | Chọn phiên và sự kiện | Có phát hiện không, độ trễ, biểu đồ sai số |

- **Seed:** mã khởi tạo để dễ lặp lại thí nghiệm, không phải điểm số.
- **Epoch:** một lượt học qua dữ liệu; 3 epoch là tối đa 3 lượt.
- **Learning rate:** mức điều chỉnh khi học; giữ mặc định 0.001 để bắt đầu.
- **Chạy nhanh:** tối đa 10.000 dòng đầu của mỗi phiên học/kiểm định.
  Phiên test được chọn vẫn chạy đầy đủ. Đây là lần chạy demo mới, không hứa ra 40/50.
- Nếu dùng CSV gốc, chọn định dạng CSV và thư mục chứa train1–3, test1–5.
  App dùng scaler đã có; các mảng mới được ghi riêng trong lần chạy.
- Log tự cập nhật. Chỉ qua bước sau khi tác vụ hoàn tất; có nút dừng nếu cần.
- Không có dữ liệu/thư viện thì app báo lỗi, không tạo kết quả giả.

## Nắm code theo thứ tự này

Không cần đọc hết Dashboard hoặc thư mục backup để chuẩn bị phần GRU.

| Đọc file/đoạn nào | Cần hiểu điều gì? |
| --- | --- |
| `src/gru_model.py` — lớp `OneStepGRU` | Nhận chuỗi 30 mẫu × 79 tín hiệu; GRU và lớp đầu ra dự đoán mẫu tiếp theo. |
| `src/gru_dataset.py` — `SessionWindowDataset.__getitem__` | Cắt 30 mẫu làm đầu vào, mẫu kế tiếp làm đáp án; không ghép qua hai phiên. |
| `src/train_gru.py` — hàm `train_single_run` | Tính sai số, cập nhật mô hình, kiểm định và lưu checkpoint tốt nhất. Chưa cần đọc toàn bộ phần vẽ hình. |
| `src/evaluate_final_test.py` — `score_gru`, các hàm `extract_*` và `match_events_one_to_one` | Tính sai số dự báo, tạo cảnh báo, ghép với sự kiện thật và tính độ trễ. |
| `demo/be/gru_runtime.py` — `prepare`, `train`, `evaluate` | BE gọi lại các hàm GRU có sẵn; mỗi lần ghi vào thư mục riêng. |

`demo/fe/gru_workflow.py` chỉ dựng các ô nhập/nút bấm và hiển thị tiến trình.
`demo/fe/pages/` chứa các trang phân tích; CSS nằm trong `demo/fe/styles/`.
`demo/app.py` chỉ khởi động FE; các tác vụ thực thi nằm trong `demo/be/`.
`configs/primary_gru.yaml` giữ các thiết lập chính: cửa sổ 30, hidden size 64,
79 đầu vào, một lớp GRU. Không cần học thuộc toàn bộ code giao diện.

## Những câu cần tự giải thích được

- **GRU học gì?** Học dữ liệu hoạt động bình thường để dự đoán tín hiệu tiếp theo.
- **Khi nào báo bất thường?** Sai số dự báo vượt ngưỡng đã chọn trên train3.
- **Dữ liệu chia ra sao?** Học train1+train2; kiểm định/chọn ngưỡng train3;
  kiểm thử trên các phiên test được chọn. Benchmark gốc dùng test1–test5.
- **40/50 nghĩa gì?** Bắt được 40 cuộc tấn công, còn bỏ sót 10; không có nghĩa
  mọi cảnh báo đều đúng. Đây là kết quả GRU seed 42 chính thức, không phải số cố định của UI.
- **Event F1:** Kết hợp khả năng bắt được tấn công và độ đúng của cảnh báo.
- **Normal FA/h:** Số đợt báo động giả mỗi giờ dữ liệu bình thường.
- **Median Delay:** Trung vị thời gian phát hiện, chỉ trên các sự kiện phát hiện được.
  Sự kiện bỏ sót có delay không áp dụng, không gán 0.

## Khi được hỏi về bài báo/DNN

Bài báo tham khảo dùng DNN phân loại bốn trạng thái; phần giữa kỳ tập trung GRU
trên HAI. DNN là thí nghiệm bổ sung có điều chỉnh, đã lưu ở `backupcuoiki/dnn/`.
Không nói bài báo gốc dùng GRU hoặc đã tái lập nguyên bản 100%.

## Lưu ý khi show

- Chuẩn bị sẵn dữ liệu trên Kali và môi trường `.venv` đang chạy được.
- Không mở lần lượt tất cả trang phân tích; đi đúng bốn bước thao tác trên.
- Tạo lần chạy mới để thay đổi thiết lập sau khi đã huấn luyện/đánh giá xong.
- Dữ liệu gốc, scaler và benchmark không bị ghi đè bởi UI.
- Tài liệu dài và báo cáo học thuật cũ nằm trong `backupcuoiki/tai_lieu/`.
- Bộ tạo cửa sổ cũ và kết quả thử 30 epoch nằm trong `backupcuoiki/legacy/`,
  không còn trộn vào mã/kết quả đang dùng cho giữa kỳ.
