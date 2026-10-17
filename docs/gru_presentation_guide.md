# Thao tác thực hiện GRU

Mở **Overview → Thực hiện GRU**. Đây là khu vực chạy GRU thật, gồm bốn bước.
Không có phần giới thiệu bài toán, bài giảng hay lời dẫn thuyết trình.
Các trang benchmark và DNN vẫn giữ riêng, sử dụng kết quả đóng băng.

## 1. Dữ liệu

- Chọn **Mảng đã xử lý (.npy)** và nhập thư mục `data/processed/hai_21_03` trên Kali.
  Trong thư mục cần có `train1/2/3_features.npy`, `train1/2/3_timestamps.npy`;
  thư mục con `final_test/` chứa features, timestamps, attack, attack_P1/P2/P3
  cho từng phiên test1–test5.
- Hoặc chọn **CSV HAI gốc** và nhập thư mục chứa train1–3 và test1–5 dạng
  `.csv.gz` hoặc `.csv`. Ví dụ `data/source/hai_repo/hai-21.03`.
- Chọn **Chạy nhanh** để dùng tối đa 10.000 dòng đầu mỗi phiên train1, train2,
  train3; hoặc chọn **Toàn bộ dữ liệu**. Không cắt bớt dữ liệu test.
- Bấm **Chuẩn bị dữ liệu**. Worker đọc file thật, kiểm tra 79 cột float32,
  NaN/Inf, nhãn 0/1, độ dài và thời gian liên tục từng giây.
- CSV được chuẩn hóa bằng scaler gốc của dự án (không fit lại). Mảng mới chỉ
  ghi vào thư mục của lần chạy. Nguồn CSV/NPY không bị sửa.
- Xong bước này mới bật nút **Tiếp theo**.

## 2. Huấn luyện GRU

- Chọn seed 42/43/44, số epoch tối đa 1–50, learning rate.
- Bấm **Huấn luyện GRU**. Đây là huấn luyện thật trên CPU; không phải phát lại log.
- Cửa sổ cố định 30, đầu vào 79, hidden size 64, một lớp GRU.
- Gọi trực tiếp `train_single_run` trong `src/train_gru.py` và mô hình/dataset
  có sẵn. Không thay đổi hàm huấn luyện cốt lõi.
- Log được cập nhật tự động khoảng 1,5 giây/lần; nút **Dừng tác vụ** kết thúc
  đúng tiến trình của lần chạy nếu cần.
- Checkpoint được chọn theo validation; sau đó tính ngưỡng Q99.9 trên train3
  của lần chạy này. Giới hạn 10.000 dòng áp dụng cả hiệu chuẩn trong chế độ nhanh.
- Xong có thể chuyển giữa **Log huấn luyện** và **Đường học**. Train Loss là
  MSE; chỉ số kiểm định là RMSE, không phải cùng thang đo.

## 3. Đánh giá

- Chọn ít nhất một phiên test, mặc định test1 để demo nhanh.
- Bấm **Chạy đánh giá**. Worker nạp checkpoint vừa huấn luyện, kiểm tra hash,
  dự đoán trên toàn bộ phiên đã chọn và dùng ngưỡng vừa hiệu chuẩn.
- Tái sử dụng các hàm scoring, trích xuất sự kiện, ghép 1–1 và normal false
  alarms trong `src/evaluate_final_test.py`; công thức tổng hợp metric giữ nguyên.
- Kết quả hiển thị lấy từ lần chạy mới. Mẫu số là số sự kiện thật trong các
  phiên đã chọn, không gán cố định 50 hoặc dùng số 40/50 của benchmark.
- Có thể tải kết quả JSON và mở log đánh giá.

## 4. Kiểm tra sự kiện

- Chọn phiên, lọc phát hiện/bỏ sót, chọn sự kiện.
- Xem thời gian, trạng thái, độ trễ và đường sai số/ngưỡng được tính từ lần chạy mới.
- Sự kiện bỏ sót có độ trễ N/A; không gán bằng 0.
- Biểu đồ không lấy hình seed 42 từ benchmark và không dựng đường tín hiệu giả.

## Quản lý lần chạy

Mỗi lần có thư mục riêng `runs/gru/<id>/` gồm settings, trạng thái, log từng bước,
manifest dữ liệu, `gru.pt`, history, ngưỡng, điểm số và kết quả đánh giá.
Thư mục này được gitignore. Các tệp chính thức trong `models/`, `results/`,
`configs/`, `src/` không được ghi bởi worker.

**Tạo lần chạy mới** để đổi dữ liệu/thông số sau khi đã hoàn tất. **Mở lần chạy
trước** để tiếp tục kiểm tra sau khi tải lại trình duyệt. Không tạo chạy mới hoặc
mở chạy khác khi tiến trình hiện tại đang làm việc.

Lệnh chạy app vẫn là `streamlit run demo/app.py`, trong môi trường Python đã
cài các thư viện GRU của dự án. Giao diện báo rõ thư viện thiếu và không giả lập
thành công khi thiếu PyTorch/dataset.

## Kiểm thử

`python -m unittest discover -s demo -p 'test_*.py' -v`

Test luồng chạy thật dùng dữ liệu tổng hợp nhỏ trong thư mục tạm, huấn luyện một
epoch rồi đánh giá hai sự kiện tổng hợp. Đây chỉ là test chức năng, không phải
kết quả HAI. Thư mục chạy test được dọn sau khi xong; hash checkpoint và kết quả
chính thức được đối chiếu trước/sau.

Chưa kiểm tra trực quan ở 1366×768 hoặc 1920×1080 vì Computer Use không có browser
khả dụng trong phiên này. Không tuyên bố đã kiểm tra bố cục hoặc mức cuộn thực tế.

### Kiểm tra ngày 2026-10-16

- 17 test qua trong môi trường kiểm tra tạm có PyTorch CPU: 7 test DNN cũ,
  10 test UI/tác vụ GRU mới.
- AppTest bấm ba nút thật: Chuẩn bị dữ liệu, Huấn luyện GRU, Chạy đánh giá.
  Worker tạo checkpoint, ngưỡng, điểm số và kết quả mới trên dữ liệu tổng hợp;
  UI đọc được kết quả/sự kiện của đúng lần chạy. Không sử dụng dữ liệu HAI thật.
- Kiểm tra thêm nhánh CSV dùng scaler đóng băng, giới hạn train/validation nhưng
  không cắt test, trạng thái lỗi, khóa tác vụ và xác minh tiến trình trước khi dừng.
- `demo/check_product.py`: PASS trong môi trường kiểm tra này.
- Băm SHA-256 của 142 file nghiên cứu trước/sau: không thay đổi.
- Môi trường kiểm tra nằm trong thư mục tạm, không sửa môi trường Python gốc.
- Các test nguồn vốn import module Linux `resource` vẫn không chạy trên Windows;
  không tuyên bố đã tái lập bộ benchmark trên Kali hay kiểm thử các module đó.
- Chưa commit/push.
