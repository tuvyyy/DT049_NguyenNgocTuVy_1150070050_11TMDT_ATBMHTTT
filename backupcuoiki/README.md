# Backup cho cuối kỳ

Giữa kỳ dùng GRU. Thư mục này giữ phần DNN và tài liệu nghiên cứu ít dùng.

- `dnn/`: 39 file code, cấu hình, test, checkpoint, kết quả và tài liệu DNN,
  giữ nguyên cấu trúc tương đối gốc và nội dung từng file.
- `tai_lieu/`: README dài trước khi rút gọn và tài liệu nghiên cứu V1/GRU cũ.
- `legacy/`: bộ tạo cửa sổ cũ không còn được GRU dùng và sáu file thử 30 epoch.
- `manifest.json`: vị trí cũ, vị trí backup và SHA-256 của cả 60 file đã lưu.

Backup giữ đúng nội dung file đã commit trên Git. Với file văn bản, dùng xuống
dòng LF của Git để tránh lệch hash khi chuyển Windows/Kali; checkpoint và hình
giữ nguyên byte. `.gitattributes` giữ nội dung backup ổn định trên cả hai máy.

Các đường dẫn trong script/manifest khoa học DNN vẫn là đường dẫn gốc, vì vậy
**không chạy script hoặc test trực tiếp trong `backupcuoiki/dnn/`**.

Khi cần DNN cho cuối kỳ, từ thư mục gốc dự án:

```sh
python backupcuoiki/khoi_phuc_dnn.py
python backupcuoiki/khoi_phuc_dnn.py --apply
```

Lệnh đầu chỉ kiểm tra và xem trước. Lệnh sau chép DNN về đúng vị trí cũ,
giữ nguyên bản backup, kiểm tra hash và từ chối ghi đè file có nội dung khác.
Khởi động lại Streamlit sau khi khôi phục; mục DNN tự xuất hiện trở lại.
Chạy test DNN đã khôi phục bằng `python -m unittest discover -s demo -p 'test_dnn_v2_ui.py' -v`.
Các test giữa kỳ dùng `python -m unittest discover -s tests -p 'test_*.py' -v`.

`demo/ui_utils.py` là cầu nối import ngắn để DNN cũ dùng các tiện ích FE mới;
nội dung UI thật đã chuyển vào `demo/fe/`. Không cần khôi phục giao diện GRU cũ.

Các tài liệu trong `tai_lieu/` giữ để tra cứu, không cần đọc hết cho giữa kỳ.
Hướng dẫn hiện tại nằm ở `docs/HUONG_DAN_GIUA_KY.md` tại gốc dự án.
