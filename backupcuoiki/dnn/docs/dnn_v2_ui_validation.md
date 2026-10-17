# DNN Replication V2 — kiểm tra giao diện

## Phạm vi

- Chỉ tổ chức lại trang DNN Replication V2; giữ sidebar hiện có.
- Thanh radio ngang có 5 tab, conditional rendering: chỉ dựng nội dung tab đang chọn.
- Bỏ header chung cỡ lớn trên riêng trang V2; giữ nguyên header của các trang khác.
- CSS yêu cầu marker `.st-key-dnn_v2`, kể cả các quy tắc nền và khoảng cách.
- Chỉ commit/push sau khi người dùng duyệt; người dùng đã yêu cầu push bản cập nhật này.

## File thay đổi

| File | Nội dung |
| --- | --- |
| `demo/app.py` | Gọi renderer V2 riêng và dùng header gọn cho V2 |
| `demo/dnn_v2.py` | 5 tab, biểu đồ chọn metric, bộ lọc, phân trang và tài liệu |
| `demo/dnn_v2.css` | Dark theme, card, thanh tab, chiều cao ảnh và vùng chi tiết |
| `demo/test_dnn_v2_ui.py` | Kiểm thử tích hợp Streamlit AppTest |
| `docs/dnn_v2_dashboard_reference.md` | Giữ lại các phần diễn giải học thuật từ trang cũ |
| `docs/dnn_v2_ui_validation.md` | Báo cáo này |

## Hành vi mới

1. **Tổng quan:** 4 card so sánh DNN/GRU seed 42: số sự kiện, Event F1, Median Delay, Normal FA/h. Không dựng bảng hoặc biểu đồ trong tab này.
2. **So sánh mô hình:** một biểu đồ tương tác; mặc định 4 mô hình; có đủ seeds 42/43/44 khi chọn. Bảng tổng hợp đầy đủ, theo phiên, confusion matrices và phân hệ nằm trong expander mặc định đóng.
3. **Sự kiện tấn công:** mặc định test4, DNN seed 42 τ*, `test4_atk_01`. Tối đa 6 sự kiện mỗi trang. Đổi bộ lọc sẽ về trang đầu; không giữ lựa chọn sự kiện đã bị lọc ra. Có trạng thái rỗng và vô hiệu hóa Trước/Tiếp tại biên. Chi tiết hiển thị thời điểm bắt đầu hiệu lực, kết thúc, trạng thái và độ trễ; thông tin điểm số/threshold để trong vùng mở rộng.
4. **Độ ổn định 3 Seeds:** card Recall/F1/FA/h, cảnh báo seed 44 phát hiện 8/17. Chọn một trong bốn hình gốc sẽ thay thế phần card, mỗi lần chỉ dựng một hình. Phân phối xác suất được ghi rõ là minh họa độ nhạy, không phải một đường quét threshold mới.
5. **Phương pháp & Giới hạn:** sơ đồ Train → Validation → Held-out Test, khác biệt supervision, giới hạn và tài liệu học thuật đầy đủ trong expander. Các manifest/audit gốc vẫn truy cập được bằng bộ chọn tài liệu.

## Kiểm thử tự động

Lệnh:

```powershell
python -X utf8 -m unittest discover -s demo -p test_dnn_v2_ui.py -v
```

Kết quả sau lần sửa khoảng cách và tương phản: **7/7 test PASS** (5.470 giây trong lần chạy cuối).

Suite kiểm tra:

- Khởi tạo app; đủ 5 tab và chuyển qua lại; tab Tổng quan không dựng bảng/chart hoặc bộ lọc của tab khác.
- KPI DNN/GRU khớp CSV V2; dữ liệu thật được gửi tới biểu đồ khớp CSV với cả 4 metric và cả lựa chọn 4/8 mô hình.
- Bảng chi tiết mặc định đóng; các bộ chọn bảng hoạt động.
- Mặc định `test4_atk_01`, thời điểm và delay đúng; phân trang 6/6/5 trên 17 sự kiện không trùng/mất sự kiện; nút biên và reset khi lọc.
- Cả 11 lựa chọn mô hình/ngưỡng trong explorer được đối chiếu trạng thái/delay với CSV; lọc phân hệ, trạng thái rỗng không gây lỗi.
- Chỉ số 3 seeds và threshold khớp CSV; chọn hình chỉ dựng một hình; tài liệu Markdown và JSON audit truy cập được.
- Cả 7 trang khác render không phát sinh exception; không mang CSS V2 hoặc radio V2 sang trang khác.
- Guard từ chối bảng V1 50 sự kiện thay vì hiển thị trong V2.

HTTP health của Streamlit chạy tại `http://127.0.0.1:8501/_stcore/health`: **200**.

Đối chiếu SHA-256 trước/sau trên **142 file** trong `results/`, `models/`, `configs/`, `src/`: **không có thay đổi**. Không chạy train, inference hoặc hiệu chuẩn ngưỡng; không ghi lại checkpoint, scaler, CSV/JSON kết quả.

## Các kiểm tra có sẵn bị chặn bởi môi trường

Đã gọi lại:

```powershell
python demo/check_product.py
python -m unittest src.test_evaluate_v2_heldout -v
python -m unittest src.test_dnn_dataset_v2 -v
python -m unittest src.test_dnn_model_v2 -v
```

- `check_product.py`: tất cả nhóm file đạt (4/4 script, 11/11 CSV, 15/15 hình, 2/2 audit, 5/5 model), nhưng **FAIL phần import** vì Python Windows thiếu `matplotlib` (3/4 thư viện).
- Ba module test nghiên cứu đều dừng khi import vì không có module Unix `resource` trên Windows; test body chưa chạy. Không thay mã nghiên cứu hoặc giả lập `resource` để che lỗi này.
- `test_dnn_model_v2` có các test thực hiện backward/optimizer step và smoke training. Các test này không được thực thi trong tác vụ chỉ sửa UI, phù hợp yêu cầu không train lại.
- Các trang cũ còn cảnh báo deprecation `use_container_width`; không thay ngoài phạm vi trang V2.

## Kiểm tra trực quan và giới hạn còn lại

**Chưa kiểm tra trực quan ở 1366×768 hoặc 1920×1080; chưa có ảnh chụp UI.** Computer Use dừng trước khi mở dashboard vì không xác định được URL trình duyệt Windows đủ tin cậy để thực thi chính sách. Không tiếp tục thao tác hoặc dùng cách khác để vượt qua chặn này.

CSS và chiều cao thành phần được thiết kế để giảm cuộn (chart chính 188/230 px, bảng sự kiện 250 px, một ảnh tối đa 310 px hoặc 260 px ở màn hình thấp, vùng tài liệu mở rộng tối đa 360 px có cuộn riêng). Đây là cấu hình bố cục, **không phải bằng chứng rằng mọi tab đã vừa chính xác một màn hình**. Cần đối chiếu thực tế hai độ phân giải, zoom 100%, gồm label trục chart, thanh tab, keyboard focus và thao tác mở/đóng expander trước khi duyệt giao diện. AppTest xác nhận cấu hình expander đóng và các điều khiển con; không thay thế thao tác mở/đóng thực tế trong trình duyệt.

## Tinh chỉnh diện mạo

- Header có phân cấp tên thí nghiệm và phạm vi held-out, thanh tab chia đều với trạng thái chọn tương phản rõ.
- Card KPI có biểu tượng SVG nét mảnh, số DNN chính và đối chiếu GRU ở hàng phụ; ngưỡng và diễn giải gói trong một panel bên dưới.
- Hồ sơ sự kiện có tiêu đề ID, trạng thái và Detection Delay nổi bật. Card seed dùng ba cột metric gọn thay cho danh sách cao.
- Màu navy/sage, viền nhẹ, không gradient; các quy tắc vẫn chỉ áp dụng trong trang V2.

## Sửa theo ảnh phản hồi: nội dung sát nhau, chữ khó đọc

- Gom bốn KPI và panel diễn giải trong một HTML grid có `row-gap: 18px`; không phụ thuộc khoảng cách giữa các block Streamlit. Khoảng cách ngang giữa card là 16px.
- Radio đặt `width="stretch"`; các selector `label` chỉ áp dụng bên trong `radiogroup`, tránh ghi đè nhãn widget đang collapsed. Chữ của mục chọn được đặt trực tiếp lên phần tử văn bản để tránh thừa kế màu trắng trên nền sáng.
- Tăng chữ nội dung lên 14px, chữ phụ 13px và nâng độ sáng các màu phụ. Header dùng `role="heading"` để giữ ngữ nghĩa mà tránh xử lý lại heading HTML trong Markdown.
- Đối chiếu tỷ lệ tương phản của các cặp màu được cấu hình: tab chọn **9.77:1**, tab chưa chọn **13.79:1**, nhãn KPI **10.71:1**, chữ phụ **8.82:1**, số KPI **13.25:1**. Đây là kiểm tra số học trên mã màu CSS; chưa phải đo CSS sau cascade trong trình duyệt.
- Kiểm thử UI 7/7 vẫn PASS; 142 tệp được bảo vệ không thay đổi. Chưa có kiểm tra trực quan sau lần sửa này.
