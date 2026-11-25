# VA DRL — Module: Credits Calculation & Evaluation

Module tính điểm và đánh giá kết quả rèn luyện sinh viên thuộc hệ thống **Virtual Assistant Điểm Rèn Luyện (VA DRL)**, xây dựng theo **Quy chế 1028/QĐ-KHTN-CTSV** (Trường ĐH Khoa học Tự nhiên, ĐHQG-HCM).

## Cấu trúc thư mục
- `credits.py`: Core module `CreditCalculationModule` (ánh xạ hoạt động theo Phụ lục, kiểm tra mức tối đa Lớp 1 theo từng Mục, giới hạn Lớp 2 tổng 100 điểm và phân loại kết quả theo Điều 7).
- `server.py`: Web server nhẹ (dùng thư viện chuẩn Python) kết nối giao diện Virtual Assistant với `CreditCalculationModule`.
- `index.html`: Giao diện Web Demo **VA ĐRL** (Chat với Virtual Assistant + Dashboard hiển thị bảng điểm 6 mục và kết quả đánh giá).
- `run_web.bat`: Script chạy nhanh Web Server trên Windows.
- `drl2025.pdf`: Tài liệu Quy chế 1028/QĐ-KHTN-CTSV.

## Cách chạy

### 1. Chạy trực tiếp module trên Terminal
```bash
python credits.py
```

### 2. Chạy Web Demo Virtual Assistant ĐRL
Chạy file `run_web.bat` hoặc dùng lệnh:
```bash
python server.py
```
Sau đó mở trình duyệt tại: **http://127.0.0.1:8080**

