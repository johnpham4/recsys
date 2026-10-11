# 🚀 Hướng Dẫn Chạy Train Model Trên Modal GPU & Quản Lý Chi Phí

Tài liệu hướng dẫn từng bước chi tiết để chạy train model trên GPU đám mây của **Modal**, kiểm soát số dư free limit $30, và tải checkpoint về máy local.

---

## 📌 1. Các Link Tài Liệu Chính Thức Của Modal

- **Trang chủ & Đăng nhập**: [https://modal.com](https://modal.com)
- **Tạo & Quản lý Token**: [https://modal.com/settings/tokens](https://modal.com/settings/tokens)
- **Kiểm tra chi phí & Tiền còn lại (Usage)**: [https://modal.com/settings/usage](https://modal.com/settings/usage)
- **Cài đặt giới hạn chi tiêu (Spend Limits)**: [https://modal.com/settings/billing](https://modal.com/settings/billing)
- **Xem các app đang chạy (Dashboard)**: [https://modal.com/apps](https://modal.com/apps)
- **Tài liệu chính thức Modal**: [https://modal.com/docs](https://modal.com/docs)
- **Bảng giá GPU của Modal**: [https://modal.com/pricing](https://modal.com/pricing)

---

## 🔑 2. Cài Đặt Token Tài Khoản (Làm 1 lần)

Mở file `.env` ở thư mục gốc của project `d:\projects\recsys\backend\.env` và thêm vào 2 dòng token bạn của bạn đưa:

```env
MODAL_TOKEN_ID=ak-xxxxxxxxxxxxxxxxxxxxxxxxx
MODAL_TOKEN_SECRET=as-xxxxxxxxxxxxxxxxxxxxxxxxx
```

> **Mẹo**: Khi hết $30 ở tài khoản này và bạn muốn đổi sang tài khoản mới, chỉ cần thay 2 dòng này trong `.env` là xong ngay lập tức!

---

## 🔍 3. Kiểm Tra Kết Nối & Số Tiền Còn Lại

Trước khi chạy train, hãy chạy công cụ kiểm tra số dư ngay tại terminal:

```powershell
python scripts/modal_track.py
```

Kết quả sẽ hiển thị:
- Thông tin xác thực Token.
- Bảng chi phí đã dùng / hạn mức $30.
- Trạng thái các container đang chạy ngầm (nếu có app nào chạy quên tắt, bạn sẽ thấy ngay).

---

## 📤 4. Upload Dữ Liệu Lên Cloud Volume (Chỉ Chạy 1 Lần Duy Nhất)

Modal sử dụng **Volume** (ổ cứng đám mây lưu trữ lâu dài). Bạn chỉ cần upload các file dữ liệu (`train_features_neg_df.parquet`, `idm.json`, v.v.) lên 1 lần duy nhất:

```powershell
uvx modal run scripts/modal_runner.py::upload_data
```

*(Thời gian tải lên phụ thuộc vào tốc độ mạng, mất khoảng 1 - 3 phút cho file 360MB).*

---

## 🎯 5. Bấm Nút Train Trên GPU Đám Mây

Bạn có thể chọn một trong các cách chạy sau:

### Cách 1: Chạy file Jupyter Notebook `023-bert4rec.ipynb` bằng Papermill (Khuyên Dùng)
```powershell
uvx modal run scripts/modal_runner.py::run --cmd "papermill notebooks/023-bert4rec.ipynb output_bert.ipynb -p max_epochs 10"
```

### Cách 2: Chạy trực tiếp Script Train BERT4Rec chuyên dụng
```powershell
uvx modal run scripts/train_bert4rec_modal.py::train --epochs 10 --batch-size 128
```

### Cách 3: Chạy bất kỳ file/model nào khác trong tương lai
```powershell
# Chạy file pipeline train tổng hợp:
uvx modal run scripts/modal_runner.py::run --cmd "python notebooks/001-train-pipeline.py"

# Hoặc chạy script model bất kỳ:
uvx modal run scripts/modal_runner.py::run --cmd "python -m model.twotower.train"
```

---

## 📥 6. Tải Model Checkpoint Về Máy Local

Sau khi Modal train xong và báo `Training completed`, bạn tải file checkpoint từ Volume về thư mục `data/` ở máy:

```powershell
uvx modal run scripts/modal_runner.py::download_checkpoints
```

---

## 🛡️ 7. Lưu Ý Bảo Vệ An Toàn Cho Ví Tiền Của Bạn

1. **GPU T4 chỉ tốn ~$0.59 / giờ**: 1 lần train 10 - 15 phút chỉ tốn khoảng **$0.10** (~2.500 VNĐ). Với $30 credit, bạn có thể train hơn **50 lần** thoải mái!
2. **Khóa an toàn Timeout**: Mọi hàm trong `modal_runner.py` đều có `timeout=1800` (tối đa 30 phút tự tắt) $\to$ Dù có crash hay đơ mạng thì Modal tự ngắt điện, không bao giờ bị tính phí ngầm qua đêm.
3. **Mở giao diện Web bất cứ lúc nào**:
   ```powershell
   uvx modal dashboard
   ```
