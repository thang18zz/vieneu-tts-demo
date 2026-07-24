# Đặc tả ứng dụng: Demo giọng nói AI (VieNeu-TTS Fine-tuned)

## 1. Mục tiêu ứng dụng

Xây dựng một ứng dụng web chạy **local** (trên máy của người dùng) bằng Python + **Gradio**, có 2 tab:

- **Tab 1 — "Giới thiệu"**: hiển thị thông tin về model đã fine-tune (thông số huấn luyện) và biểu đồ loss (nếu có).
- **Tab 2 — "Tạo giọng nói"**: nhập văn bản, bấm nút **"Tạo"** để sinh file audio bằng `merged_model`, phát lại ngay trong app, và có nút mở File Explorer tới thư mục chứa file vừa tạo.

Chọn **Gradio** vì: chạy local dễ dàng bằng `python app.py`, hỗ trợ audio player, upload/hiển thị ảnh, và cho phép gọi các hàm Python thông thường (kể cả `subprocess` để mở File Explorer) khi chạy trên máy cá nhân.

> ⚠️ Lưu ý quan trọng: nút "Mở File Explorer" **chỉ hoạt động khi chạy app trên máy local** (Windows/macOS/Linux của bạn). Nếu sau này deploy lên server/cloud (Hugging Face Spaces, Colab, v.v.) thì tính năng này sẽ không hoạt động vì ứng dụng không có quyền điều khiển máy tính người dùng từ xa.

---

## 2. Cấu trúc thư mục đề xuất

```
vieneu_tts_app/
├── app.py                     # File chính chạy ứng dụng
├── config.py                  # Nơi chèn đường dẫn model, ảnh loss, thông số
├── assets/
│   └── loss_chart.png         # ⚠️ CHÈN ảnh biểu đồ loss vào đây (có thể để trống)
├── models/
│   └── merged_model/          # ⚠️ CHÈN thư mục merged_model (từ Bước 8 notebook) vào đây
├── reference/
│   └── ref_audio.wav          # File audio mẫu dùng làm giọng tham chiếu khi sinh audio
└── outputs/                   # Nơi lưu các file audio được sinh ra (tự tạo nếu chưa có)
```

---

## 3. File cấu hình (`config.py`) — nơi chèn thông tin của bạn

Đây là nơi duy nhất bạn cần chỉnh sửa để gắn model, ảnh, và thông số vào app:

```python
# config.py

# --- Đường dẫn model & tài nguyên (⚠️ CHÈN Ở ĐÂY) ---
MERGED_MODEL_DIR = "models/merged_model"      # thư mục merged_model đã tạo ở Bước 8
LOSS_CHART_IMAGE = "assets/loss_chart.png"    # ảnh biểu đồ loss, để None hoặc "" nếu chưa có
REF_AUDIO_PATH = "reference/ref_audio.wav"    # audio mẫu để clone giọng
REF_TEXT = "Nội dung chính xác của file audio mẫu ở trên"  # text khớp 100% với ref audio

OUTPUT_DIR = "outputs"                        # thư mục lưu audio sinh ra

# --- Thông số hiển thị ở Tab "Giới thiệu" (⚠️ CHÈN/CHỈNH lại cho đúng) ---
MODEL_INFO = {
    "Tên model": "My Voice - VieNeu-TTS Fine-tuned",
    "Model gốc (base model)": "pnnbao-ump/VieNeu-TTS-0.3B",
    "Phương pháp fine-tune": "LoRA",
    "Số bước huấn luyện (max_steps)": "5000",
    "Learning rate": "2e-4",
    "Tổng thời lượng dataset": "~2.5 giờ",         # điền số thực tế từ Bước 2 notebook
    "Số lượng mẫu audio": "1200",                   # điền số thực tế
    "Phần cứng huấn luyện": "GPU T4 x2 (Kaggle)",
    "Ngày huấn luyện": "24/07/2026",
    "Ghi chú": "Model được fine-tune để mô phỏng giọng nói cá nhân.",
}
```

---

## 4. Chi tiết Tab 1 — "Giới thiệu"

### Bố cục
1. **Tiêu đề lớn**: tên app / tên model (lấy từ `MODEL_INFO["Tên model"]`).
2. **Đoạn mô tả ngắn** (2–3 câu) giới thiệu app: đây là bản demo giọng nói AI được huấn luyện dựa trên VieNeu-TTS, sử dụng kỹ thuật LoRA fine-tuning.
3. **Bảng thông số huấn luyện**: render toàn bộ dict `MODEL_INFO` thành bảng 2 cột (Markdown table hoặc `gr.Dataframe`).
4. **Biểu đồ loss**:
   - Kiểm tra `os.path.exists(LOSS_CHART_IMAGE)` khi khởi động app.
   - Nếu **tồn tại** → hiển thị bằng `gr.Image(value=LOSS_CHART_IMAGE, label="Biểu đồ Loss trong quá trình huấn luyện")`.
   - Nếu **không tồn tại** → **ẩn hẳn component** (`gr.Image(visible=False)`), **không** hiển thị lỗi hay khung ảnh trống. Có thể thay bằng dòng chữ nhỏ màu xám: *"Chưa có biểu đồ loss cho model này."*

### Hành vi
- Tab này hoàn toàn tĩnh (static) — không cần load model, hiển thị ngay khi mở app, không phụ thuộc vào Tab 2.

---

## 5. Chi tiết Tab 2 — "Tạo giọng nói"

### Thành phần giao diện
| Component | Loại | Mô tả |
|---|---|---|
| Ô nhập văn bản | `gr.Textbox` (multiline, lines=4) | Placeholder: "Nhập văn bản tiếng Việt cần chuyển thành giọng nói..." |
| Nút "Tạo" | `gr.Button("Tạo", variant="primary")` | Kích hoạt sinh audio |
| Trình phát audio | `gr.Audio(type="filepath", autoplay=True)` | Hiển thị & tự phát file vừa tạo |
| Nút "Mở thư mục chứa file" | `gr.Button("📂 Mở thư mục chứa file")` | Mở File Explorer/Finder tại `OUTPUT_DIR`, trỏ tới file vừa tạo. **Disable** cho tới khi có ít nhất 1 file được tạo trong phiên làm việc hiện tại. |
| Thông báo trạng thái | `gr.Textbox` hoặc `gr.Markdown` (ẩn mặc định) | Hiển thị lỗi nếu có (text rỗng, model lỗi khi sinh...) |

### Luồng xử lý khi bấm "Tạo"
1. Validate: nếu ô văn bản rỗng → hiện thông báo lỗi nhẹ ("Vui lòng nhập văn bản"), không gọi model.
2. Disable nút "Tạo" + hiện trạng thái "Đang tạo audio..." (Gradio tự xử lý loading state qua `gr.Button` khi hàm đang chạy).
3. Gọi model (đã được **load sẵn một lần lúc khởi động app**, không load lại mỗi lần bấm — quan trọng để tránh chậm):
   - Input: văn bản người dùng nhập + `REF_AUDIO_PATH` + `REF_TEXT` (giọng tham chiếu).
   - Output: mảng audio (numpy) hoặc file wav.
4. Lưu file vào `OUTPUT_DIR` với tên có timestamp, ví dụ: `outputs/tts_20260724_153045.wav` (tránh ghi đè file cũ).
5. Cập nhật:
   - `gr.Audio` → hiển thị & tự động phát file vừa tạo.
   - Nút "Mở thư mục" → chuyển sang trạng thái **enabled**, lưu lại đường dẫn file/thư mục vừa tạo vào biến trạng thái (session state) để dùng khi bấm nút này.
6. Nếu có lỗi trong quá trình sinh audio (model lỗi, hết VRAM, v.v.) → bắt exception, hiển thị thông báo lỗi rõ ràng, không làm crash app.

### Hành vi nút "Mở thư mục chứa file"
Khi bấm, mở File Explorer hệ điều hành tại thư mục chứa file audio vừa tạo (không cần mở đúng file, chỉ cần mở thư mục và tốt nhất là focus/highlight vào file đó nếu hệ điều hành hỗ trợ):

- **Windows**: `subprocess.run(["explorer", "/select,", file_path])`
- **macOS**: `subprocess.run(["open", "-R", file_path])`
- **Linux**: hầu hết trình quản lý file không hỗ trợ "select file", nên dùng `subprocess.run(["xdg-open", folder_path])` để mở thư mục.

Cần try/except quanh lệnh này — nếu hệ điều hành không hỗ trợ hoặc không có GUI (ví dụ chạy trong môi trường server/container không có desktop), hiển thị thông báo: *"Không thể mở File Explorer trong môi trường này. File của bạn nằm tại: `<đường dẫn>`"*.

---

## 6. Luồng khởi động ứng dụng (khi chạy `python app.py`)

1. Đọc `config.py`.
2. Load model từ `MERGED_MODEL_DIR` **một lần duy nhất** vào biến global (tránh load lại mỗi lần người dùng bấm "Tạo" — việc này chậm và tốn tài nguyên).
3. Tạo thư mục `OUTPUT_DIR` nếu chưa tồn tại.
4. Kiểm tra `LOSS_CHART_IMAGE` có tồn tại không, lưu kết quả để dùng khi build Tab 1.
5. Build giao diện Gradio với 2 tab như mô tả ở trên.
6. `demo.launch()` — mở app tại `http://127.0.0.1:7860` (mặc định).

---

## 7. Các trường hợp đặc biệt cần xử lý

- **Chưa load được model** (sai đường dẫn `MERGED_MODEL_DIR`, thiếu file): app vẫn khởi động được, Tab 1 hiển thị bình thường, nhưng Tab 2 hiển thị cảnh báo rõ ràng ("Không tìm thấy model, kiểm tra lại đường dẫn trong config.py") và disable nút "Tạo".
- **Chưa có ảnh loss**: ẩn hẳn component ảnh, không lỗi, không khung trống (đã nêu ở mục 4).
- **Văn bản quá dài**: có thể giới hạn độ dài input (ví dụ cảnh báo nếu > 500 ký tự) để tránh audio quá dài hoặc timeout, tuỳ khả năng của model.
- **Bấm "Tạo" nhiều lần liên tiếp**: disable nút trong lúc đang xử lý để tránh gọi model chồng chéo.
- **Chưa từng tạo audio nào**: nút "Mở thư mục chứa file" ở trạng thái disabled ngay từ đầu.

---

## 8. Yêu cầu triển khai

- Cài đặt: `pip install gradio` (và các thư viện cần thiết của VieNeu-TTS: `vieneu`, `torch`, `soundfile`, v.v. — dùng chung môi trường đã cài khi fine-tune).
- Chạy: `python app.py` trên máy có sẵn `merged_model` và (tuỳ chọn) file `ref_audio.wav` + `loss_chart.png` đã được chèn đúng theo cấu trúc thư mục ở mục 2.
- Không cần GPU mạnh để chạy inference (nhẹ hơn nhiều so với lúc train), nhưng có GPU sẽ sinh audio nhanh hơn.
