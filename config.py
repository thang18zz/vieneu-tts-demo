# config.py
# ============================================================================
# VieNeu-TTS Demo — Cấu hình ứng dụng
# ============================================================================
# Đây là nơi DUY NHẤT bạn cần chỉnh sửa để gắn model, ảnh, và thông số.
# ============================================================================

import os

# --- Đường dẫn model & tài nguyên (⚠️ CHÈN Ở ĐÂY) ---
MERGED_MODEL_DIR = "models/merged_model"       # thư mục merged_model đã tạo ở Bước 8
LOSS_CHART_IMAGE = "assets/loss_chart.png"      # ảnh biểu đồ loss, để None hoặc "" nếu chưa có
REF_AUDIO_PATH = "reference/ref_audio.wav"      # audio mẫu để clone giọng
REF_TEXT = "Nội dung chính xác của file audio mẫu ở trên"  # text khớp 100% với ref audio

OUTPUT_DIR = "outputs"                          # thư mục lưu audio sinh ra

# --- Giới hạn input ---
MAX_TEXT_LENGTH = 500                           # số ký tự tối đa cho input

# --- Thông số hiển thị ở Tab "Giới thiệu" (⚠️ CHÈN/CHỈNH lại cho đúng) ---
MODEL_INFO = {
    "Tên model": "My Voice — VieNeu-TTS Fine-tuned",
    "Model gốc (base model)": "capleaf/viNeuTTS",
    "Phương pháp fine-tune": "LoRA",
    "Số bước huấn luyện (max_steps)": "5000",
    "Learning rate": "2e-4",
    "Tổng thời lượng dataset": "~2.5 giờ",
    "Số lượng mẫu audio": "1200",
    "Phần cứng huấn luyện": "GPU T4 x2 (Kaggle)",
    "Ngày huấn luyện": "24/07/2026",
    "Ghi chú": "Model được fine-tune để mô phỏng giọng nói cá nhân.",
}

# --- Đường dẫn tuyệt đối (tự tính toán, không cần sửa) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MERGED_MODEL_DIR_ABS = os.path.join(BASE_DIR, MERGED_MODEL_DIR)
LOSS_CHART_IMAGE_ABS = os.path.join(BASE_DIR, LOSS_CHART_IMAGE) if LOSS_CHART_IMAGE else None
REF_AUDIO_PATH_ABS = os.path.join(BASE_DIR, REF_AUDIO_PATH)
OUTPUT_DIR_ABS = os.path.join(BASE_DIR, OUTPUT_DIR)
