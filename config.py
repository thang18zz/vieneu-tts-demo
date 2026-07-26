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
REF_AUDIO_PATH = "reference/ref_audio.wav"      # audio mẫu (ứng dụng sẽ tự đọc file .txt cùng tên)

OUTPUT_DIR = "outputs"                          # thư mục lưu audio sinh ra

# --- Chunking an toàn ---
TTS_SAFE_CHUNK_TOKENS = 120
TTS_SAFE_CHUNK_WORDS = 30
MAX_LEADING_SILENCE_MS = 80
MAX_TRAILING_SILENCE_MS = 120
SILENCE_HANGOVER_MS = 40
SILENCE_FRAME_MS = 20
SILENCE_MIN_SPEECH_MS = 60
MAX_INTERNAL_SILENCE_MS = 900
TARGET_INTERNAL_SILENCE_MS = 250
REPETITION_SECONDS_PER_WORD_TRIGGER = 0.65
REPETITION_ASR_MODEL = "tiny"
REPETITION_MIN_NGRAM_WORDS = 3
REPETITION_MAX_NGRAM_WORDS = 30

# --- Thông số hiển thị ở Tab "Giới thiệu" (⚠️ CHÈN/CHỈNH lại cho đúng) ---
MODEL_INFO = {
    "Tên model": "My Voice — VieNeu-TTS Fine-tuned",
    "Model gốc (base model)": "capleaf/viNeuTTS",
    "Phương pháp fine-tune": "LoRA",
    "Số bước huấn luyện (max_steps)": "1500",
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
