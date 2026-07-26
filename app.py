#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VieNeu-TTS Demo — Ứng dụng tạo giọng nói AI
=============================================
Ứng dụng Gradio với giao diện premium dark-theme, hỗ trợ:
  • Tab 1: Giới thiệu model + biểu đồ loss
  • Tab 2: Nhập văn bản → sinh audio → phát lại + mở thư mục

Chạy: python app.py
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import platform
import subprocess
import datetime
import traceback

import gradio as gr
import numpy as np
import soundfile as sf

from config import (
    MERGED_MODEL_DIR_ABS,
    LOSS_CHART_IMAGE_ABS,
    REF_AUDIO_PATH_ABS,
    OUTPUT_DIR_ABS,
    MODEL_INFO,
)

import re

_VI_DIGITS = (
    "không",
    "một",
    "hai",
    "ba",
    "bốn",
    "năm",
    "sáu",
    "bảy",
    "tám",
    "chín",
)


def _read_vi_triplet(value: int, force_hundreds: bool = False) -> str:
    """Đọc một nhóm số từ 0 đến 999 bằng tiếng Việt."""
    hundreds, remainder = divmod(value, 100)
    tens, units = divmod(remainder, 10)
    words = []

    if hundreds or force_hundreds:
        words.extend((_VI_DIGITS[hundreds], "trăm"))
        if tens == 0 and units:
            words.append("lẻ")

    if tens >= 2:
        words.extend((_VI_DIGITS[tens], "mươi"))
        if units == 1:
            words.append("mốt")
        elif units == 4:
            words.append("tư")
        elif units == 5:
            words.append("lăm")
        elif units:
            words.append(_VI_DIGITS[units])
    elif tens == 1:
        words.append("mười")
        if units == 5:
            words.append("lăm")
        elif units:
            words.append(_VI_DIGITS[units])
    elif units and not (hundreds or force_hundreds):
        words.append(_VI_DIGITS[units])
    elif units:
        words.append(_VI_DIGITS[units])

    return " ".join(words)


def integer_to_vietnamese(value: int) -> str:
    """Đọc số nguyên không âm bằng tiếng Việt, hỗ trợ đến hàng tỷ tỷ."""
    if value == 0:
        return _VI_DIGITS[0]

    scales = ("", "nghìn", "triệu", "tỷ", "nghìn tỷ", "triệu tỷ", "tỷ tỷ")
    groups = []
    while value:
        value, group = divmod(value, 1000)
        groups.append(group)

    parts = []
    highest = len(groups) - 1
    for index in range(highest, -1, -1):
        group = groups[index]
        if not group:
            continue
        force_hundreds = index < highest and group < 100
        spoken = _read_vi_triplet(group, force_hundreds=force_hundreds)
        scale = scales[index] if index < len(scales) else ""
        parts.append(f"{spoken} {scale}".strip())
    return " ".join(parts)


def normalize_numbers_for_tts(text: str) -> str:
    """
    Chuẩn hóa số trước khi lọc ký tự.

    Dấu `,` hoặc `.` chỉ là dấu thập phân khi nằm sát giữa hai chữ số:
    `2,3`/`2.3` -> `hai phẩy ba`; `2, 3` vẫn là một danh sách.
    Phần thập phân được đọc từng chữ số để không làm mất số 0.
    """
    if not text:
        return text

    number_pattern = re.compile(r"(?<![\w])([+-]?)(\d+)(?:([.,])(\d+))?(?![\w])")

    def _replace(match: re.Match) -> str:
        sign, integer_part, decimal_mark, fractional_part = match.groups()
        words = integer_to_vietnamese(int(integer_part))
        if sign == "-":
            words = f"âm {words}"
        elif sign == "+":
            words = f"dương {words}"
        if decimal_mark:
            fraction = " ".join(_VI_DIGITS[int(digit)] for digit in fractional_part)
            words = f"{words} phẩy {fraction}"
        return words

    return number_pattern.sub(_replace, text)


TTS_PUNCTUATION = frozenset(".,!?\u2026;:-")


def ensure_trailing_punctuation(text: str) -> str:
    """Thêm một dấu chấm nếu text không kết thúc bằng dấu câu TTS hỗ trợ."""
    if not text:
        return text
    text = text.rstrip()
    if text and text[-1] not in TTS_PUNCTUATION:
        text += "."
    return text


def normalize_annotations_for_tts(text: str) -> str:
    """
    Chuyển chú thích trong ngoặc thành một đơn vị ngữ điệu riêng.

    Ví dụ:
        "bản kỷ (chép sự tích các đế vương), còn phần"
        -> "bản kỷ, chép sự tích các đế vương. còn phần"
    """
    if not text:
        return text

    # Xử lý từ ngoặc trong cùng ra ngoài để không làm mất nội dung khi có ngoặc lồng.
    annotation_pattern = re.compile(r"\(\s*([^()]+?)\s*\)")
    previous = None
    while text != previous:
        previous = text
        text = annotation_pattern.sub(lambda match: f", {match.group(1).strip()}.", text)

    # Dấu câu ngay sau ngoặc cũ không được tạo thành chuỗi ".,", ".;"...
    text = re.sub(r"\.\s*[,;:]+\s*", ". ", text)
    text = re.sub(r",\s*,+", ", ", text)
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"\s+,", ",", text)
    return text


def normalize_text_for_tts(text: str) -> str:
    """Chuẩn hóa văn bản trước khi đưa vào TTS."""
    if not text:
        return text

    text = normalize_annotations_for_tts(text)

    # 1. expand_units
    replacements = {
        'km/h': ' ki-lô-mét trên giờ',
        'km': ' ki-lô-mét',
        'm/s': ' mét trên giây',
        'cm': ' xăng-ti-mét',
        'mm': ' mi-li-mét',
        'kg': ' ki-lô-gam',
        'mg': ' mi-li-gam',
        'ml': ' mi-li-lít',
        'm²': ' mét vuông',
        'm3': ' mét khối',
        'm³': ' mét khối',
        'đ': ' đồng',
        'vnđ': ' đồng',
        '°c': ' độ xê',
        '%': ' phần trăm',
        '$': ' đô la',
        'usd': ' đô la',
        'm': ' mét',
    }

    sorted_keys = sorted(replacements.keys(), key=len, reverse=True)

    for k in sorted_keys:
        escaped_k = re.escape(k)
        if k[-1].isalnum() and k[-1] not in ('³', '²'):
            pattern = r'(\d+(?:[.,]\d+)*)' + escaped_k + r'\b'
        else:
            pattern = r'(\d+(?:[.,]\d+)*)' + escaped_k

        text = re.sub(pattern, r'\g<1>' + replacements[k], text, flags=re.IGNORECASE)

    # 2. Chuẩn hóa số dự phòng trước khi lọc ký tự. Bước này luôn chạy để số
    # không bị mất khi môi trường không cài vinorm.
    text = normalize_numbers_for_tts(text)

    # 3. vinorm.TTSnorm
    try:
        from vinorm import TTSnorm
        text = TTSnorm(text)
    except ImportError:
        pass
    except Exception as e:
        print(f"[WARN] Lỗi khi chạy vinorm.TTSnorm: {e}")

    # 4. Lọc an toàn cuối — giữ lại dấu câu để model biết biên câu
    # (xóa số, ký hiệu đặc biệt nhưng GIỮ: . , ! ? … ; : -)
    text = re.sub(
        r'[^a-zA-Záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệ'
        r'íìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ'
        r'đÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌ'
        r'ÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ\s.,!?\u2026;:-]',
        '', text
    )
    # Dọn dấu câu bị lặp nhiều lần liên tiếp (vd "..." → "…" hoặc giữ nguyên "...")
    text = re.sub(r'([.,!?;:-])\1{2,}', r'\1\1\1', text)  # tối đa 3 dấu liên tiếp
    text = re.sub(r'\s+', ' ', text).strip()

    return ensure_trailing_punctuation(text)

# ============================================================================
# 1. GLOBAL STATE
# ============================================================================
tts_model = None
model_load_error = None
last_generated_file = None  # lưu đường dẫn file audio mới nhất trong phiên

# ============================================================================
# 1b. MULTI-SENTENCE HELPERS
# ============================================================================
# Khoảng lặng giữa các câu khi ghép audio (giây). Chỉnh tuỳ phong cách đọc:
#   0.10–0.15s → nói nhanh / podcast
#   0.18–0.25s → đọc truyện bình thường  (mặc định)
#   0.30–0.50s → nghe sách / chậm rãi
INTER_SENTENCE_SILENCE_S = 0.20
TTS_SAMPLE_RATE = 24000


def _edge_silence_seconds(wav: np.ndarray, sample_rate: int) -> tuple[float, float]:
    """Đo khoảng lặng liên tục ở đầu/cuối waveform mà không cắt audio."""
    audio = np.asarray(wav, dtype=np.float32).reshape(-1)
    if audio.size == 0 or sample_rate <= 0:
        return 0.0, 0.0

    finite_audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(finite_audio)))
    if peak == 0.0:
        duration = finite_audio.size / sample_rate
        return duration, duration

    # Ngưỡng tương đối có sàn nhỏ để bỏ noise nền nhưng không ăn vào âm cuối.
    threshold = max(1e-4, peak * 0.01)
    voiced_indices = np.flatnonzero(np.abs(finite_audio) > threshold)
    if voiced_indices.size == 0:
        duration = finite_audio.size / sample_rate
        return duration, duration

    leading = int(voiced_indices[0]) / sample_rate
    trailing = int(finite_audio.size - 1 - voiced_indices[-1]) / sample_rate
    return leading, trailing


def concatenate_with_adaptive_pauses(
    chunks: list[np.ndarray],
    sample_rate: int = TTS_SAMPLE_RATE,
    target_pause_s: float = INTER_SENTENCE_SILENCE_S,
) -> np.ndarray:
    """
    Ghép audio và chỉ bù silence còn thiếu tại mỗi biên.

    Silence sẵn có ở cuối chunk trước và đầu chunk sau được tính vào khoảng nghỉ
    mục tiêu. Không bao giờ append khoảng nghỉ liên câu sau chunk cuối.
    """
    if not chunks:
        return np.array([], dtype=np.float32)

    normalized_chunks = [
        np.nan_to_num(np.asarray(chunk, dtype=np.float32).reshape(-1))
        for chunk in chunks
    ]
    audio_parts = []
    for index, chunk in enumerate(normalized_chunks):
        audio_parts.append(chunk)
        if index == len(normalized_chunks) - 1:
            continue

        _, trailing_s = _edge_silence_seconds(chunk, sample_rate)
        leading_s, _ = _edge_silence_seconds(normalized_chunks[index + 1], sample_rate)
        existing_pause_s = trailing_s + leading_s
        padding_s = max(0.0, target_pause_s - existing_pause_s)
        padding_samples = int(round(sample_rate * padding_s))
        if padding_samples:
            audio_parts.append(np.zeros(padding_samples, dtype=np.float32))
        print(
            f"[INFO] Biên audio {index + 1}/{len(normalized_chunks) - 1}: "
            f"silence có sẵn={existing_pause_s:.3f}s, bù={padding_s:.3f}s"
        )

    return np.concatenate(audio_parts)


def split_sentences_vi(text: str) -> list:
    """
    Tách văn bản tiếng Việt thành danh sách câu đơn.
    Dùng lookbehind để giữ dấu câu kết thúc (. ! ? …) kèm với câu trước.
    Câu không kết thúc bằng dấu câu (phần còn lại) vẫn được giữ lại.

    Ví dụ:
        'Hôm nay trời đẹp. Bạn có khỏe không? Tôi khỏe!'
        → ['Hôm nay trời đẹp.', 'Bạn có khỏe không?', 'Tôi khỏe!']
    """
    if not text:
        return []
    # Tách tại vị trí SAU dấu .  !  ?  … (và khoảng trắng sau đó)
    parts = re.split(r'(?<=[.!?\u2026])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def _infer_one_sentence(text_norm: str, ref_audio: str, ref_text: str):
    """Gọi tts_model.infer() cho 1 câu, trả về numpy float32 array."""
    if ref_audio:
        wav = tts_model.infer(text=text_norm, ref_audio=ref_audio, ref_text=ref_text)
    else:
        wav = tts_model.infer(text=text_norm)
    if isinstance(wav, np.ndarray):
        return wav.astype(np.float32)
    if isinstance(wav, tuple) and len(wav) == 2:
        return wav[1].astype(np.float32)
    # fallback: đã là array-like
    return np.array(wav, dtype=np.float32)

# ============================================================================
# 2. MODEL LOADING (một lần duy nhất khi khởi động)
# ============================================================================
def load_model():
    """Load VieNeu-TTS model một lần duy nhất."""
    global tts_model, model_load_error

    if not os.path.isdir(MERGED_MODEL_DIR_ABS):
        model_load_error = (
            f"❌ Không tìm thấy thư mục model:\n"
            f"   {MERGED_MODEL_DIR_ABS}\n\n"
            f"Hãy chèn merged_model vào thư mục models/ và khởi động lại app."
        )
        print(f"[WARN] {model_load_error}")
        return

    try:
        from vieneu import Vieneu
    except ImportError as e:
        model_load_error = (
            f"❌ Không tìm thấy thư viện 'vieneu' hoặc thiếu dependencies.\n"
            f"Lỗi: {e}\n"
            "Chạy: pip install vieneu\n"
            "rồi khởi động lại app."
        )
        print(f"[WARN] {model_load_error}")
        return

    try:
        print(f"[INFO] Đang load model từ: {MERGED_MODEL_DIR_ABS}")
        tts_model = Vieneu(
            mode="standard",
            backbone_repo=MERGED_MODEL_DIR_ABS,
            gguf_filename=None,
            codec_repo="neuphonic/neucodec"
        )
        print("[INFO] ✅ Load model thành công!")
    except Exception as e:
        model_load_error = (
            f"❌ Lỗi khi load model:\n{str(e)}\n\n"
            f"Kiểm tra lại đường dẫn model trong config.py."
        )
        print(f"[ERROR] {model_load_error}")
        traceback.print_exc()


# ============================================================================
# 3. AUDIO GENERATION
# ============================================================================
def generate_audio(text, device_choice="CPU"):
    """
    Sinh audio từ văn bản.
    Tự động tách câu → infer từng câu riêng → ghép với khoảng lặng tự nhiên.
    Điều này tránh hiện tượng lặp từ và silence gap khi nhập nhiều câu.
    Returns: (audio_path | None, status_message, open_btn_interactive)
    """
    global last_generated_file

    # --- Switch Device ---
    try:
        import torch
        if tts_model is not None and hasattr(tts_model, "backbone") and hasattr(tts_model.backbone, "to"):
            if device_choice == "GPU" and torch.cuda.is_available():
                tts_model.backbone.to("cuda")
            else:
                tts_model.backbone.to("cpu")
    except Exception as e:
        print(f"[WARN] Lỗi khi đổi device: {e}")

    # --- Validate cơ bản ---
    if not text or not text.strip():
        return None, "⚠️ Vui lòng nhập văn bản.", gr.update(interactive=False)

    # Chuẩn hóa chữ thường và biên chú thích trước khi chia câu/đưa vào model.
    # casefold() xử lý Unicode ổn định và vẫn giữ nguyên dấu tiếng Việt/dấu câu.
    text = ensure_trailing_punctuation(
        normalize_annotations_for_tts(text.casefold())
    )

    if tts_model is None:
        return (
            None,
            model_load_error or "❌ Model chưa được load.",
            gr.update(interactive=False),
        )

    # --- Chuẩn bị ref audio / ref text ---
    ref_audio = REF_AUDIO_PATH_ABS if os.path.isfile(REF_AUDIO_PATH_ABS) else None
    ref_text = ""
    if ref_audio:
        ref_text_path = os.path.splitext(REF_AUDIO_PATH_ABS)[0] + ".txt"
        if os.path.isfile(ref_text_path):
            with open(ref_text_path, "r", encoding="utf-8") as f:
                ref_text = f.read().strip()
        if not ref_text:
            return (
                None,
                f"⚠️ Lỗi: Có file âm thanh mẫu nhưng chưa có văn bản. "
                f"Hãy tạo file '{os.path.basename(ref_text_path)}' "
                f"(cùng chỗ với file âm thanh) và ghi nội dung vào đó.",
                gr.update(interactive=True),
            )

    # --- Tách câu, normalize từng câu, infer từng câu ---
    sentences = split_sentences_vi(text)
    if not sentences:
        sentences = [text]  # fallback: 1 câu = toàn bộ text

    audio_chunks = []
    ok_count = 0
    err_msgs = []

    for i, sent in enumerate(sentences):
        sent_norm = normalize_text_for_tts(sent)
        if not sent_norm:
            print(f"[INFO] Bỏ qua câu {i+1} (rỗng sau chuẩn hóa): '{sent[:60]}'")
            continue
        try:
            print(f"[INFO] Đang sinh câu {i+1}/{len(sentences)}: '{sent_norm[:80]}'")
            wav_arr = _infer_one_sentence(sent_norm, ref_audio, ref_text)
            audio_chunks.append(wav_arr)
            ok_count += 1
        except Exception as e:
            err_msgs.append(f"Câu {i+1}: {str(e)[:120]}")
            print(f"[WARN] Lỗi sinh câu {i+1}: {e}")
            traceback.print_exc()
            continue

    if not audio_chunks:
        detail = "; ".join(err_msgs) if err_msgs else "Không rõ nguyên nhân."
        return (
            None,
            f"❌ Không sinh được audio nào.\nChi tiết: {detail}",
            gr.update(interactive=False),
        )

    # --- Ghép audio và lưu file ---
    try:
        full_audio = concatenate_with_adaptive_pauses(audio_chunks)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tts_{timestamp}.wav"
        filepath = os.path.join(OUTPUT_DIR_ABS, filename)
        sf.write(filepath, full_audio, samplerate=TTS_SAMPLE_RATE)
        last_generated_file = filepath

        status = f"✅ Đã tạo thành công: {filename} ({ok_count} câu)"
        if err_msgs:
            status += f"\n⚠️ {len(err_msgs)} câu bị lỗi: {'; '.join(err_msgs[:3])}"
        return (filepath, status, gr.update(interactive=True))

    except Exception as e:
        traceback.print_exc()
        return (
            None,
            f"❌ Lỗi khi ghép/lưu audio:\n{str(e)}",
            gr.update(interactive=False),
        )


# ============================================================================
# 4. FILE EXPLORER
# ============================================================================
def open_file_explorer():
    """Mở File Explorer tại thư mục chứa file audio vừa tạo."""
    global last_generated_file

    if not last_generated_file or not os.path.isfile(last_generated_file):
        return f"📂 File nằm tại: {OUTPUT_DIR_ABS}"

    try:
        system = platform.system()
        if system == "Windows":
            subprocess.run(
                ["explorer", "/select,", os.path.normpath(last_generated_file)],
                check=False,
            )
        elif system == "Darwin":
            subprocess.run(["open", "-R", last_generated_file], check=False)
        else:
            folder = os.path.dirname(last_generated_file)
            subprocess.run(["xdg-open", folder], check=False)
        return f"✅ Đã mở thư mục chứa file."
    except Exception as e:
        return (
            f"⚠️ Không thể mở File Explorer trong môi trường này.\n"
            f"File của bạn nằm tại: {last_generated_file}"
        )


# ============================================================================
# 5. CUSTOM CSS — Premium Dark Theme
# ============================================================================
CUSTOM_CSS = """
/* ═══════════════════════════════════════════════════════════════════════════
   VieNeu-TTS Premium Dark Theme
   ═══════════════════════════════════════════════════════════════════════════ */

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* --- Root Variables --- */
:root {
    --bg-primary: #0a0a0f;
    --bg-secondary: #12121a;
    --bg-card: rgba(18, 18, 30, 0.85);
    --bg-card-hover: rgba(25, 25, 40, 0.95);
    --border-glow: rgba(139, 92, 246, 0.3);
    --border-default: rgba(255, 255, 255, 0.06);
    --text-primary: #f0f0f5;
    --text-secondary: #9ca3af;
    --text-muted: #6b7280;
    --accent-purple: #8b5cf6;
    --accent-cyan: #06b6d4;
    --accent-pink: #ec4899;
    --accent-green: #10b981;
    --gradient-hero: linear-gradient(135deg, #8b5cf6 0%, #06b6d4 50%, #ec4899 100%);
    --gradient-button: linear-gradient(135deg, #8b5cf6 0%, #6d28d9 100%);
    --gradient-button-hover: linear-gradient(135deg, #a78bfa 0%, #7c3aed 100%);
    --shadow-glow: 0 0 30px rgba(139, 92, 246, 0.15), 0 0 60px rgba(6, 182, 212, 0.08);
    --shadow-card: 0 4px 24px rgba(0, 0, 0, 0.4), 0 0 1px rgba(139, 92, 246, 0.2);
    --radius-lg: 16px;
    --radius-md: 12px;
    --radius-sm: 8px;
    --transition-smooth: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

/* --- Global --- */
.gradio-container {
    background: var(--bg-primary) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    max-width: 960px !important;
    margin: 0 auto !important;
}

.dark {
    --background-fill-primary: var(--bg-primary) !important;
    --background-fill-secondary: var(--bg-secondary) !important;
}

/* --- Animated Background --- */
.gradio-container::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background:
        radial-gradient(ellipse at 20% 50%, rgba(139, 92, 246, 0.08) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 20%, rgba(6, 182, 212, 0.06) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 80%, rgba(236, 72, 153, 0.05) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
    animation: bgPulse 8s ease-in-out infinite alternate;
}

@keyframes bgPulse {
    0% { opacity: 0.6; }
    100% { opacity: 1; }
}

/* --- Hero Header --- */
.hero-section {
    text-align: center;
    padding: 48px 24px 32px;
    position: relative;
    z-index: 1;
}

.hero-icon {
    font-size: 56px;
    margin-bottom: 16px;
    filter: drop-shadow(0 0 20px rgba(139, 92, 246, 0.5));
    animation: floatIcon 3s ease-in-out infinite;
}

@keyframes floatIcon {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-8px); }
}

.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: var(--gradient-hero);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 12px 0;
    line-height: 1.2;
    letter-spacing: -0.03em;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: var(--text-secondary);
    line-height: 1.6;
    max-width: 640px;
    margin: 0 auto;
    font-weight: 400;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 16px;
    border-radius: 100px;
    background: rgba(139, 92, 246, 0.12);
    border: 1px solid rgba(139, 92, 246, 0.25);
    color: var(--accent-purple);
    font-size: 0.8rem;
    font-weight: 600;
    margin-bottom: 20px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* --- Tabs --- */
.tabs {
    position: relative;
    z-index: 1;
}

.tab-nav {
    border-bottom: 1px solid var(--border-default) !important;
    margin-bottom: 8px !important;
}

.tab-nav button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 12px 24px !important;
    color: var(--text-muted) !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
    transition: var(--transition-smooth) !important;
    letter-spacing: 0.01em !important;
}

.tab-nav button:hover {
    color: var(--text-primary) !important;
    background: rgba(139, 92, 246, 0.05) !important;
}

.tab-nav button.selected {
    color: var(--accent-purple) !important;
    border-bottom-color: var(--accent-purple) !important;
    background: rgba(139, 92, 246, 0.08) !important;
}

/* --- Cards / Panels --- */
.info-card {
    background: var(--bg-card);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-lg);
    padding: 28px;
    margin: 16px 0;
    backdrop-filter: blur(12px);
    box-shadow: var(--shadow-card);
    transition: var(--transition-smooth);
}

.info-card:hover {
    border-color: var(--border-glow);
    box-shadow: var(--shadow-glow);
    background: var(--bg-card-hover);
}

.card-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0 0 16px 0;
    display: flex;
    align-items: center;
    gap: 10px;
}

.card-title-icon {
    font-size: 1.3rem;
}

/* --- Training Info Table --- */
.training-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    border-radius: var(--radius-md);
    overflow: hidden;
    border: 1px solid var(--border-default);
}

.training-table th,
.training-table td {
    padding: 14px 20px;
    text-align: left;
    font-size: 0.9rem;
    border-bottom: 1px solid var(--border-default);
}

.training-table th {
    background: rgba(139, 92, 246, 0.08);
    color: var(--accent-purple);
    font-weight: 600;
    width: 42%;
    white-space: nowrap;
}

.training-table td {
    color: var(--text-primary);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
}

.training-table tr:last-child th,
.training-table tr:last-child td {
    border-bottom: none;
}

.training-table tr:hover td,
.training-table tr:hover th {
    background: rgba(139, 92, 246, 0.04);
}

/* --- Buttons --- */
.generate-btn {
    background: var(--gradient-button) !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 14px 32px !important;
    cursor: pointer !important;
    transition: var(--transition-smooth) !important;
    box-shadow: 0 4px 16px rgba(139, 92, 246, 0.3) !important;
    letter-spacing: 0.02em !important;
    min-height: 52px !important;
}

.generate-btn:hover {
    background: var(--gradient-button-hover) !important;
    box-shadow: 0 6px 24px rgba(139, 92, 246, 0.45) !important;
    transform: translateY(-1px) !important;
}

.generate-btn:active {
    transform: translateY(0) !important;
}

.folder-btn {
    background: rgba(6, 182, 212, 0.1) !important;
    border: 1px solid rgba(6, 182, 212, 0.3) !important;
    border-radius: var(--radius-md) !important;
    color: var(--accent-cyan) !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 12px 24px !important;
    transition: var(--transition-smooth) !important;
    min-height: 48px !important;
}

.folder-btn:hover {
    background: rgba(6, 182, 212, 0.18) !important;
    border-color: rgba(6, 182, 212, 0.5) !important;
    box-shadow: 0 0 20px rgba(6, 182, 212, 0.15) !important;
}

.folder-btn:disabled {
    opacity: 0.35 !important;
    cursor: not-allowed !important;
}

/* --- Textbox / Input --- */
.input-textbox textarea {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-default) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 16px !important;
    transition: var(--transition-smooth) !important;
    line-height: 1.7 !important;
}

.input-textbox textarea:focus {
    border-color: var(--accent-purple) !important;
    box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.12) !important;
    outline: none !important;
}

.input-textbox textarea::placeholder {
    color: var(--text-muted) !important;
    font-style: italic !important;
}

/* --- Audio Player --- */
.audio-player {
    border-radius: var(--radius-md) !important;
    overflow: hidden;
}

.audio-player audio {
    width: 100%;
    border-radius: var(--radius-md);
}

/* --- Status Messages --- */
.status-box {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
}

/* --- Warning Box --- */
.warning-box {
    background: rgba(245, 158, 11, 0.08);
    border: 1px solid rgba(245, 158, 11, 0.25);
    border-radius: var(--radius-md);
    padding: 20px 24px;
    color: #fbbf24;
    font-size: 0.9rem;
    line-height: 1.6;
}

.warning-box .warning-icon {
    font-size: 1.3rem;
    margin-right: 8px;
}

/* --- No Loss Chart --- */
.no-chart-msg {
    color: var(--text-muted);
    font-size: 0.85rem;
    font-style: italic;
    padding: 16px;
    text-align: center;
}

/* --- Loss Chart Image --- */
.loss-chart-container img {
    border-radius: var(--radius-md);
    border: 1px solid var(--border-default);
}

/* --- Footer --- */
.app-footer {
    text-align: center;
    padding: 32px 16px;
    color: var(--text-muted);
    font-size: 0.8rem;
    border-top: 1px solid var(--border-default);
    margin-top: 40px;
}

.app-footer a {
    color: var(--accent-purple);
    text-decoration: none;
}

.app-footer a:hover {
    text-decoration: underline;
}

/* --- Responsive --- */
@media (max-width: 640px) {
    .hero-title {
        font-size: 1.6rem;
    }
    .hero-subtitle {
        font-size: 0.9rem;
    }
    .info-card {
        padding: 20px;
    }
    .training-table th,
    .training-table td {
        padding: 10px 14px;
        font-size: 0.82rem;
    }
}

/* --- Misc Overrides --- */
.gradio-container .prose {
    color: var(--text-primary) !important;
}

.gradio-container .prose h1,
.gradio-container .prose h2,
.gradio-container .prose h3 {
    color: var(--text-primary) !important;
}

footer {
    display: none !important;
}

/* --- Shimmer animation for loading states --- */
@keyframes shimmer {
    0% { background-position: -200% center; }
    100% { background-position: 200% center; }
}

.loading-text {
    background: linear-gradient(90deg, var(--text-muted) 25%, var(--accent-purple) 50%, var(--text-muted) 75%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer 2s linear infinite;
}

/* --- Hidden buttons for UI state management --- */
.hidden-btn {
    display: none !important;
}
"""


# ============================================================================
# 6. BUILD UI
# ============================================================================
def build_model_info_table():
    """Tạo bảng HTML hiển thị thông số model."""
    rows = ""
    for key, value in MODEL_INFO.items():
        rows += f"<tr><th>{key}</th><td>{value}</td></tr>\n"
    return f'<table class="training-table">{rows}</table>'


def build_app():
    """Build giao diện Gradio."""
    has_gpu = False
    try:
        import torch
        has_gpu = torch.cuda.is_available()
    except Exception:
        pass

    # Kiểm tra loss chart
    has_loss_chart = (
        LOSS_CHART_IMAGE_ABS
        and os.path.isfile(LOSS_CHART_IMAGE_ABS)
    )

    model_name = MODEL_INFO.get("Tên model", "VieNeu-TTS Demo")
    model_ready = tts_model is not None

    with gr.Blocks(
        title=f"{model_name} — Demo",
    ) as demo:

        # ── Hero Section ──
        gr.HTML(f"""
        <div class="hero-section">
            <div class="hero-badge">🔬 AI Voice Synthesis</div>
            <div class="hero-icon">🎙️</div>
            <h1 class="hero-title">{model_name}</h1>
            <p class="hero-subtitle">
                Bản demo giọng nói AI được huấn luyện dựa trên VieNeu-TTS,
                sử dụng kỹ thuật LoRA fine-tuning để mô phỏng giọng nói cá nhân
                với chất lượng cao.
            </p>
        </div>
        """)

        # ── Tabs ──
        with gr.Tabs(elem_classes="tabs") as tabs:

            # ────────────────────────────────────────────────
            # TAB 1: GIỚI THIỆU
            # ────────────────────────────────────────────────
            with gr.TabItem("📋 Giới thiệu", id="tab-intro"):

                # Thông số huấn luyện
                gr.HTML(f"""
                <div class="info-card">
                    <div class="card-title">
                        <span class="card-title-icon">⚙️</span>
                        Thông số huấn luyện
                    </div>
                    {build_model_info_table()}
                </div>
                """)

                # Biểu đồ Loss
                if has_loss_chart:
                    gr.HTML("""
                    <div class="info-card">
                        <div class="card-title">
                            <span class="card-title-icon">📊</span>
                            Biểu đồ Loss trong quá trình huấn luyện
                        </div>
                    </div>
                    """)
                    gr.Image(
                        value=LOSS_CHART_IMAGE_ABS,
                        label="Training Loss",
                        show_label=False,
                        interactive=False,
                        elem_classes="loss-chart-container",
                    )
                else:
                    gr.HTML("""
                    <div class="info-card">
                        <div class="card-title">
                            <span class="card-title-icon">📊</span>
                            Biểu đồ Loss
                        </div>
                        <p class="no-chart-msg">
                            Chưa có biểu đồ loss cho model này.
                            Hãy chèn file ảnh vào <code>assets/loss_chart.png</code> để hiển thị.
                        </p>
                    </div>
                    """)

            # ────────────────────────────────────────────────
            # TAB 2: TẠO GIỌNG NÓI
            # ────────────────────────────────────────────────
            with gr.TabItem("🎤 Tạo giọng nói", id="tab-tts"):

                # Warning nếu model chưa load
                if not model_ready:
                    gr.HTML(f"""
                    <div class="warning-box">
                        <span class="warning-icon">⚠️</span>
                        {model_load_error or "Model chưa được load. Kiểm tra lại config.py."}
                    </div>
                    """)

                gr.HTML("""
                <div class="info-card" style="padding: 20px 28px;">
                    <div class="card-title">
                        <span class="card-title-icon">✨</span>
                        Nhập văn bản để tạo giọng nói AI
                    </div>
                </div>
                """)

                # Input
                text_input = gr.Textbox(
                    label="Văn bản",
                    placeholder="Nhập văn bản tiếng Việt cần chuyển thành giọng nói...",
                    lines=4,
                    max_lines=8,
                    elem_classes="input-textbox",
                    interactive=model_ready,
                )

                with gr.Row():
                    device_radio = gr.Radio(
                        choices=["GPU", "CPU"] if has_gpu else ["CPU"],
                        value="GPU" if has_gpu else "CPU",
                        label="Thiết bị sinh (Device)",
                        interactive=True,
                    )

                # Buttons row
                with gr.Row():
                    generate_btn = gr.Button(
                        "🚀 Tạo giọng nói",
                        variant="primary",
                        elem_classes="generate-btn",
                        interactive=model_ready,
                        scale=2,
                    )
                    open_folder_btn = gr.Button(
                        "📂 Mở thư mục chứa file",
                        elem_classes="folder-btn",
                        interactive=False,
                        scale=1,
                    )

                # Audio Player
                audio_output = gr.Audio(
                    label="Kết quả",
                    type="filepath",
                    autoplay=True,
                    elem_classes="audio-player",
                )

                # Status
                status_output = gr.Textbox(
                    label="Trạng thái",
                    interactive=False,
                    elem_classes="status-box",
                    visible=True,
                    value="Sẵn sàng." if model_ready else (model_load_error or ""),
                )

                # Folder status
                folder_status = gr.Textbox(
                    label="",
                    interactive=False,
                    visible=False,
                )

                # ── Trạng thái và Nút ẩn cho xử lý đồng thời ──
                is_generating = gr.State(False)
                hidden_gen_btn = gr.Button("Hidden Gen", visible=True, elem_id="hidden_gen_btn", elem_classes="hidden-btn")
                hidden_cancel_btn = gr.Button("Hidden Cancel", visible=True, elem_id="hidden_cancel_btn", elem_classes="hidden-btn")

                # Cập nhật trạng thái: Nếu đang generate mà user sửa text, mở lại nút
                text_input.change(
                    fn=lambda is_gen: gr.update(interactive=True) if is_gen else gr.update(),
                    inputs=[is_generating],
                    outputs=[generate_btn],
                )

                # Nút hiển thị
                generate_btn.click(
                    fn=None,
                    inputs=[is_generating],
                    js="""(is_gen) => {
                        function clickBtn(id) {
                            let el = document.querySelector('#' + id);
                            if (el && el.tagName !== 'BUTTON') el = el.querySelector('button') || el;
                            if (el) el.click();
                        }

                        if (is_gen) {
                            if (confirm('Bạn có muốn dừng tiến trình hiện tại để tạo giọng nói mới?')) {
                                clickBtn('hidden_cancel_btn');
                                setTimeout(() => {
                                    clickBtn('hidden_gen_btn');
                                }, 500);
                            }
                        } else {
                            clickBtn('hidden_gen_btn');
                        }
                    }"""
                )

                # Xử lý sinh âm thanh (chạy khi ẩn)
                set_state_evt = hidden_gen_btn.click(
                    fn=lambda: (True, gr.update(interactive=False)),
                    outputs=[is_generating, generate_btn],
                )

                gen_evt = set_state_evt.then(
                    fn=generate_audio,
                    inputs=[text_input, device_radio],
                    outputs=[audio_output, status_output, open_folder_btn],
                )

                gen_evt.then(
                    fn=lambda: (False, gr.update(interactive=True)),
                    outputs=[is_generating, generate_btn],
                )

                # Nút hủy
                hidden_cancel_btn.click(
                    fn=lambda: (False, gr.update(interactive=True), "⚠️ Tiến trình bị hủy để tạo mới."),
                    outputs=[is_generating, generate_btn, status_output],
                    cancels=[gen_evt],
                )

                # ── Sự kiện Mở thư mục ──
                open_folder_btn.click(
                    fn=open_file_explorer,
                    inputs=[],
                    outputs=[status_output],
                )

        # ── Footer ──
        gr.HTML("""
        <div class="app-footer">
            <p>
                Được xây dựng với ❤️ sử dụng
                <a href="https://gradio.app" target="_blank">Gradio</a> •
                Model dựa trên
                <a href="https://github.com/capleaf/viNeuTTS" target="_blank">VieNeu-TTS</a>
            </p>
            <p style="margin-top: 8px; font-size: 0.75rem;">
                ⚠️ Nút "Mở thư mục" chỉ hoạt động khi chạy app trên máy local.
            </p>
        </div>
        """)

    return demo


# ============================================================================
# 7. MAIN
# ============================================================================
def main():
    """Khởi chạy ứng dụng."""
    print("=" * 60)
    print("  VieNeu-TTS Demo — Starting...")
    print("=" * 60)

    # Tạo thư mục outputs nếu chưa có
    os.makedirs(OUTPUT_DIR_ABS, exist_ok=True)

    # Load model một lần
    load_model()

    # Build & launch
    demo = build_app()
    custom_theme = gr.themes.Base(
        primary_hue=gr.themes.colors.purple,
        secondary_hue=gr.themes.colors.cyan,
        neutral_hue=gr.themes.colors.gray,
        font=gr.themes.GoogleFont("Inter"),
        font_mono=gr.themes.GoogleFont("JetBrains Mono"),
    ).set(
        body_background_fill="*neutral_950",
        body_background_fill_dark="*neutral_950",
        block_background_fill="*neutral_900",
        block_background_fill_dark="*neutral_900",
        input_background_fill="*neutral_900",
        input_background_fill_dark="*neutral_900",
    )

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        css=CUSTOM_CSS,
        theme=custom_theme,
    )


if __name__ == "__main__":
    main()
