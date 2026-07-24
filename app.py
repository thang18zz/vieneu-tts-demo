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
    REF_TEXT,
    OUTPUT_DIR_ABS,
    MODEL_INFO,
    MAX_TEXT_LENGTH,
)

# ============================================================================
# 1. GLOBAL STATE
# ============================================================================
tts_model = None
model_load_error = None
last_generated_file = None  # lưu đường dẫn file audio mới nhất trong phiên

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
        from vieneu.tts import TTS
        print(f"[INFO] Đang load model từ: {MERGED_MODEL_DIR_ABS}")
        tts_model = TTS(model_path=MERGED_MODEL_DIR_ABS)
        print("[INFO] ✅ Load model thành công!")
    except ImportError:
        model_load_error = (
            "❌ Không tìm thấy thư viện 'vieneu'.\n"
            "Chạy: pip install vieneu\n"
            "rồi khởi động lại app."
        )
        print(f"[WARN] {model_load_error}")
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
def generate_audio(text):
    """
    Sinh audio từ văn bản.
    Returns: (audio_path | None, status_message, open_btn_interactive)
    """
    global last_generated_file

    # --- Validate ---
    if not text or not text.strip():
        return None, "⚠️ Vui lòng nhập văn bản.", gr.update(interactive=False)

    text = text.strip()

    if len(text) > MAX_TEXT_LENGTH:
        return (
            None,
            f"⚠️ Văn bản quá dài ({len(text)} ký tự). Giới hạn: {MAX_TEXT_LENGTH} ký tự.",
            gr.update(interactive=False),
        )

    if tts_model is None:
        return (
            None,
            model_load_error or "❌ Model chưa được load.",
            gr.update(interactive=False),
        )

    # --- Generate ---
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tts_{timestamp}.wav"
        filepath = os.path.join(OUTPUT_DIR_ABS, filename)

        # Kiểm tra ref audio
        ref_audio = REF_AUDIO_PATH_ABS if os.path.isfile(REF_AUDIO_PATH_ABS) else None

        if ref_audio:
            wav = tts_model.synthesize(
                text=text,
                ref_audio=ref_audio,
                ref_text=REF_TEXT,
            )
        else:
            wav = tts_model.synthesize(text=text)

        # Lưu file
        if isinstance(wav, np.ndarray):
            sf.write(filepath, wav, samplerate=24000)
        elif isinstance(wav, tuple) and len(wav) == 2:
            sr, audio_data = wav
            sf.write(filepath, audio_data, samplerate=sr)
        elif isinstance(wav, str) and os.path.isfile(wav):
            # Model trả về đường dẫn file
            import shutil
            shutil.copy2(wav, filepath)
        else:
            sf.write(filepath, wav, samplerate=24000)

        last_generated_file = filepath
        return (
            filepath,
            f"✅ Đã tạo thành công: {filename}",
            gr.update(interactive=True),
        )

    except Exception as e:
        traceback.print_exc()
        return (
            None,
            f"❌ Lỗi khi tạo audio:\n{str(e)}",
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
    # Kiểm tra loss chart
    has_loss_chart = (
        LOSS_CHART_IMAGE_ABS
        and os.path.isfile(LOSS_CHART_IMAGE_ABS)
    )

    model_name = MODEL_INFO.get("Tên model", "VieNeu-TTS Demo")
    model_ready = tts_model is not None

    with gr.Blocks(
        title=f"{model_name} — Demo",
        css=CUSTOM_CSS,
        theme=gr.themes.Base(
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
        ),
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

                # ── Events ──
                generate_btn.click(
                    fn=generate_audio,
                    inputs=[text_input],
                    outputs=[audio_output, status_output, open_folder_btn],
                )

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
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
    )


if __name__ == "__main__":
    main()
