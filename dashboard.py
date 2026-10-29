from __future__ import annotations

import base64
import csv
import io
import json
import tempfile
import time
from datetime import datetime
from pathlib import Path

import av
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image as PDFImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
from ultralytics import YOLO

# ============================================================
# APP CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Study Distraction Detector",
    page_icon="📱",
    layout="wide",
)

# ============================================================
# THEME
# ============================================================

st.markdown(
    """
    <style>
    :root {
        --bg-1: #0b0d12;
        --bg-2: #11141c;
        --panel: #171a24;
        --panel-2: #1b1f2b;
        --brand: #8b5cf6;
        --brand-2: #6d28d9;
        --text: #edf0f7;
        --muted: #aab2c5;
        --line: #2b3040;
        --green: #22c55e;
        --red: #ef4444;
        --yellow: #f59e0b;
    }

    .stApp {
        background: linear-gradient(180deg, var(--bg-1) 0%, var(--bg-2) 48%, var(--bg-1) 100%);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(11,13,18,0.92);
    }

    [data-testid="stSidebar"] {
        background: #11131b;
        border-right: 1px solid var(--line);
    }

    [data-testid="stSidebar"] * { color: var(--text); }
    [data-testid="stSidebar"] hr { border-color: var(--line); }
    [data-testid="stMarkdownContainer"], [data-testid="stText"], label, p, li { color: var(--text); }
    h1, h2, h3, h4 { color: #f7f8fc; letter-spacing: -0.025em; }
    [data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--line);
        padding: 16px 18px;
        border-radius: 15px;
        box-shadow: 0 5px 18px rgba(0,0,0,.22);
    }
    [data-testid="stMetricLabel"] { color: var(--muted); font-weight: 600; }
    [data-testid="stMetricValue"] { color: #f5f6fb; }

    div.stButton > button, div.stDownloadButton > button {
        border-radius: 10px;
        font-weight: 650;
        border: 1px solid #383e51;
        background: #202433;
        color: #f4f5fb;
        transition: transform .15s ease, box-shadow .15s ease;
    }
    div.stButton > button:hover, div.stDownloadButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 5px 14px rgba(139,92,246,.18);
        border-color: var(--brand);
        color: white;
    }

    [data-testid="stFileUploader"] {
        border: 1px dashed #555d75;
        border-radius: 14px;
        padding: 8px;
        background: rgba(25,28,39,.8);
    }

    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border-radius: 12px;
        overflow: hidden;
    }

    [data-baseweb="select"] > div, [data-baseweb="input"] > div, [data-baseweb="textarea"] > div {
        background: var(--panel-2);
        border-color: var(--line);
        color: var(--text);
    }
    input, textarea { color: var(--text) !important; }

    [data-testid="stAlert"] {
        background: #1b1f2b;
        border-color: var(--line);
    }

    .hero-panel {
        padding: 28px 30px;
        border-radius: 22px;
        background: linear-gradient(120deg, #21143f 0%, #4c1d95 62%, #7c3aed 125%);
        color: white;
        margin: 4px 0 22px 0;
        border: 1px solid #5937a1;
        box-shadow: 0 14px 30px rgba(124,58,237,.18);
    }
    .hero-panel h1 {
        color: white;
        font-size: clamp(1.75rem, 4vw, 2.55rem);
        margin: 0 0 8px 0;
        line-height: 1.15;
    }
    .hero-panel p {
        color: #e9ddff;
        margin: 0;
        font-size: 1rem;
        max-width: 850px;
    }
    .hero-chip {
        display: inline-block;
        border: 1px solid rgba(255,255,255,.35);
        background: rgba(255,255,255,.12);
        border-radius: 999px;
        padding: 5px 11px;
        margin-bottom: 13px;
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: .02em;
    }

    .section-note {
        color: var(--muted);
        font-size: .92rem;
        margin-top: -7px;
    }

    .info-card {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 15px;
        padding: 16px 18px;
        height: 100%;
        box-shadow: 0 4px 14px rgba(0,0,0,.18);
    }
    .info-card h3 {
        margin-top: 0;
        color: #c4b5fd;
        font-size: 1.02rem;
    }
    .info-card p {
        margin-bottom: 0;
        color: #c1c7d6;
        font-size: .92rem;
    }

    [data-testid="stProgressBar"] > div > div { background: var(--brand); }

    @media (max-width: 640px) {
        .hero-panel { padding: 22px 19px; border-radius: 16px; }
        [data-testid="stMetric"] { padding: 12px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# FILE LOCATIONS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
REPORT_DIR = BASE_DIR / "reports"
REPORT_FILE = REPORT_DIR / "session_summary.csv"

MODEL_NAME = "yolo11n.pt"
MODEL_FILE = BASE_DIR / MODEL_NAME

def ensure_model_exists() -> Path:
    candidate_paths = [
        MODEL_FILE,
        BASE_DIR.parent / MODEL_NAME,
        BASE_DIR.parent / "yolo11s.pt",
        BASE_DIR.parent / "yolo11n.pt",
    ]

    for candidate in candidate_paths:
        if candidate.exists():
            return candidate

    try:
        YOLO(MODEL_NAME)
        for candidate in candidate_paths:
            if candidate.exists():
                return candidate
    except Exception:
        pass

    raise FileNotFoundError(
        f"Could not find {MODEL_NAME}. Please install ultralytics and run: "
        "python -c \"from ultralytics import YOLO; YOLO('yolo11n.pt')\""
    )

REPORT_DIR.mkdir(parents=True, exist_ok=True)

CSV_COLUMNS = [
    "Session Time",
    "Video",
    "Resolution",
    "Frames Processed",
    "Frames With Phone",
    "Phone Visible Percentage",
    "Total Phone Detections",
]

NUMERIC_COLUMNS = [
    "Frames Processed",
    "Frames With Phone",
    "Phone Visible Percentage",
    "Total Phone Detections",
]

# ============================================================
# SESSION STATE
# ============================================================

if "study_timer_duration" not in st.session_state:
    st.session_state.study_timer_duration = 25
if "study_timer_running" not in st.session_state:
    st.session_state.study_timer_running = False
if "study_timer_started_at" not in st.session_state:
    st.session_state.study_timer_started_at = None
if "study_timer_elapsed_before_pause" not in st.session_state:
    st.session_state.study_timer_elapsed_before_pause = 0.0
if "study_timer_finished" not in st.session_state:
    st.session_state.study_timer_finished = False

if "live_phone_detected" not in st.session_state:
    st.session_state.live_phone_detected = False
if "last_live_warning_time" not in st.session_state:
    st.session_state.last_live_warning_time = 0.0
if "live_warning_triggered" not in st.session_state:
    st.session_state.live_warning_triggered = False

# ============================================================
# ALERT SOUND SETTINGS
# ============================================================

ALERT_COOLDOWN = 20
REQUIRED_CONSECUTIVE_DETECTIONS = 3

alert_choice = st.selectbox(
    "🔔 Choose Phone Alert Sound",
    [
        "1. Female Spoken Warning",
        "2. Gentle Warning Siren",
        "3. Deep Warning Pulse",
        "4. Emergency Warning Sweep",
        "5. Custom MP3/WAV File",
        "6. Soft Warning Bell",
        "7. Urgent Warning Tone",
        "8. Warning Chime",
        "9. Digital Warning Signal",
        "10. Safety Warning Voice",
    ],
    key="phone_alert_choice_main",
)

alert_volume = st.slider(
    "🔊 Alert Volume",
    min_value=0,
    max_value=100,
    value=75,
    step=5,
    help="Controls the warning volume.",
    key="phone_alert_volume_main",
)

volume_level = alert_volume / 100.0
custom_audio = None
custom_audio_type = None

if alert_choice == "5. Custom MP3/WAV File":
    uploaded_audio = st.file_uploader(
        "Upload your warning sound",
        type=["mp3", "wav"],
        key="custom_phone_alert_audio_main",
    )
    if uploaded_audio is not None:
        custom_audio = base64.b64encode(uploaded_audio.getvalue()).decode("utf-8")
        custom_audio_type = uploaded_audio.type or "audio/mpeg"

def play_female_warning(volume=0.8):
    safe_volume = max(0.0, min(1.0, float(volume)))
    message = "Attention! Attention! Phone detected! Please put your phone away and focus on studying!"
    js = f"""
    <script>
    try {{
        const speech = window.speechSynthesis;
        if (speech) {{
            speech.cancel();
            const utterance = new SpeechSynthesisUtterance("{message}");
            utterance.volume = {safe_volume};
            utterance.rate = 0.9;
            utterance.pitch = 1.15;

            const voices = speech.getVoices();
            const preferred = voices.find(v =>
                /female|zira|samantha|aria|jenny|hazel/i.test((v.name || "") + " " + (v.voiceURI || ""))
            );

            if (preferred) {{
                utterance.voice = preferred;
            }}

            speech.speak(utterance);
        }}
    }} catch (e) {{
        console.log("Female warning failed:", e);
    }}
    </script>
    """
    components.html(js, height=0)

def play_warning_tone(mode, volume=0.75):
    vol = max(0.0, min(1.0, float(volume)))
    configs = {
        "siren": {
            "pattern": [420, 520, 620, 760, 620, 520, 420],
            "durations": [0.40, 0.45, 0.50, 0.55, 0.50, 0.45, 0.60],
            "type": "sawtooth",
        },
        "pulse": {
            "pattern": [520, 560, 610, 660, 610, 560, 520, 450],
            "durations": [0.38, 0.35, 0.42, 0.45, 0.42, 0.35, 0.40, 0.55],
            "type": "triangle",
        },
        "sweep": {
            "pattern": [390, 470, 560, 700, 560, 470, 390],
            "durations": [0.45, 0.50, 0.55, 0.60, 0.55, 0.50, 0.65],
            "type": "sawtooth",
        },
        "bell": {
            "pattern": [880, 770, 660, 770, 660, 560],
            "durations": [0.28, 0.35, 0.42, 0.35, 0.42, 0.50],
            "type": "triangle",
        },
        "warning": {
            "pattern": [700, 760, 820, 760, 700, 680],
            "durations": [0.24, 0.28, 0.35, 0.30, 0.28, 0.42],
            "type": "sine",
        },
        "urgent": {
            "pattern": [950, 900, 980, 900, 980, 860, 980, 820],
            "durations": [0.18, 0.18, 0.20, 0.18, 0.20, 0.18, 0.25, 0.42],
            "type": "square",
        },
        "digital": {
            "pattern": [650, 720, 800, 860, 800, 720, 650],
            "durations": [0.18, 0.18, 0.22, 0.22, 0.22, 0.18, 0.28],
            "type": "triangle",
        },
        "voice": {
            "pattern": [430, 520, 610, 520, 610, 520, 430],
            "durations": [0.30, 0.35, 0.38, 0.35, 0.38, 0.35, 0.50],
            "type": "sine",
        },
    }

    cfg = configs.get(mode, configs["warning"])
    pattern = cfg["pattern"]
    durations = cfg["durations"]
    tone_type = cfg["type"]

    js = f"""
    <script>
    (function() {{
      try {{
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        if (!AudioCtx) return;

        const ctx = new AudioCtx();
        const now = ctx.currentTime;
        const vol = {vol};
        const pattern = {json.dumps(pattern)};
        const durations = {json.dumps(durations)};
        const toneType = "{tone_type}";

        pattern.forEach((freq, index) => {{
          const osc = ctx.createOscillator();
          const gain = ctx.createGain();

          osc.type = toneType;
          osc.frequency.setValueAtTime(freq, now + index * 0.22);

          const start = now + index * 0.22;
          const duration = durations[index] || 0.35;
          const peak = 0.08 + (index % 3) * 0.03;

          gain.gain.setValueAtTime(0.0001, start);
          gain.gain.linearRampToValueAtTime(vol * peak, start + 0.04);
          gain.gain.exponentialRampToValueAtTime(0.0001, start + duration);

          osc.connect(gain);
          gain.connect(ctx.destination);

          osc.start(start);
          osc.stop(start + duration + 0.04);
        }});

        setTimeout(() => {{
          try {{ ctx.close(); }} catch (e) {{}}
        }}, 7000);

      }} catch (e) {{
        console.log("Warning tone failed:", e);
      }}
    }})();
    </script>
    """
    components.html(js, height=0)

def play_custom_warning_audio():
    if custom_audio is None:
        st.info("Upload a custom MP3 or WAV file first.")
        return

    audio_html = f"""
    <audio controls autoplay style="width:100%;">
        <source src="data:{custom_audio_type};base64,{custom_audio}" type="{custom_audio_type}">
        Your browser does not support this audio file.
    </audio>
    <script>
    const player = document.querySelector("audio");
    player.volume = {volume_level};
    player.play().catch(() => {{
        console.log("Custom warning autoplay blocked by browser.");
    }});
    </script>
    """
    components.html(audio_html, height=90)

def show_phone_alert(preview=False):
    if not preview:
        st.warning("📱 Phone detected repeatedly! Please check whether you need your phone for studying.")

    if alert_choice == "1. Female Spoken Warning":
        play_female_warning(volume=volume_level)
    elif alert_choice == "2. Gentle Warning Siren":
        play_warning_tone("siren", volume=volume_level)
    elif alert_choice == "3. Deep Warning Pulse":
        play_warning_tone("pulse", volume=volume_level)
    elif alert_choice == "4. Emergency Warning Sweep":
        play_warning_tone("sweep", volume=volume_level)
    elif alert_choice == "5. Custom MP3/WAV File":
        play_custom_warning_audio()
    elif alert_choice == "6. Soft Warning Bell":
        play_warning_tone("bell", volume=volume_level)
    elif alert_choice == "7. Urgent Warning Tone":
        play_warning_tone("urgent", volume=volume_level)
    elif alert_choice == "8. Warning Chime":
        play_warning_tone("warning", volume=volume_level)
    elif alert_choice == "9. Digital Warning Signal":
        play_warning_tone("digital", volume=volume_level)
    else:
        play_warning_tone("voice", volume=volume_level)

    fallback_html = """
    <div style="margin-top: 12px;">
        <button id="manual-warning-button"
            style="padding: 10px 16px; background: #8b5cf6; color: white; border: none; border-radius: 10px; cursor: pointer;">
            🔊 Play warning sound
        </button>
    </div>
    <script>
    document.getElementById("manual-warning-button")?.addEventListener("click", function() {
        try {
            const speech = window.speechSynthesis;
            if (!speech) return;
            speech.cancel();
            const msg = new SpeechSynthesisUtterance("Attention! Attention! Phone detected! Please put your phone away and focus on studying!");
            msg.volume = 0.8;
            msg.rate = 0.9;
            msg.pitch = 1.15;
            speech.speak(msg);
        } catch (e) {
            console.log("Manual warning failed:", e);
        }
    });
    </script>
    """
    components.html(fallback_html, height=70)

if st.button("🔊 Test Selected Sound", key="test_phone_alert_main"):
    show_phone_alert(preview=True)

# ============================================================
# LIVE DETECTION HELPERS
# ============================================================

PHONE_KEYWORDS = [
    "cell phone",
    "mobile phone",
    "phone",
    "smartphone",
    "cellphone",
]

def should_alert_user(phone_count, last_alert_time, current_time):
    if phone_count <= 0:
        return False
    if current_time - last_alert_time < ALERT_COOLDOWN:
        return False
    return phone_count >= REQUIRED_CONSECUTIVE_DETECTIONS

# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero-panel">
      <span class="hero-chip">Class 12 Artificial Intelligence Project</span>
      <h1>📱 AI Study Distraction Detector</h1>
      <p>Computer vision for analysing phone visibility in study-session videos, with live camera detection, session insights, and downloadable reports.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="section-note">Phone visibility is an indicator only—not proof of distraction, attention, or study quality. Phones may also be used for learning.</p>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 📱 Study Detector")
    st.caption("AI project workspace")
    st.markdown("---")
    st.markdown("### Quick navigation")
    st.markdown("[📊 Session dashboard](#study-session-dashboard)")
    st.markdown("[🎬 Video analysis](#ai-video-analysis)")
    st.markdown("[📈 Latest results](#detection-results)")
    st.markdown("[⏱️ Study timer](#study-session-timer)")
    st.markdown("[📷 Live webcam](#live-webcam-detection)")
    st.markdown("---")
    with st.expander("ℹ️ About this project", expanded=False):
        st.write("**Objective:** identify visible phone appearances in sampled study-video frames.")
        st.write("**Technology:** Python, Streamlit, OpenCV, and the YOLO object-detection model.")
        st.write("**Output:** phone-detection metrics, charts, a session history, and a PDF report.")
        st.caption("This prototype does not measure attention and may miss or misidentify phones.")

# ============================================================
# REPORT HELPERS
# ============================================================

def load_report() -> pd.DataFrame:
    if not REPORT_FILE.exists() or REPORT_FILE.stat().st_size == 0:
        report_df = pd.DataFrame(columns=CSV_COLUMNS)
    else:
        try:
            report_df = pd.read_csv(REPORT_FILE)
        except pd.errors.EmptyDataError:
            report_df = pd.DataFrame(columns=CSV_COLUMNS)

    for column in CSV_COLUMNS:
        if column not in report_df.columns:
            report_df[column] = pd.NA

    for column in NUMERIC_COLUMNS:
        report_df[column] = pd.to_numeric(report_df[column], errors="coerce")

    return report_df[CSV_COLUMNS]

def save_session(video_name, resolution, frames_processed, frames_with_phone, percentage, detections):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    file_has_content = REPORT_FILE.exists() and REPORT_FILE.stat().st_size > 0

    with REPORT_FILE.open("a", newline="", encoding="utf-8") as report_file:
        writer = csv.writer(report_file)
        if not file_has_content:
            writer.writerow(CSV_COLUMNS)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            video_name,
            resolution,
            int(frames_processed),
            int(frames_with_phone),
            round(float(percentage), 2),
            int(detections),
        ])

def create_pdf_report(analysis):
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45,
    )

    styles = getSampleStyleSheet()
    story = []

    video_name = str(analysis["video_name"])
    resolution = str(analysis["resolution"])
    frames = int(analysis["frames_processed"])
    phone_frames = int(analysis["frames_with_phone"])
    percentage = float(analysis["percentage"])
    detections = int(analysis["detections"])
    focus_score = max(0.0, min(100.0, 100.0 - percentage))

    safe_video_name = video_name.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    story.append(Paragraph("AI Study Distraction Detector", styles["Title"]))
    story.append(Paragraph("Study Session Report", styles["Heading2"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Report generated: {datetime.now().strftime('%d %B %Y, %I:%M %p')}", styles["Normal"]))
    story.append(Spacer(1, 12))

    table_data = [
        ["Statistic", "Result"],
        ["Video", safe_video_name],
        ["Video Resolution", resolution],
        ["Frames Analyzed", f"{frames:,}"],
        ["Frames With Phone", f"{phone_frames:,}"],
        ["Total Phone Detections", f"{detections:,}"],
        ["Phone Visible Percentage", f"{percentage:.2f}%"],
        ["Phone-Free Score", f"{focus_score:.0f}/100"],
    ]

    table = Table(table_data, colWidths=[230, 230], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#24476B")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
        ("PADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))

    story.append(table)
    story.append(Spacer(1, 20))
    story.append(Paragraph("Phone Visibility Chart", styles["Heading2"]))

    chart_buffer = io.BytesIO()
    figure, axis = plt.subplots(figsize=(6, 3.2))
    axis.bar(["Phone Visible", "Phone Not Detected"], [percentage, 100.0 - percentage])
    axis.set_ylabel("Percentage (%)")
    axis.set_ylim(0, 100)
    axis.set_title("Analyzed Video Frames")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(chart_buffer, format="png", dpi=160, bbox_inches="tight")
    plt.close(figure)
    chart_buffer.seek(0)

    story.append(PDFImage(chart_buffer, width=6 * inch, height=3.2 * inch))
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "<b>Interpretation:</b> The Phone-Free Score is calculated as 100 minus the phone-visible percentage. Phone visibility does not prove distraction, and this score is not a validated measurement of concentration or study quality.",
        styles["BodyText"],
    ))

    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()

# ============================================================
# MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_yolo_model(model_path: str):
    return YOLO(model_path)

# ============================================================
# LIVE DETECTION
# ============================================================

class LivePhoneDetector(VideoProcessorBase):
    def __init__(self):
        self.model = load_yolo_model(str(ensure_model_exists()))
        self.last_warn_time = 0.0

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")

        try:
            result = self.model.predict(
                source=image,
                imgsz=640,
                conf=0.18,
                verbose=False,
                max_det=20,
            )[0]

            detections = []
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0].item())
                    class_name = self.model.names[class_id] if not isinstance(self.model.names, dict) else self.model.names.get(class_id, "")
                    class_name = str(class_name).strip().lower()

                    if any(keyword in class_name for keyword in PHONE_KEYWORDS):
                        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                        conf = float(box.conf[0].item())
                        detections.append((x1, y1, x2, y2, conf))

            if detections:
                for x1, y1, x2, y2, conf in detections:
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(
                        image,
                        f"PHONE {conf:.0%}",
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )

                now = time.monotonic()
                if now - st.session_state.last_live_warning_time >= ALERT_COOLDOWN:
                    st.session_state.last_live_warning_time = now
                    st.session_state.live_warning_triggered = True
                    st.session_state.live_phone_detected = True
                    st.error("📱 LIVE PHONE DETECTED! Put your phone away now.")
                    speak_warning()

                cv2.rectangle(image, (0, 0), (image.shape[1], 52), (0, 0, 200), -1)
                cv2.putText(
                    image,
                    "PHONE DETECTED LIVE",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            else:
                st.session_state.live_phone_detected = False
                cv2.rectangle(image, (0, 0), (image.shape[1], 42), (25, 110, 25), -1)
                cv2.putText(
                    image,
                    "NO PHONE DETECTED",
                    (18, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        except Exception:
            st.session_state.live_phone_detected = False

        return av.VideoFrame.from_ndarray(image, format="bgr24")

def speak_warning():
    js = """
    <script>
    try {
        const synth = window.speechSynthesis;
        if (!synth) return;
        synth.cancel();
        const msg = new SpeechSynthesisUtterance("Attention! Phone detected. Please put your phone away and focus on studying.");
        msg.volume = 0.9;
        msg.rate = 0.9;
        msg.pitch = 1.0;
        synth.speak(msg);
    } catch (e) {
        console.log("Speech warning failed:", e);
    }
    </script>
    """
    components.html(js, height=0)

# ============================================================
# LIVE WEBRTC SECTION
# ============================================================

st.divider()
st.subheader("📷 Live Webcam Detection")
st.caption("Use Chrome and allow camera access. This is the real-time live detector and warns immediately when a phone is detected.")

webrtc_streamer(
    key="live_phone_detector",
    video_processor_factory=LivePhoneDetector,
    media_stream_constraints={
        "video": {
            "frameRate": 10,
            "width": 640,
            "height": 480,
            "facingMode": "user",
        },
        "audio": False,
    },
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
            {
                "urls": ["turn:your-turn-server.example.com:3478"],
                "username": "your_turn_username",
                "credential": "your_turn_password",
            },
        ]
    },
    async_processing=True,
)

if st.session_state.get("live_warning_triggered", False):
    st.session_state.live_warning_triggered = False

# ============================================================
# VIDEO ANALYSIS
# ============================================================

def analyze_video(uploaded_file):
    temp_path = None
    capture = None

    try:
        resolved_model_path = ensure_model_exists()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return None

    try:
        extension = Path(uploaded_file.name).suffix.lower() or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = Path(temp_file.name)

        capture = cv2.VideoCapture(str(temp_path))
        if not capture.isOpened():
            st.error("Unable to open this video. Try another MP4, MOV, or AVI file.")
            return None

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))

        if width <= 0 or height <= 0:
            st.error("Video dimensions could not be read. Try another video.")
            return None

        model = load_yolo_model(str(resolved_model_path))

        sampled_frames = 0
        frames_with_phone = 0
        total_detections = 0
        object_counts = {}
        progress = st.progress(0.0)
        status = st.empty()

        while True:
            success, frame = capture.read()
            if not success:
                break

            if sampled_frames % 5 == 0:
                prediction = model.predict(source=frame, imgsz=640, conf=0.12, verbose=False, max_det=20)[0]
                phone_count = 0

                if prediction.boxes is not None:
                    for box in prediction.boxes:
                        cls_id = int(box.cls[0].item())
                        class_name = model.names[cls_id] if not isinstance(model.names, dict) else model.names.get(cls_id, "")
                        class_name = str(class_name).strip().lower()
                        object_counts[class_name] = object_counts.get(class_name, 0) + 1

                        if any(keyword in class_name for keyword in PHONE_KEYWORDS):
                            phone_count += 1
                            total_detections += 1

                if phone_count > 0:
                    frames_with_phone += 1

                if total_frames > 0:
                    progress.progress(min((sampled_frames + 1) / total_frames, 1.0))
                status.write(f"Analyzed {sampled_frames + 1} sampled frames...")

            sampled_frames += 1

        percentage = (frames_with_phone / sampled_frames) * 100 if sampled_frames else 0.0
        result = {
            "video_name": uploaded_file.name,
            "resolution": f"{width}x{height}",
            "frames_processed": sampled_frames,
            "frames_with_phone": frames_with_phone,
            "percentage": round(float(percentage), 2),
            "detections": total_detections,
            "object_counts": object_counts,
        }

        return result

    except Exception as error:
        st.error(f"Video analysis failed: {error}")
        return None

    finally:
        if capture is not None:
            capture.release()
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass

# ============================================================
# DASHBOARD
# ============================================================

st.divider()
st.subheader("📊 Study Session Dashboard")
st.caption("A quick overview of completed video analyses and saved session history.")

report_df = load_report()

if report_df.empty:
    st.info("No completed sessions yet. Upload a video to see the stats here.")
else:
    total_sessions = len(report_df)
    total_frames = int(report_df["Frames Processed"].fillna(0).sum())
    total_detections = int(report_df["Total Phone Detections"].fillna(0).sum())
    average_visibility = report_df["Phone Visible Percentage"].dropna().mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sessions Analyzed", total_sessions)
    col2.metric("Frames Processed", f"{total_frames:,}")
    col3.metric("Phone Detections", f"{total_detections:,}")
    col4.metric("Average Phone Visibility", f"{average_visibility:.1f}%" if pd.notna(average_visibility) else "N/A")

    st.subheader("📈 Session Comparison")
    chart_df = report_df[["Video", "Phone Visible Percentage"]].dropna()
    if not chart_df.empty:
        chart_df = chart_df.rename(columns={
            "Video": "Study Video",
            "Phone Visible Percentage": "Phone Visibility (%)",
        })
        st.bar_chart(chart_df.set_index("Study Video"))
    else:
        st.info("No visibility data yet.")

    st.subheader("📋 Session History")
    st.dataframe(report_df, use_container_width=True)

    csv_data = report_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Session Report (CSV)",
        data=csv_data,
        file_name="session_summary.csv",
        mime="text/csv",
        key="download_csv_summary_main",
    )

# ============================================================
# VIDEO UPLOAD
# ============================================================

st.divider()
st.subheader("🎬 AI Video Analysis")
st.caption("Upload a recorded study session to analyse sampled frames for visible phones.")

uploaded_video = st.file_uploader(
    "Choose a study video",
    type=["mp4", "mov", "avi"],
    key="uploaded_video_main",
)

if uploaded_video is not None:
    st.video(uploaded_video)

    if st.button("Start AI Detection", type="primary", key="start_ai_detection_main"):
        analysis_result = analyze_video(uploaded_video)
        if analysis_result is not None:
            try:
                save_session(
                    analysis_result["video_name"],
                    analysis_result["resolution"],
                    analysis_result["frames_processed"],
                    analysis_result["frames_with_phone"],
                    analysis_result["percentage"],
                    analysis_result["detections"],
                )

                st.session_state["last_analysis"] = analysis_result
                st.success("Analysis completed and session results were saved to the report.")
            except Exception as error:
                st.error(f"The analysis finished, but the report could not be saved: {error}")

# ============================================================
# LAST ANALYSIS RESULTS
# ============================================================

if "last_analysis" in st.session_state:
    saved_analysis = st.session_state["last_analysis"]
    saved_frame = saved_analysis.get("annotated_frame_rgb")

    st.divider()
    st.subheader("📸 Example Frame with Phone Detection")

    if saved_frame is not None:
        st.image(saved_frame, caption=f"Phone-detection example from {saved_analysis['video_name']}", use_container_width=True)
    else:
        st.info("No analyzed frame is available for display.")

    st.caption("This preview shows the sampled frame with the highest-confidence phone detection.")

    st.subheader("🔥 Phone Detection Heatmap")
    heatmap_data = saved_analysis.get("phone_heatmap")
    if heatmap_data is not None:
        heatmap_array = np.asarray(heatmap_data, dtype=np.float32)
        if heatmap_array.sum() > 0:
            figure, axis = plt.subplots(figsize=(8, 4.5))
            image = axis.imshow(
                heatmap_array,
                cmap="hot",
                interpolation="nearest",
                extent=[0, 100, 100, 0],
                aspect="auto",
            )
            axis.set_title("Phone Detection Locations")
            axis.set_xlabel("Horizontal Position (%)")
            axis.set_ylabel("Vertical Position (%)")
            figure.colorbar(image, ax=axis, label="Detection count")
            figure.tight_layout()
            st.pyplot(figure)
            plt.close(figure)
        else:
            st.info("No phone detections were recorded, so a heatmap cannot be displayed.")
    else:
        st.info("Run a new video analysis to generate the heatmap.")

    st.subheader("🔎 Detection Confidence and Video Quality")
    average_confidence = saved_analysis.get("average_confidence")
    highest_confidence = saved_analysis.get("highest_confidence")
    confidence_col1, confidence_col2, confidence_col3 = st.columns(3)
    confidence_col1.metric("Detection threshold", f"{saved_analysis.get('detection_threshold', 0.25):.0%}")
    confidence_col2.metric("Average phone confidence", f"{average_confidence:.1%}" if average_confidence is not None else "N/A")
    confidence_col3.metric("Highest phone confidence", f"{highest_confidence:.1%}" if highest_confidence is not None else "N/A")
    st.caption("Confidence is the model's score for each detected phone, not a guarantee of correctness.")

    warnings = saved_analysis.get("quality_warnings", [])
    if warnings:
        for warning in warnings:
            st.warning(warning)
    else:
        st.success("No obvious technical video-quality warnings were found.")

    duration = saved_analysis.get("duration")
    st.write(f"Video: {saved_analysis['resolution']} | " + (f"Duration: {duration:.1f} seconds" if duration is not None else "Duration unavailable"))

    st.subheader("📊 Detection Results")

    st.divider()
    st.subheader("🧮 Object Counter")

    object_counts = saved_analysis.get("object_counts", {})
    clutter_col1, clutter_col2 = st.columns(2)
    clutter_col1.metric("Total Object Detections", f"{saved_analysis.get('total_object_detections', 0):,}")
    clutter_col2.metric("Unique Object Classes", saved_analysis.get("unique_object_classes", 0))

    if object_counts:
        clutter_df = pd.DataFrame(
            [
                {"Object Class": name.title(), "Detection Count": count}
                for name, count in sorted(object_counts.items(), key=lambda item: item[1], reverse=True)
            ]
        )
        st.dataframe(clutter_df, use_container_width=True, hide_index=True)
        st.bar_chart(clutter_df.set_index("Object Class"))
    else:
        st.info("No objects were detected in the analyzed frames.")

    sampled_frames = saved_analysis["frames_processed"]
    phone_frames = saved_analysis["frames_with_phone"]
    percentage = saved_analysis["percentage"]
    focus_score = max(0.0, min(100.0, 100.0 - percentage))

    st.divider()
    st.subheader("🎯 Study Focus Score")

    score_col1, score_col2 = st.columns(2)
    score_col1.metric("Study Focus Score", f"{focus_score:.0f}/100")
    score_col2.metric("Phone Visible in Analyzed Frames", f"{percentage:.1f}%")

    st.progress(int(round(focus_score)))
    st.markdown("**How is the score calculated?**")
    st.write("Study Focus Score = 100 − Phone Visible Percentage")
    st.write(f"Your video had a phone visible in {phone_frames:,} out of {sampled_frames:,} analyzed frames.")
    st.write(f"Calculation: 100 − {percentage:.1f} = {focus_score:.1f}/100.")

# ============================================================
# TIMER
# ============================================================

st.divider()
st.subheader("⏱️ Study Session Timer")
st.caption("Use a simple timer to structure a focused study block.")

duration_options = [25, 45, 60]
selected_duration = st.selectbox(
    "Choose your study duration (minutes)",
    duration_options,
    index=duration_options.index(st.session_state.study_timer_duration),
    disabled=st.session_state.study_timer_running,
    key="study_timer_duration_selector_main",
)

if selected_duration != st.session_state.study_timer_duration:
    st.session_state.study_timer_duration = selected_duration
    st.session_state.study_timer_elapsed_before_pause = 0.0
    st.session_state.study_timer_started_at = None
    st.session_state.study_timer_running = False
    st.session_state.study_timer_finished = False

total_seconds = st.session_state.study_timer_duration * 60

if st.session_state.study_timer_running:
    elapsed_seconds = st.session_state.study_timer_elapsed_before_pause + time.monotonic() - st.session_state.study_timer_started_at
else:
    elapsed_seconds = st.session_state.study_timer_elapsed_before_pause

elapsed_seconds = min(elapsed_seconds, total_seconds)
remaining_seconds = max(0, total_seconds - elapsed_seconds)

if remaining_seconds <= 0:
    st.session_state.study_timer_running = False
    st.session_state.study_timer_finished = True

elapsed_minutes = int(elapsed_seconds) // 60
elapsed_remainder = int(elapsed_seconds) % 60
remaining_minutes = int(remaining_seconds) // 60
remaining_remainder = int(remaining_seconds) % 60

timer_col1, timer_col2 = st.columns(2)
timer_col1.metric("Elapsed Study Time", f"{elapsed_minutes:02d}:{elapsed_remainder:02d}")
timer_col2.metric("Time Remaining", f"{remaining_minutes:02d}:{remaining_remainder:02d}")

st.progress(min(1.0, elapsed_seconds / total_seconds))

button_col1, button_col2, button_col3 = st.columns(3)

with button_col1:
    if st.button(
        "▶️ Start / Resume",
        disabled=(st.session_state.study_timer_running or st.session_state.study_timer_finished),
        key="study_timer_start_main",
    ):
        st.session_state.study_timer_started_at = time.monotonic()
        st.session_state.study_timer_running = True

with button_col2:
    if st.button(
        "⏸️ Pause",
        disabled=not st.session_state.study_timer_running,
        key="study_timer_pause_main",
    ):
        st.session_state.study_timer_elapsed_before_pause = elapsed_seconds
        st.session_state.study_timer_running = False
        st.session_state.study_timer_started_at = None

with button_col3:
    if st.button("🔄 Reset", key="study_timer_reset_main"):
        st.session_state.study_timer_running = False
        st.session_state.study_timer_started_at = None
        st.session_state.study_timer_elapsed_before_pause = 0.0
        st.session_state.study_timer_finished = False

if st.session_state.study_timer_finished:
    st.success("🎉 Study session completed! Take a short break.")
elif st.session_state.study_timer_running:
    st.info("Your study timer is running.")

# ============================================================
# ABOUT SECTION
# ============================================================

st.divider()
st.header("ℹ️ About the Project")
st.markdown("Learn what the prototype does, how it works, and how to interpret its results.")

about1, about2, about3 = st.columns(3)
with about1:
    st.markdown(
        '<div class="info-card"><h3>🎯 Project objective</h3><p>Identify visible mobile phones in study-session videos and present the detections in a readable report.</p></div>',
        unsafe_allow_html=True,
    )
with about2:
    st.markdown(
        '<div class="info-card"><h3>🧠 AI workflow</h3><p>OpenCV reads video frames, YOLO predicts objects, and the app summarizes sampled frames where a phone is detected.</p></div>',
        unsafe_allow_html=True,
    )
with about3:
    st.markdown(
        '<div class="info-card"><h3>📊 Results and reports</h3><p>Session metrics, charts, a detection timeline, CSV history, and a downloadable PDF help review the analysis.</p></div>',
        unsafe_allow_html=True,
    )

st.markdown("#### Technology stack")
st.write("Python · Streamlit · Ultralytics YOLO · OpenCV · Pandas · Matplotlib · ReportLab · WebRTC")

st.markdown("#### Important limitations")
st.warning(
    "A phone detection does not prove that a student is distracted. The model can miss phones or detect objects incorrectly. The displayed Phone-Free Score is a simple calculation from phone visibility; it is not a validated concentration score."
)
st.caption("Designed as a Class 12 AI prototype for educational demonstration.")
