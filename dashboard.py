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
# Streamlit config
# ============================================================

st.set_page_config(
    page_title="AI Study Distraction Detector",
    page_icon="📱",
    layout="wide",
)

# ============================================================
# Styling
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
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
REPORT_DIR = BASE_DIR / "reports"
REPORT_FILE = REPORT_DIR / "session_summary.csv"

MODEL_NAME = "yolo11n.pt"
MODEL_FILE = BASE_DIR / MODEL_NAME

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
# Model loader
# ============================================================

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

@st.cache_resource(show_spinner=False)
def load_yolo_model(model_path: str):
    return YOLO(model_path)

# ============================================================
# Session state
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

if "session_snapshots" not in st.session_state:
    st.session_state.session_snapshots = []

if "last_warning_reason" not in st.session_state:
    st.session_state.last_warning_reason = "No warning yet"

if "live_status" not in st.session_state:
    st.session_state.live_status = "FOCUSED / NO PHONE"

if "live_focus_streak" not in st.session_state:
    st.session_state.live_focus_streak = 0

if "warning_summary" not in st.session_state:
    st.session_state.warning_summary = {
        "warning_count": 0,
        "last_warning": "No warning yet",
        "recent_events": [],
    }

if "analysis_warning_triggered" not in st.session_state:
    st.session_state.analysis_warning_triggered = False

if "detection_confidence" not in st.session_state:
    st.session_state.detection_confidence = 0.18

if "frame_sample_interval" not in st.session_state:
    st.session_state.frame_sample_interval = 5

if "detection_snapshots" not in st.session_state:
    st.session_state.detection_snapshots = []

if "live_stats" not in st.session_state:
    st.session_state.live_stats = {
        "status": "No camera yet",
        "phone_detected": False,
        "phone_count": 0,
        "confidence": 0.0,
        "last_check": None,
    }

if "warning_log" not in st.session_state:
    st.session_state.warning_log = []

if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = None

if "generated_live_detected_video" not in st.session_state:
    st.session_state.generated_live_detected_video = None

if "analysis_run_counter" not in st.session_state:
    st.session_state.analysis_run_counter = 0

if "last_analysis_alert_key" not in st.session_state:
    st.session_state.last_analysis_alert_key = None

if "last_live_alert_key" not in st.session_state:
    st.session_state.last_live_alert_key = None

# ============================================================
# Utilities
# ============================================================

PHONE_KEYWORDS = [
    "cell phone",
    "mobile phone",
    "phone",
    "smartphone",
    "cellphone",
]

def compute_phone_percentage(frames_with_phone, total_frames):
    if total_frames == 0:
        return 0.0
    return (frames_with_phone / total_frames) * 100.0

def get_focus_score(phone_percentage):
    return max(0.0, min(100.0, 100.0 - phone_percentage))

def get_alert_level(focus_score):
    if focus_score >= 80:
        return "Safe"
    elif focus_score >= 55:
        return "Warning"
    return "Critical"

def get_study_status(focus_score):
    if focus_score >= 80:
        return "Excellent focus"
    elif focus_score >= 60:
        return "Good focus"
    elif focus_score >= 40:
        return "Moderate distraction"
    return "High distraction"

def get_risk_band(focus_score):
    if focus_score >= 80:
        return "LOW RISK"
    elif focus_score >= 60:
        return "MEDIUM RISK"
    elif focus_score >= 40:
        return "HIGH RISK"
    return "CRITICAL"

def update_focus_streak(current_streak, phone_visible):
    if phone_visible:
        return 0
    return current_streak + 1

def calculate_focus_streak_label(streak_value):
    if streak_value >= 50:
        return "Excellent focus streak"
    elif streak_value >= 20:
        return "Strong focus streak"
    elif streak_value >= 5:
        return "Steady focus"
    return "Build focus streak"

def append_warning_log(log_list, message):
    log_list.append({
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "message": message,
    })
    return log_list

def append_warning_summary(summary_dict, message):
    summary_dict["warning_count"] += 1
    summary_dict["last_warning"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary_dict["recent_events"].append({
        "time": summary_dict["last_warning"],
        "message": message,
    })
    if len(summary_dict["recent_events"]) > 10:
        summary_dict["recent_events"] = summary_dict["recent_events"][-10:]
    return summary_dict

def generate_timeline(detection_flags):
    return [
        {"frame": idx, "phone_visible": 1 if flag else 0}
        for idx, flag in enumerate(detection_flags)
    ]

def build_phone_heatmap(detection_events):
    if not detection_events:
        return None

    df = pd.DataFrame(detection_events)
    if df.empty or "phone_detected" not in df.columns:
        return None

    df = df[["frame_index", "phone_detected"]].copy()
    df["phone_detected"] = df["phone_detected"].astype(int)

    if len(df) <= 1:
        heat = np.array([df["phone_detected"].tolist()])
    else:
        max_index = max(int(df["frame_index"].max()), 1)
        bucket_count = min(30, len(df))
        buckets = np.linspace(0, max_index, bucket_count + 1)
        df["bucket"] = pd.cut(df["frame_index"], bins=buckets, include_lowest=True)
        heat_series = df.groupby("bucket", observed=False)["phone_detected"].mean().fillna(0)
        heat = np.array([heat_series.tolist()])

    fig = px.imshow(
        heat,
        color_continuous_scale="YlOrRd",
        title="Phone Visibility Heatmap",
        labels={"x": "Sample bucket", "y": "Frame band"},
        aspect="auto",
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig

def detect_phone_in_frame(frame, confidence_threshold):
    model = load_yolo_model(str(ensure_model_exists()))
    detections = []
    max_conf = 0.0

    results = model.predict(
        source=frame,
        imgsz=640,
        conf=confidence_threshold,
        verbose=False,
        max_det=20,
    )

    if not results or len(results) == 0:
        return {
            "phone_detected": False,
            "phone_count": 0,
            "confidence": 0.0,
            "detections": [],
            "annotated_frame": frame.copy(),
        }

    result = results[0]
    if result.boxes is None:
        return {
            "phone_detected": False,
            "phone_count": 0,
            "confidence": 0.0,
            "detections": [],
            "annotated_frame": frame.copy(),
        }

    for box in result.boxes:
        cls_id = int(box.cls[0].item())
        class_name = (
            model.names[cls_id]
            if not isinstance(model.names, dict)
            else model.names.get(cls_id, "")
        )
        class_name = str(class_name).strip().lower()

        if any(keyword in class_name for keyword in PHONE_KEYWORDS):
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            conf = float(box.conf[0].item())
            detections.append({
                "class": class_name,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
            })
            max_conf = max(max_conf, conf)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(
                frame,
                f"PHONE {conf:.0%}",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

    return {
        "phone_detected": len(detections) > 0,
        "phone_count": len(detections),
        "confidence": float(max_conf),
        "detections": detections,
        "annotated_frame": frame.copy(),
    }

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

def save_session_record(video_name, resolution, frames_processed, frames_with_phone, percentage, detections):
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
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(["Phone Visible", "Phone Not Detected"], [percentage, 100.0 - percentage])
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Analyzed Video Frames")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(chart_buffer, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    chart_buffer.seek(0)

    story.append(PDFImage(chart_buffer, width=6 * inch, height=3.2 * inch))
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "<b>Interpretation:</b> The Phone-Free Score is calculated as 100 minus the phone-visible percentage. Phone visibility does not prove distraction; it is only a signal.",
        styles["BodyText"],
    ))

    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()

# ============================================================
# Alert audio functions
# ============================================================

ALERT_COOLDOWN = 20

def play_female_warning(volume=0.8):
    safe_volume = max(0.0, min(1.0, float(volume)))
    message = "Attention! Phone detected! Please put your phone away and focus on studying!"
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
            if (preferred) utterance.voice = preferred;
            speech.speak(utterance);
        }}
    }} catch (e) {{
        console.log("Female warning failed:", e);
    }}
    </script>
    """
    components.html(js, height=0)

def play_warning_tone(mode="warning", volume=0.75):
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
# Alert UI
# ============================================================

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
# Webcam detection
# ============================================================

class WebcamPhoneDetector(VideoProcessorBase):
    def __init__(self):
        self.model = load_yolo_model(str(ensure_model_exists()))
        self.last_warn_time = 0.0

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")

        try:
            result = self.model.predict(
                source=image,
                imgsz=640,
                conf=st.session_state.detection_confidence,
                verbose=False,
                max_det=20,
            )[0]

            detections = []
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0].item())
                    class_name = (
                        self.model.names[class_id]
                        if not isinstance(self.model.names, dict)
                        else self.model.names.get(class_id, "")
                    )
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
                    st.session_state.last_warning_reason = "Phone detected in webcam"
                    st.session_state.warning_summary = append_warning_summary(
                        st.session_state.warning_summary,
                        st.session_state.last_warning_reason
                    )
                    st.session_state.warning_log.append({
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "message": "Phone detected in webcam"
                    })

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

            st.session_state.live_stats = {
                "status": "PHONE DETECTED" if detections else "NO PHONE DETECTED",
                "phone_detected": bool(detections),
                "phone_count": len(detections),
                "confidence": max([round(float(item[4]), 4) for item in detections], default=0.0),
                "last_check": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            st.session_state.live_status = "PHONE DETECTED" if detections else "FOCUSED / NO PHONE"

        except Exception:
            st.session_state.live_phone_detected = False
            st.session_state.live_status = "FOCUSED / NO PHONE"

        return av.VideoFrame.from_ndarray(image, format="bgr24")

# ============================================================
# Modifier for webcam warning to avoid duplicates
# ============================================================

def trigger_live_warning_once():
    if st.session_state.get("live_warning_triggered", False):
        show_phone_alert(preview=False)
        st.session_state.live_warning_triggered = False

# ============================================================
# Hero section
# ============================================================

st.markdown(
    """
    <div class="hero-panel">
      <span class="hero-chip">Class 12 Artificial Intelligence Project</span>
      <h1>📱 AI Study Distraction Detector</h1>
      <p>Computer vision for monitoring phone visibility during study sessions, with live webcam detection, uploaded video analysis, focus scoring, warnings, and downloadable reports.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="section-note">Phone visibility is an indicator only — not proof of distraction. It can support reflection, not replace judgment.</p>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 📱 Study Detector")
    st.caption("AI project workspace")
    st.markdown("---")
    st.markdown("### Quick navigation")
    st.markdown("[📹 Webcam stream](#live-webcam-detection)")
    st.markdown("[📼 Uploaded video detection](#live-video-detection-of-uploaded-video)")
    st.markdown("[📊 Session dashboard](#study-session-dashboard)")
    st.markdown("[🎬 AI video analysis](#ai-video-analysis)")
    st.markdown("[📈 Detection results](#detection-results)")
    st.markdown("[⏱️ Study timer](#study-session-timer)")
    st.markdown("---")
    with st.expander("ℹ️ About this project", expanded=False):
        st.write("**Objective:** detect visible phone appearances during study sessions.")
        st.write("**Technology:** Python, Streamlit, OpenCV, YOLO, Pandas, Plotly.")
        st.write("**Output:** focus score, timeline, warnings, and downloadable reports.")
        st.caption("This is a prototype and may miss or misidentify objects.")

# ============================================================
# Live webcam detection
# ============================================================

st.divider()
st.subheader("📹 Live Webcam Detection")
st.caption("This is a real webcam stream with live phone detection overlay.")

webrtc_streamer(
    key="webcam_phone_detector",
    video_processor_factory=WebcamPhoneDetector,
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
        ]
    },
    async_processing=True,
)

if st.session_state.get("live_warning_triggered", False):
    trigger_live_warning_once()

live_detected = st.session_state.get("live_phone_detected", False)
st.session_state.live_status = "PHONE DETECTED" if live_detected else "FOCUSED / NO PHONE"

risk_score = 100.0 if not live_detected else max(0.0, 100.0 - 60.0)
risk_band = get_risk_band(risk_score)

live_status_col1, live_status_col2, live_status_col3 = st.columns(3)
live_status_col1.metric("Live Status", st.session_state.live_status)
live_status_col2.metric("Focus Streak", st.session_state.live_focus_streak)
live_status_col3.metric("Risk Level", risk_band)

st.progress(min(1.0, risk_score / 100.0))
st.caption(f"Focus streak message: {calculate_focus_streak_label(st.session_state.live_focus_streak)}")

if live_detected:
    st.session_state.live_focus_streak = 0
else:
    st.session_state.live_focus_streak = update_focus_streak(st.session_state.live_focus_streak, False)

st.subheader("📊 Live Camera Statistics")
live_stats = st.session_state.live_stats
stat1, stat2, stat3, stat4 = st.columns(4)
stat1.metric("Status", live_stats["status"])
stat2.metric("Phones Found", live_stats["phone_count"])
stat3.metric("Max Confidence", f"{live_stats['confidence']:.1%}")
stat4.metric("Last Check", live_stats["last_check"] or "Not checked yet")

# ============================================================
# Live detection of uploaded video
# ============================================================

def create_live_detected_video(uploaded_file, confidence_threshold, sample_interval):
    model = load_yolo_model(str(ensure_model_exists()))

    temp_video_path = None
    temp_output_path = None
    capture = None
    writer = None

    try:
        suffix = Path(uploaded_file.name).suffix.lower() or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_video_path = Path(temp_file.name)

        capture = cv2.VideoCapture(str(temp_video_path))
        if not capture.isOpened():
            st.error("The uploaded video could not be opened.")
            return None

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 24.0
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        if width <= 0 or height <= 0:
            st.error("The uploaded video has no valid frame size.")
            return None

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as out_file:
            temp_output_path = Path(out_file.name)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(temp_output_path), fourcc, fps, (width, height))

        progress = st.progress(0.0)
        status = st.empty()
        analyzed_frames = 0
        detected_frames = 0
        total_phone_hits = 0
        detection_events = []
        detection_flags = []
        snapshots = []

        frame_index = 0
        while True:
            success, frame = capture.read()
            if not success:
                break

            if frame_index % sample_interval != 0:
                frame_index += 1
                continue

            analyzed_frames += 1
            prediction = model.predict(
                source=frame,
                imgsz=640,
                conf=confidence_threshold,
                verbose=False,
                max_det=20,
            )[0]

            phone_count = 0

            if prediction.boxes is not None:
                for box in prediction.boxes:
                    cls_id = int(box.cls[0].item())
                    class_name = (
                        model.names[cls_id]
                        if not isinstance(model.names, dict)
                        else model.names.get(cls_id, "")
                    )
                    class_name = str(class_name).strip().lower()

                    if any(keyword in class_name for keyword in PHONE_KEYWORDS):
                        phone_count += 1
                        total_phone_hits += 1
                        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                        conf = float(box.conf[0].item())

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.putText(
                            frame,
                            f"PHONE {conf:.0%}",
                            (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2,
                            cv2.LINE_AA,
                        )

            if phone_count > 0:
                detected_frames += 1
                detection_flags.append(True)
                detection_events.append({
                    "frame_index": frame_index,
                    "phone_detected": True,
                    "phone_count": phone_count,
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })
            else:
                detection_flags.append(False)
                detection_events.append({
                    "frame_index": frame_index,
                    "phone_detected": False,
                    "phone_count": 0,
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })

            if len(snapshots) < 5:
                snapshots.append(frame.copy())
            else:
                snapshots.pop(0)
                snapshots.append(frame.copy())

            writer.write(frame)
            progress.progress(min((frame_index + 1) / max(total_frames, 1), 1.0))
            status.write(f"Processing video: {analyzed_frames} frames analyzed")

            frame_index += 1

        if writer is not None:
            writer.release()

        percentage = compute_phone_percentage(detected_frames, analyzed_frames)
        focus_score = get_focus_score(percentage)

        final_result = {
            "video_name": uploaded_file.name,
            "resolution": f"{width}x{height}",
            "frames_processed": analyzed_frames,
            "frames_with_phone": detected_frames,
            "percentage": round(float(percentage), 2),
            "detections": total_phone_hits,
            "focus_score": focus_score,
            "alert_level": get_alert_level(focus_score),
            "study_status": get_study_status(focus_score),
            "risk_band": get_risk_band(focus_score),
            "timeline": generate_timeline(detection_flags),
            "detection_events": detection_events,
            "snapshots": snapshots,
            "output_video_path": str(temp_output_path),
        }

        return final_result

    except Exception as exc:
        st.error(f"Live video detection failed: {exc}")
        return None

    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
        if temp_video_path is not None and temp_video_path.exists():
            try:
                temp_video_path.unlink(missing_ok=True)
            except Exception:
                pass

st.divider()
st.subheader("📼 Live Video Detection of Uploaded Video")
st.caption("Upload a video and generate a processed version with phone detection overlay.")

uploaded_video = st.file_uploader(
    "Choose a study video to process",
    type=["mp4", "mov", "avi"],
    key="uploaded_live_video_main",
)

if uploaded_video is not None:
    st.video(uploaded_video)

    if st.button("Start Live Detection Preview", type="primary", key="start_live_detect_button"):
        live_result = create_live_detected_video(
            uploaded_file=uploaded_video,
            confidence_threshold=float(st.session_state.detection_confidence),
            sample_interval=int(st.session_state.frame_sample_interval),
        )

        if live_result is not None:
            st.session_state.generated_live_detected_video = live_result["output_video_path"]
            st.session_state.last_analysis = {
                "video_name": live_result["video_name"],
                "resolution": live_result["resolution"],
                "frames_processed": live_result["frames_processed"],
                "frames_with_phone": live_result["frames_with_phone"],
                "percentage": live_result["percentage"],
                "detections": live_result["detections"],
                "focus_score": live_result["focus_score"],
                "alert_level": live_result["alert_level"],
                "study_status": live_result["study_status"],
                "risk_band": live_result["risk_band"],
                "timeline": live_result["timeline"],
                "detection_events": live_result["detection_events"],
                "snapshots": live_result["snapshots"],
                "annotated_frame": live_result["snapshots"][-1] if live_result["snapshots"] else None,
            }

            st.success("Live detection preview generated successfully.")

            if st.session_state.generated_live_detected_video:
                st.subheader("📽️ Detected Video Preview")
                st.video(st.session_state.generated_live_detected_video)

# ============================================================
# Dashboard summary
# ============================================================

st.divider()
st.subheader("📊 Study Session Dashboard")
st.caption("Quick overview of completed analysis sessions.")

report_df = load_report()

if report_df.empty:
    st.info("No completed sessions yet. Upload a study video to see the dashboard results.")
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
# Warning summary
# ============================================================

st.subheader("🧾 Recent Warning Summary")
warning_summary = st.session_state.warning_summary
st.write(f"Total warnings: {warning_summary['warning_count']}")
st.write(f"Last warning: {warning_summary['last_warning']}")

if warning_summary["recent_events"]:
    warning_df = pd.DataFrame(warning_summary["recent_events"])
    st.dataframe(warning_df, use_container_width=True)
else:
    st.info("No warnings yet.")

# ============================================================
# Controls
# ============================================================

st.divider()
st.subheader("🔧 Detection Controls")

confidence_threshold = st.slider(
    "Adjust detection confidence",
    min_value=0.05,
    max_value=0.60,
    value=0.18,
    step=0.01,
    key="conf_slider",
)
st.session_state.detection_confidence = confidence_threshold

sample_interval = st.slider(
    "Frame sampling interval",
    min_value=1,
    max_value=15,
    value=5,
    step=1,
    help="Analyze every Nth frame to keep performance efficient.",
    key="sample_slider",
)
st.session_state.frame_sample_interval = sample_interval

# ============================================================
# Video analysis function
# ============================================================

def analyze_video(uploaded_file):
    temp_path = None
    capture = None
    warning_log = []
    detection_events = []
    snapshots = []

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

        if width <= 0 or height <= 0:
            st.error("Video dimensions could not be read. Try another video.")
            return None

        model = load_yolo_model(str(resolved_model_path))
        sample_interval = int(st.session_state.frame_sample_interval)
        confidence_threshold = float(st.session_state.detection_confidence)

        sampled_frames = 0
        frames_with_phone = 0
        total_detections = 0
        object_counts = {}
        detection_confidences = []
        detection_flags = []
        progress = st.progress(0.0)
        status = st.empty()
        last_annotated_frame = None
        frame_index = 0

        while True:
            success, frame = capture.read()
            if not success:
                break

            if frame_index % sample_interval != 0:
                frame_index += 1
                continue

            sampled_frames += 1
            prediction = model.predict(
                source=frame,
                imgsz=640,
                conf=confidence_threshold,
                verbose=False,
                max_det=20,
            )[0]
            phone_count = 0

            if prediction.boxes is not None:
                for box in prediction.boxes:
                    cls_id = int(box.cls[0].item())
                    class_name = (
                        model.names[cls_id]
                        if not isinstance(model.names, dict)
                        else model.names.get(cls_id, "")
                    )
                    class_name = str(class_name).strip().lower()
                    object_counts[class_name] = object_counts.get(class_name, 0) + 1

                    if any(keyword in class_name for keyword in PHONE_KEYWORDS):
                        phone_count += 1
                        total_detections += 1
                        detection_confidences.append(float(box.conf[0].item()))
                        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.putText(
                            frame,
                            f"PHONE {box.conf[0].item():.0%}",
                            (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2,
                            cv2.LINE_AA,
                        )

            if phone_count > 0:
                frames_with_phone += 1
                detection_flags.append(True)
                warning_log = append_warning_log(warning_log, "Phone detected in sampled frame")
                detection_events.append({
                    "frame_index": frame_index,
                    "phone_detected": True,
                    "phone_count": phone_count,
                    "confidence": max(float(box.conf[0].item()) for box in prediction.boxes if any(keyword in str(model.names[int(box.cls[0].item())]).lower() for keyword in PHONE_KEYWORDS)) if prediction.boxes is not None else 0.0,
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })
                snapshots.append(frame.copy())
            else:
                detection_flags.append(False)
                detection_events.append({
                    "frame_index": frame_index,
                    "phone_detected": False,
                    "phone_count": 0,
                    "confidence": 0.0,
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })

            if len(snapshots) > 5:
                snapshots = snapshots[-5:]
            st.session_state.detection_snapshots = snapshots

            last_annotated_frame = frame.copy()
            if total_frames > 0:
                progress.progress(min((frame_index + 1) / total_frames, 1.0))
            status.write(f"Analyzing sampled frames... {sampled_frames}")

            frame_index += 1

        phone_percentage = compute_phone_percentage(frames_with_phone, sampled_frames)
        focus_score = get_focus_score(phone_percentage)
        average_confidence = float(np.mean(detection_confidences)) if detection_confidences else 0.0
        highest_confidence = float(np.max(detection_confidences)) if detection_confidences else 0.0

        result = {
            "video_name": uploaded_file.name,
            "resolution": f"{width}x{height}",
            "frames_processed": sampled_frames,
            "frames_with_phone": frames_with_phone,
            "percentage": round(float(phone_percentage), 2),
            "detections": total_detections,
            "focus_score": focus_score,
            "alert_level": get_alert_level(focus_score),
            "average_confidence": average_confidence,
            "highest_confidence": highest_confidence,
            "warnings": warning_log,
            "warning_count": len(warning_log),
            "timeline": generate_timeline(detection_flags),
            "object_counts": object_counts,
            "detection_events": detection_events,
            "annotated_frame": last_annotated_frame,
            "study_status": get_study_status(focus_score),
            "risk_band": get_risk_band(focus_score),
        }

        if result["warning_count"] > 0 and not st.session_state.analysis_warning_triggered:
            st.session_state.analysis_warning_triggered = True
            st.session_state.last_warning_reason = "Phone detected in video analysis"
            st.session_state.warning_summary = append_warning_summary(
                st.session_state.warning_summary,
                st.session_state.last_warning_reason
            )
            st.session_state.warning_log.append({
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "message": "Phone detected in video analysis"
            })
            show_phone_alert(preview=False)

        return result

    except Exception as error:
        st.error(f"Video analysis failed: {error}")
        return None

    finally:
        if capture is not None:
            capture.release()
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass

# ============================================================
# AI video analysis
# ============================================================

st.divider()
st.subheader("🎬 AI Video Analysis")
st.caption("Upload a study video to measure phone visibility during the session.")

uploaded_video_analysis = st.file_uploader(
    "Choose a study video",
    type=["mp4", "mov", "avi"],
    key="uploaded_video_main",
)

if uploaded_video_analysis is not None:
    st.video(uploaded_video_analysis)

    if st.button("Start AI Detection", type="primary", key="start_ai_detection_main"):
        st.session_state.analysis_warning_triggered = False
        analysis_result = analyze_video(uploaded_video_analysis)
        if analysis_result is not None:
            try:
                save_session_record(
                    analysis_result["video_name"],
                    analysis_result["resolution"],
                    analysis_result["frames_processed"],
                    analysis_result["frames_with_phone"],
                    analysis_result["percentage"],
                    analysis_result["detections"],
                )

                st.session_state["last_analysis"] = analysis_result
                st.success("Analysis completed and saved to the report.")
            except Exception as error:
                st.error(f"The analysis finished, but the report could not be saved: {error}")

# ============================================================
# Detection results
# ============================================================

if st.session_state.get("last_analysis") is not None:
    saved_analysis = st.session_state["last_analysis"]

    st.divider()
    st.subheader("📈 Detection Results")

    focus_score = saved_analysis["focus_score"]
    phone_percentage = saved_analysis["percentage"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Focus Score", f"{focus_score:.1f}%")
    col2.metric("Phone Visibility", f"{phone_percentage:.1f}%")
    col3.metric("Alert Level", saved_analysis["alert_level"])
    col4.metric("Study Status", saved_analysis["study_status"])

    st.subheader("📉 Distraction Timeline")
    timeline_df = pd.DataFrame(saved_analysis["timeline"])
    if not timeline_df.empty:
        fig = px.line(
            timeline_df,
            x="frame",
            y="phone_visible",
            title="Phone visibility across sampled frames",
            markers=True,
            range_y=[0, 1.2],
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No timeline is available yet.")

    st.subheader("🔥 Phone Visibility Heatmap")
    heatmap_chart = build_phone_heatmap(saved_analysis.get("detection_events", []))
    if heatmap_chart is not None:
        st.plotly_chart(heatmap_chart, use_container_width=True)
    else:
        st.info("No heatmap data yet.")

    st.subheader("🔒 Warning Log")
    warnings = saved_analysis.get("warnings", [])
    if warnings:
        warning_df = pd.DataFrame(warnings)
        st.dataframe(warning_df, use_container_width=True)
    else:
        st.info("No warnings were recorded.")

    st.subheader("🎯 Confidence and Quality")
    confidence_col1, confidence_col2, confidence_col3 = st.columns(3)
    confidence_col1.metric("Average Confidence", f"{saved_analysis.get('average_confidence', 0):.1%}")
    confidence_col2.metric("Highest Confidence", f"{saved_analysis.get('highest_confidence', 0):.1%}")
    confidence_col3.metric("Warning Count", saved_analysis.get("warning_count", 0))

    st.subheader("🧮 Object Counter")
    object_counts = saved_analysis.get("object_counts", {})
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
        st.info("No objects were detected in the sampled frames.")

    st.subheader("🧾 Detection Events")
    detection_events = pd.DataFrame(saved_analysis.get("detection_events", []))
    if not detection_events.empty:
        st.dataframe(detection_events, use_container_width=True)
        events_csv = detection_events.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Detection Events (CSV)",
            data=events_csv,
            file_name="detection_events.csv",
            mime="text/csv",
            key="download_detection_events_main",
        )
    else:
        st.info("No detection events recorded.")

    st.subheader("📸 Detection Snapshots")
    if st.session_state.detection_snapshots:
        for idx, image in enumerate(reversed(st.session_state.detection_snapshots), start=1):
            st.image(image, channels="BGR", use_container_width=True, caption=f"Snapshot {idx}")
    else:
        st.info("No detection snapshots saved yet.")

    st.download_button(
        "Download PDF Report",
        data=create_pdf_report(saved_analysis),
        file_name="study_session_report.pdf",
        mime="application/pdf",
        key="download_pdf_report_main",
    )

# ============================================================
# Warning log
# ============================================================

st.divider()
st.subheader("Warning Log")
warnings = st.session_state.get("warning_log", [])
if warnings:
    warning_df = pd.DataFrame(warnings)
    st.dataframe(warning_df, use_container_width=True)
else:
    st.info("No warnings recorded.")

# ============================================================
# Timer
# ============================================================

st.divider()
st.subheader("⏱️ Study Session Timer")
st.caption("Use the timer to manage a focused study block.")

duration_options = [25, 45, 60]
selected_duration = st.selectbox(
    "Choose study duration (minutes)",
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
# About section
# ============================================================

st.divider()
st.header("ℹ️ About the Project")
st.markdown("Learn what the prototype does, how it works, and how to interpret the results.")

about1, about2, about3 = st.columns(3)
with about1:
    st.markdown(
        '<div class="info-card"><h3>🎯 Project objective</h3><p>Identify visible mobile phones in study-session videos and summarize distraction patterns.</p></div>',
        unsafe_allow_html=True,
    )
with about2:
    st.markdown(
        '<div class="info-card"><h3>🧠 AI workflow</h3><p>OpenCV reads frames, the YOLO model runs object detection, and the dashboard summarizes phone visibility over time.</p></div>',
        unsafe_allow_html=True,
    )
with about3:
    st.markdown(
        '<div class="info-card"><h3>📊 Results and reports</h3><p>Focus score, warning log, CSV history, and downloadable PDF help explain the outcomes clearly.</p></div>',
        unsafe_allow_html=True,
    )

st.markdown("#### Technology stack")
st.write("Python · Streamlit · YOLO · OpenCV · Pandas · Plotly · ReportLab · WebRTC")

st.markdown("#### Important limitations")
st.warning(
    "A detected phone does not prove distraction. The model can miss objects or misclassify things. This app should be used as a study-support tool, not as a final judgment."
)
st.caption("Designed as a Class 12 AI prototype for educational demonstration.")


