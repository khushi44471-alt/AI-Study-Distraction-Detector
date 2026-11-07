
from __future__ import annotations
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import cv2
import numpy as np
import plotly.express as px
import base64
import csv
import io
import json
import tempfile
import time
from datetime import datetime
from pathlib import Path

from reportlab.platypus import (
    Image as PDFImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
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
# PREMIUM DASHBOARD THEME — REFINED EDITION
# ============================================================

is_dark_mode = st.toggle(
    "🌙 Dark Mode",
    value=True,
    key="dashboard_theme",
)

if is_dark_mode:
    theme = {
        "bg1": "#11101b",
        "bg2": "#171526",
        "panel": "#201e32",
        "panel2": "#28243e",
        "brand": "#a78bfa",
        "text": "#eeeafd",
        "muted": "#aaa4c2",
        "line": "#39334f",
        "heading": "#ffffff",
        "button": "#29243e",
        "button_text": "#f5f0ff",
        "input": "#211e34",
        "hero1": "#30205b",
        "hero2": "#6338a8",
        "hero3": "#9875ef",
    }

    shadow = "rgba(0, 0, 0, 0.22)"
    glow = "rgba(167, 139, 250, 0.16)"

else:
    theme = {
        "bg1": "#f7f5fc",
        "bg2": "#efebf9",
        "panel": "#ffffff",
        "panel2": "#f7f3ff",
        "brand": "#8055d9",
        "text": "#302944",
        "muted": "#77718c",
        "line": "#e6def4",
        "heading": "#241a3b",
        "button": "#ffffff",
        "button_text": "#55368f",
        "input": "#ffffff",
        "hero1": "#49318a",
        "hero2": "#7951c9",
        "hero3": "#a58aeb",
    }

    shadow = "rgba(75, 50, 120, 0.08)"
    glow = "rgba(128, 85, 217, 0.12)"


st.markdown(
    f"""
    <style>

    /* =========================================
       MAIN DASHBOARD BACKGROUND
    ========================================= */

    :root {{
        --bg-1: {theme["bg1"]};
        --bg-2: {theme["bg2"]};
        --panel: {theme["panel"]};
        --panel-2: {theme["panel2"]};
        --brand: {theme["brand"]};
        --text: {theme["text"]};
        --muted: {theme["muted"]};
        --line: {theme["line"]};
        --shadow: {shadow};
        --glow: {glow};
    }}

    .stApp {{
        background:
            radial-gradient(
                ellipse at 12% 0%,
                {glow},
                transparent 38%
            ),
            linear-gradient(
                135deg,
                var(--bg-1),
                var(--bg-2)
            );
        color: var(--text);
    }}

    .main .block-container {{
        max-width: 1480px;
        padding-top: 2.2rem;
        padding-bottom: 4rem;
    }}

    /* =========================================
       HEADER
    ========================================= */

    [data-testid="stHeader"] {{
        background: transparent;
    }}

    [data-testid="stToolbar"] {{
        right: 1rem;
    }}

    /* =========================================
       TYPOGRAPHY
    ========================================= */

    .main h1,
    .main h2,
    .main h3,
    .main h4 {{
        color: {theme["heading"]} !important;
        letter-spacing: -0.035em;
        font-weight: 750;
    }}

    .main h1 {{
        font-size: clamp(2rem, 4vw, 2.8rem);
        line-height: 1.2;
    }}

    .main h2 {{
        font-size: 1.65rem;
    }}

    .main h3 {{
        font-size: 1.2rem;
    }}

    .main p,
    .main li,
    .main label {{
        color: var(--text);
    }}

    .main [data-testid="stCaptionContainer"] {{
        color: var(--muted);
    }}

    /* =========================================
       DIVIDERS
    ========================================= */

    .main hr {{
        border-color: var(--line) !important;
        opacity: 0.85;
        margin: 1.7rem 0;
    }}

    /* =========================================
       METRIC CARDS
    ========================================= */

    .main [data-testid="stMetric"] {{
        background: linear-gradient(
            145deg,
            var(--panel),
            var(--panel-2)
        );

        border: 1px solid var(--line);
        border-radius: 20px;
        padding: 21px 20px;
        min-height: 125px;

        box-shadow: 0 8px 25px var(--shadow);

        transition:
            transform 0.22s ease,
            border-color 0.22s ease,
            box-shadow 0.22s ease;
    }}

    .main [data-testid="stMetric"]:hover {{
        transform: translateY(-4px);
        border-color: var(--brand);
        box-shadow: 0 12px 30px var(--glow);
    }}

    .main [data-testid="stMetricLabel"] {{
        color: var(--muted) !important;
        font-weight: 650;
        font-size: 0.88rem;
    }}

    .main [data-testid="stMetricValue"] {{
        color: {theme["heading"]} !important;
        font-weight: 800;
        letter-spacing: -0.04em;
    }}

    /* =========================================
       BUTTONS
    ========================================= */

    .main div.stButton > button,
    .main div.stDownloadButton > button {{
        background: linear-gradient(
            135deg,
            {theme["button"]},
            var(--panel-2)
        );

        color: {theme["button_text"]} !important;
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 0.62rem 1.15rem;
        font-weight: 650;

        transition: all 0.2s ease;
    }}

    .main div.stButton > button:hover,
    .main div.stDownloadButton > button:hover {{
        border-color: var(--brand);
        transform: translateY(-2px);
        box-shadow: 0 7px 20px var(--glow);
    }}

    .main div.stButton > button[kind="primary"] {{
        background: linear-gradient(
            110deg,
            #7046c9,
            #9870e9
        );
        color: #ffffff !important;
        border: none;
    }}

    /* =========================================
       INPUTS AND SELECT BOXES
    ========================================= */

    .main [data-baseweb="select"] > div,
    .main [data-baseweb="input"] > div,
    .main [data-baseweb="textarea"] > div {{
        background: var(--panel);
        border-color: var(--line);
        border-radius: 12px;
        color: var(--text);
    }}

    .main input,
    .main textarea {{
        color: var(--text) !important;
    }}

    .main [data-baseweb="select"] span {{
        color: var(--text);
    }}

    .main [data-baseweb="select"] > div:focus-within,
    .main [data-baseweb="input"] > div:focus-within {{
        border-color: var(--brand);
        box-shadow: 0 0 0 2px var(--glow);
    }}

    /* =========================================
       FILE UPLOADER
    ========================================= */

    .main [data-testid="stFileUploader"] {{
        background: var(--panel);
        border: 1px dashed var(--brand);
        border-radius: 18px;
        padding: 17px;
        transition: all 0.2s ease;
    }}

    .main [data-testid="stFileUploader"]:hover {{
        border-style: solid;
        box-shadow: 0 8px 22px var(--glow);
    }}

    /* =========================================
       ALERTS
    ========================================= */

    .main [data-testid="stAlert"] {{
        border-radius: 14px;
        border: 1px solid var(--line);
        box-shadow: 0 4px 15px var(--shadow);
    }}

    /* =========================================
       DATA TABLES
    ========================================= */

    .main [data-testid="stDataFrame"],
    .main [data-testid="stTable"] {{
        border: 1px solid var(--line);
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 5px 18px var(--shadow);
    }}

    /* =========================================
       PROGRESS BARS
    ========================================= */

    .main [data-testid="stProgressBar"] > div {{
        background: var(--panel-2);
        border-radius: 20px;
    }}

    .main [data-testid="stProgressBar"] > div > div {{
        background: linear-gradient(
            90deg,
            #7046c9,
            #a78bfa
        );
        border-radius: 20px;
    }}

    /* =========================================
       SLIDERS AND CHECKBOXES
    ========================================= */

    .main [data-testid="stSlider"] {{
        color: var(--brand);
    }}

    .main [data-testid="stCheckbox"],
    .main [data-testid="stRadio"] {{
        color: var(--text);
    }}

    /* =========================================
       HERO PANEL
    ========================================= */

    .hero-panel {{
        padding: 36px;
        border-radius: 25px;

        background: linear-gradient(
            120deg,
            {theme["hero1"]},
            {theme["hero2"]} 60%,
            {theme["hero3"]}
        );

        border: 1px solid rgba(255,255,255,0.20);
        box-shadow: 0 18px 42px rgba(85,55,150,0.18);
        margin: 10px 0 26px 0;
    }}

    .hero-panel h1 {{
        color: #ffffff !important;
        font-size: clamp(2rem, 4vw, 2.8rem);
        margin-bottom: 10px;
    }}

    .hero-panel p {{
        color: #f1eaff !important;
        font-size: 1rem;
        max-width: 800px;
        line-height: 1.7;
    }}

    .hero-chip {{
        display: inline-block;
        background: rgba(255,255,255,0.14);
        border: 1px solid rgba(255,255,255,0.28);
        color: #ffffff;
        border-radius: 50px;
        padding: 7px 14px;
        font-size: 0.78rem;
        font-weight: 700;
        margin-bottom: 15px;
    }}

    /* =========================================
       INFORMATION CARDS
    ========================================= */

    .info-card {{
        background: linear-gradient(
            145deg,
            var(--panel),
            var(--panel-2)
        );

        border: 1px solid var(--line);
        border-radius: 20px;
        padding: 23px;
        height: 100%;
        box-shadow: 0 8px 22px var(--shadow);

        transition: all 0.22s ease;
    }}

    .info-card:hover {{
        transform: translateY(-3px);
        border-color: var(--brand);
        box-shadow: 0 12px 28px var(--glow);
    }}

    .info-card h3 {{
        color: var(--brand) !important;
        margin-top: 0;
    }}

    .info-card p {{
        color: var(--muted) !important;
        line-height: 1.65;
    }}

    .section-note {{
        color: var(--muted);
        font-size: 0.92rem;
        line-height: 1.6;
    }}

    /* =========================================
       CHART CONTAINERS
    ========================================= */

    .main [data-testid="stPlotlyChart"] {{
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 19px;
        padding: 10px;
        box-shadow: 0 7px 22px var(--shadow);
    }}

    /* =========================================
       EXPANDERS
    ========================================= */

    .main [data-testid="stExpander"] {{
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 15px;
        overflow: hidden;
    }}

    /* =========================================
       RESPONSIVE DESIGN
    ========================================= */

    @media (max-width: 768px) {{

        .main .block-container {{
            padding: 1.1rem 0.8rem 3rem 0.8rem;
        }}

        .hero-panel {{
            padding: 23px;
            border-radius: 19px;
        }}

        .main [data-testid="stMetric"] {{
            padding: 14px;
            min-height: 100px;
            border-radius: 15px;
        }}

        .main h1 {{
            font-size: 1.9rem;
        }}

        .info-card {{
            padding: 17px;
        }}
    }}

    /* =========================================
       SMOOTH ANIMATIONS
    ========================================= */

    .main button,
    .main input,
    .main [data-testid="stMetric"] {{
        transition: all 0.2s ease;
    }}

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
# Utility functions
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

def generate_session_insight(analysis):
    phone_pct = float(analysis.get("percentage", 0.0))
    focus_score = float(analysis.get("focus_score", 0.0))
    detections = int(analysis.get("detections", 0))
    frames = int(analysis.get("frames_processed", 0))
    phone_frames = int(analysis.get("frames_with_phone", 0))

    if phone_pct >= 50:
        risk_text = "Very high phone visibility"
        recommendation = "The session shows a strong phone presence. Consider taking a short break and removing phone distractions before continuing."
    elif phone_pct >= 20:
        risk_text = "Moderate phone visibility"
        recommendation = "There were several detected phone appearances; you may want to keep the phone out of reach for the next study block."
    elif phone_pct > 0:
        risk_text = "Low phone visibility"
        recommendation = "A few phone detections were found. This is not severe, but staying fully off-phone would improve focus further."
    else:
        risk_text = "No phone detected"
        recommendation = "No phone visibility was observed in the sampled frames. Keep this focus pattern going for a longer session."

    summary = (
        f"Across {frames} sampled frames, phone visibility was detected in {phone_frames} frames "
        f"({phone_pct:.1f}%), producing a focus score of {focus_score:.1f}%. "
        f"Total detected phone events: {detections}."
    )

    return {
        "risk_text": risk_text,
        "recommendation": recommendation,
        "summary": summary,
    }

def play_female_warning(message, volume=0.75):
    safe_volume = max(0.0, min(1.0, float(volume)))

    safe_message = (
        str(message)
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", " ")
    )

    js = f"""
    <script>
    try {{
        const speech = window.speechSynthesis;
        if (speech) {{
            speech.cancel();
            const utterance = new SpeechSynthesisUtterance("{safe_message}");
            utterance.volume = {safe_volume};
            utterance.rate = 0.9;
            utterance.pitch = 1.15;

            const voices = speech.getVoices();
            const preferred = voices.find(v =>
                /female|zira|samantha|aria|jenny|hazel/i.test(
                    (v.name || "") + " " + (v.voiceURI || "")
                )
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

smart_alerts_enabled = st.checkbox(
    "🔔 Enable Smart Detection Alerts",
    value=True,
    help="Receive an alert when a phone is detected repeatedly.",
    key="smart_alerts_enabled",
)

alert_threshold = st.slider(
    "📱 Consecutive Phone Detections Required",
    min_value=1,
    max_value=20,
    value=3,
    step=1,
    help="Number of consecutive detected frames required before an alert.",
    key="smart_alert_threshold",
)

alert_cooldown = st.slider(
    "⏱️ Alert Cooldown (seconds)",
    min_value=10,
    max_value=120,
    value=30,
    step=5,
    help="Minimum waiting time between repeated alerts.",
    key="smart_alert_cooldown",
)

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
        custom_audio = base64.b64encode(
            uploaded_audio.getvalue()
        ).decode("utf-8")
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
        if "phone_alert_placeholder" not in st.session_state:
            st.session_state.phone_alert_placeholder = True

        if "phone_alert_display" not in st.session_state:
            st.session_state.phone_alert_display = False

        st.session_state.phone_alert_display = True

        st.warning(
            "📱 Phone detected repeatedly! Please check whether "
            "you need your phone for studying."
        )

    if alert_choice == "1. Female Spoken Warning":
        play_female_warning(
            message="Phone detected during your study session. Please keep your phone aside.",
            volume=volume_level
        )
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

def trigger_live_warning_once():
    if st.session_state.get("live_warning_triggered", False):
        show_phone_alert(preview=False)
        st.session_state.live_warning_triggered = False

# ============================================================
# Process uploaded video with live detection overlay
# ============================================================



def process_uploaded_video_for_live_detection(
    uploaded_file, confidence_threshold, sample_interval
):
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
            st.error("Invalid video dimensions.")
            return None

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_out:
            temp_output_path = Path(temp_out.name)

        writer = cv2.VideoWriter(
            str(temp_output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        if not writer.isOpened():
            st.error("Could not create the processed output video.")
            return None

        st.subheader("🔴 Live AI Detection")
        video_display = st.empty()
        status_display = st.empty()
        progress = st.progress(0.0)

        analyzed_frames = 0
        frames_with_phone = 0
        total_phone_hits = 0
        detection_flags = []
               # Smart alert tracking
        consecutive_phone_frames = 0
        last_alert_time = 0.0
        detection_events = []
        last_annotated_frame = None
        frame_index = 0

        while True:
            success, frame = capture.read()

            if not success:
                break

            prediction = model.predict(
                source=frame,
                imgsz=640,
                conf=confidence_threshold,
                verbose=False,
                max_det=20,
            )[0]

            phone_count = 0
            detected_objects = []

            if prediction.boxes is not None:
                for box in prediction.boxes:
                    cls_id = int(box.cls[0].item())

                    class_name = (
                        model.names[cls_id]
                        if not isinstance(model.names, dict)
                        else model.names.get(cls_id, "")
                    )

                    class_name = str(class_name).strip().lower()
                    confidence = float(box.conf[0].item())

                    x1, y1, x2, y2 = [
                        int(v) for v in box.xyxy[0].tolist()
                    ]

                    is_phone = any(
                        keyword in class_name
                        for keyword in PHONE_KEYWORDS
                    )

                    if is_phone:
                        phone_count += 1
                        total_phone_hits += 1
                        box_color = (0, 0, 255)
                    else:
                        # Consistent, different colour for each object class.
                        class_hash = sum(
                            (index + 1) * ord(char)
                            for index, char in enumerate(class_name)
                        )
                        box_color = (
                            50 + (class_hash * 37) % 206,
                            50 + (class_hash * 67) % 206,
                            50 + (class_hash * 97) % 206,
                        )

                    label = f"{class_name.upper()} {confidence:.0%}"

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        box_color,
                        3,
                    )

                    cv2.putText(
                        frame,
                        label,
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        box_color,
                        2,
                        cv2.LINE_AA,
                    )

                    detected_objects.append({
                        "class": class_name,
                        "confidence": confidence,
                        "bbox": [x1, y1, x2, y2],
                        "is_phone": is_phone,
                    })

            analyzed_frames += 1
            detected = phone_count > 0

            if detected:
                frames_with_phone += 1
                consecutive_phone_frames += 1
                status_display.error("📱 PHONE DETECTED")

                current_time = time.monotonic()

                if (
                    smart_alerts_enabled
                    and consecutive_phone_frames >= alert_threshold
                    and current_time - last_alert_time >= alert_cooldown
                ):
                    show_phone_alert()
                    last_alert_time = current_time

            else:
                consecutive_phone_frames = 0
                status_display.success("✅ NO PHONE DETECTED")

            detection_flags.append(detected)

            detection_events.append({
                "frame_index": frame_index,
                "phone_detected": detected,
                "phone_count": phone_count,
                "objects": detected_objects,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })

            video_display.image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                channels="RGB",
                use_container_width=True,
            )

            writer.write(frame)
            last_annotated_frame = frame.copy()

            if total_frames > 0:
                progress.progress(
                    min((frame_index + 1) / total_frames, 1.0)
                )

            frame_index += 1

        phone_percentage = compute_phone_percentage(
            frames_with_phone,
            analyzed_frames,
        )

        focus_score = get_focus_score(phone_percentage)

        return {
            "video_name": uploaded_file.name,
            "resolution": f"{width}x{height}",
            "frames_processed": analyzed_frames,
            "frames_with_phone": frames_with_phone,
            "percentage": round(float(phone_percentage), 2),
            "detections": total_phone_hits,
            "focus_score": focus_score,
            "alert_level": get_alert_level(focus_score),
            "study_status": get_study_status(focus_score),
            "risk_band": get_risk_band(focus_score),
            "timeline": generate_timeline(detection_flags),
            "detection_events": detection_events,
            "annotated_frame": last_annotated_frame,
            "output_video_path": str(temp_output_path),
        }

    except Exception as exc:
        st.error(f"Live processing failed: {exc}")
        return None

    finally:
        if capture is not None:
            capture.release()

        if writer is not None:
            writer.release()

        if temp_video_path is not None and temp_video_path.exists():
            try:
                temp_video_path.unlink(missing_ok=True)
            except OSError:
                pass


# ============================================================
# Analyze uploaded video
# ============================================================


def build_phone_heatmap(detection_events):
    """Build a modern heatmap of phone visibility over time."""

    if not detection_events:
        return None

    heatmap_data = []

    for event in detection_events:
        frame_number = event.get("frame_index", 0)
        detected = bool(event.get("phone_detected", False))

        heatmap_data.append({
            "Frame": frame_number,
            "Phone Visibility": 1 if detected else 0,
        })

    heatmap_df = pd.DataFrame(heatmap_data)

    if heatmap_df.empty:
        return None

    # Group frames into 20-frame intervals.
    heatmap_df["Time Block"] = (
        heatmap_df["Frame"] // 20
    ) * 20

    grouped = (
        heatmap_df.groupby("Time Block")["Phone Visibility"]
        .mean()
        .reset_index()
    )

    grouped["Visibility (%)"] = (
        grouped["Phone Visibility"] * 100
    ).round(1)

    x_labels = [
        f"{int(start)}–{int(start + 19)}"
        for start in grouped["Time Block"]
    ]

    fig = px.imshow(
        [grouped["Visibility (%)"].tolist()],
        x=x_labels,
        y=["Phone visibility"],
        color_continuous_scale=[
            [0.0, "#172554"],
            [0.25, "#4338CA"],
            [0.5, "#7C3AED"],
            [0.75, "#C084FC"],
            [1.0, "#F0ABFC"],
        ],
        zmin=0,
        zmax=100,
        aspect="auto",
        labels={
            "x": "Frame interval",
            "y": "",
            "color": "Visibility",
        },
        text_auto=".1f",
    )

    fig.update_traces(
        hovertemplate=(
            "<b>Frames %{x}</b><br>"
            "Phone visibility: %{z:.1f}%"
            "<extra></extra>"
        ),
        xgap=5,
        ygap=5,
    )

    fig.update_layout(
        title={
            "text": "📱 Phone Visibility Activity Map",
            "x": 0.02,
            "xanchor": "left",
            "font": {"size": 20},
        },
        height=280,
        margin=dict(l=25, r=25, t=70, b=65),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Arial",
            size=12,
            color="#CBD5E1",
        ),
        coloraxis_colorbar=dict(
            title="Visibility",
            ticksuffix="%",
            thickness=12,
            len=0.8,
            outlinewidth=0,
        ),
    )

    fig.update_xaxes(
        title="Video timeline (frame intervals)",
        tickangle=-30,
        showgrid=False,
        zeroline=False,
        tickfont=dict(size=10),
    )

    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        tickfont=dict(size=12),
    )

    return fig


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
        sample_interval = max(1, int(st.session_state.frame_sample_interval))
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
            phone_detections = []

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
                        confidence = float(box.conf[0].item())
                        detection_confidences.append(confidence)

                        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

                        phone_detections.append({
                            "class": class_name,
                            "confidence": confidence,
                            "bbox": [x1, y1, x2, y2],
                        })

                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            (0, 0, 255),
                            3,
                        )

                        cv2.putText(
                            frame,
                            f"PHONE {confidence:.0%}",
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

                warning_log = append_warning_log(
                    warning_log,
                    "Phone detected in sampled frame",
                )

                detection_events.append({
                    "frame_index": frame_index,
                    "phone_detected": True,
                    "phone_count": phone_count,
                    "confidence": max(
                        item["confidence"] for item in phone_detections
                    ),
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
                progress.progress(
                    min((frame_index + 1) / total_frames, 1.0)
                )

            status.write(
                f"Analyzing sampled frames... {sampled_frames}"
            )

            frame_index += 1

        phone_percentage = compute_phone_percentage(
            frames_with_phone,
            sampled_frames,
        )

        focus_score = get_focus_score(phone_percentage)

        average_confidence = (
            float(np.mean(detection_confidences))
            if detection_confidences else 0.0
        )

        highest_confidence = (
            float(np.max(detection_confidences))
            if detection_confidences else 0.0
        )

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

        if (
            result["warning_count"] > 0
            and not st.session_state.analysis_warning_triggered
        ):
            st.session_state.analysis_warning_triggered = True
            st.session_state.last_warning_reason = (
                "Phone detected in video analysis"
            )

            st.session_state.warning_summary = append_warning_summary(
                st.session_state.warning_summary,
                st.session_state.last_warning_reason,
            )

            st.session_state.warning_log.append({
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "message": "Phone detected in video analysis",
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

                
def save_session_record(
    video_name,
    resolution,
    frames_processed,
    frames_with_phone,
    percentage,
    detections
):
    report_path = Path("reports/session_summary.csv")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    record = pd.DataFrame([{
        "Date": datetime.now().strftime("%Y-%m-%d"),
        "Session Time": datetime.now().strftime("%H:%M:%S"),
        "Video Name": video_name,
        "Resolution": resolution,
        "Frames Processed": frames_processed,
        "Frames With Phone": frames_with_phone,
        "Phone Visible Percentage": percentage,
        "Total Phone Detections": detections
    }])

    if report_path.exists():
        try:
            old_records = pd.read_csv(report_path)
            record = pd.concat([old_records, record], ignore_index=True)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            pass

    record.to_csv(report_path, index=False)


def create_pdf_report(analysis):
    """Generate a PDF summary of one study video analysis."""

    pdf_buffer = BytesIO()

    document = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(
        "AI Study Distraction Detector System",
        styles["Title"]
    ))
    story.append(Spacer(1, 15))

    story.append(Paragraph(
        "Study Session Analysis Report",
        styles["Heading2"]
    ))
    story.append(Spacer(1, 12))

    report_data = [
        ["Metric", "Result"],
        ["Video Name", str(analysis.get("video_name", "N/A"))],
        ["Resolution", str(analysis.get("resolution", "N/A"))],
        ["Frames Processed", str(analysis.get("frames_processed", 0))],
        ["Frames With Phone", str(analysis.get("frames_with_phone", 0))],
        ["Phone Visibility", f'{analysis.get("percentage", 0):.2f}%'],
        ["Total Phone Detections", str(analysis.get("detections", 0))],
        ["Focus Score", f'{analysis.get("focus_score", 0):.2f}%'],
        ["Alert Level", str(analysis.get("alert_level", "N/A"))],
        ["Study Status", str(analysis.get("study_status", "N/A"))],
        ["Risk Band", str(analysis.get("risk_band", "N/A"))],
    ]

    report_table = Table(report_data, colWidths=[210, 270])

    report_table.setStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ])

    story.append(report_table)
    story.append(Spacer(1, 20))

    story.append(Paragraph("Analysis Summary", styles["Heading2"]))

    insight = generate_session_insight(analysis)

    story.append(Paragraph(
        insight["summary"],
        styles["BodyText"]
    ))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Recommendation", styles["Heading2"]))

    story.append(Paragraph(
        insight["recommendation"],
        styles["BodyText"]
    ))
    story.append(Spacer(1, 20))

    story.append(Paragraph(
        "<b>Important note:</b> Phone visibility is an indicator only. "
        "It does not directly measure a person's attention, concentration, "
        "or actual distraction.",
        styles["Italic"]
    ))

    document.build(story)

    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()







# ============================================================
# PREMIUM HERO SECTION — SIMPLE HTML TEST
# ============================================================

st.markdown(
    """
    <div style="background-color:#4c1d95; color:white; padding:35px; border-radius:20px;">
        <h1 style="color:white;">AI Study Distraction Detector</h1>
        <p style="color:white;">Computer vision for monitoring phone visibility during study sessions.</p>
        <p style="color:white;">✦ CLASS 12 ARTIFICIAL INTELLIGENCE PROJECT</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# PREMIUM SIDEBAR NAVIGATION
# ============================================================

# Sidebar styling
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #11101d 0%, #19152b 100%);
    border-right: 1px solid rgba(167, 139, 250, 0.18);
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
}

[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #f5f3ff !important;
}

[data-testid="stSidebar"] p {
    color: #b8b2cc;
}

.sidebar-brand {
    background: linear-gradient(135deg, #382568, #211b3b);
    border: 1px solid rgba(167, 139, 250, 0.3);
    border-radius: 16px;
    padding: 18px 14px;
    margin-bottom: 18px;
}

.sidebar-brand h2 {
    color: white !important;
    margin: 0;
    font-size: 21px;
}

.sidebar-brand p {
    color: #c4b5fd !important;
    font-size: 12px;
    margin: 7px 0 0 0;
}

.sidebar-section {
    color: #a78bfa;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin: 22px 0 9px 3px;
}

.sidebar-link {
    display: block;
    text-decoration: none !important;
    color: #e9e5f5 !important;
    background: rgba(255,255,255,0.045);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 11px;
    padding: 11px 13px;
    margin: 6px 0;
    font-size: 13px;
    font-weight: 600;
    transition: all 0.2s ease;
}

.sidebar-link:hover {
    background: linear-gradient(100deg, #49317b, #30234f);
    border-color: #a78bfa;
    color: white !important;
    transform: translateX(4px);
    box-shadow: 0 4px 15px rgba(124, 58, 237, 0.18);
}

.sidebar-note {
    background: rgba(139, 92, 246, 0.10);
    border: 1px solid rgba(167, 139, 250, 0.20);
    border-radius: 12px;
    padding: 12px;
    color: #ddd6fe;
    font-size: 12px;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)


with st.sidebar:

    # BRAND HEADER
    st.markdown("""
    <div class="sidebar-brand">
        <h2>📱 Study Detector</h2>
        <p>AI-powered study workspace</p>
    </div>
    """, unsafe_allow_html=True)

    # --------------------------------------------------------
    # QUICK NAVIGATION
    # --------------------------------------------------------

    st.markdown(
        '<div class="sidebar-section">⚡ Quick Navigation</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <a class="sidebar-link" href="#study-session-dashboard">
        🏠 Dashboard
    </a>
    <a class="sidebar-link" href="#live-webcam-detection">
        📹 Live Detection
    </a>
    <a class="sidebar-link" href="#ai-video-analysis">
        🎬 Video Analysis
    </a>
    <a class="sidebar-link" href="#session-history-and-advanced-filters">
        📚 Session History
    </a>
    """, unsafe_allow_html=True)

    # --------------------------------------------------------
    # STUDY TOOLS
    # --------------------------------------------------------

    st.markdown(
        '<div class="sidebar-section">🎯 Study Tools</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <a class="sidebar-link" href="#study-session-timer">
        ⏱️ Focus Timer
    </a>
    <a class="sidebar-link" href="#study-progress-report">
        📈 Study Progress
    </a>
    <a class="sidebar-link" href="#session-comparison">
        ⚖️ Session Comparison
    </a>
    """, unsafe_allow_html=True)

    # --------------------------------------------------------
    # REPORTS AND ANALYTICS
    # --------------------------------------------------------

    st.markdown(
        '<div class="sidebar-section">📊 Reports & Analytics</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <a class="sidebar-link" href="#study-session-dashboard">
        📄 PDF Study Report
    </a>
    <a class="sidebar-link" href="#detection-results">
        📊 Detection Analytics
    </a>
    """, unsafe_allow_html=True)

    # --------------------------------------------------------
    # QUICK ACTIONS
    # --------------------------------------------------------

    st.markdown(
        '<div class="sidebar-section">🛠️ Quick Actions</div>',
        unsafe_allow_html=True
    )

    if st.button("🔄 Refresh Dashboard", use_container_width=True):
        st.rerun()

    if st.button("🧹 Reset Filters", use_container_width=True):
        filter_terms = ("filter", "date_range", "search", "sort")

        for key in list(st.session_state.keys()):
            if any(term in key.lower() for term in filter_terms):
                del st.session_state[key]

        st.rerun()

    # --------------------------------------------------------
    # HELP AND PROJECT INFORMATION
    # --------------------------------------------------------

    st.markdown(
        '<div class="sidebar-section">ℹ️ Help & Information</div>',
        unsafe_allow_html=True
    )

    with st.expander("📖 About this project", expanded=False):
        st.write(
            "**Objective:** Identify visible phone appearances "
            "during study sessions."
        )

        st.write(
            "**Technology:** Python, Streamlit, OpenCV, "
            "YOLO, Pandas and Plotly."
        )

        st.write(
            "**Features:** Live detection, video analysis, "
            "focus score, session history and reports."
        )

        st.caption(
            "This is a prototype. Phone visibility is only "
            "an indicator and does not directly measure attention."
        )

    with st.expander("💡 How to use the dashboard", expanded=False):
        st.write("1. Open Live Detection or Video Analysis.")
        st.write("2. Start a study session or upload a video.")
        st.write("3. Review detection results.")
        st.write("4. Compare sessions and track progress.")
        st.write("5. View or download available reports.")

    # FOOTER
    st.markdown("---")

    st.markdown("""
    <div class="sidebar-note">
        ✨ <b>Study smarter, stay focused.</b><br>
        Your AI study workspace is ready.
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# Live webcam detection
# ============================================================

st.divider()
st.subheader("📹 Live Webcam Detection")
st.caption("This is a real webcam stream with a live phone detection overlay.")

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
# AI video analysis section
# ============================================================

st.divider()
st.subheader("🎬 AI Video Analysis")
st.caption("Upload a study video, run the AI detector, and view the processed live detection preview directly here.")

uploaded_video_analysis = st.file_uploader(
    "Choose a study video",
    type=["mp4", "mov", "avi"],
    key="uploaded_video_main",
)

if uploaded_video_analysis is not None:
    st.video(uploaded_video_analysis)

    if st.button("Start AI Detection", type="primary", key="start_ai_detection_main"):
        st.session_state.analysis_warning_triggered = False

        live_preview = process_uploaded_video_for_live_detection(
            uploaded_video_analysis,
            confidence_threshold=float(st.session_state.detection_confidence),
            sample_interval=int(st.session_state.frame_sample_interval),
        )

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
                    analysis_result["focus_score"],
                )


                st.session_state["last_analysis"] = analysis_result
                st.success("AI video analysis completed and saved to the report.")
            except Exception as error:
                st.error(f"The analysis finished, but the report could not be saved: {error}")

        if live_preview is not None and live_preview.get("output_video_path"):
            st.subheader("📽️ Processed Video Preview")
            st.video(live_preview["output_video_path"])

            processed_col1, processed_col2 = st.columns(2)
            processed_col1.metric("Frames Processed", live_preview["frames_processed"])
            processed_col2.metric("Frames With Phone", live_preview["frames_with_phone"])

            if live_preview.get("annotated_frame") is not None:
                st.subheader("📸 Sample Processed Frame")
                st.image(
                    cv2.cvtColor(live_preview["annotated_frame"], cv2.COLOR_BGR2RGB),
                    channels="RGB",
                    use_container_width=True,
                    caption="Example processed frame with phone bounding box",
                )

# ============================================================
# Session report saving function
# ============================================================


def save_session_record(
    video_name,
    resolution,
    frames_processed,
    frames_with_phone,
    percentage,
    detections,
    focus_score
):
    report_path = Path("reports/session_summary.csv")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    record = pd.DataFrame([{
        "Date": datetime.now().strftime("%Y-%m-%d"),
        "Session Time": datetime.now().strftime("%H:%M:%S"),
        "Video Name": video_name,
        "Resolution": resolution,
        "Frames Processed": frames_processed,
        "Frames With Phone": frames_with_phone,
        "Phone Visible Percentage": percentage,
        "Total Phone Detections": detections,
        "Focus Score": focus_score
    }])

    if report_path.exists():
        try:
            old_records = pd.read_csv(report_path)
            record = pd.concat([old_records, record], ignore_index=True)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            pass

    record.to_csv(report_path, index=False)




# ============================================================
# PREMIUM DASHBOARD ANALYTICS
# ============================================================

st.divider()

st.markdown(
    """
    <div class="analytics-heading">
        <h2>📊 Study Analytics Overview</h2>
        <p>Your study-session statistics and phone visibility insights.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Load saved session records
def load_report():
    report_path = Path("reports/session_summary.csv")

    if not report_path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(report_path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()


report_df = load_report()

if report_df.empty:

    st.info(
        "No completed sessions yet. Upload and analyze a study video "
        "to see your analytics here."
    )

else:
    # Ensure older report files use the expected column name
    if (
        "Video Name" not in report_df.columns
        and "Video" in report_df.columns
    ):
        report_df = report_df.rename(
            columns={"Video": "Video Name"}
        )

    # Convert numeric columns safely
    numeric_columns = [
        "Frames Processed",
        "Frames With Phone",
        "Phone Visible Percentage",
        "Total Phone Detections",
    ]

    for column in numeric_columns:
        if column in report_df.columns:
            report_df[column] = pd.to_numeric(
                report_df[column],
                errors="coerce",
            ).fillna(0)

    # Calculate dashboard statistics
    total_sessions = len(report_df)

    total_frames = int(
        report_df["Frames Processed"].sum()
    ) if "Frames Processed" in report_df.columns else 0

    total_detections = int(
        report_df["Total Phone Detections"].sum()
    ) if "Total Phone Detections" in report_df.columns else 0

    average_visibility = (
        report_df["Phone Visible Percentage"].mean()
        if "Phone Visible Percentage" in report_df.columns
        else float("nan")
    )


    # --------------------------------------------------------
    # TOP ANALYTICS CARDS — PREMIUM KPI DESIGN
    # --------------------------------------------------------

    st.markdown("### 📌 Performance at a Glance")

    st.markdown(
        """
        <style>
        div[data-testid="stMetric"] {
            background: linear-gradient(145deg, #211b38, #30234f);
            border: 1px solid rgba(167, 139, 250, 0.35);
            border-radius: 18px;
            padding: 20px 16px;
            box-shadow: 0 8px 22px rgba(0, 0, 0, 0.15);
            min-height: 125px;
        }

        div[data-testid="stMetricLabel"] p {
            color: #c4b5fd !important;
            font-size: 14px !important;
            font-weight: 600 !important;
        }

        div[data-testid="stMetricValue"] {
            color: #ffffff !important;
            font-size: 28px !important;
            font-weight: 700 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric(
        "📚 Sessions Analyzed",
        f"{total_sessions:,}",
    )

    metric2.metric(
        "🎞️ Frames Processed",
        f"{total_frames:,}",
    )

    metric3.metric(
        "📱 Phone Detections",
        f"{total_detections:,}",
    )

    metric4.metric(
        "📊 Average Phone Visibility",
        f"{average_visibility:.1f}%"
        if pd.notna(average_visibility)
        else "N/A",
    )

    st.divider()

   
    # --------------------------------------------------------
    # PHONE VISIBILITY DONUT CHART — PREMIUM DESIGN
    # --------------------------------------------------------

    st.markdown("### 📱 Phone Visibility Distribution")

    if pd.notna(average_visibility):

        phone_visible = max(
            0.0,
            min(float(average_visibility), 100.0),
        )

        phone_not_detected = 100.0 - phone_visible

        visibility_df = pd.DataFrame({
            "Category": [
                "Phone Visible",
                "Phone Not Detected",
            ],
            "Percentage": [
                phone_visible,
                phone_not_detected,
            ],
        })

        chart_col1, chart_col2 = st.columns([1.3, 1])

        chart_theme = (
            "plotly_dark"
            if is_dark_mode
            else "plotly_white"
        )

        with chart_col1:

            donut_fig = px.pie(
                visibility_df,
                names="Category",
                values="Percentage",
                hole=0.70,
                color="Category",
                color_discrete_map={
                    "Phone Visible": "#a78bfa",
                    "Phone Not Detected": "#22c55e",
                },
            )

            donut_fig.update_traces(
                textinfo="percent",
                textposition="inside",
                hole=0.70,
                marker=dict(
                    line=dict(
                        color="rgba(255,255,255,0.15)",
                        width=3,
                    )
                ),
                hovertemplate=(
                    "<b>%{label}</b><br>"
                    "%{value:.1f}%<extra></extra>"
                ),
            )

            donut_fig.update_layout(
                template=chart_theme,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                title=dict(
                    text="Average Phone Visibility",
                    x=0.5,
                    xanchor="center",
                    font=dict(size=18),
                ),
                legend_title_text="",
                margin=dict(l=15, r=15, t=65, b=15),
                font=dict(size=13),
                showlegend=True,
            )

            st.plotly_chart(
                donut_fig,
                use_container_width=True,
                config={"displayModeBar": False},
            )

        with chart_col2:

            st.markdown("#### 📊 Visibility Summary")

            st.markdown(
                """
                <style>
                .visibility-card {
                    background: linear-gradient(145deg, #211b38, #30234f);
                    border: 1px solid rgba(167,139,250,0.3);
                    border-radius: 16px;
                    padding: 18px;
                    margin: 12px 0;
                }
                .visibility-label {
                    color: #c4b5fd;
                    font-size: 14px;
                    font-weight: 600;
                }
                .visibility-value {
                    color: white;
                    font-size: 29px;
                    font-weight: 700;
                    margin-top: 5px;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="visibility-card">
                    <div class="visibility-label">
                        📱 Phone Visible
                    </div>
                    <div class="visibility-value">
                        {phone_visible:.1f}%
                    </div>
                </div>

                <div class="visibility-card">
                    <div class="visibility-label">
                        ✅ Phone Not Detected
                    </div>
                    <div class="visibility-value">
                        {phone_not_detected:.1f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.caption(
                "These percentages describe phone visibility "
                "in analyzed video frames. They do not measure "
                "or prove a student's attention."
            )

    else:
        st.info("No phone visibility data is available yet.")

    st.divider()


   
    # ------------------------------------------------------------
    # MONTHLY STUDY ACTIVITY CALENDAR
    # ------------------------------------------------------------

    st.markdown(
        """
        <div style="margin-bottom:15px;">
            <div style="font-size:12px;color:#a78bfa;
                        font-weight:700;letter-spacing:2px;">
                STUDY CONSISTENCY
            </div>
            <div style="font-size:25px;font-weight:750;
                        margin-top:5px;">
                📅 Monthly Activity Calendar
            </div>
            <div style="font-size:13px;color:gray;">
                Track your recorded study sessions day by day.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "Date" in report_df.columns:

        activity_dates = pd.to_datetime(
            report_df["Date"],
            errors="coerce",
        ).dropna().dt.normalize()

        today = pd.Timestamp.now().normalize()
        month_start = today.replace(day=1)
        next_month = month_start + pd.offsets.MonthBegin(1)
        days_in_month = (next_month - month_start).days
        first_weekday = month_start.weekday()

        date_counts = (
            activity_dates.dt.date.value_counts().to_dict()
        )

        month_name = today.strftime("%B %Y")

        calendar_bg = (
            "#11141c" if is_dark_mode else "#ffffff"
        )
        empty_bg = (
            "#242033" if is_dark_mode else "#eef2f7"
        )
        border_color = (
            "#302a43" if is_dark_mode else "#dbe2ea"
        )
        text_color = theme["text"]

        calendar_cells = []

        # Empty cells before the first day
        for _ in range(first_weekday):
            calendar_cells.append(
                '<div class="month-empty"></div>'
            )

        # Actual days of the current month
        for day in range(1, days_in_month + 1):

            current_date = month_start.replace(day=day)
            session_count = date_counts.get(
                current_date.date(),
                0,
            )

            if session_count == 0:
                level = 0
            elif session_count == 1:
                level = 1
            elif session_count <= 3:
                level = 2
            else:
                level = 3

            count_label = (
                f"{session_count} session"
                if session_count == 1
                else f"{session_count} sessions"
            )

            calendar_cells.append(
                f"""
                <div class="month-day level-{level}"
                     title="{current_date.strftime('%d %B %Y')} · {count_label}">
                    <span class="day-number">{day}</span>
                    <span class="day-count">
                        {session_count if session_count else ""}
                    </span>
                </div>
                """
            )

        # Empty cells after the last day
        while len(calendar_cells) % 7 != 0:
            calendar_cells.append(
                '<div class="month-empty"></div>'
            )

        calendar_html = f"""
        <style>
            .month-calendar {{
                background: {calendar_bg};
                border: 1px solid {border_color};
                border-radius: 18px;
                padding: 20px;
                color: {text_color};
            }}

            .month-title {{
                font-size: 22px;
                font-weight: 750;
                margin-bottom: 20px;
            }}

            .month-grid {{
                display: grid;
                grid-template-columns: repeat(7, minmax(0, 1fr));
                gap: 8px;
            }}

            .month-weekday {{
                text-align: center;
                color: #a78bfa;
                font-size: 12px;
                font-weight: 700;
                padding-bottom: 10px;
            }}

            .month-day {{
                min-height: 65px;
                border-radius: 10px;
                padding: 9px 5px;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                transition: transform .15s ease;
                cursor: default;
            }}

            .month-day:hover {{
                transform: translateY(-3px);
                outline: 1px solid #c4b5fd;
            }}

            .month-empty {{
                min-height: 65px;
            }}

            .level-0 {{
                background: {empty_bg};
            }}

            .level-1 {{
                background: #4c1d95;
                color: white;
            }}

            .level-2 {{
                background: #7c3aed;
                color: white;
            }}

            .level-3 {{
                background: #a78bfa;
                color: #171326;
            }}

            .day-number {{
                font-size: 15px;
                font-weight: 700;
            }}

            .day-count {{
                font-size: 10px;
                margin-top: 4px;
                opacity: .9;
            }}

            @media(max-width:600px) {{
                .month-calendar {{
                    padding: 10px;
                }}
                .month-grid {{
                    gap: 4px;
                }}
                .month-day, .month-empty {{
                    min-height: 48px;
                }}
                .day-number {{
                    font-size: 12px;
                }}
            }}
        </style>

        <div class="month-calendar">
            <div class="month-title">📅 {month_name}</div>

            <div class="month-grid">
                <div class="month-weekday">MON</div>
                <div class="month-weekday">TUE</div>
                <div class="month-weekday">WED</div>
                <div class="month-weekday">THU</div>
                <div class="month-weekday">FRI</div>
                <div class="month-weekday">SAT</div>
                <div class="month-weekday">SUN</div>

                {''.join(calendar_cells)}
            </div>
        </div>
        """

        st.components.v1.html(
            calendar_html,
            height=480,
            scrolling=False,
        )

        # Monthly statistics
        month_dates = activity_dates[
            (activity_dates >= month_start)
            & (activity_dates < next_month)
        ]

        active_days = month_dates.dt.date.nunique()
        monthly_sessions = len(month_dates)

        st.markdown("")

        calendar_col1, calendar_col2 = st.columns(2)

        calendar_col1.metric(
            "Active Study Days",
            active_days,
        )

        calendar_col2.metric(
            "Sessions This Month",
            monthly_sessions,
        )

        st.caption(
            "Purple intensity represents the number of recorded "
            "sessions. It does not measure attention or productivity."
        )

    else:
        st.info("Your session report does not contain a Date column.")

    st.divider()

   
    # --------------------------------------------------------
    # SESSION VISIBILITY TREND
    # --------------------------------------------------------

    st.markdown("### 📈 Session Visibility Trend")

    if (
        "Phone Visible Percentage" in report_df.columns
        and not report_df.empty
    ):
        trend_df = report_df.copy()

        if "Video Name" in trend_df.columns:
            trend_df["Session"] = (
                trend_df["Video Name"]
                .fillna("Study Session")
                .astype(str)
            )
        else:
            trend_df["Session"] = [
                f"Session {i + 1}"
                for i in range(len(trend_df))
            ]

        if "Session Time" in trend_df.columns:
            trend_df["Session"] = (
                trend_df["Session"]
                + " ("
                + trend_df["Session Time"].fillna("").astype(str)
                + ")"
            )

        trend_df["Session Number"] = range(
            1,
            len(trend_df) + 1,
        )

        trend_fig = px.line(
            trend_df,
            x="Session Number",
            y="Phone Visible Percentage",
            markers=True,
            hover_name="Session",
            title="Phone Visibility Across Sessions",
        )

        trend_fig.update_traces(
            line=dict(color="#8b5cf6", width=3),
            marker=dict(size=9),
        )

        trend_fig.update_layout(
            template=(
                "plotly_dark"
                if is_dark_mode
                else "plotly_white"
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Study Session",
            yaxis_title="Phone Visibility (%)",
            yaxis_range=[0, 100],
            margin=dict(l=10, r=10, t=55, b=10),
        )

        st.plotly_chart(
            trend_fig,
            use_container_width=True,
        )

    else:
        st.info("No session trend data is available.")

    st.divider()


    # --------------------------------------------------------
    # VIDEO COMPARISON
    # --------------------------------------------------------

    st.markdown("### 🎬 Study Video Comparison")

    if (
        "Video Name" in report_df.columns
        and "Phone Visible Percentage" in report_df.columns
    ):
        comparison_df = report_df[
            ["Video Name", "Phone Visible Percentage"]
        ].copy()

        comparison_df = comparison_df.dropna(
            subset=["Phone Visible Percentage"]
        )

        if not comparison_df.empty:

            comparison_df["Video Name"] = (
                comparison_df["Video Name"]
                .fillna("Unknown Video")
                .astype(str)
            )

            comparison_fig = px.bar(
                comparison_df,
                x="Video Name",
                y="Phone Visible Percentage",
                color="Phone Visible Percentage",
                color_continuous_scale=[
                    "#312e81",
                    "#7c3aed",
                    "#c084fc",
                ],
                title="Phone Visibility by Study Video",
            )

            comparison_fig.update_layout(
                template=(
                    "plotly_dark"
                    if is_dark_mode
                    else "plotly_white"
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Study Video",
                yaxis_title="Phone Visibility (%)",
                yaxis_range=[0, 100],
                coloraxis_showscale=False,
                margin=dict(l=10, r=10, t=55, b=10),
            )

            st.plotly_chart(
                comparison_fig,
                use_container_width=True,
            )

        else:
            st.info("No video comparison data is available.")

    else:
        st.info("The report is missing required comparison columns.")

    st.divider()

   
    # --------------------------------------------------------
    # STUDY PROGRESS REPORT — PREMIUM DESIGN
    # --------------------------------------------------------

    st.markdown("### 📊 Study Progress Report")
    st.caption(
        "Track study-session statistics across weeks and months."
    )

    if not report_df.empty and "Date" in report_df.columns:

        progress_df = report_df.copy()

        progress_df["_Date"] = pd.to_datetime(
            progress_df["Date"],
            errors="coerce",
        )

        progress_df = progress_df.dropna(subset=["_Date"])

        if not progress_df.empty:

            progress_df["_Visibility"] = pd.to_numeric(
                progress_df.get(
                    "Phone Visible Percentage",
                    pd.Series(index=progress_df.index, dtype=float),
                ),
                errors="coerce",
            )

            progress_df["_Focus"] = pd.to_numeric(
                progress_df.get(
                    "Focus Score",
                    pd.Series(index=progress_df.index, dtype=float),
                ),
                errors="coerce",
            )

            progress_df["_Detections"] = pd.to_numeric(
                progress_df.get(
                    "Total Phone Detections",
                    pd.Series(index=progress_df.index, dtype=float),
                ),
                errors="coerce",
            ).fillna(0)

            report_view = st.radio(
                "Choose Progress Report",
                ["Weekly", "Monthly"],
                horizontal=True,
                key="study_progress_view",
            )

            if report_view == "Weekly":
                progress_df["_Period"] = (
                    progress_df["_Date"]
                    .dt.to_period("W")
                    .dt.start_time
                )
            else:
                progress_df["_Period"] = (
                    progress_df["_Date"]
                    .dt.to_period("M")
                    .dt.start_time
                )

            progress_summary = (
                progress_df.groupby("_Period")
                .agg(
                    Sessions=("Date", "count"),
                    Average_Visibility=("_Visibility", "mean"),
                    Average_Focus=("_Focus", "mean"),
                    Total_Detections=("_Detections", "sum"),
                )
                .reset_index()
                .sort_values("_Period")
            )

            latest_period = progress_summary.iloc[-1]

            st.markdown(
                f"#### 📅 Latest {report_view} Summary"
            )

            # Progress summary metrics
            progress_col1, progress_col2, progress_col3, progress_col4 = (
                st.columns(4)
            )

            progress_col1.metric(
                "📚 Sessions Completed",
                int(latest_period["Sessions"]),
            )

            progress_col2.metric(
                "📱 Average Phone Visibility",
                (
                    f'{latest_period["Average_Visibility"]:.1f}%'
                    if pd.notna(latest_period["Average_Visibility"])
                    else "N/A"
                ),
            )

            progress_col3.metric(
                "🎯 Average Focus Score",
                (
                    f'{latest_period["Average_Focus"]:.1f}%'
                    if pd.notna(latest_period["Average_Focus"])
                    else "N/A"
                ),
            )

            progress_col4.metric(
                "🔍 Phone Detections",
                int(latest_period["Total_Detections"]),
            )

            st.markdown("---")

            # Prepare period labels
            progress_summary["Period"] = (
                progress_summary["_Period"].dt.strftime(
                    "%d %b %Y"
                    if report_view == "Weekly"
                    else "%b %Y"
                )
            )

            st.markdown("#### 📈 Progress Over Time")

            chart_data = progress_summary.melt(
                id_vars=["Period"],
                value_vars=[
                    "Average_Visibility",
                    "Average_Focus",
                ],
                var_name="Metric",
                value_name="Percentage",
            )

            chart_data["Metric"] = chart_data["Metric"].map({
                "Average_Visibility": "Phone Visibility",
                "Average_Focus": "Focus Score",
            })

            chart_data = chart_data.dropna(
                subset=["Percentage"]
            )

            if not chart_data.empty:

                progress_fig = px.line(
                    chart_data,
                    x="Period",
                    y="Percentage",
                    color="Metric",
                    markers=True,
                    color_discrete_map={
                        "Phone Visibility": "#a78bfa",
                        "Focus Score": "#22c55e",
                    },
                    title=f"{report_view} Study Progress",
                )

                progress_fig.update_traces(
                    line=dict(width=3),
                    marker=dict(size=9),
                    hovertemplate=(
                        "<b>%{fullData.name}</b><br>"
                        "%{x}<br>"
                        "%{y:.1f}%<extra></extra>"
                    ),
                )

                progress_fig.update_layout(
                    template=(
                        "plotly_dark"
                        if is_dark_mode
                        else "plotly_white"
                    ),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(size=13),
                    title=dict(
                        x=0.5,
                        xanchor="center",
                        font=dict(size=19),
                    ),
                    yaxis_title="Percentage (%)",
                    yaxis_range=[0, 100],
                    xaxis_title=report_view,
                    legend_title_text="",
                    hovermode="x unified",
                    margin=dict(l=15, r=15, t=65, b=20),
                )

                progress_fig.update_xaxes(
                    showgrid=False,
                    zeroline=False,
                )

                progress_fig.update_yaxes(
                    showgrid=True,
                    gridcolor=(
                        "rgba(255,255,255,0.10)"
                        if is_dark_mode
                        else "rgba(0,0,0,0.08)"
                    ),
                    zeroline=False,
                )

                st.plotly_chart(
                    progress_fig,
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

            else:
                st.info(
                    "Focus and phone visibility data are not "
                    "available for the selected periods yet."
                )

            st.markdown("---")
            st.markdown("#### 📋 Progress History")
            st.caption(
                "A breakdown of your recorded study sessions "
                "for each selected period."
            )

            progress_display = progress_summary[
                [
                    "Period",
                    "Sessions",
                    "Average_Visibility",
                    "Average_Focus",
                    "Total_Detections",
                ]
            ].rename(columns={
                "Average_Visibility": "Average Phone Visibility (%)",
                "Average_Focus": "Average Focus Score (%)",
                "Total_Detections": "Total Phone Detections",
            })

            progress_display["Average Phone Visibility (%)"] = (
                progress_display["Average Phone Visibility (%)"].round(1)
            )

            progress_display["Average Focus Score (%)"] = (
                progress_display["Average Focus Score (%)"].round(1)
            )

            st.dataframe(
                progress_display,
                use_container_width=True,
                hide_index=True,
            )

        else:
            st.info("No valid session dates are available.")

    else:
        st.info(
            "Complete at least one study session to generate "
            "your progress report."
        )

    st.divider()

    # --------------------------------------------------------
    # SESSION HISTORY AND ADVANCED FILTERS
    # --------------------------------------------------------

    st.markdown("### 📋 Recent Session History")
    st.caption("Search, filter, sort, and download your study sessions.")

    history_df = report_df.copy()

    # Prepare date and visibility columns
    if "Date" in history_df.columns:
        history_df["_Date"] = pd.to_datetime(
            history_df["Date"],
            errors="coerce"
        )
    else:
        history_df["_Date"] = pd.NaT

    if "Phone Visible Percentage" in history_df.columns:
        history_df["_Visibility"] = pd.to_numeric(
            history_df["Phone Visible Percentage"],
            errors="coerce"
        )
    else:
        history_df["_Visibility"] = pd.NA

    # Default filter values
    valid_dates = history_df["_Date"].dropna()

    if not valid_dates.empty:
        default_date_range = (
            valid_dates.min().date(),
            valid_dates.max().date()
        )
    else:
        default_date_range = None

    # Reset filters
    if st.button("🔄 Reset All Filters", key="reset_history_filters"):
        st.session_state["history_video_search"] = ""
        st.session_state["history_visibility_range"] = (0.0, 100.0)
        st.session_state["history_sort_order"] = "Newest First"

        if default_date_range is not None:
            st.session_state["history_date_range"] = default_date_range

    # Filter controls
    st.markdown("#### 🔎 Filter Sessions")

    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:
        if default_date_range is not None:
            selected_dates = st.date_input(
                "📅 Select Date Range",
                value=default_date_range,
                key="history_date_range"
            )
        else:
            selected_dates = None
            st.info("No valid session dates are available.")

        if "Video Name" in history_df.columns:
            search_term = st.text_input(
                "🎬 Search Video Name",
                placeholder="Enter a video name...",
                key="history_video_search"
            )
        else:
            search_term = ""

    with filter_col2:
        visibility_range = st.slider(
            "📱 Phone Visibility Range (%)",
            min_value=0.0,
            max_value=100.0,
            value=(0.0, 100.0),
            step=1.0,
            key="history_visibility_range"
        )

        sort_order = st.selectbox(
            "↕️ Sort Sessions",
            [
                "Newest First",
                "Oldest First",
                "Highest Visibility",
                "Lowest Visibility"
            ],
            key="history_sort_order"
        )

    # Apply date filter
    if (
        selected_dates is not None
        and isinstance(selected_dates, (tuple, list))
        and len(selected_dates) == 2
    ):
        start_date, end_date = selected_dates

        history_df = history_df[
            history_df["_Date"].dt.date.between(
                start_date,
                end_date
            )
        ]

    # Apply video search
    if search_term and "Video Name" in history_df.columns:
        history_df = history_df[
            history_df["Video Name"].fillna("").astype(str).str.contains(
                search_term,
                case=False,
                regex=False
            )
        ]

    # Apply visibility filter
    if "_Visibility" in history_df.columns:
        history_df = history_df[
            history_df["_Visibility"].between(
                visibility_range[0],
                visibility_range[1]
            )
        ]

    # Apply sorting
    if sort_order == "Newest First":
        history_df = history_df.sort_values(
            "_Date",
            ascending=False,
            na_position="last"
        )

    elif sort_order == "Oldest First":
        history_df = history_df.sort_values(
            "_Date",
            ascending=True,
            na_position="last"
        )

    elif sort_order == "Highest Visibility":
        history_df = history_df.sort_values(
            "_Visibility",
            ascending=False,
            na_position="last"
        )

    elif sort_order == "Lowest Visibility":
        history_df = history_df.sort_values(
            "_Visibility",
            ascending=True,
            na_position="last"
        )

    # --------------------------------------------------------
    # SIDE-BY-SIDE SESSION COMPARISON
    # --------------------------------------------------------

    st.divider()
    st.markdown("### ⚖️ Side-by-Side Session Comparison")
    st.caption(
        "Select two saved sessions to compare their detection statistics."
    )

    if len(history_df) >= 2:

        comparison_options = {}

        for session_index, session_row in history_df.iterrows():
            video = str(session_row.get("Video Name", "Study Session"))
            date = str(session_row.get("Date", "Unknown date"))
            session_time = str(session_row.get("Session Time", ""))
            
            label = (
                f"{video} | {date} | {session_time} "
                f"| Record {session_index + 1}"
            )

            comparison_options[label] = session_index

        comparison_labels = list(comparison_options.keys())

        compare_col1, compare_col2 = st.columns(2)

        with compare_col1:
            first_session_label = st.selectbox(
                "🟣 Select First Session",
                comparison_labels,
                index=0,
                key="comparison_session_one",
            )

        second_options = [
            label for label in comparison_labels
            if label != first_session_label
        ]

        with compare_col2:
            second_session_label = st.selectbox(
                "🔵 Select Second Session",
                second_options,
                index=0,
                key="comparison_session_two",
            )

        first_session = history_df.loc[
            comparison_options[first_session_label]
        ]

        second_session = history_df.loc[
            comparison_options[second_session_label]
        ]

        comparison_data = []

        metrics_to_compare = [
            ("Phone Visibility (%)", "Phone Visible Percentage"),
            ("Frames Processed", "Frames Processed"),
            ("Total Phone Detections", "Total Phone Detections"),
            ("Focus Score (%)", "Focus Score"),
        ]

        for display_name, column_name in metrics_to_compare:
            first_value = pd.to_numeric(
                first_session.get(column_name, pd.NA),
                errors="coerce",
            )

            second_value = pd.to_numeric(
                second_session.get(column_name, pd.NA),
                errors="coerce",
            )

            comparison_data.append({
                "Metric": display_name,
                "First Session": (
                    round(float(first_value), 2)
                    if pd.notna(first_value) else "Not available"
                ),
                "Second Session": (
                    round(float(second_value), 2)
                    if pd.notna(second_value) else "Not available"
                ),
            })

        comparison_table = pd.DataFrame(comparison_data)

        st.markdown("#### 📊 Comparison Results")

        st.dataframe(
            comparison_table,
            use_container_width=True,
            hide_index=True,
        )

        chart_data = []

        for session_label, session_row in [
            (first_session_label, first_session),
            (second_session_label, second_session),
        ]:
            for metric_name, column_name in [
                ("Phone Visibility", "Phone Visible Percentage"),
                ("Focus Score", "Focus Score"),
            ]:
                value = pd.to_numeric(
                    session_row.get(column_name, pd.NA),
                    errors="coerce",
                )

                if pd.notna(value):
                    chart_data.append({
                        "Metric": metric_name,
                        "Percentage": float(value),
                        "Session": session_label,
                    })

        if chart_data:
            comparison_fig = px.bar(
                pd.DataFrame(chart_data),
                x="Metric",
                y="Percentage",
                color="Session",
                barmode="group",
                title="Session Performance Comparison",
                text_auto=".1f",
            )

            comparison_fig.update_layout(
                template=(
                    "plotly_dark"
                    if is_dark_mode
                    else "plotly_white"
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis_title="Percentage (%)",
                yaxis_range=[0, 100],
                margin=dict(l=10, r=10, t=55, b=10),
                legend_title="Study Sessions",
            )

            st.plotly_chart(
                comparison_fig,
                use_container_width=True,
            )

        st.caption(
            "Older sessions without a saved focus score will show "
            "'Not available'. Phone visibility is an indicator only, "
            "not a validated measurement of attention."
        )

    else:
        st.info(
            "At least two saved sessions are required for comparison."
        )


    # Remove temporary columns
    display_history = history_df.drop(
        columns=["_Date", "_Visibility"],
        errors="ignore"
    )

    # Display filtered results
    st.markdown("#### 📊 Filtered Results")

    result_col1, result_col2 = st.columns(2)

    result_col1.metric(
        "Matching Sessions",
        len(display_history)
    )

    result_col2.metric(
        "Total Sessions",
        len(report_df)
    )

    if not display_history.empty:
        st.dataframe(
            display_history,
            use_container_width=True,
            hide_index=True
        )

        filtered_csv = display_history.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download Filtered Sessions (CSV)",
            data=filtered_csv,
            file_name="filtered_session_report.csv",
            mime="text/csv",
            key="download_filtered_sessions"
        )

    else:
        st.info("No sessions match your selected filters.")

    # Download complete report
    csv_data = report_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download Complete Session Report (CSV)",
        data=csv_data,
        file_name="session_summary.csv",
        mime="text/csv",
        key="download_csv_summary_main"
    )


st.caption(
    "Class 12 AI Project | Phone visibility is an indicator only, "
    "not a validated measurement of attention."
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
# Detection controls
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

    st.subheader("💡 Session Insight")
    insight = generate_session_insight(saved_analysis)
    st.info(insight["risk_text"])
    st.write(insight["summary"])
    st.write(insight["recommendation"])

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


