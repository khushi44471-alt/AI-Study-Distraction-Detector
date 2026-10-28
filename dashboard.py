import tempfile
from datetime import datetime
from pathlib import Path
import csv
import cv2
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="AI Study Distraction Detector",
    page_icon="📱",
    layout="wide",
)

import time

# =====================================
# STUDY SESSION TIMER
# =====================================

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

from ultralytics import YOLO
import threading
import pyttsx3
import io
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as PDFImage,
)




# =====================================
# CUSTOM PHONE ALERT SETTINGS
# =====================================

import base64
import json
import threading
import streamlit.components.v1 as components

ALERT_COOLDOWN = 30
REQUIRED_CONSECUTIVE_DETECTIONS = 3

# Choose the alert sound.
alert_choice = st.selectbox(
    "🔔 Choose Phone Alert Sound",
    [
        "1. Female Spoken Warning",
        "2. Gentle Notification Chime",
        "3. Rising Emergency Siren",
        "4. Retro 8-Bit Game Melody",
        "5. Custom MP3/WAV File",
    ],
    key="phone_alert_choice",
)

# Adjustable volume: 0% to 100%.
alert_volume = st.slider(
    "🔊 Alert Volume",
    min_value=0,
    max_value=100,
    value=70,
    step=5,
    help="Controls the alert volume where supported.",
    key="phone_alert_volume",
)

volume_level = alert_volume / 100.0

custom_audio = None
custom_audio_type = None

# Upload a custom sound if option 5 is selected.
if alert_choice == "5. Custom MP3/WAV File":
    uploaded_audio = st.file_uploader(
        "Upload your alert sound",
        type=["mp3", "wav"],
        key="custom_phone_alert_audio",
    )

    if uploaded_audio is not None:
        custom_audio = base64.b64encode(
            uploaded_audio.getvalue()
        ).decode("utf-8")

        custom_audio_type = (
            uploaded_audio.type or "audio/mpeg"
        )


def speak_phone_warning(volume=0.70):
    """Speak the phone warning using an available voice."""
    try:
        import pyttsx3

        engine = pyttsx3.init()
        engine.setProperty("volume", max(0.0, min(1.0, volume)))

        voices = engine.getProperty("voices")
        female_voice = None

        for voice in voices:
            info = (
                str(voice.name) + " " + str(voice.id)
            ).lower()

            if any(name in info for name in [
                "female", "zira", "hazel", "susan",
                "samantha", "aria", "jenny",
            ]):
                female_voice = voice
                break

        if female_voice is not None:
            engine.setProperty("voice", female_voice.id)
        elif voices:
            engine.setProperty("voice", voices[0].id)

        engine.say(
            "Attention! Attention! Phone detected! "
            "Please put your phone away and focus on studying!"
        )
        engine.runAndWait()
        engine.stop()

    except Exception as error:
        print(f"Speech warning could not play: {error}")


def show_phone_alert(preview=False):
    """Display a warning and play the selected alert."""

    if not preview:
        st.warning(
            "📱 Phone detected repeatedly! "
            "Please check whether you need your phone for studying."
        )

    # Option 1: Spoken warning.
    if alert_choice == "1. Female Spoken Warning":
        threading.Thread(
            target=speak_phone_warning,
            args=(volume_level,),
            daemon=True,
        ).start()

        if preview:
            st.info("Female spoken warning preview started.")
        return

    # Option 5: Uploaded MP3 or WAV.
    if alert_choice == "5. Custom MP3/WAV File":
        if not custom_audio:
            st.info(
                "Upload an MP3 or WAV file before testing "
                "the custom alert."
            )
            return

        audio_html = f"""
        <audio controls autoplay style="width:100%;">
            <source
                src="data:{custom_audio_type};base64,{custom_audio}"
                type="{custom_audio_type}">
            Your browser does not support this audio file.
        </audio>
        <script>
        const player = document.querySelector("audio");
        player.volume = {volume_level};
        player.play().catch(() => {{
            console.log("Press Play on the audio player.");
        }});
        </script>
        """

        components.html(audio_html, height=75)
        return

    # Options 2–4: browser-generated sounds.
    if alert_choice == "2. Gentle Notification Chime":
        mode = "chime"
    elif alert_choice == "3. Rising Emergency Siren":
        mode = "siren"
    else:
        mode = "retro"

    config = json.dumps({
        "mode": mode,
        "volume": volume_level,
    })

    sound_html = """
    <div style="font-family:Arial,sans-serif;">
      <button id="playPreview"
        style="padding:9px 15px; cursor:pointer;">
        ▶ Play Sound Preview
      </button>
      <span id="soundStatus" style="margin-left:8px;"></span>
    </div>

    <script>
    const config = CONFIG_PLACEHOLDER;
    const button = document.getElementById("playPreview");
    const status = document.getElementById("soundStatus");

    button.addEventListener("click", async function() {
      try {
        const ctx = new (
          window.AudioContext || window.webkitAudioContext
        )();

        await ctx.resume();

        function tone(freq, start, duration, peak, type) {
          const osc = ctx.createOscillator();
          const gain = ctx.createGain();

          osc.type = type || "sine";
          osc.frequency.setValueAtTime(freq, start);

          gain.gain.setValueAtTime(0.001, start);
          gain.gain.linearRampToValueAtTime(
            peak * config.volume, start + 0.025
          );
          gain.gain.setValueAtTime(
            peak * config.volume, start + duration * 0.70
          );
          gain.gain.linearRampToValueAtTime(
            0.001, start + duration
          );

          osc.connect(gain);
          gain.connect(ctx.destination);
          osc.start(start);
          osc.stop(start + duration + 0.02);
        }

        const now = ctx.currentTime;

        if (config.mode === "chime") {
          tone(659.25, now, 0.22, 0.22);
          tone(783.99, now + 0.24, 0.30, 0.22);

        } else if (config.mode === "siren") {
          const osc = ctx.createOscillator();
          const gain = ctx.createGain();

          osc.type = "sawtooth";
          osc.frequency.setValueAtTime(520, now);
          osc.frequency.linearRampToValueAtTime(1050, now + 0.65);
          osc.frequency.linearRampToValueAtTime(520, now + 1.30);
          osc.frequency.linearRampToValueAtTime(1050, now + 1.95);
          osc.frequency.linearRampToValueAtTime(520, now + 2.60);

          gain.gain.setValueAtTime(0.001, now);
          gain.gain.linearRampToValueAtTime(
            0.10 * config.volume, now + 0.10
          );
          gain.gain.linearRampToValueAtTime(
            0.001, now + 2.70
          );

          osc.connect(gain);
          gain.connect(ctx.destination);
          osc.start(now);
          osc.stop(now + 2.75);

        } else {
          const notes = [
            [523.25, 0.00],
            [659.25, 0.14],
            [783.99, 0.28],
            [1046.50, 0.42],
            [783.99, 0.60],
            [1046.50, 0.78]
          ];

          for (const note of notes) {
            tone(note[0], now + note[1], 0.11, 0.20, "square");
          }
        }

        status.textContent = "Preview playing";
        setTimeout(() => {
          status.textContent = "";
          ctx.close();
        }, 3200);

      } catch (error) {
        status.textContent = "Sound blocked by browser.";
        console.log("Preview unavailable:", error);
      }
    });
    </script>
    """

    sound_html = sound_html.replace(
        "CONFIG_PLACEHOLDER", config
    )
    components.html(sound_html, height=55)


# Test the currently selected alert without analyzing a video.
if st.button("🔊 Test Selected Sound", key="test_phone_alert"):
    show_phone_alert(preview=True)








# ============================================================
# AI STUDY DISTRACTION DETECTOR
# ============================================================


st.title("📱 AI Study Distraction Detector")

st.write(
    "AI-powered phone-visibility monitoring during recorded study sessions."
)

st.caption(
    "Phone visibility is only an indicator, not proof of distraction. "
    "A phone may be used for studying, and the detector may make mistakes."
)


# ============================================================
# FILE PATHS AND REPORT SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
REPORT_DIR = BASE_DIR / "reports"
REPORT_FILE = REPORT_DIR / "session_summary.csv"
MODEL_FILE = BASE_DIR / "yolo11n.pt"

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

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# REPORT FUNCTIONS
# ============================================================

def load_report():
    """Load existing session reports from CSV."""

    if not REPORT_FILE.exists() or REPORT_FILE.stat().st_size == 0:
        report_df = pd.DataFrame(columns=CSV_COLUMNS)
    else:
        try:
            report_df = pd.read_csv(REPORT_FILE)
        except pd.errors.EmptyDataError:
            report_df = pd.DataFrame(columns=CSV_COLUMNS)

    # Ensure all required columns exist.
    for column in CSV_COLUMNS:
        if column not in report_df.columns:
            report_df[column] = pd.NA

    # Convert report values to numeric types where appropriate.
    for column in NUMERIC_COLUMNS:
        report_df[column] = pd.to_numeric(
            report_df[column],
            errors="coerce",
        )

    return report_df[CSV_COLUMNS]


def create_pdf_report(analysis):
    """Create a downloadable PDF report with statistics and a chart."""

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

    safe_video_name = (
        video_name.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

    story.append(
        Paragraph("AI Study Distraction Detector", styles["Title"])
    )
    story.append(
        Paragraph("Study Session Report", styles["Heading2"])
    )
    story.append(Spacer(1, 12))

    story.append(
        Paragraph(
            f"Report generated: {datetime.now().strftime('%d %B %Y, %I:%M %p')}",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 12))

    report_data = [
        ["Statistic", "Result"],
        ["Video", safe_video_name],
        ["Video Resolution", resolution],
        ["Frames Analyzed", f"{frames:,}"],
        ["Frames With Phone", f"{phone_frames:,}"],
        ["Total Phone Detections", f"{detections:,}"],
        ["Phone Visible Percentage", f"{percentage:.2f}%"],
        ["Phone-Free Score", f"{focus_score:.0f}/100"],
    ]

    table = Table(report_data, colWidths=[230, 230], repeatRows=1)
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
    story.append(
        Paragraph("Phone Visibility Chart", styles["Heading2"])
    )

    chart_buffer = io.BytesIO()
    figure, axis = plt.subplots(figsize=(6, 3.2))

    axis.bar(
        ["Phone Visible", "Phone Not Detected"],
        [percentage, 100.0 - percentage],
    )
    axis.set_ylabel("Percentage (%)")
    axis.set_ylim(0, 100)
    axis.set_title("Analyzed Video Frames")
    axis.grid(axis="y", alpha=0.25)

    figure.tight_layout()
    figure.savefig(
        chart_buffer,
        format="png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)
    chart_buffer.seek(0)

    story.append(
        PDFImage(chart_buffer, width=6 * inch, height=3.2 * inch)
    )
    story.append(Spacer(1, 12))

    story.append(
        Paragraph(
            "<b>Interpretation:</b> The Phone-Free Score is calculated "
            "as 100 minus the phone-visible percentage. Phone visibility "
            "does not prove distraction, and this score is not a validated "
            "measure of concentration or study quality.",
            styles["BodyText"],
        )
    )

    doc.build(story)
    pdf_buffer.seek(0)

    return pdf_buffer.getvalue()



def save_session(
    video_name,
    resolution,
    frames_processed,
    frames_with_phone,
    percentage,
    detections,
):
    """Append one completed analysis to the CSV report."""

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    file_has_content = (
        REPORT_FILE.exists()
        and REPORT_FILE.stat().st_size > 0
    )

    with REPORT_FILE.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as report_file:

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


# ============================================================
# LOAD YOLO MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_yolo_model(model_path):
    """Load and cache the YOLO model."""

    return YOLO(model_path)

# ============================================================
# VIDEO ANALYSIS
# ============================================================


def analyze_video(uploaded_file):
    """Sample every fifth frame and report phone detections and video-quality indicators."""
    temp_path = None
    capture = None
    if not MODEL_FILE.exists():
        st.error("Missing yolo11n.pt. Place it beside dashboard.py.")
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
        duration = total_frames / fps if fps > 0 and total_frames > 0 else None
        quality_warnings = []
        if min(width, height) < 480:
            quality_warnings.append("Low resolution: small or distant phones may be missed.")
        if fps <= 0:
            quality_warnings.append("Frame rate unavailable: timeline timestamps may be missing.")
        elif fps < 10:
            quality_warnings.append("Low frame rate: short phone appearances may be missed.")
        if duration is not None and duration < 5:
            quality_warnings.append("Very short video (under 5 seconds): results may not represent a session.")

        with st.spinner("Loading AI model..."):
            model = load_yolo_model(str(MODEL_FILE))

        sampled_frames = frames_with_phone = total_detections = 0

        # Desk Clutter Counter
        object_counts = {}
        consecutive_phone_frames = 0
        last_alert_time = 0.0
        frame_index = 0
        timeline = []
        confidence_values = []
        blur_values = []
        last_annotated_rgb = None
        best_phone_frame_rgb = None
        best_phone_confidence = -1.0
        progress = st.progress(0.0)
        status = st.empty()
        # LIVE AI DETECTION DISPLAY
        st.subheader("🔴 Live AI Detection")

        live_video = st.empty()

        live_col1, live_col2, live_col3, live_col4 = st.columns(4)

        live_frame_metric = live_col1.empty()
        live_phone_frames_metric = live_col2.empty()
        live_detections_metric = live_col3.empty()
        live_visibility_metric = live_col4.empty()
        while True:
            success, frame = capture.read()
            if not success:
                break
            if frame_index % 5 == 0:
                sampled_frames += 1
                # Laplacian variance is a rough blur indicator, not a definitive quality test.
                if len(blur_values) < 30:
                    small = cv2.resize(frame, (min(width, 640), min(height, 360)))
                    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                    blur_values.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))

                prediction = model.predict(source=frame, imgsz=640, conf=0.25, verbose=False)[0]
                
                annotated = frame.copy()
                phones_in_frame = 0
                frame_best_confidence = -1.0

                if prediction.boxes is not None:
                    for box in prediction.boxes:
                        class_id = int(box.cls[0].item())

                        if isinstance(model.names, dict):
                            class_name = model.names.get(
                                class_id, "Unknown"
                            )
                        else:
                            class_name = model.names[class_id]

                       
                        class_name = str(class_name).strip()

                        # Count every object class recognized by YOLO.
                        object_counts[class_name] = (
                            object_counts.get(class_name, 0) + 1
                        )

                        confidence = float(box.conf[0].item())

                        x1, y1, x2, y2 = [
                            int(v) for v in box.xyxy[0].tolist()
                        ]

                        is_phone = class_name.lower() in (
                            "cell phone",
                            "mobile phone",
                        )

                        if is_phone:
                            box_color = (0, 210, 0)
                            label_name = "Phone"

                            confidence_values.append(confidence)
                            phones_in_frame += 1
                            total_detections += 1

                            frame_best_confidence = max(
                                frame_best_confidence,
                                confidence,
                            )
                        else:
                            box_color = (255, 120, 0)
                            label_name = class_name.title()

                        # Draw a box around the detected object.
                        cv2.rectangle(
                            annotated,
                            (x1, y1),
                            (x2, y2),
                            box_color,
                            3,
                        )

                        # Show the object name and confidence.
                        label = f"{label_name} {confidence:.0%}"

                        cv2.putText(
                            annotated,
                            label,
                            (x1, max(22, y1 - 9)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            box_color,
                            2,
                            cv2.LINE_AA,
                        )

                phone_found = phones_in_frame > 0
                if phone_found:
                    frames_with_phone += 1
                    consecutive_phone_frames += 1
                else:
                    consecutive_phone_frames = 0

                elapsed_seconds = frame_index / fps if fps > 0 else None
                timeline.append({
                    "frame": frame_index,
                    "seconds": elapsed_seconds,
                    "phone_visible": int(phone_found),
                    "phone_detections": phones_in_frame,
                    "max_confidence": frame_best_confidence if phone_found else None,
                })

                if consecutive_phone_frames >= REQUIRED_CONSECUTIVE_DETECTIONS:
                    current_time = time.monotonic()
                    if current_time - last_alert_time >= ALERT_COOLDOWN:
                        show_phone_alert()
                        last_alert_time = current_time
                        consecutive_phone_frames = 0

                
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                last_annotated_rgb = annotated_rgb

                # Update the live annotated video frame.
                if live_video is not None:
                    live_video.image(
                        annotated_rgb,
                        caption=f"Analysing video frame {frame_index + 1}",
                        use_container_width=True,
                    )

                # Update live phone-detection statistics.
                current_visibility = (
                    frames_with_phone / sampled_frames * 100
                    if sampled_frames > 0
                    else 0.0
                )

                live_frame_metric.metric(
                    "Frames Analysed",
                    f"{sampled_frames:,}",
                )
                live_phone_frames_metric.metric(
                    "Frames With Phone",
                    f"{frames_with_phone:,}",
                )
                live_detections_metric.metric(
                    "Phone Detections",
                    f"{total_detections:,}",
                )
                live_visibility_metric.metric(
                    "Phone Visible",
                    f"{current_visibility:.1f}%",
                )

                if phone_found and frame_best_confidence > best_phone_confidence:
                    best_phone_confidence = frame_best_confidence
                    best_phone_frame_rgb = annotated_rgb.copy()

                status.write(f"Analyzed {sampled_frames} sampled frames. Phone detections: {total_detections}")
                if total_frames > 0:
                    progress.progress(min((frame_index + 1) / total_frames, 1.0))
            frame_index += 1

        if sampled_frames == 0:
            st.error("No readable frames were found in this video.")
            return None
        if sampled_frames < 5:
            quality_warnings.append("Too few sampled frames for a meaningful comparison.")
        mean_blur = sum(blur_values) / len(blur_values) if blur_values else None
        if mean_blur is not None and mean_blur < 35:
            quality_warnings.append("Possible blur or low image detail detected; verify the video visually.")
        if width >= 3840 or height >= 2160:
            quality_warnings.append("High-resolution video: analysis is resized to 640 pixels for YOLO, so tiny phones may be missed.")
        progress.progress(1.0)
        status.write("Video analysis completed.")
        percentage = frames_with_phone / sampled_frames * 100
        return {
            "video_name": uploaded_file.name,
            "resolution": f"{width}x{height}",
            "frames_processed": sampled_frames,
            "frames_with_phone": frames_with_phone,
            "percentage": round(percentage, 2),
            "detections": total_detections,
            "annotated_frame_rgb": best_phone_frame_rgb if best_phone_frame_rgb is not None else last_annotated_rgb,
            "timeline": timeline,
            "fps": fps,
            "duration": duration,
            "quality_warnings": quality_warnings,
            "mean_blur": mean_blur,
            "average_confidence": sum(confidence_values) / len(confidence_values) if confidence_values else None,
            "highest_confidence": max(confidence_values) if confidence_values else None,
            "detection_threshold": 0.25,
            "object_counts": dict(object_counts),
"total_object_detections": sum(object_counts.values()),
"unique_object_classes": len(object_counts),
            "has_phone_example": best_phone_frame_rgb is not None,
        }
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
# DASHBOARD SUMMARY AND CHARTS
# ============================================================

st.divider()
st.subheader("📊 Study Session Dashboard")

report_df = load_report()

if report_df.empty:
    st.info(
        "No completed sessions yet. Analyze a video to see "
        "your statistics and charts here."
    )
else:
    total_sessions = len(report_df)
    total_frames = int(
        report_df["Frames Processed"].fillna(0).sum()
    )
    total_detections = int(
        report_df["Total Phone Detections"].fillna(0).sum()
    )
    average_visibility = (
        report_df["Phone Visible Percentage"].dropna().mean()
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Sessions Analyzed", total_sessions)
    col2.metric("Frames Processed", f"{total_frames:,}")
    col3.metric("Phone Detections", f"{total_detections:,}")

    if pd.notna(average_visibility):
        col4.metric(
            "Average Phone Visibility",
            f"{average_visibility:.1f}%",
        )
    else:
        col4.metric("Average Phone Visibility", "N/A")

    st.subheader("📱 Phone Visibility Overview")

    if pd.notna(average_visibility):
        phone_visible = max(
            0.0, min(float(average_visibility), 100.0)
        )

        pie_df = pd.DataFrame({
            "Category": ["Phone Visible", "Phone Not Detected"],
            "Percentage": [
                phone_visible,
                100.0 - phone_visible,
            ],
        })

        st.caption(
            "This chart shows average phone visibility across "
            "analyzed sessions, not a measure of attention."
        )

        st.plotly_chart(
            __import__("plotly.express", fromlist=["express"]).pie(
                pie_df,
                names="Category",
                values="Percentage",
                title="Average Phone Visibility",
            ),
            use_container_width=True,
        )

    st.subheader("📈 Session Comparison")

    chart_df = report_df[
        ["Video", "Phone Visible Percentage"]
    ].dropna()

    if not chart_df.empty:
        chart_df = chart_df.rename(columns={
            "Video": "Study Video",
            "Phone Visible Percentage": "Phone Visibility (%)",
        })

        st.bar_chart(
            chart_df.set_index("Study Video")
        )
    else:
        st.info("No visibility percentages are available yet.")

    st.subheader("📋 Session History")
    st.dataframe(report_df, use_container_width=True)

    csv_data = report_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Session Report (CSV)",
        data=csv_data,
        file_name="session_summary.csv",
        mime="text/csv",
    )


st.caption(
    "Class 12 AI Project | Phone visibility analysis only; "
    "not a validated attention measurement."
)


# ============================================================
# VIDEO UPLOAD AND ANALYSIS
# ============================================================

st.divider()
st.subheader("AI Video Analysis")

st.write(
    "Upload a recorded study-session video (MP4, MOV, or AVI), "
    "then click Start AI Detection."
)

uploaded_video = st.file_uploader(
    "Choose a study video",
    type=["mp4", "mov", "avi"],
)

if uploaded_video is not None:
    st.video(uploaded_video)

    if st.button("Start AI Detection", type="primary"):

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

                st.success(
                    "Analysis completed and session results "
                    "were saved to the report."
                )

                # Refresh the dashboard with the new report.
                st.rerun()
            except Exception as error:
                st.error(
                    "The analysis finished, but the report "
                    f"could not be saved: {error}" 
                )


# ============================================================
# DISPLAY LAST ANALYZED FRAME AND RESULTS
# ============================================================

if "last_analysis" in st.session_state:
    saved_analysis = st.session_state["last_analysis"]
    saved_frame = saved_analysis.get("annotated_frame_rgb")

    st.divider()
    st.subheader("📸 Example Frame with Phone Detection")

    if saved_frame is not None:
        st.image(
            saved_frame,
            caption=f"Phone-detection example from {saved_analysis['video_name']}",
            use_container_width=True,
        )
    else:
        st.info("No analyzed frame is available for display.")

    
    st.caption(
        "This preview shows the sampled frame with the highest-confidence phone detection "
        "(or the last sampled frame if no phone was detected). Green boxes mark phone detections only."
    )
    st.subheader("🔎 Detection Confidence and Video Quality")
    average_confidence = saved_analysis.get("average_confidence")
    highest_confidence = saved_analysis.get("highest_confidence")
    confidence_col1, confidence_col2, confidence_col3 = st.columns(3)
    confidence_col1.metric("Detection threshold", f"{saved_analysis.get('detection_threshold', 0.25):.0%}")
    confidence_col2.metric("Average phone confidence", f"{average_confidence:.1%}" if average_confidence is not None else "N/A")
    confidence_col3.metric("Highest phone confidence", f"{highest_confidence:.1%}" if highest_confidence is not None else "N/A")
    st.caption("Confidence is the model's score for each detected phone, not a guarantee of correctness. N/A means no phone detections met the threshold.")
    warnings = saved_analysis.get("quality_warnings", [])
    if warnings:
        for warning in warnings:
            st.warning(warning)
    else:
        st.success("No obvious technical video-quality warnings from the basic checks. This does not guarantee accurate detection.")
    duration = saved_analysis.get("duration")
    st.write(f"Video: {saved_analysis['resolution']} | " + (f"Duration: {duration:.1f} seconds" if duration is not None else "Duration unavailable"))

    st.subheader("📊 Detection Results")



    # =====================================
    # DESK CLUTTER COUNTER
    # =====================================

    st.divider()
    st.subheader("🧮 Desk Clutter Counter")

    object_counts = saved_analysis.get("object_counts", {})

    clutter_col1, clutter_col2 = st.columns(2)

    clutter_col1.metric(
        "Total Object Detections",
        f"{saved_analysis.get('total_object_detections', 0):,}",
    )

    clutter_col2.metric(
        "Unique Object Classes",
        saved_analysis.get("unique_object_classes", 0),
    )

    if object_counts:
        clutter_df = pd.DataFrame(
            [
                {
                    "Object Class": name.title(),
                    "Detection Count": count,
                }
                for name, count in sorted(
                    object_counts.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            ]
        )

        st.dataframe(
            clutter_df,
            use_container_width=True,
            hide_index=True,
        )

        st.bar_chart(
            clutter_df.set_index("Object Class")
        )
    else:
        st.info("No objects were detected in the analyzed frames.")

    st.caption(
        "Counts represent detections across sampled frames. "
        "The same object may be counted repeatedly, so these are "
        "not counts of unique physical objects or a validated measure "
        "of desk clutter."
    )

    # =====================================
    # STUDY FOCUS SCORE
    # =====================================

    sampled_frames = saved_analysis["frames_processed"]
    phone_frames = saved_analysis["frames_with_phone"]
    percentage = saved_analysis["percentage"]

    focus_score = max(
        0.0,
        min(100.0, 100.0 - percentage),
    )

    st.divider()
    st.subheader("🎯 Study Focus Score")

    score_col1, score_col2 = st.columns(2)

    score_col1.metric(
        "Study Focus Score",
        f"{focus_score:.0f}/100",
    )

    score_col2.metric(
        "Phone Visible in Analyzed Frames",
        f"{percentage:.1f}%",
    )

    st.progress(int(round(focus_score)))

    st.markdown("**How is the score calculated?**")

    st.write(
        "Study Focus Score = 100 − Phone Visible Percentage"
    )

    st.write(
        f"Your video had a phone visible in "
        f"{phone_frames:,} out of {sampled_frames:,} "
        f"analyzed frames."
    )

    st.write(
        f"Calculation: 100 − {percentage:.1f} "
        f"= {focus_score:.1f}/100."
    )

    st.caption(
        "This is a phone-visibility-based indicator, not a "
        "measured attention or concentration score. A phone "
        "may be used for studying, and the AI may miss or "
        "misidentify phones."
    )


# =====================================
# STUDY SESSION TIMER UI
# =====================================

st.divider()
st.subheader("⏱️ Study Session Timer")

duration_options = [25, 45, 60]

selected_duration = st.selectbox(
    "Choose your study duration (minutes)",
    duration_options,
    index=duration_options.index(
        st.session_state.study_timer_duration
    ),
    disabled=st.session_state.study_timer_running,
    key="study_timer_duration_selector",
)

if selected_duration != st.session_state.study_timer_duration:
    st.session_state.study_timer_duration = selected_duration
    st.session_state.study_timer_elapsed_before_pause = 0.0
    st.session_state.study_timer_started_at = None
    st.session_state.study_timer_running = False
    st.session_state.study_timer_finished = False

total_seconds = st.session_state.study_timer_duration * 60

if st.session_state.study_timer_running:
    elapsed_seconds = (
        st.session_state.study_timer_elapsed_before_pause
        + time.monotonic()
        - st.session_state.study_timer_started_at
    )
else:
    elapsed_seconds = (
        st.session_state.study_timer_elapsed_before_pause
    )

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

timer_col1.metric(
    "Elapsed Study Time",
    f"{elapsed_minutes:02d}:{elapsed_remainder:02d}",
)

timer_col2.metric(
    "Time Remaining",
    f"{remaining_minutes:02d}:{remaining_remainder:02d}",
)

st.progress(
    min(1.0, elapsed_seconds / total_seconds)
)

button_col1, button_col2, button_col3 = st.columns(3)

with button_col1:
    if st.button(
        "▶️ Start / Resume",
        disabled=(
            st.session_state.study_timer_running
            or st.session_state.study_timer_finished
        ),
        key="study_timer_start",
    ):
        st.session_state.study_timer_started_at = time.monotonic()
        st.session_state.study_timer_running = True
        st.rerun()

with button_col2:
    if st.button(
        "⏸️ Pause",
        disabled=not st.session_state.study_timer_running,
        key="study_timer_pause",
    ):
        st.session_state.study_timer_elapsed_before_pause = elapsed_seconds
        st.session_state.study_timer_running = False
        st.session_state.study_timer_started_at = None
        st.rerun()

with button_col3:
    if st.button(
        "🔄 Reset",
        key="study_timer_reset",
    ):
        st.session_state.study_timer_running = False
        st.session_state.study_timer_started_at = None
        st.session_state.study_timer_elapsed_before_pause = 0.0
        st.session_state.study_timer_finished = False
        st.rerun()

if st.session_state.study_timer_finished:
    st.success("🎉 Study session completed! Take a short break.")

elif st.session_state.study_timer_running:
    st.info("Your study timer is running.")
    time.sleep(1)
    st.rerun()

st.caption(
    "The timer tracks your study session independently of video analysis. "
    "Phone detection is based on the analyzed video, not live monitoring."
)


# =====================================
# DISTRACTION TIMELINE
# =====================================

st.divider()
st.subheader("📍 Distraction Timeline")

saved_analysis = st.session_state.get("last_analysis")

if saved_analysis is None:
    st.info("Analyze a video first to view its distraction timeline and download the report.")

else:
    timeline_data = saved_analysis.get("timeline", [])
    timeline_df = pd.DataFrame(timeline_data)

    if (
        not timeline_df.empty
        and "seconds" in timeline_df.columns
        and timeline_df["seconds"].notna().any()
    ):
        chart_df = timeline_df.dropna(subset=["seconds"]).copy()
        chart_df["Time (seconds)"] = chart_df["seconds"]

        st.write(
            "This chart shows phone detections over video time. "
            "A value of 1 means a phone was detected in that sampled "
            "frame; 0 means no phone was detected."
        )

        chart_df = chart_df.set_index("Time (seconds)")[["phone_visible"]]
        chart_df.columns = ["Phone detected (1 = yes, 0 = no)"]

        st.line_chart(chart_df)

        detection_df = timeline_df[
            timeline_df["phone_visible"] == 1
        ].copy()

        if not detection_df.empty:
            detection_df["Timestamp"] = detection_df["seconds"].apply(
                lambda value: (
                    f"{int(value // 60):02d}:"
                    f"{value % 60:05.2f}"
                )
            )

            if "max_confidence" in detection_df.columns:
                detection_df["Confidence"] = detection_df["max_confidence"].apply(
                    lambda value: f"{value:.1%}" if pd.notna(value) else "N/A"
                )
            event_columns = ["Timestamp", "frame", "phone_detections"]
            if "Confidence" in detection_df.columns:
                event_columns.append("Confidence")
            detection_df = detection_df[event_columns].rename(
                columns={
                    "frame": "Video Frame Number",
                    "phone_detections": "Phones Detected",
                }
            )

            st.markdown("**Phone Detection Events**")
            st.dataframe(
                detection_df,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.success("No phone was detected in the sampled frames.")

        st.caption(
            "The detector analyzes sampled video frames, so brief phone "
            "appearances may be missed. Phone visibility does not prove "
            "distraction."
        )

    elif timeline_data:
        st.warning("The video did not provide usable timestamps.")
    else:
        st.info("No timeline data is available. Analyze a video again.")

    st.divider()
    st.subheader("📄 Downloadable Session Report")

    pdf_data = create_pdf_report(saved_analysis)

    st.download_button(
        label="Download Session Report (PDF)",
        data=pdf_data,
        file_name="study_session_report.pdf",
        mime="application/pdf",
        key="pdf_session_report",
    )

    st.caption(
        "Phone-Free Score = 100 − Phone Visible Percentage. "
        "This indicates phone visibility only; it does not measure "
        "actual concentration or study quality."
    )
# ============================================================
# LIVE WEBCAM PHONE DETECTION
# ============================================================

import av
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer


@st.cache_resource(show_spinner=False)
def load_webcam_model():
    """Reuse the existing YOLO model file."""
    possible_paths = [
        MODEL_FILE,
        BASE_DIR.parent / "yolo11n.pt",
    ]

    for model_path in possible_paths:
        if model_path.exists():
            return load_yolo_model(str(model_path))

    raise FileNotFoundError(
        "Could not find yolo11n.pt beside dashboard.py "
        "or in the repository's parent folder."
    )


class LivePhoneDetector(VideoProcessorBase):
    def __init__(self):
        self.model = load_webcam_model()

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")

        # Run YOLO on the browser webcam frame.
        result = self.model.predict(
            source=image,
            imgsz=640,
            conf=0.25,
            verbose=False,
        )[0]

        # Draw boxes only for phone detections.
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0].item())

                if isinstance(self.model.names, dict):
                    class_name = self.model.names.get(
                        class_id, ""
                    )
                else:
                    class_name = self.model.names[class_id]

                if str(class_name).strip().lower() not in (
                    "cell phone",
                    "mobile phone",
                ):
                    continue

                confidence = float(box.conf[0].item())
                x1, y1, x2, y2 = [
                    int(value)
                    for value in box.xyxy[0].tolist()
                ]

                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    (0, 210, 0),
                    3,
                )

                cv2.putText(
                    image,
                    f"Phone {confidence:.0%}",
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 210, 0),
                    2,
                    cv2.LINE_AA,
                )

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24",
        )


st.divider()
st.header("📷 Live Webcam Detection")

st.write(
    "Use the webcam of the computer opening this website. "
    "Allow camera access when your browser asks."
)

st.caption(
    "This detects visible phones, not attention or distraction."
)

webrtc_streamer(
    key="study-phone-webcam",
    video_processor_factory=LivePhoneDetector,
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
    async_processing=True,
)
