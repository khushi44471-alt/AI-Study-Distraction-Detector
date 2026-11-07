
import cv2
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

from project_utils import append_session_record, build_session_record

# =========================================
# AI STUDY DISTRACTION DETECTOR
# Session Tracking and Report Generation
# =========================================

BASE_DIR = Path(__file__).resolve().parent
VIDEO_PATH = BASE_DIR / "videos" / "study_test.mp4"
MODEL_PATH = BASE_DIR / "yolo11n.pt"

CONFIDENCE = 0.08
IMAGE_SIZE = 1280

REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_FILE = REPORT_DIR / "session_summary.csv"

print("=" * 45)
print("AI STUDY DISTRACTION DETECTOR")
print("=" * 45)

# Load the AI model
print("Loading YOLO model...")
model = YOLO(MODEL_PATH)

# Open the recorded video
print("Opening recorded video...")
cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    print("ERROR: Cannot open video:", VIDEO_PATH.resolve())
    raise SystemExit

print("Video opened successfully!")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if width <= 0 or height <= 0:
    print("ERROR: The video has invalid dimensions.")
    cap.release()
    raise SystemExit

if fps <= 0:
    fps = 25

total_frames = 0
phone_frames = 0
total_phone_detections = 0

print(f"Video resolution: {width} x {height}")
print("Press Q to stop the session.")

window = "AI Study Distraction Detector"
preview_enabled = True

try:
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, min(width, 1280), min(height, 720))
except cv2.error:
    preview_enabled = False
    print(
        "OpenCV preview is unavailable in this environment. "
        "The video will still be processed and saved."
    )

try:
    while True:
        success, frame = cap.read()

        if not success:
            print("Video finished.")
            break

        total_frames += 1

        # Detect objects in the original frame
        result = model.predict(
            source=frame,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE,
            verbose=False
        )[0]

        display = frame.copy()
        phone_count = 0

        # Draw detection boxes and labels
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            name = model.names[class_id]

            x1, y1, x2, y2 = map(
                int, box.xyxy[0].tolist()
            )

            if name == "cell phone":
                phone_count += 1
                color = (0, 0, 255)
                label = f"PHONE {confidence:.0%}"
            else:
                color = (255, 180, 0)
                label = f"{name} {confidence:.0%}"

            cv2.rectangle(
                display,
                (x1, y1),
                (x2, y2),
                color,
                3
            )

            cv2.putText(
                display,
                label,
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA
            )

        # Count frames containing at least one phone
        if phone_count > 0:
            phone_frames += 1
            total_phone_detections += phone_count

        percentage = (
            phone_frames / total_frames * 100
            if total_frames else 0
        )

        if phone_count > 0:
            status = f"PHONE VISIBLE: {phone_count}"
            status_color = (0, 0, 255)
        else:
            status = "NO PHONE DETECTED"
            status_color = (0, 200, 0)

        # Add statistics banner
        cv2.rectangle(
            display,
            (0, 0),
            (width, 125),
            (25, 25, 25),
            -1
        )

        cv2.putText(
            display,
            status,
            (15, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            display,
            f"Frames processed: {total_frames}",
            (15, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            f"Frames with phone: {phone_frames} ({percentage:.1f}%)",
            (15, 98),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        # Resize for display only; AI uses original frame
        scale = min(
            1280 / width,
            720 / height,
            1.0
        )

        if scale < 1.0:
            screen = cv2.resize(
                display,
                (int(width * scale), int(height * scale)),
                interpolation=cv2.INTER_AREA
            )
        else:
            screen = display

        if preview_enabled:
            try:
                cv2.imshow(window, screen)
                key = cv2.waitKey(max(1, int(1000 / fps))) & 0xFF
            except cv2.error:
                preview_enabled = False
                print(
                    "OpenCV preview stopped because GUI support is unavailable. "
                    "Continuing video analysis without the preview."
                )
            else:
                if key == ord("q"):
                    print("Session stopped by user.")
                    break

finally:
    cap.release()
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass

# =========================================
# SAVE SESSION REPORT
# =========================================

session_time = datetime.now().strftime(
    "%Y-%m-%d %H:%M:%S"
)

percentage = (
    phone_frames / total_frames * 100
    if total_frames else 0
)

session_record = build_session_record(
    video_name=VIDEO_PATH.name,
    resolution=f"{width}x{height}",
    frames_processed=total_frames,
    frames_with_phone=phone_frames,
    percentage=percentage,
    detections=total_phone_detections,
    timestamp=datetime.now(),
)

append_session_record(
    REPORT_FILE,
    session_record,
)

print("\nSESSION REPORT")
print("-" * 35)
print("Frames processed:", total_frames)
print("Frames containing a phone:", phone_frames)
print(f"Phone-visible percentage: {percentage:.1f}%")
print("Report saved to:", REPORT_FILE.resolve())
print("\nSession completed.")
