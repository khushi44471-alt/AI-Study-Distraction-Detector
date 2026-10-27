
import cv2
import csv
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

# =========================================
# AI STUDY DISTRACTION DETECTOR
# Session Tracking and Report Generation
# =========================================

VIDEO_PATH = Path("videos") / "study_test.mp4"
MODEL_PATH = "yolo11n.pt"

CONFIDENCE = 0.08
IMAGE_SIZE = 1280

REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)

REPORT_FILE = REPORT_DIR / "session_summary.csv"

print("=" * 45)
print("AI STUDY DISTRACTION DETECTOR")
print("=" * 45)

# Load the AI model
print("Loading YOLO model...")
model = YOLO(MODEL_PATH)

# Open the recorded video
print("NEW VERSION TEST - 123")
print("Opening recorded video...")
cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    print("ERROR: Cannot open video:", VIDEO_PATH.resolve())
    raise SystemExit

print("Video opened successfully!")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 25

total_frames = 0
phone_frames = 0
total_phone_detections = 0

print(f"Video resolution: {width} x {height}")
print("Press Q to stop the session.")

window = "AI Study Distraction Detector"
cv2.namedWindow(window, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window, min(width, 1280), min(height, 720))

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

        cv2.imshow(window, screen)

        if cv2.waitKey(max(1, int(1000 / fps))) & 0xFF == ord("q"):
            print("Session stopped by user.")
            break

finally:
    cap.release()
    cv2.destroyAllWindows()

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

file_exists = (
    REPORT_FILE.exists()
    and REPORT_FILE.stat().st_size > 0
)

with open(
    REPORT_FILE,
    "a",
    newline="",
    encoding="utf-8"
) as file:
    writer = csv.writer(file)

    if not file_exists:
        writer.writerow([
            "Session Time",
            "Video",
            "Resolution",
            "Frames Processed",
            "Frames With Phone",
            "Phone Visible Percentage",
            "Total Phone Detections"
        ])

    writer.writerow([
        session_time,
        VIDEO_PATH.name,
        f"{width}x{height}",
        total_frames,
        phone_frames,
        round(percentage, 2),
        total_phone_detections
    ])

print("\nSESSION REPORT")
print("-" * 35)
print("Frames processed:", total_frames)
print("Frames containing a phone:", phone_frames)
print(f"Phone-visible percentage: {percentage:.1f}%")
print("Report saved to:", REPORT_FILE.resolve())
print("\nSession completed.")
