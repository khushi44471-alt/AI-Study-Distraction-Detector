# AI Study Distraction Detector

A Class 12 Artificial Intelligence project that uses computer vision to identify visible mobile phones in study-session videos and summarize phone-visibility patterns.

> **Important:** A detected phone does not prove that a student is distracted. The dashboard's focus indicator is a simple project metric derived from confirmed phone visibility; it is not a scientifically validated measure of attention. The temporal filter reduces isolated frame-level detections but cannot eliminate misclassification.

## Features

- Live webcam phone detection using YOLO and Streamlit WebRTC.
- Uploaded-video analysis with annotated frame previews and an exported processed video.
- Configurable detection confidence, sampling interval, alert threshold, cooldown, and warning sound.
- Session history, comparison charts, study calendar, weekly/monthly progress, recent-versus-previous insights, and visibility trends.
- Temporal 2-of-3 sample confirmation to reduce isolated phone false positives in live and uploaded-video detection.
- Pomodoro timer, eight achievement milestones, study levels, warning log, and downloadable PDF analysis report.
- Built-in presentation flow, common viva answers, and a downloadable project demonstration brief.
- Desktop video-processing entry point in main.py.

## Project structure

- dashboard.py: Streamlit interface, live detection, analytics, timer, and PDF report.
- dashboard_enhancements.py: temporal detection filter, analytics helpers, study levels, and presentation guide.
- main.py: optional desktop video-processing runner.
- project_utils.py: shared validation, history normalization, and safe report persistence.
- tests/: automated tests for shared utility functions.
- requirements.txt: Python dependencies.
- packages.txt: system packages used by Streamlit hosting.

## Run the dashboard

Use Python 3.11 or newer.

1. Create and activate a virtual environment.
2. Install dependencies with python -m pip install -r requirements.txt.
3. Start the app with streamlit run dashboard.py.
4. Open the local Streamlit URL shown in the terminal.

The YOLO model file yolo11n.pt is included in this repository.

## Run the desktop script

Place a test video at videos/study_test.mp4, then run python main.py from the project directory. The desktop preview is optional: when the installed OpenCV build has no GUI support, the script continues processing and saves the session report without opening a preview window.

## Session data

Session summaries are saved in reports/session_summary.csv. The shared helper module normalizes the older desktop CSV layout and dashboard layout so analytics can read existing history. Writes use a temporary file followed by an atomic replacement to reduce the chance of a partially written CSV.

On hosted platforms, local files may be reset when an app is rebuilt or its container is replaced. Keep downloaded copies of reports that you need to retain.

## Automated checks

Run these checks locally:

~~~bash
python -m pip install pandas
python -m py_compile dashboard.py main.py project_utils.py
python -m unittest discover -s tests -v
~~~

GitHub Actions runs syntax and utility checks on Python 3.11, 3.12, and 3.13.
