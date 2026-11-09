# AI Study Distraction Detector

A Class 12 Artificial Intelligence project that uses computer vision to identify visible mobile phones in study-session videos and summarize phone-visibility patterns.

> **Important:** A detected phone does not prove that a student is distracted. The dashboard's focus indicator is a simple project metric derived from confirmed phone visibility; it is not a scientifically validated measure of attention. The temporal filter reduces isolated frame-level detections but cannot eliminate misclassification.

## Features

- A responsive home dashboard with consistent light/dark styling and quick project statistics.
- Live webcam phone detection using YOLO and Streamlit WebRTC, with camera permission guidance, connection status, and a built-in START/STOP control.
- Uploaded-video analysis with source duration, processing progress, sampled-frame counts, annotated preview, and a downloadable processed MP4.
- Configurable detection confidence, sampling interval, alert threshold, cooldown, and warning sound.
- Session history with date/time-aware sorting and filters, comparison charts, study calendar, weekly/monthly progress, recent-versus-previous insights, and visibility trends.
- Temporal 2-of-3 sample confirmation to reduce isolated phone false positives in live and uploaded-video detection.
- Clearly separated metrics for raw phone-candidate frames, temporally confirmed phone frames, and individual phone bounding-box detections.
- Pomodoro timer, eight achievement milestones, study levels, warning log, and a paginated PDF analysis report.
- Built-in presentation flow, common viva answers, and a downloadable project demonstration brief.
- Desktop video-processing entry point in main.py.

## Understanding the detection metrics

- **Sampled frames analysed:** the number of frames passed to the YOLO model, based on the selected sampling interval.
- **Frames with a raw phone candidate:** sampled frames where YOLO returned at least one phone bounding box before temporal confirmation.
- **Frames with a confirmed phone:** sampled frames where the temporal 2-of-3 filter confirmed phone visibility.
- **Phone box detections:** the total number of individual phone bounding boxes across the sampled frames. Multiple boxes in one frame are not multiple distraction events.
- **Phone visibility percentage:** confirmed phone frames divided by sampled frames analysed.

Phone visibility is an indicator only. It does not establish that a student was distracted or measure attention, concentration, or learning.

## Webcam connection notes

The webcam workspace uses the public STUN servers already configured in the app. A TURN relay is optional and requires provider-issued settings; placeholder values are ignored. Some school, mobile, or restrictive networks may still prevent a direct WebRTC connection when no TURN relay is available. In that case, use the recorded-video workspace—no additional relay account is needed for video analysis.

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
