"""Shared validation, normalization, and persistence helpers."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

SESSION_COLUMNS = [
    "Date", "Session Time", "Video Name", "Resolution",
    "Frames Processed", "Frames With Phone", "Phone Visible Percentage",
    "Total Phone Detections", "Focus Score",
]
NUMERIC_SESSION_COLUMNS = [
    "Frames Processed", "Frames With Phone", "Phone Visible Percentage",
    "Total Phone Detections", "Focus Score",
]


def resolve_model_checkpoint(
    candidate_paths: Any,
    downloaded_checkpoint: Any = None,
) -> Path | None:
    """Return the first existing local or downloaded model checkpoint."""
    paths = list(candidate_paths or [])
    if downloaded_checkpoint:
        paths.append(downloaded_checkpoint)

    for candidate in paths:
        if candidate is None:
            continue
        try:
            path = Path(candidate).expanduser()
            if path.is_file():
                return path.resolve()
        except (OSError, TypeError, ValueError):
            continue
    return None


def safe_float(value: Any, default: float = 0.0,
               minimum: float | None = None,
               maximum: float | None = None) -> float:
    """Convert a value to a finite float and optionally clamp its range."""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(number):
        return default
    if minimum is not None:
        number = max(minimum, number)
    if maximum is not None:
        number = min(maximum, number)
    return number


def format_duration(seconds: Any) -> str:
    """Format a non-negative duration as HH:MM:SS, safely handling missing data."""
    value = safe_float(seconds, default=-1.0)
    if value < 0:
        return "Not available"
    total_seconds = int(round(value))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}"


def normalize_warning_entries(warnings: Any) -> list[dict[str, str]]:
    """Normalize warning strings and dictionaries for tables and PDF reports."""
    if warnings is None:
        return []
    if isinstance(warnings, (Mapping, str)):
        items = [warnings]
    else:
        try:
            items = list(warnings)
        except TypeError:
            items = [warnings]

    normalized = []
    for item in items:
        if isinstance(item, Mapping):
            message = item.get("message", item.get("warning", "Detection warning recorded"))
            timestamp = item.get("time", item.get("timestamp", ""))
        else:
            message, timestamp = item, ""

        message_text = str(message).strip() if message is not None else ""
        timestamp_text = str(timestamp).strip() if timestamp is not None else ""
        if message_text:
            normalized.append({"message": message_text, "time": timestamp_text})
    return normalized


def validate_analysis_result(analysis: Any) -> list[str]:
    """Return validation errors; an empty list means the result is usable."""
    if not isinstance(analysis, Mapping):
        return ["The analysis result is not a dictionary-like object."]

    required = (
        "video_name", "resolution", "frames_processed",
        "frames_with_phone", "percentage", "detections",
    )
    errors = [
        f"Missing required analysis field: {field}"
        for field in required if field not in analysis
    ]
    if errors:
        return errors

    processed = safe_float(analysis.get("frames_processed"), default=-1)
    with_phone = safe_float(analysis.get("frames_with_phone"), default=-1)
    detections = safe_float(analysis.get("detections"), default=-1)
    percentage = safe_float(analysis.get("percentage"), default=float("nan"))

    if processed < 0:
        errors.append("Frames processed must be a non-negative number.")
    if with_phone < 0:
        errors.append("Frames with phone must be a non-negative number.")
    if processed >= 0 and with_phone > processed:
        errors.append("Frames with phone cannot exceed frames processed.")
    if detections < 0:
        errors.append("Phone detections must be a non-negative number.")
    if not math.isfinite(percentage) or not 0 <= percentage <= 100:
        errors.append("Phone visibility percentage must be between 0 and 100.")

    if "focus_score" in analysis:
        score = safe_float(analysis.get("focus_score"), default=float("nan"))
        if not math.isfinite(score) or not 0 <= score <= 100:
            errors.append("Focus score must be between 0 and 100.")

    if not str(analysis.get("video_name", "")).strip():
        errors.append("The video name cannot be empty.")
    if not str(analysis.get("resolution", "")).strip():
        errors.append("The video resolution cannot be empty.")
    return errors


def normalize_session_history(frame: pd.DataFrame | None) -> pd.DataFrame:
    """Normalize old desktop and dashboard CSV layouts without discarding rows."""
    if frame is None:
        return pd.DataFrame(columns=SESSION_COLUMNS)
    data = frame.copy()
    if data.empty and len(data.columns) == 0:
        return pd.DataFrame(columns=SESSION_COLUMNS)

    if "Video Name" in data.columns and "Video" in data.columns:
        names = data["Video Name"]
        missing = names.isna() | (names.astype(str).str.strip() == "")
        data.loc[missing, "Video Name"] = data.loc[missing, "Video"]
    elif "Video Name" not in data.columns and "Video" in data.columns:
        data = data.rename(columns={"Video": "Video Name"})

    if "Session Time" not in data.columns:
        data["Session Time"] = pd.NA
    raw_time = data["Session Time"].astype("string")
    full_timestamp = raw_time.str.match(
        r"^\s*\d{4}[-/]\d{1,2}[-/]\d{1,2}", na=False
    )
    parsed_time = pd.to_datetime(
        raw_time.where(full_timestamp), errors="coerce"
    )

    if "Date" not in data.columns:
        data["Date"] = parsed_time.dt.strftime("%Y-%m-%d")
    else:
        parsed_date = pd.to_datetime(data["Date"], errors="coerce")
        missing_date = parsed_date.isna() & parsed_time.notna()
        parsed_date.loc[missing_date] = parsed_time.loc[missing_date]
        data["Date"] = parsed_date.dt.strftime("%Y-%m-%d")

    full_datetime_rows = full_timestamp & parsed_time.notna()
    data.loc[full_datetime_rows, "Session Time"] = (
        parsed_time.loc[full_datetime_rows].dt.strftime("%H:%M:%S")
    )

    for column in NUMERIC_SESSION_COLUMNS:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
        else:
            data[column] = pd.NA
    for column in SESSION_COLUMNS:
        if column not in data.columns:
            data[column] = pd.NA

    ordered = SESSION_COLUMNS + [
        column for column in data.columns if column not in SESSION_COLUMNS
    ]
    return data.reindex(columns=ordered)


def build_session_record(video_name: str, resolution: str,
                         frames_processed: int, frames_with_phone: int,
                         percentage: float, detections: int,
                         focus_score: float | None = None,
                         timestamp: datetime | None = None) -> dict[str, Any]:
    """Build one canonical session row for both entry points."""
    recorded_at = timestamp or datetime.now()
    visibility = safe_float(percentage, minimum=0.0, maximum=100.0)
    calculated_focus = 100.0 - visibility
    if focus_score is not None:
        calculated_focus = safe_float(
            focus_score, default=calculated_focus,
            minimum=0.0, maximum=100.0,
        )
    return {
        "Date": recorded_at.strftime("%Y-%m-%d"),
        "Session Time": recorded_at.strftime("%H:%M:%S"),
        "Video Name": str(video_name).strip(),
        "Resolution": str(resolution).strip(),
        "Frames Processed": max(0, int(safe_float(frames_processed))),
        "Frames With Phone": max(0, int(safe_float(frames_with_phone))),
        "Phone Visible Percentage": round(visibility, 2),
        "Total Phone Detections": max(0, int(safe_float(detections))),
        "Focus Score": round(calculated_focus, 2),
    }


def append_session_record(report_path: str | Path,
                          record: Mapping[str, Any]) -> pd.DataFrame:
    """Append one report row with an atomic file replacement."""
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            previous = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            previous = pd.DataFrame(columns=SESSION_COLUMNS)
        except pd.errors.ParserError as error:
            raise ValueError(
                f"Session history is not a valid CSV file: {path.name}"
            ) from error
    else:
        previous = pd.DataFrame(columns=SESSION_COLUMNS)

    previous = normalize_session_history(previous)
    new_row = normalize_session_history(pd.DataFrame([dict(record)]))

    # Make saving a completed analysis idempotent. A rerun can safely retry
    # the same write without adding the exact same analysis twice.
    if "Session ID" in new_row.columns:
        new_session_id = new_row.iloc[0].get("Session ID")
        if pd.notna(new_session_id) and str(new_session_id).strip():
            existing_ids = previous.get("Session ID", pd.Series(dtype="string"))
            if existing_ids.astype("string").eq(str(new_session_id)).fillna(False).any():
                return previous

    combined = pd.concat([previous, new_row], ignore_index=True, sort=False)
    ordered = SESSION_COLUMNS + [
        column for column in combined.columns if column not in SESSION_COLUMNS
    ]
    combined = combined.reindex(columns=ordered)

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", suffix=".tmp",
            prefix=f".{path.stem}_", dir=path.parent, delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            combined.to_csv(temporary_file, index=False)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
    return combined


def load_session_history(report_path: str | Path) -> pd.DataFrame:
    """Load and normalize a report, safely handling missing and empty files."""
    path = Path(report_path)
    if not path.exists():
        return pd.DataFrame(columns=SESSION_COLUMNS)
    try:
        loaded = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=SESSION_COLUMNS)
    return normalize_session_history(loaded)


def calculate_study_streaks(session_dates: Any,
                            today: date | datetime | None = None) -> dict[str, int]:
    """Calculate current and longest consecutive study-day streaks."""
    reference_day = today.date() if isinstance(today, datetime) else today
    reference_day = reference_day or date.today()
    if session_dates is None:
        values = []
    elif isinstance(session_dates, (str, date, datetime)):
        values = [session_dates]
    else:
        try:
            values = list(session_dates)
        except TypeError:
            values = [session_dates]

    unique_dates: set[date] = set()
    for value in values:
        try:
            if isinstance(value, datetime):
                study_day = value.date()
            elif isinstance(value, date):
                study_day = value
            else:
                study_day = pd.Timestamp(value).date()
        except (TypeError, ValueError, OverflowError):
            continue
        if study_day <= reference_day:
            unique_dates.add(study_day)

    ordered_dates = sorted(unique_dates)
    longest = 0
    running = 0
    previous_day: date | None = None
    for study_day in ordered_dates:
        if previous_day is not None and study_day == previous_day + timedelta(days=1):
            running += 1
        else:
            running = 1
        longest = max(longest, running)
        previous_day = study_day

    current = 0
    if ordered_dates and ordered_dates[-1] >= reference_day - timedelta(days=1):
        expected = ordered_dates[-1]
        date_set = set(ordered_dates)
        while expected in date_set:
            current += 1
            expected -= timedelta(days=1)

    return {"current": current, "longest": longest, "active_days": len(ordered_dates)}


def load_integer_setting(
    setting_path: str | Path,
    key: str,
    default: int = 0,
) -> int:
    """Read a non-negative integer setting from a small JSON file."""
    path = Path(setting_path)
    try:
        with path.open("r", encoding="utf-8") as setting_file:
            data = json.load(setting_file)
        value = int(data.get(key, default)) if isinstance(data, Mapping) else default
    except (OSError, json.JSONDecodeError, TypeError, ValueError, OverflowError):
        value = default
    return max(0, value)


def save_integer_setting(
    setting_path: str | Path,
    key: str,
    value: int,
) -> None:
    """Persist an integer setting through an atomic JSON file replacement."""
    if not str(key).strip():
        raise ValueError("The JSON setting key cannot be empty.")

    path = Path(setting_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    settings: dict[str, Any] = {}

    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as setting_file:
                loaded = json.load(setting_file)
            if isinstance(loaded, dict):
                settings.update(loaded)
        except (OSError, json.JSONDecodeError):
            settings = {}

    settings[key] = max(0, int(safe_float(value)))
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".tmp",
            prefix=f".{path.stem}_",
            dir=path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(settings, temporary_file, indent=4)

        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass

