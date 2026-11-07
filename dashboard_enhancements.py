"""Polished analytics, temporal detection checks, levels, and demo support."""
from __future__ import annotations

from collections import deque
from typing import Any

import pandas as pd


class TemporalPhoneFilter:
    """Confirm a phone only when it appears in at least 2 of the last 3 samples."""

    def __init__(self, window_size: int = 3, min_hits: int = 2) -> None:
        if window_size < 1:
            raise ValueError("window_size must be at least 1")
        if min_hits < 1 or min_hits > window_size:
            raise ValueError("min_hits must be between 1 and window_size")
        self.votes: deque[bool] = deque(maxlen=window_size)
        self.min_hits = min_hits

    def update(self, raw_detected: bool) -> bool:
        self.votes.append(bool(raw_detected))
        return sum(self.votes) >= self.min_hits

    def reset(self) -> None:
        self.votes.clear()


STUDY_LEVELS = [
    {"level": 1, "name": "Starter", "start": 0, "next": 5},
    {"level": 2, "name": "Routine Builder", "start": 5, "next": 10},
    {"level": 3, "name": "Consistency Keeper", "start": 10, "next": 25},
    {"level": 4, "name": "Focus Explorer", "start": 25, "next": 50},
    {"level": 5, "name": "Momentum Pro", "start": 50, "next": 75},
    {"level": 6, "name": "Deep Work Champion", "start": 75, "next": 100},
    {"level": 7, "name": "Century Scholar", "start": 100, "next": 150},
    {"level": 8, "name": "Study Legend", "start": 150, "next": None},
]


def get_study_level(completed_sessions: int) -> dict[str, Any]:
    """Return a stable level, next milestone, and progress fraction."""
    completed = max(0, int(completed_sessions))
    current = STUDY_LEVELS[0]
    for level in STUDY_LEVELS:
        if completed >= level["start"]:
            current = level
        else:
            break

    next_target = current["next"]
    if next_target is None:
        progress = 1.0
        remaining = 0
    else:
        span = max(1, next_target - current["start"])
        progress = min(1.0, max(0.0, (completed - current["start"]) / span))
        remaining = max(0, next_target - completed)

    return {
        "level": current["level"],
        "name": current["name"],
        "next_target": next_target,
        "remaining": remaining,
        "progress": progress,
    }


def build_session_insights(history: pd.DataFrame | None) -> dict[str, Any]:
    """Summarize recent phone-visibility history without claiming to measure attention."""
    empty = {
        "has_data": False,
        "session_count": 0,
        "recent_average": None,
        "previous_average": None,
        "change": None,
        "best_visibility": None,
        "best_video": "Not available",
        "active_days_7d": None,
        "median_visibility": None,
    }
    if history is None or not isinstance(history, pd.DataFrame) or history.empty:
        return empty
    if "Phone Visible Percentage" not in history.columns:
        return empty

    data = history.copy()
    data["_visibility"] = pd.to_numeric(
        data["Phone Visible Percentage"], errors="coerce"
    )
    data = data.dropna(subset=["_visibility"])
    data = data[data["_visibility"].between(0, 100)]
    if data.empty:
        return empty

    if "Date" in data.columns:
        data["_date"] = pd.to_datetime(data["Date"], errors="coerce")
        if "Session Time" in data.columns:
            combined = pd.to_datetime(
                data["Date"].astype("string").fillna("")
                + " "
                + data["Session Time"].astype("string").fillna(""),
                errors="coerce",
            )
            data["_sort_time"] = combined.fillna(data["_date"])
        else:
            data["_sort_time"] = data["_date"]
        if data["_sort_time"].notna().any():
            data = data.sort_values("_sort_time", na_position="first")

    values = data["_visibility"].astype(float)
    recent = values.tail(5)
    previous = (
        values.iloc[len(values) - 10:len(values) - 5]
        if len(values) >= 10 else values.iloc[0:0]
    )
    recent_average = float(recent.mean())
    previous_average = float(previous.mean()) if len(previous) == 5 else None
    change = recent_average - previous_average if previous_average is not None else None

    best_index = values.idxmin()
    best_row = data.loc[best_index]
    best_video = best_row.get("Video Name", "Unnamed session")
    if pd.isna(best_video) or not str(best_video).strip():
        best_video = "Unnamed session"

    active_days = None
    if "_date" in data.columns and data["_date"].notna().any():
        latest_date = data["_date"].max().normalize()
        week_start = latest_date - pd.Timedelta(days=6)
        active_days = int(
            data.loc[
                data["_date"].between(week_start, latest_date),
                "_date",
            ].dt.date.nunique()
        )

    return {
        "has_data": True,
        "session_count": int(len(values)),
        "recent_average": recent_average,
        "previous_average": previous_average,
        "change": change,
        "best_visibility": float(values.min()),
        "best_video": str(best_video),
        "active_days_7d": active_days,
        "median_visibility": float(values.median()),
    }


def render_professional_polish(st: Any) -> None:
    """Add a main-content-only visual layer; the sidebar is deliberately untouched."""
    st.markdown(
        """
        <style>
        .main .project-status-strip {
            display:flex;align-items:center;justify-content:space-between;
            gap:16px;flex-wrap:wrap;margin:0 0 22px 0;padding:17px 21px;
            border:1px solid rgba(167,139,250,.34);border-radius:18px;
            background:linear-gradient(112deg,rgba(76,29,149,.96),rgba(49,46,129,.92));
            box-shadow:0 12px 30px rgba(30,20,65,.18);
        }
        .main .project-status-eyebrow {
            color:#ddd6fe;font-size:10px;font-weight:800;letter-spacing:2px;
            margin-bottom:5px;
        }
        .main .project-status-title {
            display:block;color:#fff;font-size:19px;font-weight:800;
            letter-spacing:-.35px;
        }
        .main .project-status-subtitle {
            display:block;color:#e9ddff;font-size:11px;margin-top:4px;
            letter-spacing:1px;font-weight:650;
        }
        .main .project-status-pill {
            color:#ecfdf5;background:rgba(16,185,129,.18);
            border:1px solid rgba(167,243,208,.42);border-radius:999px;
            padding:8px 12px;font-size:11px;font-weight:800;letter-spacing:.7px;
        }
        .main .analytics-heading {
            border-left:4px solid #a78bfa;padding:7px 0 7px 15px;
            margin:8px 0 20px 0;
        }
        .main .analytics-heading h2 {margin:0 0 4px 0;}
        .main .analytics-heading p {margin:0;opacity:.78;}
        .main [data-testid="stPlotlyChart"],
        .main [data-testid="stDataFrame"],
        .main [data-testid="stExpander"] {
            transition:border-color .18s ease,box-shadow .18s ease;
        }
        .main [data-testid="stPlotlyChart"]:hover,
        .main [data-testid="stDataFrame"]:hover {
            border-color:rgba(167,139,250,.65);
            box-shadow:0 10px 26px rgba(80,55,140,.12);
        }
        .main button:focus-visible,
        .main a:focus-visible {
            outline:3px solid #c4b5fd!important;outline-offset:3px!important;
        }
        @media (max-width:650px) {
            .main .project-status-strip {padding:14px;border-radius:14px;}
            .main .project-status-title {font-size:16px;}
        }
        @media (prefers-reduced-motion:reduce) {
            .main *, .main *::before, .main *::after {
                animation:none!important;transition:none!important;scroll-behavior:auto!important;
            }
        }
        </style>
        <div class="project-status-strip">
            <div>
                <div class="project-status-eyebrow">CLASS 12 ARTIFICIAL INTELLIGENCE · COMPUTER VISION</div>
                <span class="project-status-title">Study Intelligence Workspace</span>
                <span class="project-status-subtitle">DETECT · REVIEW · TRACK · PRESENT</span>
            </div>
            <div class="project-status-pill">● DASHBOARD READY</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_smart_insights(st: Any, history: pd.DataFrame) -> None:
    """Render readable recent-versus-previous session insights."""
    insights = build_session_insights(history)
    st.markdown("### 🧠 Smart Study Insights")
    st.caption(
        "Compares recorded phone visibility across sessions. "
        "It is a video-based indicator, not a measurement of attention."
    )
    if not insights["has_data"]:
        st.info("Complete a video analysis to unlock smart session insights.")
        return

    recent = insights["recent_average"]
    change = insights["change"]
    previous_text = (
        f"{change:+.1f} percentage points"
        if change is not None else "Need 6 sessions"
    )
    active_days = insights["active_days_7d"]
    active_text = str(active_days) if active_days is not None else "N/A"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Recent 5-session average", f"{recent:.1f}%")
    col2.metric(
        "Change vs previous 5",
        previous_text,
        delta=f"{change:+.1f} pp" if change is not None else None,
        delta_color="inverse",
    )
    col3.metric(
        "Lowest visibility session",
        f'{insights["best_visibility"]:.1f}%',
    )
    col4.metric("Active days in last recorded 7 days", active_text)

    if change is None:
        st.info(
            "Record at least ten sessions to compare the latest five "
            "with the five before them."
        )
    elif change <= -2:
        st.success(
            "Recorded phone visibility is lower in the latest sessions "
            "than in the previous comparison window."
        )
    elif change >= 2:
        st.warning(
            "Recorded phone visibility is higher in the latest sessions. "
            "Review the videos and session conditions before drawing conclusions."
        )
    else:
        st.info("Recorded phone visibility is broadly stable across the two windows.")

    st.caption(
        f'Lowest recorded visibility: {insights["best_video"]} '
        f'({insights["best_visibility"]:.1f}%). '
        f'Median across {insights["session_count"]} valid sessions: '
        f'{insights["median_visibility"]:.1f}%.'
    )
