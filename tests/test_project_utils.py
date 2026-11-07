import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from dashboard_enhancements import (
    TemporalPhoneFilter, build_session_insights, get_study_level,
)

from project_utils import (
    append_session_record, build_session_record, calculate_study_streaks,
    load_integer_setting, load_session_history, normalize_session_history,
    normalize_warning_entries, safe_float, save_integer_setting,
    validate_analysis_result,
)


class ProjectUtilsTests(unittest.TestCase):
    def test_safe_float_rejects_non_finite_values_and_clamps(self):
        self.assertEqual(safe_float("12.5"), 12.5)
        self.assertEqual(safe_float("bad", default=7), 7)
        self.assertEqual(safe_float(float("inf"), default=3), 3)
        self.assertEqual(safe_float(140, minimum=0, maximum=100), 100)

    def test_normalize_warning_entries_supports_strings_and_dicts(self):
        result = normalize_warning_entries([
            "Phone detected",
            {"time": "18:00:00", "message": "Repeated detection"},
            {"warning": "Fallback message"},
        ])
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]["message"], "Phone detected")
        self.assertEqual(result[1]["time"], "18:00:00")
        self.assertEqual(result[2]["message"], "Fallback message")

    def test_validate_analysis_result_reports_invalid_metrics(self):
        valid = {
            "video_name": "study.mp4", "resolution": "1280x720",
            "frames_processed": 100, "frames_with_phone": 20,
            "percentage": 20.0, "detections": 25, "focus_score": 80.0,
        }
        self.assertEqual(validate_analysis_result(valid), [])
        invalid = dict(valid, frames_with_phone=101, percentage=140)
        errors = validate_analysis_result(invalid)
        self.assertTrue(any("cannot exceed" in item for item in errors))
        self.assertTrue(any("between 0 and 100" in item for item in errors))

    def test_normalize_legacy_desktop_csv(self):
        legacy = pd.DataFrame([{
            "Session Time": "2026-11-07 18:25:00", "Video": "study.mp4",
            "Resolution": "1280x720", "Frames Processed": 100,
            "Frames With Phone": 10, "Phone Visible Percentage": 10,
            "Total Phone Detections": 12,
        }])
        normalized = normalize_session_history(legacy)
        self.assertEqual(normalized.loc[0, "Date"], "2026-11-07")
        self.assertEqual(normalized.loc[0, "Session Time"], "18:25:00")
        self.assertEqual(normalized.loc[0, "Video Name"], "study.mp4")

    def test_build_session_record_uses_consistent_columns(self):
        record = build_session_record(
            "study.mp4", "1280x720", 100, 25, 25, 30,
            timestamp=datetime(2026, 11, 7, 18, 30, 0),
        )
        self.assertEqual(record["Date"], "2026-11-07")
        self.assertEqual(record["Session Time"], "18:30:00")
        self.assertEqual(record["Focus Score"], 75.0)

    def test_append_preserves_legacy_rows_and_loads_history(self):
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "session_summary.csv"
            pd.DataFrame([{
                "Session Time": "2026-11-06 09:00:00", "Video": "older.mp4",
                "Resolution": "640x480", "Frames Processed": 50,
                "Frames With Phone": 5, "Phone Visible Percentage": 10,
                "Total Phone Detections": 5,
            }]).to_csv(report_path, index=False)
            new_record = build_session_record(
                "new.mp4", "1280x720", 100, 20, 20, 24,
                timestamp=datetime(2026, 11, 7, 10, 0, 0),
            )
            append_session_record(report_path, new_record)
            loaded = load_session_history(report_path)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded.loc[0, "Video Name"], "older.mp4")
            self.assertEqual(loaded.loc[1, "Video Name"], "new.mp4")
            self.assertEqual(loaded.loc[1, "Focus Score"], 80.0)

    def test_integer_setting_survives_reload_and_preserves_other_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            setting_path = Path(directory) / "study_stats.json"
            save_integer_setting(setting_path, "completed_focus_sessions", 7)
            save_integer_setting(setting_path, "other_counter", 3)
            self.assertEqual(
                load_integer_setting(setting_path, "completed_focus_sessions"),
                7,
            )
            self.assertEqual(load_integer_setting(setting_path, "other_counter"), 3)
            save_integer_setting(setting_path, "completed_focus_sessions", -4)
            self.assertEqual(
                load_integer_setting(setting_path, "completed_focus_sessions"),
                0,
            )

    def test_streaks_handle_duplicates_gaps_and_future_dates(self):
        result = calculate_study_streaks(
            ["2026-11-01", "2026-11-02", "2026-11-02",
             "2026-11-04", "2026-11-05", "2026-11-09"],
            today=date(2026, 11, 5),
        )
        self.assertEqual(result["current"], 2)
        self.assertEqual(result["longest"], 2)
        self.assertEqual(result["active_days"], 4)

    def test_temporal_phone_filter_rejects_single_frame_false_positive(self):
        phone_filter = TemporalPhoneFilter(window_size=3, min_hits=2)
        self.assertFalse(phone_filter.update(True))
        self.assertFalse(phone_filter.update(False))
        self.assertFalse(phone_filter.update(False))

    def test_temporal_phone_filter_confirms_two_of_three_samples(self):
        phone_filter = TemporalPhoneFilter(window_size=3, min_hits=2)
        self.assertFalse(phone_filter.update(True))
        self.assertTrue(phone_filter.update(True))
        self.assertTrue(phone_filter.update(False))
        phone_filter.reset()
        self.assertFalse(phone_filter.update(False))

    def test_study_levels_and_final_level(self):
        self.assertEqual(get_study_level(0)["level"], 1)
        self.assertEqual(get_study_level(10)["level"], 3)
        self.assertEqual(get_study_level(150)["name"], "Study Legend")
        self.assertEqual(get_study_level(200)["progress"], 1.0)

    def test_smart_insights_compare_latest_five_with_previous_five(self):
        dates = [f"2026-11-{day:02d}" for day in range(1, 11)]
        history = pd.DataFrame({
            "Date": dates,
            "Session Time": ["09:00:00"] * 10,
            "Video Name": [f"session-{index}.mp4" for index in range(10)],
            "Phone Visible Percentage": [50, 40, 30, 20, 10, 10, 9, 8, 7, 6],
        })
        insights = build_session_insights(history)
        self.assertTrue(insights["has_data"])
        self.assertEqual(insights["session_count"], 10)
        self.assertAlmostEqual(insights["recent_average"], 8.0)
        self.assertAlmostEqual(insights["previous_average"], 30.0)
        self.assertAlmostEqual(insights["change"], -22.0)
        self.assertEqual(insights["active_days_7d"], 7)
        self.assertEqual(insights["best_video"], "session-9.mp4")

    def test_smart_insights_wait_for_two_complete_windows(self):
        history = pd.DataFrame({
            "Phone Visible Percentage": [25, 20, 15, 10, 5, 2],
        })
        insights = build_session_insights(history)
        self.assertIsNone(insights["previous_average"])
        self.assertIsNone(insights["change"])


if __name__ == "__main__":
    unittest.main()
