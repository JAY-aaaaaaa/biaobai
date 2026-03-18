import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from negative_pressure_analysis import (
    AnalysisError,
    DEFAULT_INPUT_CSV_NAME,
    analyze,
    build_summary,
    discover_default_csv_path,
    format_dirty_row_message,
    is_negative_pressure,
    load_fences,
    point_in_polygon_ray_casting,
    resolve_time_column,
    write_error_log,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_CSV = REPO_ROOT / "tests" / "data" / "sample_input.csv"
FENCES_JSON = REPO_ROOT / "port_fences.json"
SCRIPT = REPO_ROOT / "negative_pressure_analysis.py"


class NegativePressureAnalysisTests(unittest.TestCase):
    def test_point_in_polygon_matches_expected_sample(self):
        polygon = [
            (0.0, 0.0),
            (2.0, 0.0),
            (2.0, 2.0),
            (0.0, 2.0),
            (0.0, 0.0),
        ]
        self.assertTrue(point_in_polygon_ray_casting((1.0, 1.0), polygon))
        self.assertFalse(point_in_polygon_ray_casting((3.0, 3.0), polygon))

    def test_point_in_polygon_works_for_real_fence_inside_and_outside(self):
        fence = load_fences(FENCES_JSON)[1]
        inside_point = (113.88865775, 22.458952375)
        outside_point = (110.0, 20.0)

        self.assertTrue(point_in_polygon_ray_casting(inside_point, fence.points))
        self.assertFalse(point_in_polygon_ray_casting(outside_point, fence.points))

    def test_negative_pressure_rule(self):
        self.assertTrue(is_negative_pressure(99.0, 100.0))
        self.assertFalse(is_negative_pressure(100.0, 100.0))
        self.assertFalse(is_negative_pressure(101.0, 100.0))

    def test_discover_default_csv_path_prefers_input_csv_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_dir = Path(temp_dir)
            first_csv = csv_dir / "a.csv"
            preferred_csv = csv_dir / DEFAULT_INPUT_CSV_NAME
            first_csv.write_text("demo", encoding="utf-8")
            preferred_csv.write_text("demo", encoding="utf-8")

            self.assertEqual(discover_default_csv_path(csv_dir), preferred_csv)

    def test_discover_default_csv_path_raises_when_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(AnalysisError):
                discover_default_csv_path(temp_dir)

    def test_format_dirty_row_message_contains_reason_and_row(self):
        message = format_dirty_row_message(
            5,
            AnalysisError("字段 envp_p 无法解析"),
            {"vin": "VIN001", "envp_p": "NULL"},
        )

        self.assertIn("第 5 行已丢弃", message)
        self.assertIn("字段 envp_p 无法解析", message)
        self.assertIn('"envp_p": "NULL"', message)

    def test_analyze_marks_fence_and_negative_flags_per_row(self):
        detail_rows, _, dirty_rows = analyze(SAMPLE_CSV, FENCES_JSON)

        self.assertEqual(dirty_rows, [])
        self.assertEqual([row.in_fence for row in detail_rows], [True, True, True, True, False])
        self.assertEqual(
            [row.is_negative_pressure for row in detail_rows],
            [True, False, True, True, True],
        )
        self.assertEqual(
            [row.fence_id for row in detail_rows],
            ["fence002", "fence002", "fence004", "fence001", ""],
        )

    def test_analyze_returns_expected_summary(self):
        detail_rows, summary, dirty_rows = analyze(SAMPLE_CSV, FENCES_JSON)

        self.assertEqual(dirty_rows, [])
        self.assertEqual(len(detail_rows), 5)
        self.assertEqual(summary["total_points"], 5)
        self.assertEqual(summary["processed_points"], 5)
        self.assertEqual(summary["dropped_dirty_rows"], 0)
        self.assertEqual(summary["in_fence_points"], 4)
        self.assertEqual(summary["negative_pressure_points_in_fence"], 3)
        self.assertAlmostEqual(summary["negative_pressure_ratio_in_fence"], 0.75)
        self.assertEqual(summary["vin_count"], 3)
        self.assertEqual(summary["time_range"]["start"], "2026-03-18 08:00:00")
        self.assertEqual(summary["time_range"]["end"], "2026-03-18 08:04:00")
        self.assertEqual(
            summary["fence_hit_counts"],
            {"fence001": 1, "fence002": 2, "fence004": 1},
        )

    def test_analyze_skips_dirty_rows_and_collects_error_log_messages(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dirty_csv = Path(temp_dir) / "dirty.csv"
            dirty_csv.write_text(
                "vin,clctm,mdt_po_lon,mdt_po_lat,asmod_pintmnf,envp_p\n"
                "VIN001,2026-03-18 08:00:00,113.889500,22.458500,98,100\n"
                "VIN002,2026-03-18 08:01:00,113.890000,22.459000,NULL,100\n"
                "VIN003,2026-03-18 08:02:00,113.870000,22.490000,99,\n",
                encoding="utf-8",
            )

            detail_rows, summary, dirty_rows = analyze(dirty_csv, FENCES_JSON)

            self.assertEqual(len(detail_rows), 1)
            self.assertEqual(summary["total_points"], 3)
            self.assertEqual(summary["processed_points"], 1)
            self.assertEqual(summary["dropped_dirty_rows"], 2)
            self.assertEqual(summary["in_fence_points"], 1)
            self.assertEqual(summary["negative_pressure_points_in_fence"], 1)
            self.assertAlmostEqual(summary["negative_pressure_ratio_in_fence"], 1.0)
            self.assertEqual(len(dirty_rows), 2)
            self.assertIn("'NULL'", dirty_rows[0])
            self.assertIn("''", dirty_rows[1])

    def test_build_summary_returns_none_ratio_when_no_points_in_fence(self):
        detail_rows, _, _ = analyze(SAMPLE_CSV, FENCES_JSON)
        outside_only_rows = [row for row in detail_rows if not row.in_fence]

        summary = build_summary(outside_only_rows, SAMPLE_CSV, FENCES_JSON)

        self.assertEqual(summary["total_points"], 1)
        self.assertEqual(summary["in_fence_points"], 0)
        self.assertEqual(summary["negative_pressure_points_in_fence"], 0)
        self.assertIsNone(summary["negative_pressure_ratio_in_fence"])

    def test_resolve_time_column_supports_alias_and_explicit_override(self):
        self.assertEqual(resolve_time_column(["vin", "time"], None), "time")
        self.assertEqual(resolve_time_column(["vin", "timestamp"], "timestamp"), "timestamp")
        with self.assertRaises(AnalysisError):
            resolve_time_column(["vin", "time"], "clctm")

    def test_write_error_log_outputs_messages_or_clean_notice(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            error_log = Path(temp_dir) / "error_log.txt"
            write_error_log(error_log, ["第 2 行已丢弃", "第 3 行已丢弃"])
            self.assertEqual(
                error_log.read_text(encoding="utf-8"),
                "第 2 行已丢弃\n第 3 行已丢弃\n",
            )

            write_error_log(error_log, [])
            self.assertEqual(error_log.read_text(encoding="utf-8"), "未发现脏数据。\n")

    def test_cli_runs_with_hardcoded_default_paths(self):
        detail_path = REPO_ROOT / "csv" / "detail_output.csv"
        summary_path = REPO_ROOT / "csv" / "result.json"
        error_log_path = REPO_ROOT / "csv" / "error_log.txt"
        try:
            completed = subprocess.run(
                [sys.executable, str(SCRIPT)],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn('"negative_pressure_ratio_in_fence": 0.75', completed.stdout)
            self.assertIn("error_log_file:", completed.stdout)
            self.assertTrue(detail_path.exists())
            self.assertTrue(summary_path.exists())
            self.assertTrue(error_log_path.exists())
        finally:
            for output_path in [detail_path, summary_path, error_log_path]:
                if output_path.exists():
                    output_path.unlink()

    def test_cli_writes_detail_summary_and_error_log_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            detail_path = Path(temp_dir) / "detail_output.csv"
            summary_path = Path(temp_dir) / "result.json"
            error_log_path = Path(temp_dir) / "error_log.txt"
            dirty_csv = Path(temp_dir) / "dirty.csv"
            dirty_csv.write_text(
                "vin,clctm,mdt_po_lon,mdt_po_lat,asmod_pintmnf,envp_p\n"
                "VIN001,2026-03-18 08:00:00,113.889500,22.458500,98,100\n"
                "VIN002,2026-03-18 08:01:00,113.890000,22.459000,NULL,100\n",
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--csv",
                    str(dirty_csv),
                    "--fences",
                    str(FENCES_JSON),
                    "--output-detail",
                    str(detail_path),
                    "--output-summary",
                    str(summary_path),
                    "--error-log",
                    str(error_log_path),
                ],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn('"dropped_dirty_rows": 1', completed.stdout)
            self.assertTrue(detail_path.exists())
            self.assertTrue(summary_path.exists())
            self.assertTrue(error_log_path.exists())

            with summary_path.open("r", encoding="utf-8") as file:
                summary = json.load(file)
            self.assertEqual(summary["processed_points"], 1)
            self.assertEqual(summary["dropped_dirty_rows"], 1)

            with detail_path.open("r", encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 1)

            error_log_content = error_log_path.read_text(encoding="utf-8")
            self.assertIn("第 3 行已丢弃", error_log_content)
            self.assertIn("'NULL'", error_log_content)


if __name__ == "__main__":
    unittest.main()
