#!/usr/bin/env python3
"""港牵车围栏内负压占比分析脚本。"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CSV_DIR = BASE_DIR / "csv"
DEFAULT_FENCES_PATH = BASE_DIR / "port_fences.json"
DEFAULT_DETAIL_PATH = DEFAULT_CSV_DIR / "detail_output.csv"
DEFAULT_SUMMARY_PATH = DEFAULT_CSV_DIR / "result.json"
DEFAULT_ERROR_LOG_PATH = DEFAULT_CSV_DIR / "error_log.txt"
DEFAULT_INPUT_CSV_NAME = "input.csv"
BASE_REQUIRED_COLUMNS = {
    "vin",
    "mdt_po_lon",
    "mdt_po_lat",
    "asmod_pintmnf",
    "envp_p",
}
TIME_COLUMN_CANDIDATES = ("clctm", "time", "collect_time", "timestamp")


@dataclass(frozen=True)
class Fence:
    fence_id: str
    points: list[tuple[float, float]]


@dataclass(frozen=True)
class DetailRow:
    vin: str
    clctm: str
    lon: float
    lat: float
    asmod_pintmnf: float
    envp_p: float
    pressure_diff: float
    in_fence: bool
    fence_id: str
    is_negative_pressure: bool


class AnalysisError(Exception):
    """Raised when analysis inputs are invalid."""


def point_in_polygon_ray_casting(
    point: tuple[float, float], polygon: list[tuple[float, float]]
) -> bool:
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def is_negative_pressure(asmod_pintmnf: float, envp_p: float) -> bool:
    return (asmod_pintmnf - envp_p) < 0


def discover_default_csv_path(csv_dir: str | Path = DEFAULT_CSV_DIR) -> Path:
    csv_dir_path = Path(csv_dir)
    preferred_path = csv_dir_path / DEFAULT_INPUT_CSV_NAME
    if preferred_path.exists():
        return preferred_path

    csv_files = sorted(
        path
        for path in csv_dir_path.glob("*.csv")
        if path.name not in {DEFAULT_DETAIL_PATH.name, DEFAULT_SUMMARY_PATH.name}
    )
    if not csv_files:
        raise AnalysisError(
            f"未找到输入 CSV 文件。请将待分析文件放到 {csv_dir_path} 目录下，"
            f"并优先命名为 {DEFAULT_INPUT_CSV_NAME}。"
        )
    return csv_files[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "计算港牵车在电子围栏内的负压占比（围栏内负压点数 / 围栏内总点数）。"
            "默认直接读取脚本同级 csv/ 目录中的 input.csv。"
        )
    )
    parser.add_argument(
        "--csv",
        default=None,
        help=(
            "输入车辆明细 CSV 文件路径。默认读取脚本同级 csv/input.csv；"
            "若不存在则自动选取 csv/ 目录下的首个 CSV。"
        ),
    )
    parser.add_argument(
        "--fences",
        default=str(DEFAULT_FENCES_PATH),
        help="电子围栏 JSON 文件路径，默认使用脚本同级的 port_fences.json",
    )
    parser.add_argument(
        "--time-column",
        default=None,
        help="可选：显式指定时间列名。默认按 clctm/time/collect_time/timestamp 自动识别。",
    )
    parser.add_argument(
        "--output-detail",
        default=str(DEFAULT_DETAIL_PATH),
        help="输出明细 CSV 文件路径，默认写入脚本同级 csv/detail_output.csv",
    )
    parser.add_argument(
        "--output-summary",
        default=str(DEFAULT_SUMMARY_PATH),
        help="输出汇总 JSON 文件路径，默认写入脚本同级 csv/result.json",
    )
    parser.add_argument(
        "--error-log",
        default=str(DEFAULT_ERROR_LOG_PATH),
        help="脏数据日志输出路径，默认写入脚本同级 csv/error_log.txt",
    )
    return parser.parse_args()


def load_fences(path: str | Path) -> list[Fence]:
    with open(path, "r", encoding="utf-8") as file:
        raw_fences = json.load(file)

    fences: list[Fence] = []
    for item in raw_fences:
        fence_id = item.get("fenceId")
        points = item.get("points")
        if not fence_id or not points:
            raise AnalysisError(f"围栏配置缺少 fenceId 或 points: {item}")
        fences.append(
            Fence(
                fence_id=str(fence_id),
                points=[(float(lon), float(lat)) for lon, lat in points],
            )
        )
    if not fences:
        raise AnalysisError("围栏配置为空，无法进行分析")
    return fences


def resolve_time_column(
    fieldnames: Optional[Iterable[str]], preferred_time_column: str | None = None
) -> str:
    if not fieldnames:
        raise AnalysisError("CSV 文件缺少表头")
    available_columns = set(fieldnames)
    if preferred_time_column:
        if preferred_time_column not in available_columns:
            raise AnalysisError(
                f"指定的时间列不存在: {preferred_time_column!r}，可用字段为: {sorted(available_columns)}"
            )
        return preferred_time_column

    for candidate in TIME_COLUMN_CANDIDATES:
        if candidate in available_columns:
            return candidate
    raise AnalysisError(
        "CSV 文件缺少时间字段，默认支持: " + ", ".join(TIME_COLUMN_CANDIDATES)
    )


def validate_headers(fieldnames: Optional[Iterable[str]], time_column: str) -> None:
    if not fieldnames:
        raise AnalysisError("CSV 文件缺少表头")
    missing = (BASE_REQUIRED_COLUMNS | {time_column}) - set(fieldnames)
    if missing:
        raise AnalysisError(f"CSV 文件缺少必要字段: {sorted(missing)}")


def parse_float(row: dict[str, str], column: str, row_number: int) -> float:
    value = row.get(column, "")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise AnalysisError(
            f"第 {row_number} 行字段 {column} 的值无法解析为数值: {value!r}"
        ) from exc


def format_dirty_row_message(row_number: int, error: AnalysisError, row: dict[str, str]) -> str:
    return f"第 {row_number} 行已丢弃，原因: {error}，原始数据: {json.dumps(row, ensure_ascii=False)}"


def point_fence_id(point: tuple[float, float], fences: list[Fence]) -> str:
    for fence in fences:
        if point_in_polygon_ray_casting(point, fence.points):
            return fence.fence_id
    return ""


def safe_ratio(numerator: int, denominator: int) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator


def normalize_time(value: str) -> str:
    candidate = value.strip()
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S.%f",
    ):
        try:
            return datetime.strptime(candidate, fmt).isoformat(sep=" ")
        except ValueError:
            continue
    return candidate


def build_summary(
    detail_rows: list[DetailRow],
    csv_path: str | Path,
    fences_path: str | Path,
    dropped_rows_count: int = 0,
) -> dict:
    vins = sorted({row.vin for row in detail_rows if row.vin})
    timestamps = [normalize_time(row.clctm) for row in detail_rows if row.clctm]
    fence_counter: Counter[str] = Counter(row.fence_id for row in detail_rows if row.in_fence)
    in_fence_rows = [row for row in detail_rows if row.in_fence]
    negative_in_fence_rows = [row for row in in_fence_rows if row.is_negative_pressure]

    return {
        "csv_file": str(csv_path),
        "fences_file": str(fences_path),
        "total_points": len(detail_rows) + dropped_rows_count,
        "processed_points": len(detail_rows),
        "dropped_dirty_rows": dropped_rows_count,
        "in_fence_points": len(in_fence_rows),
        "negative_pressure_points_in_fence": len(negative_in_fence_rows),
        "negative_pressure_ratio_in_fence": safe_ratio(
            len(negative_in_fence_rows), len(in_fence_rows)
        ),
        "vin_count": len(vins),
        "vin_list": vins,
        "time_range": {
            "start": min(timestamps) if timestamps else None,
            "end": max(timestamps) if timestamps else None,
        },
        "fence_hit_counts": dict(sorted(fence_counter.items())),
    }


def analyze(
    csv_path: str | Path,
    fences_path: str | Path,
    time_column: str | None = None,
) -> tuple[list[DetailRow], dict, list[str]]:
    fences = load_fences(fences_path)
    detail_rows: list[DetailRow] = []
    dirty_row_messages: list[str] = []

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        resolved_time_column = resolve_time_column(reader.fieldnames, time_column)
        validate_headers(reader.fieldnames, resolved_time_column)

        for row_number, row in enumerate(reader, start=2):
            try:
                vin = (row.get("vin") or "").strip()
                clctm = (row.get(resolved_time_column) or "").strip()
                lon = parse_float(row, "mdt_po_lon", row_number)
                lat = parse_float(row, "mdt_po_lat", row_number)
                asmod_pintmnf = parse_float(row, "asmod_pintmnf", row_number)
                envp_p = parse_float(row, "envp_p", row_number)
                pressure_diff = asmod_pintmnf - envp_p
                fence_id = point_fence_id((lon, lat), fences)
                in_fence = bool(fence_id)
                negative_pressure = is_negative_pressure(asmod_pintmnf, envp_p)
            except AnalysisError as exc:
                dirty_row_messages.append(format_dirty_row_message(row_number, exc, row))
                continue

            detail_rows.append(
                DetailRow(
                    vin=vin,
                    clctm=clctm,
                    lon=lon,
                    lat=lat,
                    asmod_pintmnf=asmod_pintmnf,
                    envp_p=envp_p,
                    pressure_diff=pressure_diff,
                    in_fence=in_fence,
                    fence_id=fence_id,
                    is_negative_pressure=negative_pressure,
                )
            )

    return detail_rows, build_summary(
        detail_rows,
        csv_path,
        fences_path,
        dropped_rows_count=len(dirty_row_messages),
    ), dirty_row_messages


def write_detail_csv(path: str | Path, detail_rows: list[DetailRow]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "vin",
                "clctm",
                "mdt_po_lon",
                "mdt_po_lat",
                "asmod_pintmnf",
                "envp_p",
                "pressure_diff",
                "in_fence",
                "fence_id",
                "is_negative_pressure",
            ],
        )
        writer.writeheader()
        for row in detail_rows:
            writer.writerow(
                {
                    "vin": row.vin,
                    "clctm": row.clctm,
                    "mdt_po_lon": row.lon,
                    "mdt_po_lat": row.lat,
                    "asmod_pintmnf": row.asmod_pintmnf,
                    "envp_p": row.envp_p,
                    "pressure_diff": row.pressure_diff,
                    "in_fence": str(row.in_fence).lower(),
                    "fence_id": row.fence_id,
                    "is_negative_pressure": str(row.is_negative_pressure).lower(),
                }
            )


def write_summary_json(path: str | Path, summary: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)


def write_error_log(path: str | Path, dirty_row_messages: list[str]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        if dirty_row_messages:
            file.write("\n".join(dirty_row_messages))
            file.write("\n")
        else:
            file.write("未发现脏数据。\n")


def main() -> int:
    args = parse_args()
    csv_path = args.csv or discover_default_csv_path()
    detail_rows, summary, dirty_row_messages = analyze(
        csv_path,
        args.fences,
        time_column=args.time_column,
    )
    write_detail_csv(args.output_detail, detail_rows)
    write_summary_json(args.output_summary, summary)
    write_error_log(args.error_log, dirty_row_messages)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"error_log_file: {args.error_log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
