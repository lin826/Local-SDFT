"""JSON / JSONL persistence helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, TypeVar

from .schema import CollectedRecord, PerformanceResult

T = TypeVar("T")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, obj: dict) -> None:
    _ensure_parent(path)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: dict) -> None:
    _ensure_parent(path)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def append_collected_record(path: Path, record: CollectedRecord) -> None:
    append_jsonl(path, record.to_dict())


def load_collected_records(path: Path) -> list[CollectedRecord]:
    return [CollectedRecord.from_dict(row) for row in read_jsonl(path)]


def export_training_jsonl(
    records: Iterable[CollectedRecord],
    path: Path,
    *,
    require_output: bool = True,
) -> int:
    """Write Alpaca-style rows; returns number of rows written."""
    _ensure_parent(path)
    kept = 0
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            row = record.to_training_row()
            if not row["instruction"]:
                continue
            if require_output and not row["output"]:
                continue
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
    return kept


def save_performance_result(path: Path, result: PerformanceResult) -> None:
    write_json(path, result.to_dict())


def load_performance_result(path: Path) -> PerformanceResult:
    return PerformanceResult.from_dict(read_json(path))


def append_performance_index(index_path: Path, result: PerformanceResult) -> None:
    append_jsonl(
        index_path,
        {
            "id": result.id,
            "run_at": result.run_at,
            "benchmark": result.benchmark,
            "model": result.model,
            "path": str(performance_result_basename(result.id)),
            "tokens_per_second": result.metrics.tokens_per_second,
            "latency_ms_mean": result.metrics.latency_ms_mean,
        },
    )


def performance_result_basename(run_id: str) -> str:
    return f"{run_id}.json"


def load_performance_index(index_path: Path) -> list[dict]:
    return read_jsonl(index_path)


def list_performance_results(benchmark_dir: Path) -> list[PerformanceResult]:
    if not benchmark_dir.is_dir():
        return []
    results: list[PerformanceResult] = []
    for path in sorted(benchmark_dir.glob("bench-*.json")):
        results.append(load_performance_result(path))
    return results
