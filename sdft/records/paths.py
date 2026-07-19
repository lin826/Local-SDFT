"""Filesystem layout for collected records and benchmark results."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def project_root(start: Path | None = None) -> Path:
    """Best-effort repo root (directory containing ``pyproject.toml``)."""
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return here


def collected_records_path(root: Path | None = None) -> Path:
    """Append-only JSONL of :class:`CollectedRecord` entries."""
    return project_root(root) / "data" / "collected" / "records.jsonl"


def collected_export_path(name: str, root: Path | None = None) -> Path:
    """Training-ready JSONL export (Alpaca / geek-jokes shape)."""
    safe = name.replace("/", "-").strip() or "export"
    return project_root(root) / "data" / "collected" / f"{safe}.jsonl"


def performance_dir(root: Path | None = None) -> Path:
    """Directory of per-run benchmark JSON files."""
    return project_root(root) / "outputs" / "benchmarks"


def performance_index_path(root: Path | None = None) -> Path:
    """Append-only JSONL index of all benchmark runs (lightweight listing)."""
    return performance_dir(root) / "index.jsonl"


def performance_result_path(run_id: str, root: Path | None = None) -> Path:
    return performance_dir(root) / f"{run_id}.json"


def new_run_id(prefix: str = "bench") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{ts}-{uuid4().hex[:8]}"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
