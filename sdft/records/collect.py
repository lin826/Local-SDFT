"""Create and validate collected training records."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .paths import collected_records_path, new_run_id, utc_now_iso
from .schema import CollectedRecord
from .store import (
    append_collected_record,
    export_training_jsonl,
    load_collected_records,
)


def validate_collected_record(record: CollectedRecord) -> list[str]:
    """Return human-readable validation errors (empty list == valid)."""
    errors: list[str] = []
    if not record.instruction.strip():
        errors.append("instruction is required")
    if record.source not in {"cli", "web", "manual", "import"}:
        errors.append(f"unknown source: {record.source!r}")
    return errors


def new_collected_record(
    instruction: str,
    *,
    input: str = "",
    output: str = "",
    source: str = "manual",
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    record_id: str | None = None,
) -> CollectedRecord:
    return CollectedRecord(
        id=record_id or new_run_id("rec"),
        created_at=utc_now_iso(),
        instruction=instruction,
        input=input,
        output=output,
        source=source,
        tags=list(tags or []),
        metadata=dict(metadata or {}),
    )


def collect_record(
    instruction: str,
    *,
    input: str = "",
    output: str = "",
    source: str = "manual",
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    path: Path | None = None,
) -> CollectedRecord:
    """Validate, persist, and return a new collected record."""
    record = new_collected_record(
        instruction,
        input=input,
        output=output,
        source=source,
        tags=tags,
        metadata=metadata,
    )
    errors = validate_collected_record(record)
    if errors:
        raise ValueError("; ".join(errors))
    append_collected_record(path or collected_records_path(), record)
    return record


def import_training_row(
    row: dict[str, Any],
    *,
    source: str = "import",
    path: Path | None = None,
) -> CollectedRecord:
    """Import an Alpaca/geek-jokes JSONL row into the collected store."""
    return collect_record(
        str(row.get("instruction", "")),
        input=str(row.get("input", "")),
        output=str(row.get("output", "")),
        source=source,
        path=path,
    )


def export_collected_for_training(
    export_name: str,
    *,
    records_path: Path | None = None,
    require_output: bool = True,
) -> tuple[Path, int]:
    """Export collected records to training JSONL; returns (path, row_count)."""
    from .paths import collected_export_path

    records = load_collected_records(records_path or collected_records_path())
    out = collected_export_path(export_name)
    count = export_training_jsonl(records, out, require_output=require_output)
    return out, count
