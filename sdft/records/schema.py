"""Shared schemas for collected training data and performance benchmark results."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


SCHEMA_VERSION = "1"


@dataclass
class CollectedRecord:
    """One user-collected training example (CLI/web/manual).

    Field names align with ``configs/geek_jokes.yaml`` / Alpaca-style JSONL so
    records can be exported directly for ``sdft.generate`` / ``sdft.train``.
    """

    instruction: str
    input: str = ""
    output: str = ""
    id: str = ""
    created_at: str = ""
    source: str = "manual"  # cli | web | manual | import
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> CollectedRecord:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in raw.items() if k in known})

    def to_training_row(self) -> dict[str, str]:
        """Alpaca / geek-jokes JSONL row for the SDFT pipeline."""
        return {
            "instruction": self.instruction.strip(),
            "input": self.input.strip(),
            "output": self.output.strip(),
        }

    def prompt(self, prompt_fields: list[str] | None = None) -> str:
        """Build a user prompt from configured fields (default: instruction + input)."""
        fields = prompt_fields or ["instruction", "input"]
        row = self.to_training_row()
        parts = [row[f] for f in fields if row.get(f)]
        return "\n\n".join(parts)


@dataclass
class PerformanceMetrics:
    """Timing and throughput for one benchmark measurement."""

    latency_ms_mean: float
    latency_ms_p50: float
    latency_ms_p95: float
    tokens_per_second: float
    samples: int
    batch_size: int
    input_tokens_total: int
    output_tokens_total: int
    device: str
    warmup_samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PerformanceMetrics:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in raw.items() if k in known})


@dataclass
class PerformanceResult:
    """One persisted benchmark run."""

    benchmark: str  # generate | inference | geek_jokes | train_smoke
    model: str
    metrics: PerformanceMetrics
    id: str = ""
    run_at: str = ""
    config_path: str | None = None
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PerformanceResult:
        metrics_raw = raw.get("metrics", {})
        known = {f.name for f in cls.__dataclass_fields__.values()} - {"metrics"}
        return cls(
            metrics=PerformanceMetrics.from_dict(metrics_raw),
            **{k: v for k, v in raw.items() if k in known},
        )
