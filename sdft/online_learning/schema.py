"""Schemas for online-learning sessions and per-turn latency."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from sdft.records.schema import SCHEMA_VERSION


@dataclass
class TurnLatency:
    """Wall-clock latency for one online-learning turn."""

    total_ms: float
    tone_ms: float | None = None
    generate_ms: float | None = None
    train_ms: float | None = None
    inference_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TurnLatency:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in raw.items() if k in known})


@dataclass
class OnlineTurn:
    """One chat turn: user message, SDFT target, assistant reply, and feedback metadata."""

    turn_index: int
    instruction: str
    input: str
    output: str
    sdft_response: str
    record_id: str
    created_at: str
    latency: TurnLatency
    assistant_reply: str = ""
    preview: str = ""
    latency_phases: list[dict[str, Any]] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    feedback_tone: str | None = None
    feedback_reward: int | None = None
    feedback_source: str = "none"
    preference_action: str = ""
    trained_on: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["latency"] = self.latency.to_dict()
        return data

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> OnlineTurn:
        latency_raw = raw.get("latency") or {}
        known = {f.name for f in cls.__dataclass_fields__.values()} - {"latency"}
        data = {k: v for k, v in raw.items() if k in known}
        if "sdft_response" not in data:
            data["sdft_response"] = raw.get("sdft_response") or raw.get("preview") or ""
        if "assistant_reply" not in data:
            data["assistant_reply"] = raw.get("assistant_reply") or raw.get("preview") or ""
        if "preview" not in data:
            data["preview"] = data.get("assistant_reply") or ""
        return cls(
            latency=TurnLatency.from_dict(latency_raw),
            **data,
        )


@dataclass
class OnlineLearningSession:
    """Persisted online-learning run with adapter path and turn history."""

    id: str
    created_at: str
    updated_at: str
    config_path: str
    adapter_dir: str
    model: str
    turns: list[OnlineTurn] = field(default_factory=list)
    latency_summary: dict[str, Any] = field(default_factory=dict)
    design_summary: dict[str, str] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "config_path": self.config_path,
            "adapter_dir": self.adapter_dir,
            "model": self.model,
            "turn_count": self.turn_count,
            "turns": [t.to_dict() for t in self.turns],
            "latency_summary": self.latency_summary,
            "design_summary": self.design_summary,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> OnlineLearningSession:
        turns = [OnlineTurn.from_dict(t) for t in raw.get("turns") or []]
        known = {f.name for f in cls.__dataclass_fields__.values()} - {"turns"}
        return cls(turns=turns, **{k: v for k, v in raw.items() if k in known})
