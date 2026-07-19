"""Latency aggregation and design summaries for online-learning sessions."""

from __future__ import annotations

import statistics
from typing import Any

from sdft.records.benchmark import _percentile

from .schema import OnlineTurn, TurnLatency


def _latency_stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "count": 0}
    return {
        "mean": round(statistics.mean(values), 3),
        "p50": round(_percentile(values, 50), 3),
        "p95": round(_percentile(values, 95), 3),
        "count": len(values),
    }


def aggregate_turn_latencies(turns: list[OnlineTurn]) -> dict[str, Any]:
    """Summarize generate / train / infer / total latency across session turns."""
    generate = [t.latency.generate_ms for t in turns if t.latency.generate_ms is not None]
    infer = [t.latency.inference_ms for t in turns if t.latency.inference_ms is not None]
    train = [t.latency.train_ms for t in turns if t.latency.train_ms is not None]
    total = [t.latency.total_ms for t in turns if t.latency.total_ms is not None]
    return {
        "turn_count": len(turns),
        "generate_ms": _latency_stats(generate),
        "train_ms": _latency_stats(train),
        "inference_ms": _latency_stats(infer),
        "total_ms": _latency_stats(total),
    }


def build_design_summary(
    *,
    session_id: str,
    config_path: str,
    model: str,
    turn_count: int,
    adapter_dir: str,
) -> dict[str, str]:
    return {
        "purpose": "Online LoRA learning from /data turns (tiny SDFT)",
        "session_id": session_id,
        "config_path": config_path,
        "model": model,
        "adapter_dir": adapter_dir,
        "turn_count": str(turn_count),
        "update_recipe": "SDFT teacher rewrite + 1–few LoRA steps on sdft_response (not user gold)",
    }


def turn_latency_from_phases(
    phases: list[dict[str, Any]],
    *,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    preview_input_tokens: int | None = None,
    preview_output_tokens: int | None = None,
) -> TurnLatency:
    """Derive TurnLatency from LatencyPhases-style phase list."""
    by_name = {p["name"]: float(p["duration_ms"]) for p in phases}
    generate = by_name.get("generate_sdft")
    infer = by_name.get("inference_preview")
    train = by_name.get("train_update")
    total = max((float(p["end_ms"]) for p in phases), default=0.0)
    if total <= 0 and phases:
        total = sum(float(p["duration_ms"]) for p in phases)
    return TurnLatency(
        total_ms=round(total, 3),
        generate_ms=round(generate, 3) if generate is not None else None,
        train_ms=round(train, 3) if train is not None else None,
        inference_ms=round(infer, 3) if infer is not None else None,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
