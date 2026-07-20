"""Wall-clock latency helpers shared by benchmarks and online learning."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(round((pct / 100) * (len(ordered) - 1)))))
    return ordered[idx]


def ms(seconds: float) -> float:
    return round(seconds * 1000.0, 3)


class LatencyPhases:
    """Wall-clock phase timer; ``start_ms`` / ``end_ms`` are relative to construction."""

    def __init__(self, t0: float | None = None) -> None:
        self.t0 = time.perf_counter() if t0 is None else t0
        self.phases: list[dict[str, Any]] = []

    @contextmanager
    def span(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            end = time.perf_counter()
            self.phases.append(
                {
                    "name": name,
                    "start_ms": ms(start - self.t0),
                    "end_ms": ms(end - self.t0),
                    "duration_ms": ms(end - start),
                }
            )

    def to_list(self) -> list[dict[str, Any]]:
        return list(self.phases)

    def total_ms(self) -> float:
        if not self.phases:
            return 0.0
        return max(float(p["end_ms"]) for p in self.phases)
