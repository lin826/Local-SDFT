"""Smoke-test schemas and persistence without loading the model."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from .benchmark import persist_performance_result
from .collect import collect_record, export_collected_for_training
from .paths import collected_records_path, performance_dir
from .schema import PerformanceMetrics, PerformanceResult
from .store import (
    list_performance_results,
    load_collected_records,
    load_performance_index,
)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        records_path = collected_records_path(root)

        collect_record(
            "What is SDFT?",
            output="Self-distillation fine-tuning.",
            source="cli",
            path=records_path,
        )
        collect_record(
            "Summarize LoRA",
            input="In one sentence.",
            output="Low-rank adapters for parameter-efficient fine-tuning.",
            source="web",
            path=records_path,
        )

        loaded = load_collected_records(records_path)
        assert len(loaded) == 2
        assert loaded[0].schema_version == "1"

        export_path, count = export_collected_for_training(
            "verify", records_path=records_path
        )
        assert count == 2
        row = json.loads(export_path.read_text().splitlines()[0])
        assert set(row) == {"instruction", "input", "output"}

        result = PerformanceResult(
            id="bench-verify-00000000",
            run_at="2026-07-19T00:00:00+00:00",
            benchmark="inference",
            model="test-model",
            metrics=PerformanceMetrics(
                latency_ms_mean=10.0,
                latency_ms_p50=9.0,
                latency_ms_p95=12.0,
                tokens_per_second=100.0,
                samples=1,
                batch_size=1,
                input_tokens_total=10,
                output_tokens_total=5,
                device="cpu",
            ),
        )
        bench_dir = performance_dir(root)
        persist_performance_result(result, root=root)

        index = load_performance_index(bench_dir / "index.jsonl")
        assert len(index) >= 1
        listed = list_performance_results(bench_dir)
        assert len(listed) == 1

    print("sdft.records.verify: OK")


if __name__ == "__main__":
    main()
