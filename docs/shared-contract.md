# Shared contract: `sdft.records`

CLI (`sdft.cli`) and the web UI (`web.app`) share one persistence layer:
[`sdft/records/`](../sdft/records/). Do not invent parallel schemas or file
layouts in either surface.

## Schemas

| Type | Module | Purpose |
|---|---|---|
| `CollectedRecord` | `schema.py` | User-collected training rows (instruction / input / output / tags) |
| `PerformanceResult` | `schema.py` | One benchmark or chat-inference run + metrics + metadata |
| `PerformanceMetrics` | `schema.py` | Latency, tokens/sec, sample counts |

`SCHEMA_VERSION` is bumped when on-disk JSON shape changes.

## Paths

Resolved via `sdft.records.paths` (project-root relative):

| Helper | Default location |
|---|---|
| `collected_records_path()` | `data/collected/records.jsonl` |
| `collected_export_path(name)` | `data/collected/exports/<name>.jsonl` |
| `performance_dir()` | `outputs/benchmarks/` |
| `performance_result_path(id)` | `outputs/benchmarks/<id>.json` |
| `performance_index_path()` | `outputs/benchmarks/index.jsonl` |

Online-learning sessions live under `outputs/online-learning/<session_id>/`
(see `sdft.online_learning.paths`) and are **not** part of this contract, but
their chat turns also call `collect_record` so rows flow into the shared store.

## Public API

Import from `sdft.records` only:

```python
from sdft.records import (
    collect_record,
    export_collected_for_training,
    measure_chat,
    run_benchmark,
    persist_performance_result,
)
```

Latency timing uses `sdft.records.latency.LatencyPhases` (also re-exported from
`sdft.records.benchmark` for back-compat).

## Who writes what

| Writer | Reads | Writes |
|---|---|---|
| `sdft.cli collect/export` | collected jsonl | collected jsonl / exports |
| `sdft.cli bench` | configs + model | performance results + index |
| `web` `/data` | sessions + collected | online sessions + collected |
| `web` `/perf` | adapters + configs | performance results (chat runs) |
