# OpenClaw-style tool-use SDFT

Curated synthetic math trajectories teach LFM2.5-230M the tool-call + boxed-answer protocol
using **LFM-native** chat templates (eval `format: lfm`).

Training is two-turn and eval-aligned:

1. Turn 1 — emit `<|tool_call_start|>…<|tool_call_end|>` only
2. Turn 2 — after the tool observation, emit `Answer: \boxed{...}`

## Data files / split

| File | Purpose |
|---|---|
| `data/openclaw_tooluse.jsonl` | Alpaca rows — generate input |
| `data/openclaw_tooluse_sdft.jsonl` | Identity SDFT turn pairs (LFM-rendered prefixes) |
| `data/openclaw_eval_heldout.jsonl` | **Canonical held-out eval** (`id`, `question`, `answer`) |
| `data/openclaw_demo.jsonl` | Mirror of held-out (question/answer only) |
| `data/openclaw_split_manifest.json` | Split documentation (overlap=0) |

Held-out questions are a fixed bank with **different numbers** than train. The builder and
`scripts/run_openclaw_ablation.py` fail closed if any eval question overlaps train inputs
or the reserved few-shot prompt `What is 3 + 5?`.

```bash
uv run python scripts/build_openclaw_tooluse_data.py --write-sdft
```

## Pipeline

```bash
uv sync --extra toolcall
uv run python scripts/build_openclaw_tooluse_data.py --write-sdft
uv run python -m sdft.train --config configs/openclaw_tooluse_sdft.yaml \
  --data data/openclaw_tooluse_sdft.jsonl
uv run python -m sdft.merge --config configs/openclaw_tooluse_sdft.yaml \
  --out outputs/openclaw-tooluse-merged

# Full ablation (ZS / OS / OS+CoT / CoT-only / SDFT-*)
uv run python scripts/run_openclaw_ablation.py --skip-data --skip-train --format lfm
```

Outputs: `outputs/benchmarks/openclaw-rl/ablation/comparison.json` and
`outputs/benchmarks/openclaw-rl/ablation/demo_only_sdft.json`.

## Why identity SDFT?

The base 230M model does not emit reliable tool calls. Running `sdft.generate` would
rewrite gold trajectories into plain text. For bootstrap we train on gold completions.
