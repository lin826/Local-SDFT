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

## Generation limits (eval / ablation)

Defaults (see `configs/openclaw_demo_eval.yaml`, `ToolLoopConfig` in `sdft/toolcall/loop.py`):

| Knob | Default | Role |
|---|---|---|
| `max_new_tokens` | 512 | Per-turn decode cap |
| `max_rounds` | 8 (eval yaml) / 16 (code default) | Tool-loop iterations |
| `max_context_chars` | 12000 | Prompt size guard per round |

Override from CLI:

```bash
uv run python -m sdft.toolcall.openclaw_eval \
  --config configs/openclaw_demo_eval.yaml \
  --max-new-tokens 1024 --max-rounds 16 --max-context-chars 16384

uv run python scripts/run_openclaw_ablation.py --skip-data --skip-train \
  --format lfm --max-rounds 16
```

### Missing `\boxed{}`: truncation vs format quality

Ablation on held-out eval (`outputs/benchmarks/openclaw-rl/ablation/`, `format: lfm`):

- **22/29** SDFT-ZS failures have `finish_reason=max_rounds`, **0** `context_overflow`.
- Failures are mostly **prose answers** (`**391**`, “the result is 391”) without `\boxed{}`.
- Raising limits (16 rounds, 1024 tokens/turn, 16k context) on idx=0 still ends at
  `max_rounds` with no box — more budget just yields longer repetition.

OpenClaw-format demo runs (`demo-sdft`, `format: openclaw`):

- Many stops are `context_overflow` (prompt > `max_context_chars` after gibberish + hint loops).
- Real `\boxed{…}` (excluding hint template) is rare; `pred='answer'` often comes from
  the literal `\boxed{answer}` in invalid-action hints.
- Higher context (16k) on a demo item **increased** response length (7k → 17k chars) without
  producing a valid box.

**Conclusion:** missing `\boxed{}` is primarily **format / training quality**, not per-turn
token truncation. Modest `max_context_chars` alignment (8192 → 12000) avoids premature overflow
on shorter runs; large `max_new_tokens` / `max_rounds` bumps alone will not fix boxing and
can amplify degenerate loops.
