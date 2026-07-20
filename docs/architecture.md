# Architecture

Local-SDFT is a small, Apple-Silicon-friendly toolkit for **Self-Distillation
Fine-Tuning** (SDFT) on Liquid AI's LFM2.5-230M, plus comparable baselines
(gold SFT, GRPO) and a live online-learning demo.

```mermaid
flowchart LR
  subgraph data [Data]
    HF[HF / JSONL datasets]
    Collect[Collected records]
  end
  subgraph pipeline [Offline pipeline]
    Gen[sdft.generate]
    Train[sdft.train]
    GRPO[sdft.grpo_train]
    Merge[sdft.merge]
  end
  subgraph apps [Apps]
    Web[web.app]
    Online[online_learning]
    CLI[sdft.cli]
  end
  HF --> Gen
  Gen -->|sdft_response jsonl| Train
  HF -->|gold response| Train
  HF -->|prompts + gold| GRPO
  Train --> Merge
  GRPO --> Merge
  Collect --> Train
  Web --> Online
  Online --> Train
  CLI --> Collect
  Web --> CLI
```

## Package map

| Path | Role |
|---|---|
| `sdft/generate.py` | Teacher pass: rewrite targets in-distribution |
| `sdft/train.py` | LoRA SFT on `sdft_response` or gold `response` (`--target`) |
| `sdft/grpo_train.py` | LoRA GRPO baseline via TRL `GRPOTrainer` |
| `sdft/merge.py` | Fold adapter into base weights |
| `sdft/config.py` | YAML → dataclasses (single source of knobs) |
| `sdft/rewards.py` | Local reward fns for GRPO (`instruction`, `boxed`) |
| `sdft/peft_utils.py` | Shared `adapter_ready` / chat model loading |
| `sdft/online_learning/` | Per-turn tone feedback → tiny SDFT → reply |
| `sdft/toolcall/` | ReTool-style tool loop + OpenClaw eval |
| `sdft/records/` | Shared collect + benchmark persistence |
| `web/` | FastAPI + HTMX UI (`/`, `/data`, `/perf`) |
| `configs/compare/` | Batch-size-1 baseline suite |
| `scripts/run_batch1_comparison.py` | Train + score base / SFT / SDFT / GRPO |

## Entry points

```bash
# Offline SDFT
uv run python -m sdft.generate --config configs/default.yaml
uv run python -m sdft.train    --config configs/default.yaml
uv run python -m sdft.merge    --config configs/default.yaml

# Gold SFT baseline (same jsonl, train on response field)
uv run python -m sdft.train --config configs/compare/batch1_sft_gold.yaml --target gold

# GRPO baseline
uv run python -m sdft.grpo_train --config configs/compare/batch1_grpo.yaml --data data/compare/batch1_grpo.jsonl

# Web demo
uv run python -m web.app   # http://127.0.0.1:8765

# Batch-size-1 comparison (blog / notebook numbers)
uv run python scripts/run_batch1_comparison.py --num-train 32 --num-eval 16
```

## Batch-size-1 philosophy

The online-learning demo updates LoRA with `batch_size: 1` / `grad_accum: 1`
after every chat turn. The `configs/compare/batch1_*.yaml` suite uses the same
update style offline so SFT, SDFT, and GRPO are comparable to that demo:

- **SFT / SDFT:** `per_device_train_batch_size=1`
- **GRPO:** TRL requires `batch_size % num_generations == 0`; we use
  `batch_size=2`, `num_generations=2` (smallest group that still is GRPO)

## Web modules

| Module | Responsibility |
|---|---|
| `web/app.py` | FastAPI routes + uvicorn entry |
| `web/chat_context.py` | History / instruction UI assembly |
| `web/perf_runtime.py` | Chat inference + SSE streaming |
| `web/perf_models.py` | Model / adapter selection for `/perf` |
| `web/transcript_parse.py` | Tool-call transcript rendering |
