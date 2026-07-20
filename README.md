# Local-SDFT

Using LoRA to decrease the load of fine-tuning, and challenge the capability of
self-distillation on geek jokes (PHD comics) — locally, on
[Liquid AI LFM2.5-230M](https://huggingface.co/LiquidAI/LFM2.5-230M).
The 230M base weights stay frozen; only small LoRA adapters are trained.

Recipe per [Yang et al., 2024 — *Self-Distillation Bridges Distribution Gap in
Language Model Fine-Tuning*](https://arxiv.org/abs/2406.13629):

1. **Generate** — the model rewrites each training target *in its own distribution*:
   for every example it sees a few in-context demonstrations (gold pairs from the
   same dataset) and answers the prompt itself.
2. **Train** — LoRA SFT on `(prompt → model-generated response)`, loss on the
   completion tokens only.
3. **Merge** — fold the trained adapter back into the base weights for deployment.

## Requirements

- Python ≥ 3.11, [uv](https://docs.astral.sh/uv/)
- Tested on Apple Silicon (M2 Max, trains on MPS in fp32); CUDA and CPU work too
  (device is auto-detected: MPS → CUDA → CPU)

## Setup

```bash
uv sync
```

Stack: torch 2.13 · transformers 5.14 (`Lfm2ForCausalLM`, no trust-remote-code) ·
trl 1.8 (`SFTTrainer`) · peft 0.19.

## Usage

```bash
# 1. Self-distillation: teacher pass over the task dataset
uv run python -m sdft.generate --config configs/default.yaml

# 2. LoRA fine-tune on the self-generated data
uv run python -m sdft.train --config configs/default.yaml

# 3. (Optional) merge adapter into base weights -> standalone model
uv run python -m sdft.merge --config configs/default.yaml
```

All knobs live in the config YAML (dataset + field mapping, few-shot count,
decoding, LoRA rank/targets, training hyperparameters); sections map to the
dataclasses in `sdft/config.py`. Common overrides are also CLI flags
(`--num-examples`, `--out`, `--data`, `--output-dir`, `--epochs`).

### Geek jokes (PHD comics)

`configs/geek_jokes.yaml` is a ready template: drop your data at
`data/geek_jokes.jsonl` (one JSON per line: `{"instruction": ..., "input": ...,
"output": ...}`) and run the three steps with `--config configs/geek_jokes.yaml`.

### Smoke test (~2 min end to end)

```bash
uv run python -m sdft.generate --num-examples 16 --out data/smoke.jsonl
uv run python -m sdft.train --data data/smoke.jsonl --output-dir outputs/smoke --epochs 1
uv run python -m sdft.merge --adapter outputs/smoke --out outputs/smoke-merged
```

Utility: `uv run python -m sdft.inspect` prints the model's LoRA-targetable
module names.

### OpenClaw-RL tool-calling eval

See [docs/openclaw-rl-eval.md](docs/openclaw-rl-eval.md) for the ReTool-style
tool loop and AIME-2024 benchmark adapter. Quick smoke:

```bash
uv sync --extra toolcall
bash scripts/run_openclaw_rl_eval.sh
```

## Evaluation results

Numbers below are from local runs on Apple Silicon (MPS). Artifacts live under
`outputs/benchmarks/` (gitignored). Official AlpacaEval 2 **LC win-rate** was
**not** computed (no `OPENAI_API_KEY` / GPT-4 Turbo judge).

### AlpacaEval 2.0 (held-out generations)

- **Train:** `yahma/alpaca-cleaned` (128 SDFT pairs) — never AlpacaEval prompts
- **Eval:** first **64 / 805** instructions from `tatsu-lab/alpaca_eval`
- **Merged LFM checkpoint:** `outputs/lfm25-230m-alpacaeval2-sdft-merged`
  (`configs/lfm25_alpacaeval2_trained.yaml`)

| Condition | Model | n | Mean output chars | Identical to base | Train loss | Notes |
|---|---|---:|---:|---:|---|---|
| LFM base | `LiquidAI/LFM2.5-230M` | 64 | 1173 | — | — | Healthy greedy generations |
| LFM SDFT | LFM + LoRA merge | 64 | 1118 | **6 / 64** | 0.599 | Real merge (`adapter maxabs≈0.032`, no NaNs); **58 / 64** outputs differ |
| Qwen base | `Qwen/Qwen3.5-0.8B` | 64 | 1828 | — | — | Healthy |
| Qwen SDFT | Qwen + early LoRA (`checkpoint-2`) | 64 | 1816 | 41 / 64 | 0.076 (final) | Final adapter collapsed to NaNs on MPS — score `checkpoint-2` only |

**LC win-rate / raw win-rate:** not available yet. Score existing generations with:

```bash
export OPENAI_API_KEY=sk-...
uv sync --extra alpacaeval
# From the alpacaeval worktree that produced the merges:
SCORE=1 SKIP_SDFT=1 NUM_EVAL=64 bash scripts/run_lfm25_alpacaeval2.sh
```

#### `/perf` qualitative (AlpacaEval-faithful ZS)

Side-by-side chat on open-ended prompts (base vs SDFT merge):

| Prompt | Base LFM2.5-230M | LFM SDFT merge |
|---|---|---|
| Sew a button on a shirt | Refusal: *“I'm sorry, but I can't assist with that.”* | Step-by-step sewing guide (~246 tok, ~124 tok/s) |
| How do I make apple juice? | Refusal / capability hedge | Ingredient list + recipe (~168 tok, ~117 tok/s) |

Prompt-strategy ablations in `/perf` (ZS / FS1 / FS3 / CoT / FS+CoT / SysHelpful)
use train-side ICL demos only, with a leakage guard against AlpacaEval instructions.

### OpenClaw tool-use (held-out math, `format: lfm`)

Identity SDFT on curated tool-call trajectories → `outputs/openclaw-tooluse-merged`.
Held-out bank: **29** questions (`outputs/benchmarks/openclaw-rl/ablation/comparison.json`).

| Arm | Model | pass@1 | Mean tool calls | Mean score |
|---|---|---:|---:|---:|
| ZS | base | 20.7% | 0.03 | −0.586 |
| OS | base | 17.2% | 1.00 | −0.655 |
| OS+CoT | base | 20.7% | 0.00 | −0.586 |
| CoT-only | base | 20.7% | 0.00 | −0.586 |
| SDFT-ZS | SDFT merge | 20.7% | 0.03 | −0.586 |
| SDFT+OS | SDFT merge | 13.8% | 1.24 | −0.724 |
| **SDFT+OS+CoT** | SDFT merge | **24.1%** | 0.03 | **−0.517** |

Best arm is **SDFT+OS+CoT** (+3.4 pp over ZS). Most failures are format quality
(prose / missing `\boxed{}`), not token truncation — see
[docs/openclaw-tooluse-sdft.md](docs/openclaw-tooluse-sdft.md).

#### Smoke / demos (smaller sets)

| Run | n | pass@1 | Notes |
|---|---:|---:|---|
| AIME-2024 smoke (ZS / OS / SDFT) | 3 | 0% | Expect low; smoke wiring only |
| Demo bank, base / ZS / SDFT | 20 | 0% | OpenClaw-format tool loop |
| Demo bank, one-shot (OS) | 20 | **15%** | Best small-bank prompt-only arm |

Reproduce the full ablation:

```bash
uv sync --extra toolcall
uv run python scripts/run_openclaw_ablation.py --skip-data --skip-train --format lfm
```

## LoRA targets for the LFM2 architecture

LFM2.5-230M is a hybrid: **6 attention blocks + 8 short-convolution blocks**.
Module leaf names:

| Module | Location | Count |
|---|---|---|
| `self_attn.{q,k,v,out}_proj` | attention blocks | 6× |
| `conv.{in_proj,out_proj,conv}` | conv blocks | 8× |
| `feed_forward.{w1,w2,w3}` | SwiGLU MLPs | 14× |

Default targets all four attention projections via a regex over full paths:
`.*self_attn\.(q|k|v|out)_proj`. Note the leaf name `out_proj` exists in **both**
attention and conv blocks — a plain list target `["out_proj"]` would adapt both.
To also adapt MLPs or conv blocks, extend the regex, e.g.
`.*(self_attn\.(q|k|v|out)_proj|feed_forward\.w[123])`.

## Notes

- **Default dataset** is `yahma/alpaca-cleaned` as a stand-in; point
  `data.dataset` at any HF dataset, or use `dataset: json` + `data_files: ...`
  for a local `.jsonl` (see `configs/geek_jokes.yaml`).
- Self-generated targets are in-distribution but not necessarily *correct* —
  that is the expected SDFT trade-off. For a 230M model, more few-shot demos
  (`generation.num_shots`) or a task-matched demo pool helps.
- To *cross*-distill instead (bigger teacher → 230M student), run `sdft.generate`
  with a config whose `model.name` is e.g. `LiquidAI/LFM2.5-1.2B`, then train
  with the 230M config.
- LFM2.5 is released under the [LFM Open License v1.0](https://huggingface.co/LiquidAI/LFM2.5-230M)
  (free below $10M annual revenue).
