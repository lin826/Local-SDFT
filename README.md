# Local-SDFT

Fine-tune a **230M** or **1.2B** language model on your laptop — with
**Self-Distillation Fine-Tuning (SDFT)**, plain LoRA SFT, and **GRPO** —
and watch it learn live in a browser.

Built on [Liquid AI LFM2.5-230M](https://huggingface.co/LiquidAI/LFM2.5-230M) and
[LFM2.5-1.2B-Instruct](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct).
Base weights stay frozen; only small LoRA adapters train.

Recipe per [Shenfeld et al., 2026 — *Self-Distillation Enables Continual
Learning*](https://self-distillation.github.io/SDFT)
([arXiv](https://arxiv.org/abs/2601.19897)):

1. **Generate** — the model rewrites each training target *in its own distribution*:
   for every example it sees a few in-context demonstrations (gold pairs from the
   same dataset) and answers the prompt itself.
2. **Train** — LoRA SFT on `(prompt → model-generated response)`, loss on the
   completion tokens only.
3. **Merge** — fold the trained adapter back into the base weights for deployment.

Also in this repo:

- **Gold SFT** and **GRPO** baselines at batch size 1 (same style as the online demo)
- **Online learning** chat (`/data`) — tone as implicit feedback, tiny LoRA updates
- **Perf chat** (`/perf`) — base vs SDFT side-by-side with streaming + ablations
- **Colab notebook** — [`notebooks/local_sdft_colab.ipynb`](notebooks/local_sdft_colab.ipynb)
  (`SMOKE=True` for a tiny in-sample run; `False` for full AE2 **805** train + same-pool score)

See [docs/architecture.md](docs/architecture.md) for the package map.

## Requirements

- Python ≥ 3.11, [uv](https://docs.astral.sh/uv/)
- Tested on Apple Silicon (M2 Max, trains on MPS in fp32); CUDA and CPU work too
  (device is auto-detected: MPS → CUDA → CPU)

## Setup

```bash
uv sync
```

Dependencies: `torch>=2.6` · `transformers>=4.54` (`Lfm2ForCausalLM`, no
trust-remote-code) · `trl>=0.19` (`SFTTrainer` / `GRPOTrainer`) · `peft>=0.15`.

## Quick start — offline SDFT

```bash
# 1. Self-distillation: teacher pass over the task dataset
uv run python -m sdft.generate --config configs/default.yaml

# 2. LoRA fine-tune on the self-generated data
uv run python -m sdft.train    --config configs/default.yaml

# 3. (Optional) merge adapter into base weights -> standalone model
uv run python -m sdft.merge    --config configs/default.yaml
```

All knobs live in the config YAML (dataset + field mapping, few-shot count,
decoding, LoRA rank/targets, training hyperparameters); sections map to the
dataclasses in `sdft/config.py`. Common overrides are also CLI flags
(`--num-examples`, `--out`, `--data`, `--output-dir`, `--epochs`, `--target`).

Optional end-to-end smoke (~2 min):

```bash
uv run python -m sdft.generate --num-examples 16 --out data/smoke.jsonl
uv run python -m sdft.train --data data/smoke.jsonl --output-dir outputs/smoke --epochs 1
uv run python -m sdft.merge --adapter outputs/smoke --out outputs/smoke-merged
```

Utility: `uv run python -m sdft.inspect` prints the model's LoRA-targetable
module names.

## Showcase results

Numbers are from local Apple Silicon runs (MPS, ~32 GB). Artifacts under
`outputs/compare/` and `outputs/benchmarks/` (gitignored). BFCL scores are a
**local AST subset**, not the official Berkeley Function-Calling Leaderboard.
Official AlpacaEval 2 LC win-rate was not computed (no GPT-4 judge key).

### BFCL-v3 local AST — where SDFT shines

Local harness (`sdft.bfcl`) loads
[gorilla-llm/Berkeley-Function-Calling-Leaderboard](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard),
generates with transformers + LFM `apply_chat_template(..., tools=...)`, and
scores AST accuracy against `possible_answer` lists. Categories:
`simple` · `multiple` · `parallel` · `irrelevance`.

**Split:** eval = first N / category; train = the following M / category
(overlap guarded by id + normalized question). Gold targets are LFM JSON
tool-call blocks from `possible_answer`; GRPO uses `bfcl_reward` (+1/−1).

#### LFM2.5-230M — BFCL-trained (`16` train + `32` eval / cat, GRPO `max_steps=32`, fp32)

| Arm | Overall | Simple | Multiple | Parallel | Irrelevance | Mean lat (s) | tok/s | Train s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | **71.1%** | 84.4% | 71.9% | 50.0% | 78.1% | 0.40 | 106 | — |
| gold SFT | 54.7% | 84.4% | 75.0% | 0.0% | 59.4% | 0.58 | 97 | 18.1 |
| SDFT | 68.0% | 81.2% | 78.1% | 59.4% | 53.1% | 0.38 | 100 | 18.2 |
| GRPO | 69.5% | 84.4% | 75.0% | 56.2% | 62.5% | 0.42 | 98 | 40.8 |

Adapters: `outputs/compare/bfcl-{sft-gold,sdft,grpo}`. Small-n gold SFT collapses
**parallel** (0% — format overfitting); SDFT recovers parallel above base (59.4%
vs 50.0%) while trading some irrelevance. That parallel recovery is the cleanest
local SDFT signal in this repo.

#### LFM2.5-1.2B-Instruct — BFCL-trained (`8` train + `16` eval / cat, GRPO `max_steps=8`, fp16)

| Arm | Overall | Simple | Multiple | Parallel | Irrelevance | Mean lat (s) | tok/s | Train s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 75.0% | 75.0% | 62.5% | 68.8% | 93.8% | 0.72 | 69 | — |
| gold SFT | **78.1%** | 75.0% | 62.5% | 81.2% | 93.8% | 0.70 | 64 | 28.1 |
| SDFT | 76.6% | 81.2% | 62.5% | 68.8% | 93.8% | 0.67 | 64 | 24.0 |
| GRPO | 75.0% | 75.0% | 62.5% | 68.8% | 93.8% | 0.78 | 64 | 27.4 |

On this smaller 1.2B slice, gold edges overall; SDFT stays near base without the
230M-style parallel collapse.

### Qualitative — `/perf` (refusals → answers)

Side-by-side chat on open-ended prompts (base vs SDFT merge,
AlpacaEval-faithful zero-shot):

| Prompt | Base LFM2.5-230M | LFM SDFT merge |
|---|---|---|
| Sew a button on a shirt | Refusal: *“I'm sorry, but I can't assist with that.”* | Step-by-step sewing guide (~246 tok, ~124 tok/s) |
| How do I make apple juice? | Refusal / capability hedge | Ingredient list + recipe (~168 tok, ~117 tok/s) |

## Reproduce

Flags below match the published tables. Artifacts land in `outputs/compare/`
(gitignored). Configs: [`configs/compare/`](configs/compare/).

```bash
# BFCL 230M (showcase table)
uv run python scripts/run_bfcl_baselines.py \
  --suite 230m --num-train-per-cat 16 --num-eval-per-cat 32 --max-grpo-steps 32

# BFCL 1.2B Instruct
uv run python scripts/run_bfcl_baselines.py \
  --suite 1_2b --num-train-per-cat 8 --num-eval-per-cat 16 --max-grpo-steps 8

# Eval only (base or any adapter)
uv run python scripts/run_bfcl_eval.py --suite 230m --num-examples 32
uv run python scripts/run_bfcl_eval.py --suite 230m --adapter outputs/compare/bfcl-sdft --arm sdft

# Batch-size-1 Alpaca heuristic (secondary table below)
uv run python scripts/run_batch1_comparison.py --num-train 32 --num-eval 16 --max-grpo-steps 16

# /perf qualitative
uv run python -m web.app   # → http://127.0.0.1:8765/perf
```

Custom local JSONL: point a config at an Alpaca-style file (`dataset: json` +
`data_files: ...`) and run the Quick-start three steps with `--config`.

Colab (in-sample AE2): [`notebooks/local_sdft_colab.ipynb`](notebooks/local_sdft_colab.ipynb)
— `SMOKE=False` trains and scores all **805** AlpacaEval 2.0 instructions on the
same pool (train-set average, not held-out).

## Web demo

```bash
uv run python -m web.app
# → http://127.0.0.1:8765
```

| Route | What it does |
|---|---|
| `/` | Overview + recent benchmarks |
| `/data` | Online learning: each message is tone feedback, then a tiny LoRA SDFT update, then a reply |
| `/perf` | Base vs adapter chat with SSE streaming, prompt-strategy ablations, latency Gantt |
| `/perf/{run_id}` | Run detail |

Online learning defaults: [`configs/online_learning.yaml`](configs/online_learning.yaml)
(`batch_size: 1`, `train_steps: 2`).

## Other runs

Secondary numbers — useful for wiring checks, not the main SDFT story.

### Batch-size-1 heuristic (Alpaca held-out)

Score = `instruction_reward` (non-refusal + length + gold lexical overlap) — no
API judge. Arms stay close without a GPT-4 judge.

| Arm | Mean reward | Refusal rate | Mean chars | Train s |
|---|---:|---:|---:|---:|
| base | 1.179 | 0.000 | 488.3 | — |
| gold SFT | **1.190** | 0.000 | 476.2 | 10.7 |
| SDFT | 1.176 | 0.000 | 516.5 | 10.3 |
| GRPO | 1.177 | 0.000 | 519.6 | 35.5 |

LFM2.5-230M, `num_train=32`, `num_eval=16`, GRPO `max_steps=16`, fp32.
Same update style as `/data` (`batch_size=1`; GRPO group size 2).

### OpenClaw tool-use (best arm)

Identity SDFT on curated tool-call trajectories → `outputs/openclaw-tooluse-merged`.
Held-out bank: 29 questions. Best arm **SDFT+OS+CoT**: **24.1%** pass@1
(+3.4 pp over base ZS 20.7%). Details:
[docs/openclaw-tooluse-sdft.md](docs/openclaw-tooluse-sdft.md).

```bash
uv sync --extra toolcall
uv run python scripts/run_openclaw_ablation.py --skip-data --skip-train --format lfm
```

## LoRA targets for the LFM2 architecture

LFM2.5-230M and LFM2.5-1.2B share the same hybrid layout pattern:
**attention (GQA) blocks + short-convolution blocks**. 230M: 6 attn + 8 conv;
1.2B: 6 GQA + 10 LIV conv (16 layers total). Module leaf names:

| Module | Location | Count |
|---|---|---|
| `self_attn.{q,k,v,out}_proj` | attention / GQA blocks | 6× (both sizes) |
| `conv.{in_proj,out_proj,conv}` | conv / LIV blocks | 8× (230M) / 10× (1.2B) |
| `feed_forward.{w1,w2,w3}` | SwiGLU MLPs | per block |

Default targets all four attention projections via a regex over full paths:
`.*self_attn\.(q|k|v|out)_proj`. Note the leaf name `out_proj` exists in **both**
attention and conv blocks — a plain list target `["out_proj"]` would adapt both.
To also adapt MLPs or conv blocks, extend the regex, e.g.
`.*(self_attn\.(q|k|v|out)_proj|feed_forward\.w[123])`.

## Docs

- [Architecture](docs/architecture.md) — package map + entry points
- [Shared contract](docs/shared-contract.md) — CLI/web persistence schemas
- [OpenClaw-RL eval](docs/openclaw-rl-eval.md)
- [OpenClaw tool-use SDFT](docs/openclaw-tooluse-sdft.md)

## Notes

- **Default dataset** is `yahma/alpaca-cleaned` as a stand-in; point
  `data.dataset` at any HF dataset, or use `dataset: json` + `data_files: ...`
  for a local `.jsonl` (Alpaca-style `instruction` / `input` / `output`).
- Self-generated targets are in-distribution but not necessarily *correct* —
  that is the expected SDFT trade-off. For a 230M model, more few-shot demos
  (`generation.num_shots`) or a task-matched demo pool helps.
- To *cross*-distill instead (bigger teacher → 230M student), run `sdft.generate`
  with a config whose `model.name` is `LiquidAI/LFM2.5-1.2B-Instruct`, then train
  with the 230M config. On MPS, prefer `dtype: float16` for 1.2B.
- GRPO's `instruction` reward is a local heuristic (refusal / length / gold
  overlap). For verifiable math/tool tasks, set `grpo.reward: boxed`.
- LFM2.5 is released under the [LFM Open License v1.0](https://huggingface.co/LiquidAI/LFM2.5-230M)
  (free below $10M annual revenue).
- This project code is MIT (see `LICENSE`).
