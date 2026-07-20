# Local-SDFT

Fine-tune a **230M** or **1.2B** language model on your laptop — with
**Self-Distillation Fine-Tuning (SDFT)**, plain LoRA SFT, and **GRPO** —
and watch it learn live in a browser.

Built on [Liquid AI LFM2.5-230M](https://huggingface.co/LiquidAI/LFM2.5-230M) and
[LFM2.5-1.2B-Instruct](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct)
(HF has no bare `LFM2.5-1.2B` id — use the Instruct checkpoint).
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
  (setup cell uninstalls Colab's preinstalled `torchao<0.16`, which otherwise breaks peft≥0.19 LoRA)

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

### Smoke test (~2 min end to end)

```bash
uv run python -m sdft.generate --num-examples 16 --out data/smoke.jsonl
uv run python -m sdft.train --data data/smoke.jsonl --output-dir outputs/smoke --epochs 1
uv run python -m sdft.merge --adapter outputs/smoke --out outputs/smoke-merged
```

Utility: `uv run python -m sdft.inspect` prints the model's LoRA-targetable
module names.

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

## Batch-size-1 baselines (SFT / SDFT / GRPO)

Same update style as the online demo — small LoRA, `batch_size=1` for SFT/SDFT,
and the smallest valid GRPO group (`num_generations=2`):

```bash
# 230M (default; --max-grpo-steps 16 matches the published table below)
uv run python scripts/run_batch1_comparison.py --num-train 32 --num-eval 16 --max-grpo-steps 16
# omit --max-grpo-steps for a full uncapped GRPO run (may differ from the table)
# 1.2B Instruct (float16, smaller context — see configs/compare/batch1_1_2b_*.yaml)
uv run python scripts/run_batch1_comparison.py --suite 1_2b --num-train 16 --num-eval 8 --max-grpo-steps 8
# faster smoke:
uv run python scripts/run_batch1_comparison.py --num-train 16 --num-eval 8 --max-grpo-steps 8
```

Configs live under [`configs/compare/`](configs/compare/). Results land in
`outputs/compare/batch1_comparison.json` / `batch1_1_2b_comparison.json` (gitignored).

Gold SFT / GRPO alone:

```bash
uv run python -m sdft.train --config configs/compare/batch1_sft_gold.yaml --target gold
uv run python -m sdft.grpo_train --config configs/compare/batch1_grpo.yaml --data data/compare/batch1_grpo.jsonl
```

### BFCL (local AST subset)

Local BFCL-v3 single-turn subset (no vLLM / official leaderboard). Categories:
`simple`, `multiple`, `parallel`, `irrelevance`.

Train gold SFT / SDFT / GRPO **on BFCL** (held-out first-N/category for eval), then
score with the AST harness:

```bash
# 230M: 16 train + 32 eval per category (64 / 128 total), GRPO capped at 32 steps
uv run python scripts/run_bfcl_baselines.py --suite 230m --num-train-per-cat 16 --num-eval-per-cat 32 --max-grpo-steps 32
# 1.2B Instruct smoke (fp16): 8 train + 16 eval per category, GRPO 8 steps
uv run python scripts/run_bfcl_baselines.py --suite 1_2b --num-train-per-cat 8 --num-eval-per-cat 16 --max-grpo-steps 8
```

Eval only (base or any adapter):

```bash
uv run python scripts/run_bfcl_eval.py --suite 230m --num-examples 32
uv run python scripts/run_bfcl_eval.py --suite 1_2b --num-examples 32
uv run python scripts/run_bfcl_eval.py --suite 230m --adapter outputs/compare/bfcl-sdft --arm sdft
```

Configs: `configs/compare/bfcl_*.yaml`. Results:
`outputs/compare/bfcl_comparison.json` / `bfcl_1_2b_comparison.json`.

### Custom local JSONL

Point a config at your own Alpaca-style file (`dataset: json` +
`data_files: data/my_dataset.jsonl`, one JSON object per line with
`instruction` / `input` / `output`) and run the three steps with
`--config path/to/your.yaml`.

### OpenClaw-RL tool-calling eval

See [docs/openclaw-rl-eval.md](docs/openclaw-rl-eval.md) for the ReTool-style
tool loop and AIME-2024 benchmark adapter. Quick smoke:

```bash
uv sync --extra toolcall
bash scripts/run_openclaw_rl_eval.sh
```

## Evaluation results

Numbers below are from local runs on Apple Silicon (MPS, ~32 GB unified memory).
Artifacts live under `outputs/benchmarks/` and `outputs/compare/` (gitignored).
Official AlpacaEval 2 **LC win-rate** was **not** computed (no `OPENAI_API_KEY` /
GPT-4 Turbo judge). BFCL scores are a **local AST subset**, not the official
Berkeley Function-Calling Leaderboard.

### Batch-size-1 comparison (held-out heuristic reward)

Score = `instruction_reward` (non-refusal + length + gold lexical overlap) — no API judge.

#### LFM2.5-230M (`num_train=32`, `num_eval=16`, GRPO `max_steps=16`, fp32)

| Arm | Mean reward | Refusal rate | Mean chars | Train s |
|---|---:|---:|---:|---:|
| base | 1.179 | 0.000 | 488.3 | — |
| gold SFT | **1.190** | 0.000 | 476.2 | 10.7 |
| SDFT | 1.176 | 0.000 | 516.5 | 10.3 |
| GRPO | 1.177 | 0.000 | 519.6 | 35.5 |

#### LFM2.5-1.2B-Instruct smoke (`num_train=16`, `num_eval=8`, GRPO `max_steps=8`, fp16)

| Arm | Mean reward | Refusal rate | Mean chars | Train s |
|---|---:|---:|---:|---:|
| base | 1.400 | 0.000 | 333.0 | — |
| gold SFT | 1.399 | 0.000 | 333.1 | 19.5 |
| SDFT | **1.408** | 0.000 | 338.0 | 11.9 |
| GRPO | 1.400 | 0.000 | 333.0 | 30.9 |

On the local heuristic the arms stay close (expected without a GPT-4 judge).
Reproduce (include `--max-grpo-steps` to match the caps above; omit it for a full uncapped GRPO run):

```bash
uv run python scripts/run_batch1_comparison.py --num-train 32 --num-eval 16 --max-grpo-steps 16
uv run python scripts/run_batch1_comparison.py --suite 1_2b --num-train 16 --num-eval 8 --max-grpo-steps 8
```

Also see the [Colab notebook](notebooks/local_sdft_colab.ipynb).

### BFCL-v3 local AST subset

**Not** official BFCL leaderboard scores. Local harness (`sdft.bfcl`) loads
[gorilla-llm/Berkeley-Function-Calling-Leaderboard](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard),
generates with transformers + LFM `apply_chat_template(..., tools=...)`, and
scores AST accuracy against `possible_answer` lists. Categories run:
`simple` · `multiple` · `parallel` · `irrelevance`. Skipped: live, multi-turn,
executable, web-search, Java/JS.

**Split:** eval = first N examples/category (same slice as default BFCL eval);
train = the following M examples/category. Overlap guarded by example id and
normalized question text. Gold targets are LFM JSON tool-call blocks derived from
`possible_answer`; GRPO uses `bfcl_reward` (+1/−1 AST / irrelevance).

#### 230M — BFCL-trained (`16` train + `32` eval / category, GRPO `max_steps=32`, fp32 MPS)

| Arm | Overall | Simple | Multiple | Parallel | Irrelevance | Mean lat (s) | tok/s | Train s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | **71.1%** | 84.4% | 71.9% | 50.0% | 78.1% | 0.40 | 106 | — |
| gold SFT | 54.7% | 84.4% | 75.0% | 0.0% | 59.4% | 0.58 | 97 | 18.1 |
| SDFT | 68.0% | 81.2% | 78.1% | 59.4% | 53.1% | 0.38 | 100 | 18.2 |
| GRPO | 69.5% | 84.4% | 75.0% | 56.2% | 62.5% | 0.42 | 98 | 40.8 |

Adapters: `outputs/compare/bfcl-{sft-gold,sdft,grpo}`. Small-n gold SFT collapsed
parallel (format overfitting); SDFT/GRPO lift parallel vs base while trading some
irrelevance accuracy.

#### 1.2B-Instruct — BFCL-trained smoke (`8` train + `16` eval / category, GRPO `max_steps=8`, fp16)

| Arm | Overall | Simple | Multiple | Parallel | Irrelevance | Mean lat (s) | tok/s | Train s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 75.0% | 75.0% | 62.5% | 68.8% | 93.8% | 0.72 | 69 | — |
| gold SFT | **78.1%** | 75.0% | 62.5% | 81.2% | 93.8% | 0.70 | 64 | 28.1 |
| SDFT | 76.6% | 81.2% | 62.5% | 68.8% | 93.8% | 0.67 | 64 | 24.0 |
| GRPO | 75.0% | 75.0% | 62.5% | 68.8% | 93.8% | 0.78 | 64 | 27.4 |

Reproduce:

```bash
uv run python scripts/run_bfcl_baselines.py --suite 230m --num-train-per-cat 16 --num-eval-per-cat 32 --max-grpo-steps 32
uv run python scripts/run_bfcl_baselines.py --suite 1_2b --num-train-per-cat 8 --num-eval-per-cat 16 --max-grpo-steps 8
```

#### Reference — Alpaca-trained adapters on BFCL (not tool-call SDFT)

Earlier smoke scored Alpaca batch1 adapters on the same first-32/category slice.
They barely move BFCL (expected — instruction SFT, not tool-call training):

| Arm | Overall (230M, n=32/cat) | Note |
|---|---:|---|
| base | **71.1%** | same held-out slice |
| gold SFT† / SDFT† / GRPO† | 68.8–71.1% | `outputs/compare/batch1-*` |

† Alpaca batch1 compare adapters. Full per-category tables live in git history /
`outputs/benchmarks/bfcl/` when re-run with `--adapter outputs/compare/batch1-sdft`.

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
Model: **LFM2.5-230M** (1.2B OpenClaw ablation not re-run in this pass).

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
