# Local-SDFT

Fine-tune a **230M** or **1.2B** language model on your laptop — with
**Self-Distillation Fine-Tuning (SDFT)**, plain LoRA SFT, and **GRPO** —
and watch it learn live in a browser.

Built on [Liquid AI LFM2.5-230M](https://huggingface.co/LiquidAI/LFM2.5-230M) and
[LFM2.5-1.2B-Thinking](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking).
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
  (**standalone**, no repo clone): few-step curated demo — seeded
  `alpaca-cleaned[:STEP_COUNT]` with `batch_size=1`, ZS / ICL / CoT + gold SFT
  + SDFT LoRA, fast heuristic scoreboard + side-by-side table (no slow AE2
  judge); no GRPO)

See [docs/architecture.md](docs/architecture.md) for the package map.

## Requirements

- Python ≥ 3.11, [uv](https://docs.astral.sh/uv/)
- Tested on Apple Silicon (M2 Max, trains on MPS in fp32); CUDA and CPU work too
  (device is auto-detected: MPS → CUDA → CPU)

## Setup

```bash
uv sync
# optional: official AlpacaEval 2 judge
uv sync --extra alpacaeval
```

Dependencies: `torch>=2.6` · `transformers>=4.54` (`Lfm2ForCausalLM`, no
trust-remote-code) · `trl>=0.19` (`SFTTrainer` / `GRPOTrainer`) · `peft>=0.15`.
Optional: `alpaca-eval` via the `alpacaeval` extra. Local judge needs no API key
(`JUDGE=local`, default); official GPT-4-Turbo needs `JUDGE=openai` + `OPENAI_API_KEY`.

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
AlpacaEval-style scoring: local open judge ≈ AE2 pairwise protocol (not
leaderboard-equivalent); official GPT-4-Turbo LC win-rate needs `JUDGE=openai`
+ `OPENAI_API_KEY` via `scripts/run_alpaca_eval.py`. Colab uses the local
Qwen/Qwen3.5-9B 4-bit judge only.

### BFCL-v3 local AST — where SDFT shines

Local harness (`sdft.bfcl`) loads
[gorilla-llm/Berkeley-Function-Calling-Leaderboard](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard),
generates with transformers + LFM `apply_chat_template(..., tools=...)`, and
scores AST accuracy against `possible_answer` lists. Categories:
`simple` · `multiple` · `parallel` · `irrelevance`.

**Local category bank sizes** (cached under `data/bfcl/`): simple **400**,
multiple **200**, parallel **200**, irrelevance **240**.

**Split:** eval = first N / category; train = the following M / category
(overlap guarded by id + normalized question). Showcase tables use
`N=32`, `M=64` (128 eval / 256 train total). Gold targets are LFM JSON
tool-call blocks from `possible_answer`; GRPO uses `bfcl_reward` (+1/−1).

#### LFM2.5-230M — BFCL-trained (`64` train + `32` eval / cat, GRPO `max_steps=256`, fp32)

| Arm | Overall | Simple | Multiple | Parallel | Irrelevance | Mean lat (s) | tok/s | Train s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | **71.1%** | 84.4% | 71.9% | 50.0% | 78.1% | 0.38 | 113 | — |
| gold SFT | 58.6% | 81.2% | 65.6% | 0.0% | 87.5% | 0.47 | 103 | 61.3 |
| SDFT | 66.4% | 81.2% | 81.2% | 53.1% | 50.0% | 0.36 | 102 | 55.2 |
| GRPO | **71.1%** | 81.2% | 75.0% | 62.5% | 65.6% | 0.40 | 103 | 243.9 |

Adapters: `outputs/compare/bfcl-{sft-gold,sdft,grpo}`
(`outputs/compare/bfcl_comparison_full.json`). Gold SFT still collapses
**parallel** (0% — format overfitting); SDFT recovers parallel above base
(53.1% vs 50.0%) while trading irrelevance. GRPO matches base overall and
lifts parallel to 62.5%.

#### LFM2.5-1.2B-Thinking — BFCL-trained (`64` train + `32` eval / cat, GRPO `max_steps=256`, fp16)

| Arm | Overall | Simple | Multiple | Parallel | Irrelevance | Mean lat (s) | tok/s | Train s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 26.6% | 9.4% | 3.1% | 0.0% | 93.8% | 2.98 | 64 | — |
| gold SFT | **66.4%** | 90.6% | 81.2% | 0.0% | 93.8% | 0.76 | 62 | 157.2 |
| SDFT | 24.2% | 0.0% | 0.0% | 0.0% | 96.9% | 3.74 | 46 | 201.9 |
| GRPO | 27.3% | 12.5% | 3.1% | 0.0% | 93.8% | 3.51 | 55 | 1473.7 |

Adapters: `outputs/compare/bfcl-1_2b-{sft-gold,sdft,grpo}`
(`outputs/compare/bfcl_1_2b_comparison_full.json`). Same `max_new_tokens=192`
budget as the Instruct configs: **base / SDFT / GRPO often exhaust the budget
inside `<think>…</think>`** (mean ~191 gen tokens) and emit no tool call, so
AST scores collapse except irrelevance. **Gold SFT** short-circuits to tool
calls (~47 tok) and jumps to 66.4% overall — but still 0% parallel. Raising
the decode budget is the obvious follow-up for a fair Thinking bake-off.

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
# BFCL 230M (showcase table → outputs/compare/bfcl_comparison_full.json)
uv run python scripts/run_bfcl_baselines.py \
  --suite 230m --num-train-per-cat 64 --num-eval-per-cat 32 --max-grpo-steps 256 \
  --out outputs/compare/bfcl_comparison_full.json

# BFCL 1.2B Thinking (showcase → outputs/compare/bfcl_1_2b_comparison_full.json)
uv run python scripts/run_bfcl_baselines.py \
  --suite 1_2b --num-train-per-cat 64 --num-eval-per-cat 32 --max-grpo-steps 256 \
  --out outputs/compare/bfcl_1_2b_comparison_full.json

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

Colab (few-step demo): [`notebooks/local_sdft_colab.ipynb`](notebooks/local_sdft_colab.ipynb)
— **standalone notebook** (no Local-SDFT clone/import). Shuffle
`yahma/alpaca-cleaned` with a fixed seed, take `[:STEP_COUNT]` (default 32),
train gold SFT / SDFT with `batch_size=1` for that many steps, then batched
ZS / ICL / CoT / adapter generations on the **same** slice. Primary metrics
are refusal rate / length / instruction reward + a qualitative table (not a
full AE2 / Qwen judge pass). One base load covers teacher → train → eval.
No GRPO.

```bash
uv sync --extra alpacaeval
# default local judge: Qwen/Qwen3.5-9B 4-bit (Colab T4); optional override:
# export ALPACA_EVAL_LOCAL_JUDGE=...
export JUDGE=local
# or official: export JUDGE=openai OPENAI_API_KEY=...
uv run python scripts/run_alpaca_eval.py \
  --model-outputs outputs/alpacaeval/sdft/model_outputs.json \
  --name sdft --output-dir outputs/alpacaeval/sdft
```

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
  with a config whose `model.name` is `LiquidAI/LFM2.5-1.2B-Thinking`, then train
  with the 230M config. On MPS, prefer `dtype: float16` for 1.2B.
- GRPO's `instruction` reward is a local heuristic (refusal / length / gold
  overlap). For verifiable math/tool tasks, set `grpo.reward: boxed`.
- LFM2.5 is released under the [LFM Open License v1.0](https://huggingface.co/LiquidAI/LFM2.5-230M)
  (free below $10M annual revenue).
- This project code is MIT (see `LICENSE`).
