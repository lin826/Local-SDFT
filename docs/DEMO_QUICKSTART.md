# Demo quickstart — QA, tool use, database interaction

Three self-contained, narrated demos of **online SDFT** (learn-while-serving) on
`LiquidAI/LFM2.5-230M`. Each one starts from the generic base model, coaches it
live with reward-selected self-distillation, and shows a success curve climbing
on **held-out** inputs it was never coached on — all on-device and offline.

| # | Demo | Script | What the model learns |
|---|---|---|---|
| 1 | **Simple QA** | `scripts/demo_correct_once.py` | a Q&A *habit* (answer in one sentence) that transfers across topics |
| 2 | **Tool use** | `scripts/demo_toolcall.py` | to call a calculator: `question → calc(...)` |
| 3 | **Database** | `scripts/demo_sqlite.py` | to answer questions by writing SQL a real engine runs |

All three share the same recipe: sample a few replies, keep the best-rewarded one
(reshaped into a guaranteed-correct target by a *shaper*), take a few LoRA steps,
save a versioned adapter. The only thing that changes per demo is the reward
function (`configs/demo_*.yaml → online.reward_fn`).

## Setup (once)

```bash
uv sync --extra online          # adds rich + fastapi to the base stack

# Pre-download the model so the demo can run offline (required on an offline box):
python -c "from huggingface_hub import snapshot_download as d; d('LiquidAI/LFM2.5-230M')"
```

Runs on CPU, but the numbers below were validated on a single H100; each demo is
1–3 minutes on GPU. Device is auto-detected (MPS → CUDA → CPU).

---

## 1. Simple QA — "correct it once, never again"

You want one-sentence answers. You fix a few replies in plain language; the habit
sticks — and generalizes. The honest proof it *learned* rather than *memorized*:
corrections are given **only on cooking questions**, and success is measured on
**held-out programming questions**.

```bash
python scripts/demo_correct_once.py
```

What you'll see (validated, LFM2.5-230M, offline):

```
Before:                                 0%  one-sentence on held-out programming
4 plain corrections (cooking only):   → 50%
consolidation (replays those 4):        50% → 83% → 100%
adapter ON: 100%      adapter OFF (base): 0%
```

It also runs the **"why not just prompt/retrieve?"** head-to-head — the base model
gets the same corrections in-context (ICL) or retrieved (RAG), no training:

| approach | held-out accuracy | extra tokens/call |
|---|---|---|
| base (no help) | 0% | 0 |
| ICL (rule + all corrections) | 67% | +144, every call |
| RAG (rule + top-3 retrieved) | 100% | +107, every call |
| **finetuned (ours)** | **100%** | **+0** |

The point to say out loud: finetuning folds the corrections into the weights —
equal-or-better accuracy at **zero** per-call context cost.

> Prefer an interactive version? `python -m sdft.online.cli serve --config
> configs/demo_house_style.yaml` opens a web UI with a live success sparkline and
> an adapter A/B toggle (see [DEMO.md](DEMO.md)).

---

## 2. Tool use — "it learns to use a calculator"

Teach the model to answer arithmetic by emitting `<tool>calc("…")</tool>` instead
of doing freehand math. **Coach on small numbers, test on large numbers that never
appear in coaching** (disjointness is asserted in code), so a correct held-out
answer can only come from learning the skill, not from memorizing an answer.

```bash
python scripts/demo_toolcall.py --rounds 6
```

What you'll see (validated, LFM2.5-230M, offline):

```
1. base answers freehand           → 0%   held-out correct (and confidently wrong)
2. coach ~2 rounds                 → 100% held-out correct
3. adapter OFF (base) again        → 0%
```

The appear/disappear on the **same unseen inputs** as you toggle the adapter is the
"it learned" moment. Reward = a valid `calc()` call whose expression evaluates to
the right answer (so freehand-correct scores 0 — it rewards *tool use*); arithmetic
is checked with an AST-safe evaluator (`sdft/online/tools.py`), never `eval`.

---

## 3. Database interaction — text→SQL against a real SQLite engine

The "it drives real software" demo. The model answers natural-language questions
about an actual database by writing a query; a **real SQLite engine runs it**; the
reward is whether the returned rows match a gold query's rows. Coach and test use
**disjoint** categories/cities (`Books`/`Seattle` in coaching, `Toys`/`Denver` at
test), so a correct executed answer means it wrote correct SQL, not that it
memorized a value.

```bash
python scripts/demo_sqlite.py     # prints the SQL it wrote + the real rows
```

What you'll see (validated, LFM2.5-230M, offline):

```
base (no adapter):   0%   — writes no query at all
after coaching:     50%   (peaked 83% in an earlier run; run-to-run variance)

Held-out questions, executed against the real DB:
  "products in the 'Toys' category?"  → SELECT COUNT(*) ... WHERE category='Toys'  → (2)      ✓
  "customers from Denver"             → SELECT name ... WHERE city='Denver'         → Bob, Eve ✓
```

Honest edge: the 230M reliably learns **COUNT / WHERE-filter** patterns and
transfers them to unseen values; **superlatives** (`ORDER BY … LIMIT 1`) and
**JOINs** are hazier and account for most misses (the 1.2B is the lever there).

**Safety — the model's SQL is untrusted, so execution is jailed**
(`sdft/online/sqlenv.py`): read-only connection (`mode=ro`) + `PRAGMA query_only`
+ a **SQLite authorizer that permits only reads** (SELECT/READ/FUNCTION; denies
every write/DDL/`ATTACH`/pragma even if the parser is fooled) + one statement per
call + an instruction-budget timeout. `tests/test_sqlenv.py` asserts that
`DROP`/`UPDATE`/`INSERT`/stacked-statement/`ATTACH` all fail and leave the database
byte-for-byte unchanged.

Served **schema-less**: the schema is only a sampling-time teacher hint
(`coach_instruction`), so a correct query means the model learned the schema into
its weights.

---

## Running on the cluster (GPU node)

Each demo has a launcher that sets the offline environment (`unset HF_TOKEN`,
`PYTHONNOUSERSITE=1`, `HF_HUB_OFFLINE=1`, cache under `HF_HOME`). Submit to a GPU
node and point `PYTHON` at the CUDA venv:

```bash
GPU_PY=/proj/inf-scaling/zwhong/projs/local_online_sdft/.venv-gpu/bin/python

bsub -G grp_preemptable -q preemptable -gpu "num=1" -J sdft_qa \
     -o bsub_outputs/%J.out env PYTHON=$GPU_PY bash scripts/demo_correct_once.sh
bsub -G grp_preemptable -q preemptable -gpu "num=1" -J sdft_tool \
     -o bsub_outputs/%J.out env PYTHON=$GPU_PY bash scripts/demo_toolcall.sh
bsub -G grp_preemptable -q preemptable -gpu "num=1" -J sdft_sql \
     -o bsub_outputs/%J.out env PYTHON=$GPU_PY bash scripts/demo_sqlite.sh
```

Read the run with `bpeek <jobid>` (live) or `bsub_outputs/<jobid>.out` (after).

## Notes

- **Fresh each run:** every script wipes its own `online.db_path` and
  `adapters_dir` first, so the curve reflects only this run.
- **Small held-out sets (5–6 prompts) are noisy** — expect run-to-run variance in
  the exact percentages; the *direction* (base → learned, adapter OFF → ON) is the
  robust result.
- **Swap the task** by pointing `online.reward_fn` at a different `@reward("name")`
  in `sdft/online/reward.py`.
- More demos (continual learning, lifelong skill-accumulation + experience replay,
  inbox triage) are documented in [DEMO.md](DEMO.md).
```