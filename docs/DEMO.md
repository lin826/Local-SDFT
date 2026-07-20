# Airplane-Mode Coach — feel online on-device RL in 3 minutes

The demo that makes on-device online learning *visceral*: start generic, give the model
light feedback, and watch it acquire a new behavior — live, offline, with a success curve
climbing on prompts it was never coached on. (Design discussion: issue #1.)

## Run it

```bash
uv sync --extra online          # fastapi/uvicorn/rich for the web UI
# pre-download the model once (needed if your training box is offline):
python -c "from huggingface_hub import snapshot_download as d; d('LiquidAI/LFM2.5-230M')"

# Web UI (the show): success sparkline, adapter A/B toggle, ✈︎ OFFLINE badge
python -m sdft.online.cli serve --config configs/demo_house_style.yaml
# open http://127.0.0.1:8080  → click "Coach ×10", watch the curve climb,
# then toggle the adapter OFF/ON to A/B the same held-out prompts.

# Headless twin (cluster/CI): prints the climbing success@held-out
python -m sdft.online.cli demo --config configs/demo_house_style.yaml --rounds 6
```

## The four "wow" beats

1. **Live success curve** — `success@held-out` climbs as you coach (RL legibility).
2. **Held-out generalization** — measured on prompts never coached on → it learned a
   *skill*, not a lookup. This is the beat RAG can't match.
3. **Adapter A/B toggle** — one control flips LoRA off/on; **same prompt**, generic ↔
   learned, a second apart.
4. **✈︎ OFFLINE** — inference *and* training happen on-device; feedback never leaves it.

The web UI also has a **"Why not just prompt / retrieve?"** card: one click runs a
live base/ICL/RAG/finetuned head-to-head (`/v1/demo/compare`) and shows held-out
accuracy alongside the per-call **token tax** — ICL/RAG get the same learned
examples in-context, finetuning matches or beats them at **+0 tokens/call**.

## The task and the learning rule

Default task `house_style` (in `sdft/online/reward.py`): a reply must open with a **TL;DR**,
have **≤3 bullets**, and **end with a question**. The base 230M doesn't do this; it's
objectively checkable (so the curve is hands-free); and it obviously generalizes.

Learning rule = **reward-selected on-policy self-distillation** (RAFT-flavored, honest
"online RL"): sample N rollouts, keep the best-rewarded reply, distill the model onto it
with a few LoRA steps, save a versioned adapter. Swap the task by setting `online.reward_fn`
to `five_words`, `terse`, or your own `@reward("name")`.

## Why finetuning, not RAG (say it during the demo)

- RAG injects **facts**; it can't change **behavior/style/skill** — this demo changes behavior.
- RAG re-pays the **prompt tax** every call (latency + context budget) — brutal on a phone;
  a learned adapter is free at inference.
- The behavior **generalizes** to unseen inputs with no hand-crafted system prompt.
- **Private**: the feedback and the resulting skill stay on the device.

## Receipts to show

`success@held-out` before/after · #coaching rounds to competence · adapter size (MBs) ·
tokens/sec · peak RAM — all on-device.

## Variant: tool-calling — "it learns to use a calculator" (the LEARNS-not-memorizes demo)

The most rigorous version. Teach the model to answer arithmetic by emitting
`<tool>calc("…")</tool>`. **Coach on small numbers, test on large numbers that
never appear in coaching** (disjointness is asserted in code + a unit test), so a
correct held-out answer can only come from learning the *skill* (question → tool
call), never from memorized answers.

```bash
python scripts/demo_toolcall.py --rounds 6          # narrated walkthrough
python -m sdft.online.cli demo --config configs/demo_toolcall.yaml   # task-generic
```

Validated on LFM2.5-230M (H100, offline): base does freehand math and is **0%**
correct; after ~2 coaching rounds it emits correct tool calls and is **100%** on
held-out problems; toggling the adapter off returns it to **0%**. The
appear/disappear on the *same unseen inputs* is the "it learned" moment.

Reward = a valid `calc()` call whose expression evaluates to the right answer
(rewards tool *use*, so freehand-correct scores 0); a shaper supplies a
guaranteed-correct tool call as the SFT target; arithmetic is evaluated with an
AST-safe evaluator (`sdft/online/tools.py`), never `eval`.

## Variant: continual learning — adapt now, recover old skills fast

Answers the plasticity/fast-recovery story ("forgetting is fine as long as you
pick up fast"). The assistant's response mode changes with context: mode A =
structured briefing, mode B = a direct one-line answer. It runs **A → B → A**
and reports success on *both* modes each phase.

```bash
python scripts/demo_continual.py
```

Validated on LFM2.5-230M (H100, offline):

```
Phase 1  learn briefing (A):   A 0% → 100% in 3 passes
Phase 2  switch to direct (B): B → 67%, A fades 100% → 33%   (forgetting is fine)
Phase 3  back to briefing (A): A recovers to 83% in ONE pass
Savings: A reached competence in 3 passes first time, 1 on return.
```

The point is not "never forget" — it's cheap recovery. A small on-device replay
buffer plus primed weights bring an old task back in a fraction of the original
steps. Switch mode B via `MODE_B` in the script (`direct`, `terse`, `five_words`,
`house_style`, or your own reward+shaper).

## Flagship: lifelong learning — a growing skill repertoire, and when replay matters

The long-horizon continual demo: your assistant picks up **four distinct,
trigger-keyed skills** the way it would over a week of real use, introduced one
at a time —

| you type | it learns to |
|---|---|
| `Summarize: <text>` | reply with a one-line summary |
| `What is A op B?` | emit a `calc()` tool call |
| `List <topic>` | answer as three bullet points |
| `Reply to: <message>` | end the reply with your fixed sign-off |

After each new skill we re-measure **every** skill learned so far on held-out
prompts, so you watch the repertoire *accumulate* (or overwrite itself). The
identical curriculum runs with replay OFF and ON, swept across LoRA capacity.

```bash
python scripts/demo_lifelong.py                      # sweeps r=32 and r=2 (GPU-validated)
python scripts/demo_lifelong.py --ranks 2 --rounds 4 # quick single-capacity look
```

Validated on LFM2.5-230M (H100, offline; ~14 min for the full sweep):

```
Capacity × replay — mean held-out across all 4 skills
  LoRA r=32 (ample):  no-replay 95%  ==  replay 95%   (+0 pts)
  LoRA r=2  (tight):  no-replay 45%  <   replay 65%   (+20 pts)
```

The lesson is **capacity-dependent, and that's the honest point**:

- **Ample capacity (r=32):** the four skills key off different triggers, so they
  occupy different input subspaces and don't interfere — all four reach ~100%
  *with or without replay*. A 230M genuinely accumulates a 4-skill repertoire
  on-device, no rehearsal required.
- **Tight capacity (r=2, 0.17% of params — the realistic always-on adapter
  budget on a phone):** now the skills compete for the same scarce weights.
  Without replay, teaching later skills **overwrites earlier ones** (the `list`
  skill drops to 0%); with replay — rehearsing a few past examples of every skill
  each update — the repertoire holds (`list` recovers to 60%, mean 65% vs 45%).

So **experience replay is what keeps online SDFT stable exactly where edge
devices live**: scarce capacity, learning continually, no second chance at old
data. The replay buffer tags each demonstration by skill and balances rehearsal
across them (`sdft/online/buffer.py`, `max_per_topic_per_batch`). Run-to-run
numbers vary (small held-out sets, stochastic sampling); the direction — replay
≥ no-replay, with the gap opening as capacity tightens — is robust.

This is the deeper cousin of the A→B→A continual demo above: that one shows fast
*recovery* of an overwritten skill; this one shows replay *preventing* the
overwrite in the first place, over a longer horizon and a bigger repertoire.

## A note on task difficulty (honest)

Format/style/tool-call skills (the demos above) imprint reliably on LFM2.5-230M
because they're pattern/policy learning. The **inbox triage** demo
(`scripts/demo_inbox.py`) is a harder *semantic* task (understand an email →
your action across 5 categories): the 230M learns only the lexical categories
(newsletter/manager) and overfits; it wants the larger on-device model
(LFM2-1.2B, still laptop-class) and more coaching. Use it to show the current
capability edge, not a polished 100%.

## Flagship: "Correct it once, never again" (cross-domain generalization)

The demo for the most universal AI frustration — repeating yourself. Your pet
peeve: one-sentence answers. You fix a few replies in plain language (the
correction path), and the habit transfers to a topic it was never corrected on.
Proof it's a learned behavior, not memorized answers: correct only **cooking**
answers, measure on held-out **programming** questions.

```bash
python scripts/demo_correct_once.py
```

Validated on LFM2.5-230M (H100, offline):

```
Before:                                 0%  one-sentence on held-out programming
4 plain corrections (cooking only):   → 50%
consolidation (replays those 4):        50% → 83% → 100%
adapter ON: 100%      adapter OFF (base): 0%
```

Mechanism (honest): a few corrections quickly solve the coaching domain; then the
loop **consolidates** (replays those corrections, no new user input) until the
habit generalizes. You told it once; nothing left the device.

### Why not just prompt (ICL) or retrieve (RAG)?

The correct-once demo includes a fair head-to-head: the *base* model gets the
same corrections in-context (ICL = rule + all corrections; RAG = rule + top-3
retrieved) vs the finetuned adapter. On LFM2.5-230M (held-out programming, the
one-sentence habit):

| approach | held-out accuracy | extra tokens/call |
|---|---|---|
| base (no help) | 0% | 0 |
| ICL (rule + all corrections) | 67% | +144, every call |
| RAG (rule + top-3 retrieved) | 100% | +107, every call |
| **finetuned (ours)** | **100%** | **+0** |

ICL is worse on accuracy (small models follow long prompts unreliably); RAG can
match accuracy but pays a permanent per-call token tax + a retrieval step, and
degrades when retrieval misses or preferences accumulate. Finetuning folds the
corrections into the weights: equal-or-better quality at constant zero context
cost, on-device.
