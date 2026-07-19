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
