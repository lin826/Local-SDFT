# Relationship to OpenClaw-RL, and harness interop

[OpenClaw-RL](https://github.com/Gen-Verse/OpenClaw-RL) also does *online* learning
from live model traffic, so it's worth being precise about how this project differs
and how the two interoperate.

## What OpenClaw-RL is

A **fully asynchronous online RL framework** that turns live agent conversations into
training signals. It wraps a self-hosted policy as an OpenAI-compatible proxy
(`/v1/chat/completions` on `:30000`), intercepts multi-turn agent traffic, and uses the
**next environment/tool/user message as a "next-state" reward signal** to score the
previous turn — no human labels. A Process Reward Model (majority vote over `\boxed{1|-1|0}`)
produces a scalar reward; the policy is optimized in the background by one of three methods:

- **Binary RL (GRPO)** — PPO-style clipped surrogate, reward broadcast across response tokens.
- **On-Policy Distillation (OPD)** — a judge extracts a textual *hint* from the next-state,
  a teacher is queried with `prompt + hint`, and the per-token directional advantage is
  `log π_teacher(a_t | s+hint) − log π_θ(a_t | s)` (optionally top-K distribution distillation).
- **Hybrid** of the two.

It is built on **Slime + Megatron-LM + SGLang**, targets **Qwen3/Qwen3.5 (4B–32B)**, and
its default launch uses an **8×GPU node** (multi-node for the SWE setting). There is a
cloud LoRA path via Tinker/Fireworks. It is emphatically a **cloud / multi-GPU-server**
system.

## How this project differs

| Axis | OpenClaw-RL | this project (`sdft.online`) |
|---|---|---|
| Hardware | 8×GPU node (Megatron/SGLang) | single laptop / phone-class device |
| Model | Qwen3/3.5, 4B–32B | LFM2.5-230M (also LFM2-1.2B) |
| Method | RL (GRPO / OPD / hybrid) | on-policy SDFT (forward-KL to a demonstration-conditioned teacher) |
| Reward source | automated PRM over environment "next-state" | the **user's own** interaction — corrections, accepted replies, or a local reward_fn |
| "Online" means | async RL-in-the-loop vs. env feedback | learn from the live **human** while serving |
| Human feedback | none (auto next-state) | central |
| Teacher | separate hint-conditioned teacher model | the **same** model, demonstration-conditioned (no 2nd copy in RAM) |
| Framework | Slime + Megatron + SGLang | transformers + PEFT, one process |

**One-line positioning:** OpenClaw-RL is *RL from automated environment rewards on a GPU
cluster*; this is *self-distillation from live human feedback on an edge device*. The
nearest conceptual overlap is OpenClaw-RL's OPD (distill from a hint-informed teacher) —
we distill on-device from a demonstration-conditioned teacher, driven by the user rather
than a judge.

## Supporting their harness (interop)

OpenClaw-RL's plug-in surface is an **OpenAI-compatible proxy plus three custom fields**
per turn (header or JSON body):

| field | header | meaning |
|---|---|---|
| `session_id` | `X-Session-Id` | trajectory grouping key |
| `turn_type` | `X-Turn-Type` | `main` (train on it) or `side` (housekeeping; skip) |
| `session_done` | `X-Session-Done` | conversation boundary |

Our server (`sdft.online.serve`) honors the **same contract**, so it is a drop-in target
for an OpenClaw client (or any client that emits these fields):

- `turn_type: "side"` turns are served but **excluded** from learning.
- `session_done` closes the conversation, harvesting accepted-reply self-demonstrations.
- `session_id` maps to our `conversation_id`.

```bash
# point an OpenClaw client's provider baseUrl at this server, or test directly:
curl -s localhost:8080/v1/chat/completions \
  -H 'X-Session-Id: sess-1' -H 'X-Turn-Type: main' \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"deploy command?"}]}'
```

This lets the same client stream traffic to *either* an OpenClaw-RL cluster proxy or this
on-device server, and get the learning style appropriate to the deployment: cluster RL from
environment reward, or laptop SDFT from user feedback. We deliberately do **not** reimplement
their Slime trainer / PRM / losses — interop is purely the wire protocol + three headers.
