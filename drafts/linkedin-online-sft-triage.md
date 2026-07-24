# LinkedIn drafts — Online SFT triage

Synced from LinkedIn Pulse preview (`LinkedIn_post_preview.pdf`, 2026-07-23).
Narrative framed as **online SFT** (behavior-supervised LoRA); teacher/demo framing removed — that loop is not what earns the result here.

URLs used (repo slug still `online-sdft-demo`; prose says online SFT):
- LinkedIn Pulse draft: https://www.linkedin.com/pulse/draft/preview/7485799152388317184/
- Colab: https://colab.research.google.com/github/lin826/online-sdft-demo/blob/main/online_sdft_colab.ipynb
- Repo: https://github.com/lin826/online-sdft-demo

---

## Article (Pulse) — synced + SFT framing

# Your taps are the training set: on-device personalization from one bit of feedback

*Online LoRA SFT on a 230M model: notification triage that follows your policy when it drifts. No gold labels, no growing prompt, no retrieval index. A seeded 60-item demo you can rerun in a 3-minute Colab.*

**TL;DR.** A 230M model ([Liquid AI LFM2.5-230M](https://huggingface.co/LiquidAI/LFM2.5-230M)) learns how you triage notifications from the only supervision you already emit: whether you opened each one. Every item triggers a few `batch_size=1` LoRA steps into a ~1.4 MB adapter, so the policy lives in the weights and serving stays a bare ~90-token prompt. In a seeded week with policy shifts twice, this run made **18** streamed mistakes; the best-tuned ICL and RAG baselines made **30** and **32**, at **13×** and **7×** tokens per query. This is a demo (60 items, one seed), built to be broken. Reproduce it in a 3-minute Colab.

---

Most "personalized" assistants keep the personalization in the prompt. Your history gets packed into a context window or a retrieval index and re-sent on every call. That has a recurring cost in tokens, and when the model isn't on your device, a recurring cost in privacy.

The alternative is old-fashioned: put the adaptation in the weights. Two things quietly made that practical.

**First, local fine-tuning stopped being exotic.** A 230M chat model trains LoRA adapters on a laptop, a Colab T4, or a phone-class NPU. Freeze the base, train a couple of MB of low-rank deltas, and you're done in seconds. The weights live next to the app, not in someone's datacenter.

**Second, the interesting update size is one.** The default mental model of fine-tuning is a batch job: collect ten thousand examples, rent a GPU, wait. But personalization doesn't arrive in batches. It arrives one notification, one correction, one "ugh, not this again" at a time. A model that tracks a single human wants a batch size of **1** and a continual schedule.

The blocker was always: *update on what?* Classic supervised fine-tuning assumes a gold target for every prompt — and you are never going to hand-label the correct triage decision for 500 of your own notifications.

You don't need to. The supervision is already there: one implicit bit you leak by tapping, or not tapping. That is enough for **online SFT**.

### The job: triage that keeps up with your week

The on-device assistant sees a stream of inbox items (emails, Slack pings, calendar nudges, system alerts) and makes the most basic attention decision there is: **INTERRUPT** (buzz now), **LATER** (hold for the evening digest), or **ARCHIVE** (never surface). You never label anything. Opened immediately, opened at the digest, or never opened leaks exactly one bit per item: implicit, noisy, and free.

What makes it hard is that the mapping isn't fixed. The same week has three owners of your attention, each rewriting the policy over the same vocabulary:

- **Weekday** (items 1–30, 50%): manager pings about the blocking project → INTERRUPT; monitoring alerts → ARCHIVE.
- **On-call** (items 31–42, 20%): the world inverts. Every payment-latency/5xx/pager alert → INTERRUPT; the same manager ping → LATER.
- **Off-hours** (items 43–60, 30%): a friend tagging you on Saturday is the whole point of a phone → INTERRUPT; the manager ping is Monday-you's problem → ARCHIVE.

Full disclosure: I built this stream, and I built it to be hostile in specific ways. Drift points are hard cuts. History is 50% weekday by construction, so any method leaning on past decisions inherits a stale majority. Items are two bare lines whose vocabulary overlaps across classes ("your payment was successful" is a receipt; "payment latency" is an alert), with no channel field and no role tags. It's a stress test for staleness and keyword-matching, not a sample of anyone's real inbox. You have to learn the policy, not the keywords.

### One item, four ways

Here is the shape of the whole argument in a single row. An off-hours push arrives: *5 people liked your post*, two anonymous lines, nothing else. It's Saturday, so the right call is INTERRUPT.

| Arm | Decision | Cost |
| --- | --- | --- |
| Zero-shot | LATER (✗) | base prior that notifications can always wait |
| ICL (k=12) | INTERRUPT ✓ | +1,104 tokens of recent-decision window |
| RAG (k=6) | ARCHIVE ✗ | +558 tokens that fetch the weekday verdict for social pushes |
| **Online SFT** | INTERRUPT ✓ | bare ~90-token prompt |

RAG's miss is the staleness argument in one row: its store holds two regimes' worth of "social → ARCHIVE" and a sliver of Saturday, so retrieval confidently serves the past. ICL gets it right here and charges 13× for the privilege. The rest of the post consists of these four columns, measured over the week.

### The online SFT loop

For each incoming item:

1. **The model decides.** With the same bare prompt it serves, it proposes INTERRUPT / LATER / ARCHIVE.
2. **Your behavior referees, one bit.** Matched what you did? Keep its answer as the target. Didn't? The target becomes the action you actually took.
3. **A few one-row LoRA steps.** The kept-or-corrected action is trained into the adapter (`batch_size=1`).
4. **Serve the next item bare.** The adapter carries the policy. No demos, no retrieval.

Step 2 is the "no gold" heart. Either way, the only supervision is a bit you already emitted. The same reinforce-or-correct shape works for any implicit signal (a thumbs-up, dwell time, conversational tone); here, it's whether you opened the notification.

Two training tricks earn their keep in the 3-way setting. Every update pairs the fresh item with one replayed item from each other class, so no single action quietly takes over the logits. And a probe guardrail snapshots the adapter at its best and rolls back before serving (more on that below, because it deserves scrutiny, not a code comment).

```python
model = get_peft_model(base, LoraConfig(
  r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
  target_modules=LORA_TARGET, task_type="CAUSAL_LM"))
buffer, best, mistakes = [], None, 0
for pos, item in enumerate(stream, start=1):
  guess = parse_action(generate(model, item))
  mistakes += (guess != observed_action(item))
  buffer = (buffer + [item])[-REPLAY:]
  batch = [item] + one_of_each_other_action(buffer)
  sft_step(model, batch, batch_size=1, steps=STEPS_PER_ITEM,
         completion_only_loss=True)
  if at_checkpoint(pos) and in_current_regime(pos):
    best = keep_if_best(model, probe_acc(model))
# probe guardrail: roll back to the best snapshot before serving
restore(model, best)
```

At inference, there are no demonstrations and no retrieval. The learned policy is in the weights, and a decision costs the bare ~90-token prompt.

### Results, with the caveats attached

All numbers are from a single-seeded run of `LiquidAI/LFM2.5-230M` on Apple Silicon (MPS, fp32). Seeding makes the run reproducible to the decimal on the same device; it does not make it robust. Sixty items and one seed is a demo, and the honest reading of every number below is "in this run." The Colab takes a seed argument. If the gap doesn't survive your seeds, I want the issue filed.

Grading is the online kind, in three ways. **Whole-week accuracy:** each regime's held-out set (12 items) is scored while that regime is live, at its block-end checkpoint, then averaged; a week has three owners of your attention, and you don't get to skip two of them. **Tracking curves:** the same score at every checkpoint along the stream. **Regret:** cumulative mistakes on the streamed items themselves, each predicted before its label lands (prequential evaluation, the online-learning standard). Every method walks the same causal history: at any moment, it can only use the decisions already made, no future items, no hand-curated cheat sheet.

How the baselines were tuned is stated plainly. ICL and RAG are not pinned to an arbitrary context size; `k` is swept, and each baseline is reported at the `k` that best flatters it for the metric at hand. Note what that means: the sweep picks `k` after seeing the whole stream, an oracle that no deployment has. The SFT recipe went through a sweep on this stream too. So read every comparison as "each method near its ceiling on this stream," not as a tuned-versus-untuned upset, and not as a benchmark.

In this run, ICL's window gives it sawtooth spikes: strong exactly when the window happens to be regime-pure, dragged back down through every drift by its tail of stale demos, and by week's end, it has quietly forgotten Monday. RAG hugs its store's majority: it grinds up to 0.92 by late weekday, then never beats 0.50 off-hours, because the weekday bulk outvotes the weekend for the entire block. The zero-shot floor sits at 0.17 in the weekday regime, below chance: with two-line items, the base model's priors are actively wrong, not just uninformed.

The live-stream scoreboard, every method predicting each item before seeing its label: **Online SFT 18, ICL 30, RAG 32, zero-shot 37.** The frozen baselines shave five to seven mistakes off the zero-shot 37. Learning the stream, in this run, cut it by roughly half.

The probe guardrail is in the open. Three-way online training can overshoot and decay past its peak, so at each checkpoint, the loop snapshots the adapter whenever a small held-out probe of the current regime performs best, and rolls back to the best snapshot before serving. Two honest notes. First, that probe draws on held-out items from the current regime, which is adjacent to the distribution on which the accuracy panels are scored; a skeptical reader should treat it as model selection that the frozen baselines don't get. Second, the regret curve remains sequentially honest (each streamed item is predicted before its own label arrives), but the guardrail itself consumes probe labels, so it is not free supervision. The repo has a flag to turn it off. If you want to know how much it's worth, that is one command.

### What this demo doesn't show

- **Scale.** One task, one model size, 60 items, one seed. Nothing here is a claim about your inbox.
- **Real behavior.** The feedback bit is clean by construction. Real open/ignore signals are noisy and confounded (you were in a meeting; you fat-fingered a dismiss).
- **Real devices.** MPS fp32 on a Mac is not a phone NPU. No latency, memory, or battery numbers yet.
- **Baseline ceilings.** `k` was oracle-swept, but smarter retrieval (recency-weighted, regime-aware) would close some of the gap. The claim is about recurring costs and drift tracking, not that retrieval is unfixable.
- **Free lunch.** The probe guardrail spends held-out labels; the honest deployment question is what replaces it.

### Try it, then break it

Clone and run locally (15 minutes on an M2-Max Mac or any CUDA GPU), or skip the clone and spend a few free T4 minutes: the standalone Colab fetches the seeded dataset from the repo, serves ICL and RAG at their swept-best `k`, runs all four arms, and redraws the three-panel figure.

Then change things. Move the drift points. Add noise to the feedback bit. Try five seeds and see whether 18-versus-30 holds. Turn off the probe guardrail and watch what happens. Widen the `k` sweep and watch RAG keep importing your weekday self into your weekend. The fun of a 230M model is that curiosity, not VRAM, is the bottleneck, and the loop is small enough to hold in your head while you break it.

Big thanks to Kai-Chi Huang and Zhang-Wei Hong for the 12-hour Sundai hackathon.

If this was useful: star ⭐ the [online SFT demo](https://github.com/lin826/online-sdft-demo) repo.

---

## Full post (feed)

Most "personalized" assistants keep the personalization in the prompt — a context window or a retrieval index, re-sent on every call. Recurring token cost. Recurring privacy cost.

The other option: put the adaptation in the weights.

I wrote up a small demo of **online LoRA SFT** on a phone-class 230M model ([LFM2.5-230M](https://huggingface.co/LiquidAI/LFM2.5-230M)). The job is notification triage — INTERRUPT / LATER / ARCHIVE — across a week with two policy drifts. Supervision is implicit (did you open it?), not gold labels. Loop per item:

1. bare-prompt guess (the same call you'd serve)
2. behavior confirms or corrects
3. a few `batch_size=1` LoRA steps into a ~1.4 MB adapter

Serving stays a ~90-token prompt. No demo window. No retrieval index. The policy lives in the weights.

Compared under the same causal rules to zero-shot, ICL, and RAG on a seeded 60-item stream: **18** streamed mistakes vs **30 / 32** for the best-tuned ICL / RAG, at **13× / 7×** their token bill. Useful for measuring adaptation, not a leaderboard claim.

Write-up → LinkedIn article (this Pulse draft)
3-min Colab (free T4) → link in first comment
Repo → https://github.com/lin826/online-sdft-demo

If you work on on-device / continual / personalization: curious what you'd break first — noisier feedback, longer regimes, or a real inbox.

#MachineLearning #ContinualLearning #OnDeviceAI #LLM #LoRA

---

## Short version

Most personalization is a growing prompt or a retrieval index — re-read every call, taxed forever, stale when you drift.

I demo'd **online SFT** on a 230M model: one `batch_size=1` LoRA update per notification, supervised only by whether you opened it. The policy ends up in a ~1.4 MB adapter; serving is a bare ~90-token prompt.

Scenario: INTERRUPT / LATER / ARCHIVE across a week that flips twice (weekday → on-call → off-hours). Causal ICL/RAG baselines: **18** stream mistakes vs **30 / 32**, at a fraction of the tokens.

Article → LinkedIn Pulse
Colab → first comment
Repo → https://github.com/lin826/online-sdft-demo

#ContinualLearning #OnDeviceAI #LLM

---

## Suggested first comment

Runnable notebook (free Colab T4, ~few minutes):

https://colab.research.google.com/github/lin826/online-sdft-demo/blob/main/online_sdft_colab.ipynb

It pulls the seeded stream from the repo, runs ZS / ICL / RAG / online SFT, and redraws the tracking + regret figures.

Code: https://github.com/lin826/online-sdft-demo
