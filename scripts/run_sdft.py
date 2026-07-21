"""Online SDFT on the drifting inbox stream — the loop the blog post describes.

The four steps, exactly as in drafts/blog-online-sdft-triage.md:

  1. A tiny dataset: a stream that drifts twice
       seeded synthetic inbox; the 3-way policy flips at DRIFTS (triage_common)
  2. The feedback loop: the model supervises itself
       the model makes its own call; the observed action is the only supervision:
       feedback = 1 (reinforce its own on-policy answer) or 0 (correct toward yours)
  3. The online update: attach LoRA once, stream batch_size=1 steps with replay,
       and keep the best adapter on the current regime (the probe guardrail)
  4. Serving: bare prompt — the adapter carries the policy

Reads outputs/triage-showcase/baselines.json (run scripts/run_baselines.py first)
and writes results.json, the trained adapter, and the two-panel blog figure.

Run:  uv run python scripts/run_sdft.py
"""

from __future__ import annotations

import copy
import json
import random

import torch
from peft import (LoraConfig, get_peft_model, get_peft_model_state_dict,
                  set_peft_model_state_dict)

from triage_common import (
    ACTIONS, BASELINES_JSON, DRIFTS, FIG_DIR, MODEL_NAME, OUT_DIR, REGIMES, SEED,
    STREAM_LEN, accuracy, build_eval, build_msgs, build_stream, generate,
    load_base_model, load_tokenizer, parse_action, pick_device, render_prompt,
)

# --- training knobs (the blog's knob table) --------------------------------- #
LORA_R = 16                                      # adapter rank (~2.8 MB on disk)
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET = r".*self_attn\.(q|k|v|out)_proj"   # LFM2 attention projections
LR = 1e-3            # one persistent AdamW across the whole stream: the completion
                     # is 1-2 tokens (tiny loss), so it wants a larger step than a
                     # scheduled batch trainer (2e-4 stalls, 3e-3 diverges)
TEACHER_SHOTS = 2    # recent decisions in context when the model makes its own call
REPLAY = 16          # sliding replay-buffer size (items)
STEPS_PER_ITEM = 5   # batch_size=1 update steps per incoming item — 3-way wants
                     # more gradient than binary (3 stalls on the cold start)
CHECKPOINTS = tuple(range(6, STREAM_LEN + 1, 6))   # eval every 6 streamed items

ADAPTER_DIR = OUT_DIR / "adapter-online-sdft"


def make_updater(model, tok):
    """One persistent AdamW across the whole stream — Adam momentum carries between
    items, so each batch_size=1 step nudges rather than lurches. Loss is on the
    completion tokens only (exact split by concatenating token ids)."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LR)
    eos = tok.eos_token or ""
    device = next(model.parameters()).device

    def update(batch: list[dict], steps: int) -> None:
        model.train()
        model.config.use_cache = False
        for step_idx in range(steps):
            row = batch[step_idx % len(batch)]    # cycle the (item + replay) mini-batch
            prompt_text = tok.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                tokenize=False, add_generation_prompt=True)
            prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
            target_ids = tok(row["target"] + eos, add_special_tokens=False)["input_ids"]
            input_ids = torch.tensor([prompt_ids + target_ids], device=device)
            labels = torch.tensor([[-100] * len(prompt_ids) + target_ids], device=device)
            loss = model(input_ids=input_ids, labels=labels).loss   # completion-only loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)   # keep bs=1 steps from diverging
            optimizer.step()
            optimizer.zero_grad()
        model.config.use_cache = True
        model.eval()

    return update


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if not BASELINES_JSON.exists():
        raise SystemExit(f"{BASELINES_JSON} not found — run `uv run python scripts/run_baselines.py` "
                         "first (the figure compares against the ZS / ICL / RAG arms it writes).")
    baselines = json.loads(BASELINES_JSON.read_text())
    config = baselines["config"]
    if (config["model"], config["seed"], config["stream_len"]) != (MODEL_NAME, SEED, STREAM_LEN):
        raise SystemExit("baselines.json was produced with a different model/seed/stream — "
                         "re-run scripts/run_baselines.py")

    device = pick_device()
    print(f"device={device}  model={MODEL_NAME}", flush=True)
    torch.manual_seed(SEED)   # LoRA init + dropout masks — makes the run repeatable

    # -- 1. a tiny dataset: a stream that drifts twice ------------------------ #
    stream = build_stream(random.Random(SEED))
    evals = {phase: build_eval(random.Random(SEED + phase), phase) for phase in (1, 2, 3)}

    tok = load_tokenizer()
    base = load_base_model(device)

    # -- 2. the feedback loop: the model supervises itself -------------------- #
    print("\n== self-distillation: the model guesses, your behaviour confirms/corrects ==",
          flush=True)

    def recent_decisions(i: int) -> list[tuple[dict, str]]:
        return [(stream[j], stream[j]["action"]) for j in range(max(0, i - TEACHER_SHOTS), i)]

    teacher_msgs = [build_msgs(stream[i], recent_decisions(i)) for i in range(len(stream))]
    guesses = generate(base, tok, teacher_msgs, label="self-guess")

    # feedback = 1 -> its guess matched what you did (reinforce its own answer)
    # feedback = 0 -> it missed (correct: train on your action instead)
    rows = []
    for item, guess in zip(stream, guesses):
        prediction = parse_action(guess)
        rows.append({
            "prompt": render_prompt(item),
            "target": item["action"],    # when feedback == 1 this IS the model's own answer
            "action": item["action"],
            "pred": prediction,
            "feedback": 1 if prediction == item["action"] else 0,
        })
    n_reinforced = sum(row["feedback"] for row in rows)   # numeric feedback -> a plain sum
    reinforce_frac = n_reinforced / len(rows)
    with (OUT_DIR / "sdft_targets.jsonl").open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    print(f"  reinforced (model already right): {n_reinforced}/{len(rows)}", flush=True)

    # -- 3. the online update: attach LoRA once, stream batch_size=1 ---------- #
    print("\n== online SDFT: per-item batch_size=1 updates with replay + checkpoints ==",
          flush=True)
    model = get_peft_model(base, LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET, task_type="CAUSAL_LM"))
    update = make_updater(model, tok)

    def heldout_accuracy(phase: int, label: str) -> float:
        return accuracy(evals[phase], generate(
            model, tok, [build_msgs(item) for item in evals[phase]], label=label))

    curve = {"pos": [], "acc_p1": [], "acc_p2": [], "acc_p3": []}
    replay_buffer: list[dict] = []
    sampler = random.Random(SEED)
    # Probe guardrail: 3-way over-trains and *decays* past its peak, so during the
    # final regime snapshot the adapter whenever the current-policy probe is at its
    # best, and roll back to that snapshot before serving (auto-rollback on decay).
    best = {"acc": -1.0, "pos": None, "state": None}
    for i, row in enumerate(rows):
        replay_buffer = (replay_buffer + [row])[-REPLAY:]
        # pair the fresh item with one replayed item from EACH other class, so
        # every batch_size=1 update cycles all three actions (binary's
        # pair-with-the-opposite trick, generalised — kills majority collapse)
        batch = [row]
        for action in ACTIONS:
            pool = [b for b in replay_buffer[:-1] if b["action"] == action]
            if action != row["action"] and pool:
                batch.append(sampler.sample(pool, 1)[0])
        update(batch, STEPS_PER_ITEM)

        pos = i + 1
        if pos in CHECKPOINTS:
            curve["pos"].append(pos)
            for phase in (1, 2, 3):
                curve[f"acc_p{phase}"].append(
                    heldout_accuracy(phase, f"sdft@{pos}/p{phase}"))
            report = "  ".join(f"{regime}={curve[f'acc_p{phase}'][-1]:.2f}"
                               for phase, regime in zip((1, 2, 3), REGIMES))
            print(f"  checkpoint {pos}: {report}", flush=True)
            if pos > DRIFTS[1] and curve["acc_p3"][-1] >= best["acc"]:
                best = {"acc": curve["acc_p3"][-1], "pos": pos,
                        "state": copy.deepcopy(get_peft_model_state_dict(model))}

    if best["state"] is not None:            # roll back to the probe-kept best
        set_peft_model_state_dict(model, best["state"])
        note = ("" if best["pos"] == CHECKPOINTS[-1]
                else " instead of the decayed final one")
        print(f"  probe guardrail: serving the adapter from item {best['pos']} "
              f"({REGIMES[2]} acc {best['acc']:.2f}){note}", flush=True)

    # -- 4. serving: bare prompt, adapter carries the policy ------------------ #
    model.save_pretrained(str(ADAPTER_DIR))
    adapter_bytes = (ADAPTER_DIR / "adapter_model.safetensors").stat().st_size
    sdft_cur = best["acc"] if best["state"] is not None else curve["acc_p3"][-1]

    # "One item, four minds": among the off-hours social pushes the baselines
    # answered, pick one the served adapter gets right where zero-shot doesn't.
    qualitative = None
    for candidate in baselines["qualitative_base"]:
        item = candidate["item"]
        assert candidate["prompt"] == render_prompt(item), \
            "baselines.json is stale — re-run scripts/run_baselines.py"
        reply = generate(model, tok, [build_msgs(item)], label="q/sdft", batch_size=1)[0]
        picked = {"prompt": candidate["prompt"], "gold": candidate["gold"],
                  "zs": candidate["zs"], "icl": candidate["icl"],
                  "rag": candidate["rag"], "sdft": reply}
        if qualitative is None:
            qualitative = picked                     # fallback: first candidate
        if (parse_action(reply) == item["action"]
                and parse_action(candidate["zs"]) != item["action"]):
            qualitative = picked                     # the showcase pick
            break

    arms = {name: dict(arm) for name, arm in baselines["arms"].items()}
    arms["Online-SDFT"] = {
        "acc_by_regime": {regime: curve[f"acc_p{phase}"][-1]
                          for phase, regime in zip((1, 2, 3), REGIMES)},
        "acc_cur": sdft_cur, "acc_old": curve["acc_p1"][-1],
        "tok_per_query": arms["ZS"]["tok_per_query"],   # served bare
        "labels_needed": 0,
    }
    arms["Online-SDFT"]["acc_by_regime"][REGIMES[2]] = sdft_cur   # probe-kept adapter

    results = {
        "config": {**config, "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
                   "lora_dropout": LORA_DROPOUT, "lr": LR, "teacher_shots": TEACHER_SHOTS,
                   "replay": REPLAY, "steps_per_item": STEPS_PER_ITEM,
                   "checkpoints": list(CHECKPOINTS)},
        "arms": arms,
        "sweeps": baselines["sweeps"],
        "curve": curve,
        "sdft_best": {"pos": best["pos"], "acc": best["acc"]},
        "qualitative": qualitative,
        "adapter_bytes": adapter_bytes,
        "teacher_self_acc": reinforce_frac,
        "reinforce_frac": reinforce_frac,  # fraction of the stream that was pure self-distillation
    }
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2))
    print("\nwrote", OUT_DIR / "results.json", flush=True)

    make_figure(results)
    print("DONE", flush=True)


def make_figure(results: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.transforms import blended_transform_factory

    arms = results["arms"]
    curve = results["curve"]
    drifts = results["config"]["drifts"]
    stream_len = results["config"]["stream_len"]
    zs_tokens = arms["ZS"]["tok_per_query"]
    colors = {"ZS": "#9aa0a6", "ICL": "#e8710a", "RAG": "#d93025", "Online-SDFT": "#1a73e8"}
    regime_colors = {"weekday": "#7b3fa0", "on-call": "#1a73e8", "off-hours": "#0b8043"}

    def arm_color(name: str) -> str:
        for prefix, color in colors.items():
            if name.startswith(prefix):
                return color
        return "#5f6368"

    fig, (ax_cost, ax_drift) = plt.subplots(1, 2, figsize=(12.6, 4.9))

    # Panel A: accuracy on the CURRENT policy vs the recurring prompt-token bill
    for name, arm in arms.items():
        x, y = arm["tok_per_query"], arm["acc_cur"] * 100
        ax_cost.scatter(x, y, s=170, color=arm_color(name), zorder=3,
                        edgecolor="white", linewidth=1.5)
        dy = 3.2 if not name.startswith("RAG") else -5.0
        ax_cost.annotate(name, (x, y), textcoords="offset points", xytext=(8, dy),
                         fontsize=10.5, fontweight="bold", color=arm_color(name))
    ax_cost.set_xlabel("Recurring prompt tokens / query  (on-device cost, every notification)")
    ax_cost.set_ylabel("Accuracy on current policy  (%)")
    ax_cost.set_title("A.  Best-of-sweep baselines vs a bare prompt", fontsize=12,
                      fontweight="bold")
    ax_cost.grid(True, alpha=0.25)
    ax_cost.set_ylim(0, 105)
    ax_cost.axvspan(0, zs_tokens + 22, color="#1a73e8", alpha=0.05)
    ax_cost.text((zs_tokens + 22) / 2, 6, "bare-prompt zone\n(weights carry the policy)",
                 ha="center", fontsize=8.5, color="#1a73e8", style="italic")

    # Panel B: continual adaptation across TWO drifts (three regimes, one adapter)
    icl_rag = {n: a for n, a in arms.items() if n.startswith(("ICL", "RAG"))}
    ref_arm = max(icl_rag.values(), key=lambda arm: arm["acc_cur"])
    ref = ref_arm["acc_cur"] * 100   # the winning frozen baseline, priced at ITS tokens
    mult = max(1, round(ref_arm["tok_per_query"] / max(zs_tokens, 1)))
    ax_drift.axhline(ref, color="#e8710a", ls=":", lw=1.6)
    ax_drift.text(1, ref - 2.5 if ref > 93 else ref + 2,   # pin the label on the line
                  f"best ICL / RAG on off-hours  (+{mult}× tokens every call)",
                  fontsize=7.6, color="#e8710a", ha="left",
                  va="top" if ref > 93 else "bottom")
    for x in drifts:
        ax_drift.axvline(x, color="#5f6368", ls="--", lw=1.2)
    # No legend box: the curves share colors with the phase sub-titles below the
    # axis, so those labels double as the key.
    for phase, regime in zip((1, 2, 3), REGIMES):
        ax_drift.plot(curve["pos"], [v * 100 for v in curve[f"acc_p{phase}"]], "-o",
                      color=regime_colors[regime], lw=2.2, ms=4.5)
    kept = results.get("sdft_best") or {}
    if kept.get("pos"):   # star the checkpoint the probe guardrail serves
        ax_drift.scatter([kept["pos"]], [kept["acc"] * 100], marker="*", s=300,
                         color=regime_colors[REGIMES[2]], edgecolor="white",
                         linewidth=1.2, zorder=5)
        near_top = kept["acc"] > 0.88          # dodge below the star near the ceiling
        ax_drift.annotate("probe keeps\nthis adapter", (kept["pos"], kept["acc"] * 100),
                          textcoords="offset points",
                          xytext=(0, -24 if near_top else 10), fontsize=7.5,
                          color=regime_colors[REGIMES[2]], ha="center", fontweight="bold")

    # Per-phase sub-titles along the x-axis: tint each regime's span and label it.
    phase_axis = blended_transform_factory(ax_drift.transData, ax_drift.transAxes)
    bounds = [0, *drifts, stream_len]
    for start, end, regime in zip(bounds, bounds[1:], REGIMES):
        ax_drift.axvspan(start, end, color=regime_colors[regime], alpha=0.045, zorder=0)
        ax_drift.text((start + end) / 2, -0.115, f"{regime}\nitems {start + 1}–{end}",
                      transform=phase_axis, ha="center", va="top", fontsize=8.3,
                      color=regime_colors[regime], fontweight="bold")
    ax_drift.set_xlim(0, stream_len + 2)
    ax_drift.set_xlabel("Items streamed  (one batch_size=1 update each)", labelpad=36)
    ax_drift.set_ylabel("Held-out accuracy  (%)")
    ax_drift.set_title("B.  Weights track each regime — two drifts, one adapter",
                       fontsize=12, fontweight="bold")
    ax_drift.set_ylim(0, 105)
    ax_drift.grid(True, alpha=0.25)

    adapter_mb = results["adapter_bytes"] / 1e6
    fig.suptitle(
        f"On-device 3-way triage across 3 regimes · LFM2.5-230M · policy lives in a "
        f"{adapter_mb:.1f} MB LoRA adapter, no gold labels",
        fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "online_sdft_triage.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
