"""Baseline arms for the continual-triage demo: zero-shot, ICL, and RAG — with a k sweep.

Serves the *current* (off-hours) policy on held-out items three ways, none of
which update any weights:

  ZS       bare prompt, base priors
  ICL k    k current-policy demos prepended to every call (token tax; needs labels)
  RAG k    k nearest past decisions retrieved from the full decision history

ICL and RAG each get a sweep over K_SWEEP; the best k on the current-regime eval
becomes that baseline's headline arm (ties go to the smaller, cheaper k).

Writes outputs/triage-showcase/baselines.json — the sweep table, per-regime
accuracies and the per-query prompt-token bill for the winning arms, and the
baseline replies on the qualitative off-hours items. Run this before
scripts/run_sdft.py, which reads that file to draw the comparison figure.

Run:  uv run python scripts/run_baselines.py
"""

from __future__ import annotations

import json
import random
import re

from triage_common import (
    ACTIONS, BASELINES_JSON, DATA_OUT, DRIFTS, EVAL_N, MODEL_NAME, OUT_DIR, REGIMES,
    SEED, STREAM_LEN, accuracy, build_eval, build_msgs, build_stream, export_dataset,
    generate, load_base_model, load_tokenizer, pick_device, prompt_tokens, render_prompt,
)

# --- baseline knobs --------------------------------------------------------- #
K_SWEEP = (3, 6, 9, 12)   # context sizes tried for BOTH baselines; multiples of 3
                          # so ICL's cheat-sheet stays one exemplar per action


def build_icl_demos(stream: list[dict], k: int) -> list[tuple[dict, str]]:
    """ICL's cheat-sheet: k current-policy exemplars from the off-hours block,
    round-robin over the three actions (k=6 -> 2 per action), hand-kept fresh."""
    current = [item for item in stream if item["phase"] == 3]
    per_action = {action: [item for item in current if item["action"] == action]
                  for action in ACTIONS}
    demos = []
    for slot in range(k):
        exemplar = per_action[ACTIONS[slot % len(ACTIONS)]][slot // len(ACTIONS)]
        demos.append((exemplar, exemplar["action"]))
    return demos


def make_retriever(store: list[dict]):
    """RAG's index: bag-of-words overlap against the full decision history
    (a stand-in for an on-device vector index)."""
    vocab = [set(re.findall(r"\w+", render_prompt(item).lower())) for item in store]

    def retrieve(item: dict, k: int) -> list[tuple[dict, str]]:
        query = set(re.findall(r"\w+", render_prompt(item).lower()))
        ranked = sorted(range(len(store)), key=lambda i: -len(query & vocab[i]))
        return [(store[i], store[i]["action"]) for i in ranked[:k]]

    return retrieve


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    print(f"device={device}  model={MODEL_NAME}", flush=True)

    # The same seeded drifting stream and held-out sets run_sdft.py uses.
    stream = build_stream(random.Random(SEED))
    evals = {phase: build_eval(random.Random(SEED + phase), phase) for phase in (1, 2, 3)}
    eval_cur = evals[3]   # the *current* policy is the final regime: off-hours

    export_dataset(stream, evals)   # the committed copy the Colab notebook fetches
    print(f"wrote dataset -> {DATA_OUT} ({len(stream)} stream + 3x{EVAL_N} eval items; "
          f"regimes {REGIMES}; drifts@{DRIFTS})", flush=True)

    tok = load_tokenizer()
    base = load_base_model(device)
    retrieve = make_retriever(stream)   # the personal decision store: all 60 items

    def demos_for(method: str, item: dict, k: int) -> list[tuple[dict, str]]:
        return build_icl_demos(stream, k) if method == "ICL" else retrieve(item, k)

    def sweep_arm(method: str) -> tuple[int, dict]:
        """Try every k on the current-regime eval; return (best k, sweep table)."""
        table = {}
        for k in K_SWEEP:
            msgs = [build_msgs(item, demos_for(method, item, k)) for item in eval_cur]
            acc = accuracy(eval_cur, generate(base, tok, msgs, label=f"{method.lower()} k={k}"))
            tokens = sum(prompt_tokens(tok, m) for m in msgs) / len(msgs)
            table[k] = {"acc_cur": acc, "tok_per_query": tokens}
            print(f"  {method} k={k:2d}: acc_cur={acc:.2f}  tok/query={tokens:.0f}", flush=True)
        best = max(K_SWEEP, key=lambda k: (table[k]["acc_cur"], -k))   # ties -> cheaper k
        return best, table

    print("\n== sweep: ICL / RAG context size on the current policy (off-hours) ==", flush=True)
    icl_k, icl_sweep = sweep_arm("ICL")
    rag_k, rag_sweep = sweep_arm("RAG")
    print(f"  best: ICL k={icl_k}, RAG k={rag_k}", flush=True)

    # Per-regime accuracy for ZS and the winning ICL/RAG arms (regime 1 doubles
    # as the interference check; regime 3 re-uses the sweep numbers above).
    print("\n== per-regime accuracy for the headline arms ==", flush=True)

    def per_regime(method: str | None, k: int = 0) -> dict:
        accs = {}
        for phase, regime in zip((1, 2, 3), REGIMES):
            msgs = [build_msgs(item, None if method is None else demos_for(method, item, k))
                    for item in evals[phase]]
            label = f"{method.lower() if method else 'zs'}/{regime}"
            accs[regime] = accuracy(evals[phase], generate(base, tok, msgs, label=label))
        return accs

    zs_accs = per_regime(None)
    icl_accs = per_regime("ICL", icl_k)
    rag_accs = per_regime("RAG", rag_k)

    def mean_prompt_tokens(build) -> float:
        return sum(prompt_tokens(tok, build(item)) for item in eval_cur) / len(eval_cur)

    tok_zs = mean_prompt_tokens(lambda item: build_msgs(item))

    # The qualitative drifted items — off-hours `social` pushes that should now
    # INTERRUPT. Capture every candidate's baseline replies; run_sdft.py picks
    # the one the served adapter gets right for "one item, four minds".
    social_items = [item for item in eval_cur if item["category"] == "social"]
    qualitative = [{
        "item": item,
        "prompt": render_prompt(item),
        "gold": item["action"],
        "zs": generate(base, tok, [build_msgs(item)], label="q/zs", batch_size=1)[0],
        "icl": generate(base, tok, [build_msgs(item, build_icl_demos(stream, icl_k))],
                        label="q/icl", batch_size=1)[0],
        "rag": generate(base, tok, [build_msgs(item, retrieve(item, rag_k))],
                        label="q/rag", batch_size=1)[0],
    } for item in social_items]

    baselines = {
        "config": {"model": MODEL_NAME, "seed": SEED, "stream_len": STREAM_LEN,
                   "drifts": list(DRIFTS), "regimes": list(REGIMES), "eval_n": EVAL_N,
                   "k_sweep": list(K_SWEEP), "icl_k": icl_k, "rag_k": rag_k},
        "sweeps": {"ICL": icl_sweep, "RAG": rag_sweep},
        "arms": {
            "ZS": {"acc_by_regime": zs_accs, "acc_cur": zs_accs[REGIMES[2]],
                   "acc_old": zs_accs[REGIMES[0]], "tok_per_query": tok_zs,
                   "labels_needed": 0},
            f"ICL k={icl_k}": {"acc_by_regime": icl_accs, "acc_cur": icl_accs[REGIMES[2]],
                               "acc_old": icl_accs[REGIMES[0]],
                               "tok_per_query": icl_sweep[icl_k]["tok_per_query"],
                               "labels_needed": icl_k},
            f"RAG k={rag_k}": {"acc_by_regime": rag_accs, "acc_cur": rag_accs[REGIMES[2]],
                               "acc_old": rag_accs[REGIMES[0]],
                               "tok_per_query": rag_sweep[rag_k]["tok_per_query"],
                               "labels_needed": STREAM_LEN},
        },
        "qualitative_base": qualitative,
    }
    BASELINES_JSON.write_text(json.dumps(baselines, indent=2))
    print(f"\nwrote {BASELINES_JSON}", flush=True)
    for name, arm in baselines["arms"].items():
        regime_report = "  ".join(f"{regime}={arm['acc_by_regime'][regime]:.2f}"
                                  for regime in REGIMES)
        print(f"  {name:10s} {regime_report}  tok/query={arm['tok_per_query']:.0f}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
