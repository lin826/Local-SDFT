"""On-device continual triage — online SDFT vs ZS / ICL / RAG, with a policy drift.

Scenario: a phone-hosted 230M model learns a user's *drifting* attention policy
— should this notification INTERRUPT you now, or wait for the DIGEST? — from
implicit feedback (open now vs let it wait), no gold labels. Halfway through the
stream the user goes on-call, so the policy DRIFTS: automated monitoring alerts
(payment / latency / pager) flip DIGEST -> INTERRUPT, and non-incident manager
pings flip INTERRUPT -> DIGEST.

The self-distillation loop (per the repo's online_learning/feedback.py): the
model makes its own decision; your behaviour reinforces it when right and
corrects it when wrong. The target is always a bare action you actually took —
never a hand-written gold answer.

We compare four ways to serve the *current* policy on held-out items:
  ZS           bare prompt, base priors
  ICL k        k current-policy demos prepended every call (token tax; needs labels)
  RAG k        k nearest past decisions retrieved from a store (5x tokens + a disk index)
  Online-SDFT  stream -> confirmed/corrected action -> a few batch_size=1 LoRA steps; serve bare

Outputs: outputs/triage-showcase/results.json + a 2-panel figure.
Run:  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/run_triage_showcase.py
"""

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path

# Model is a normal HF download (cached after first run). If you're offline or
# rate-limited, export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 before running.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402
from peft import LoraConfig as PeftLoraConfig  # noqa: E402
from peft import get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

MODEL_NAME = "LiquidAI/LFM2.5-230M"
SEED = 7
# Binary attention decision — the purest form of "what deserves to break your focus".
# Class-balanced per phase so a tiny on-device LoRA learns the mapping, not the prior.
ACTIONS = ("INTERRUPT", "DIGEST")

# stream / eval sizes
STREAM_LEN = 40
DRIFT_AT = 20            # items 0..19 = phase 1, 20..39 = phase 2
EVAL_N = 12             # held-out items per phase policy
CHECKPOINTS = (5, 10, 15, 20, 25, 30, 35, 40)

# knobs (match repo online-learning config: bs=1, small replay, few steps/item)
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
LORA_TARGET = r".*self_attn\.(q|k|v|out)_proj"
LR = 1e-3   # persistent-optimizer online loop: completion is 1-2 tokens (tiny loss),
            # so it wants a larger step than a scheduled batch trainer (2e-4 stalls, 3e-3 diverges)
MAX_LEN = 512
MAX_NEW = 40
ICL_K = 4
RAG_K = 4
TEACHER_SHOTS = 2
REPLAY = 16             # sliding replay buffer size
STEPS_PER_ITEM = 3      # batch_size=1 optimizer steps per incoming item

OUT_DIR = Path("outputs/triage-showcase")
ADAPTER_DIR = OUT_DIR / "adapter-online-sdft"
DATA_OUT = Path("data/inbox_triage.jsonl")
FIG_DIR = Path("docs/assets")

# --------------------------------------------------------------------------- #
# Synthetic inbox generator — policy is keyword/sender driven so a 230M LoRA
# can pick it up in a few dozen steps; the drift is a clean regime change.
# --------------------------------------------------------------------------- #
NAMES = ["Priya", "Marcus", "Lena", "Diego", "Aisha", "Tom", "Yuki", "Sam"]
PROJECTS = ["Atlas", "the Q3 launch", "the billing rewrite", "Project Nomad", "the search revamp"]
PRODUCTS = ["Nimbus", "the mobile app", "SoundOff", "Kettle", "Zephyr"]
INCIDENT_SIG = ["payment latency", "checkout 5xx errors", "a pager alert", "an API outage",
                "elevated error rate", "payment gateway timeouts", "a DB failover"]


def _pick(rng, xs):
    return xs[rng.randrange(len(xs))]


def gen_item(category: str, rng: random.Random) -> dict:
    """Return {channel, sender, subject, snippet, category}."""
    if category == "mgr_project":
        proj = _pick(rng, PROJECTS)
        return {
            "channel": "email", "sender": f"{_pick(rng, NAMES)} (your manager)",
            "subject": f"Need your call on {proj}",
            "snippet": _pick(rng, [
                f"Can you weigh in before the standup? It's blocking {proj}.",
                f"Quick decision needed on {proj} — reviewers are waiting on you.",
                f"Are we shipping {proj} today? Need your sign-off.",
            ]), "category": category}
    if category == "teammate_fyi":
        return {
            "channel": "slack", "sender": f"{_pick(rng, NAMES)} (teammate)",
            "subject": "fyi / no rush",
            "snippet": _pick(rng, [
                "Left a couple of comments on your PR whenever you get a sec.",
                "Shared some notes from the sync — no action needed today.",
                "Thinking about refactoring the utils, curious what you think sometime.",
            ]), "category": category}
    if category == "calendar_soon":
        who = _pick(rng, NAMES)
        return {
            "channel": "calendar", "sender": "Calendar",
            "subject": f"Starts in 20 min: 1:1 with {who}",
            "snippet": f"Reminder: your meeting with {who} begins soon.",
            "category": category}
    if category == "promo":
        return {
            "channel": "email", "sender": f"{_pick(rng, PRODUCTS)} Team",
            "subject": _pick(rng, ["48-hour sale — 40% off", "New features you'll love",
                                    "We miss you! Come back for 20% off"]),
            "snippet": "Limited time only. Unsubscribe anytime.", "category": category}
    if category == "social":
        return {
            "channel": "push", "sender": "Social",
            "subject": _pick(rng, ["5 people liked your post", "You have 3 new followers",
                                    "Someone mentioned you in a comment"]),
            "snippet": "Tap to see the activity.", "category": category}
    if category == "receipt":
        prod = _pick(rng, PRODUCTS)
        return {
            "channel": "email", "sender": "Receipts",
            "subject": f"Your payment to {prod} was successful",
            "snippet": _pick(rng, [
                "Payment of $12.00 received. Thanks for your order.",
                "Your invoice is attached. No action required.",
                "Charge confirmed. View your billing history anytime.",
            ]), "category": category}
    if category == "monitoring":
        sig = _pick(rng, INCIDENT_SIG)
        return {
            "channel": "system", "sender": "Monitoring Bot",
            "subject": f"ALERT: {sig}",
            "snippet": _pick(rng, [
                f"Automated alert: {sig} detected in production.",
                f"Threshold breached — {sig}. Dashboard link inside.",
                f"{sig} on the checkout service. Auto-generated.",
            ]), "category": category}
    raise ValueError(category)


CATEGORIES = ["mgr_project", "teammate_fyi", "calendar_soon", "promo",
              "social", "receipt", "monitoring"]


def true_action(category: str, phase: int) -> str:
    """Latent per-phase policy. phase 1 = focus week, phase 2 = on-call."""
    base = {
        "mgr_project": "INTERRUPT",   # manager, blocking the active project
        "calendar_soon": "INTERRUPT",  # meeting in <30 min
        "teammate_fyi": "DIGEST",
        "promo": "DIGEST",
        "social": "DIGEST",
        "receipt": "DIGEST",
        "monitoring": "DIGEST",        # muted noise during focus week
    }
    if phase == 1:
        return base[category]
    # phase 2 — the drift: you go on-call for a payments incident
    drift = dict(base)
    drift["monitoring"] = "INTERRUPT"   # every alert now deserves a buzz
    drift["mgr_project"] = "DIGEST"     # heads-down on the incident; pings wait
    return drift[category]


def item_block(item: dict) -> str:
    return (f"Channel: {item['channel']}\n"
            f"From: {item['sender']}\n"
            f"Subject: {item['subject']}\n"
            f"{item['snippet']}")


def render_prompt(item: dict) -> str:
    return (
        "Triage this inbox item. Should it buzz the user now or wait for the digest? "
        "Answer with exactly one of INTERRUPT or DIGEST.\n\n" + item_block(item)
    )


# Deterministic category composition so both phases carry enough of the
# drift-sensitive categories (monitoring, mgr_project) to actually be learned.
# Balanced ~10 INTERRUPT / 10 DIGEST in each phase (see true_action).
PHASE1_SPEC = {"mgr_project": 6, "calendar_soon": 4, "monitoring": 4,
               "teammate_fyi": 2, "promo": 2, "social": 1, "receipt": 1}    # 20 (10/10)
PHASE2_SPEC = {"monitoring": 6, "calendar_soon": 4, "mgr_project": 6,
               "teammate_fyi": 2, "promo": 1, "social": 1}                  # 20 (10/10)
EVAL_SPEC = {"monitoring": 3, "calendar_soon": 3, "mgr_project": 3,
             "teammate_fyi": 1, "promo": 1, "social": 1}                    # 12 (6/6 both phases)


def _build_from_spec(spec: dict, phase: int, rng: random.Random) -> list[dict]:
    items = []
    for cat, cnt in spec.items():
        for _ in range(cnt):
            it = gen_item(cat, rng)
            it["phase"] = phase
            it["action"] = true_action(cat, phase)
            items.append(it)
    rng.shuffle(items)
    return items


def build_stream(rng: random.Random) -> list[dict]:
    """Phase-1 block, then phase-2 block; labels drift at the block boundary."""
    return _build_from_spec(PHASE1_SPEC, 1, rng) + _build_from_spec(PHASE2_SPEC, 2, rng)


def build_eval(rng: random.Random, phase: int, n: int) -> list[dict]:
    items = _build_from_spec(EVAL_SPEC, phase, rng)
    return items[:n] if n else items


# --------------------------------------------------------------------------- #
# Model helpers
# --------------------------------------------------------------------------- #
def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_tok():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def load_base(device: str):
    dtype = torch.float16 if device == "cuda" else torch.float32
    m = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=dtype)
    return m.to(device) if device != "cuda" else m


def to_dev(enc, model):
    dev = next(model.parameters()).device
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in enc.items()}


def demo_msg(item: dict, action: str) -> list[dict]:
    return [{"role": "user", "content": render_prompt(item)},
            {"role": "assistant", "content": action}]


def build_msgs(item: dict, demos: list[tuple[dict, str]] | None = None) -> list[dict]:
    msgs: list[dict] = []
    for d_item, d_act in demos or []:
        msgs += demo_msg(d_item, d_act)
    msgs.append({"role": "user", "content": render_prompt(item)})
    return msgs


def prompt_tokens(tok, msgs) -> int:
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return len(tok(text, add_special_tokens=False)["input_ids"])


ACTION_RE = re.compile(r"\b(INTERRUPT|DIGEST)\b", re.IGNORECASE)


def parse_action(text: str) -> str:
    m = ACTION_RE.search(text or "")
    return m.group(1).upper() if m else "NONE"


@torch.inference_mode()
def generate(model, tok, items, demos_fn, *, max_new=MAX_NEW, label="gen", bs=8) -> list[str]:
    model.eval()
    outs: list[str] = []
    for start in range(0, len(items), bs):
        batch = items[start:start + bs]
        texts = [tok.apply_chat_template(demos_fn(it), tokenize=False, add_generation_prompt=True)
                 for it in batch]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = to_dev(enc, model)
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        new = out[:, enc["input_ids"].shape[1]:]
        for t in tok.batch_decode(new, skip_special_tokens=True):
            outs.append(t.strip())
        print(f"  [{label}] {min(start + bs, len(items))}/{len(items)}", flush=True)
    return outs


def accuracy(items, gens) -> float:
    ok = sum(parse_action(g) == it["action"] for it, g in zip(items, gens))
    return ok / max(len(items), 1)


def retriever(store: list[dict]):
    toks = [set(re.findall(r"\w+", render_prompt(s).lower())) for s in store]

    def retrieve(item, k):
        q = set(re.findall(r"\w+", render_prompt(item).lower()))
        order = sorted(range(len(store)), key=lambda i: -len(q & toks[i]))
        return [(store[i], store[i]["action"]) for i in order[:k]]
    return retrieve


def make_updater(model, tok):
    """A persistent-optimizer online updater — Adam momentum carries across the
    stream, so each bs=1 step nudges rather than lurches (true online learning,
    and far smoother than re-initialising a Trainer per item). Loss is on the
    completion tokens only (exact split by concatenating token ids)."""
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    trainable = [p for p in model.parameters() if p.requires_grad]
    eos = tok.eos_token or ""
    dev = next(model.parameters()).device

    def step(rows: list[dict], steps: int) -> None:
        model.train()
        model.config.use_cache = False
        for k in range(steps):
            r = rows[k % len(rows)]
            ptxt = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                                           tokenize=False, add_generation_prompt=True)
            ids_p = tok(ptxt, add_special_tokens=False)["input_ids"]
            ids_c = tok(r["target"] + eos, add_special_tokens=False)["input_ids"]
            ids = torch.tensor([ids_p + ids_c], device=dev)
            labels = torch.tensor([[-100] * len(ids_p) + ids_c], device=dev)  # completion-only loss
            loss = model(input_ids=ids, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)  # keep bs=1 steps from diverging
            opt.step()
            opt.zero_grad()
        model.config.use_cache = True
        model.eval()

    return step


# --------------------------------------------------------------------------- #
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_OUT.parent.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    print(f"device={device}  model={MODEL_NAME}", flush=True)

    rng = random.Random(SEED)
    stream = build_stream(rng)
    eval_p1 = build_eval(random.Random(SEED + 1), phase=1, n=EVAL_N)
    eval_p2 = build_eval(random.Random(SEED + 2), phase=2, n=EVAL_N)

    with DATA_OUT.open("w") as fh:
        for it in stream:
            fh.write(json.dumps(it) + "\n")
    print(f"wrote stream -> {DATA_OUT} ({len(stream)} items; drift@{DRIFT_AT})", flush=True)

    tok = load_tok()
    base = load_base(device)

    # ICL cheat-sheet: representative current-policy (phase 2) exemplars, one per
    # distinct category/action — a *fair, strong* baseline (but hand-labeled).
    phase2 = [it for it in stream if it["phase"] == 2]

    def _first(cat):
        return next(it for it in phase2 if it["category"] == cat)
    # balanced cheat-sheet: 2 INTERRUPT + 2 DIGEST exemplars of the current policy
    icl_demos = [(_first(c), _first(c)["action"])
                 for c in ("monitoring", "calendar_soon", "mgr_project", "teammate_fyi")][:ICL_K]
    retrieve = retriever(stream)                                   # personal decision store

    results: dict = {"config": {
        "model": MODEL_NAME, "seed": SEED, "stream_len": STREAM_LEN, "drift_at": DRIFT_AT,
        "eval_n": EVAL_N, "icl_k": ICL_K, "rag_k": RAG_K, "lora_r": LORA_R}}

    # ---- baseline arms on the *current* policy (eval_p2) --------------------
    def eval_arm(items, demos_fn, label):
        gens = generate(base, tok, items, demos_fn, label=label)
        return gens, accuracy(items, gens)

    print("\n== baseline arms (current policy = eval_p2) ==", flush=True)
    zs_g2, zs_a2 = eval_arm(eval_p2, lambda it: build_msgs(it), "zs/p2")
    icl_g2, icl_a2 = eval_arm(eval_p2, lambda it: build_msgs(it, icl_demos), "icl/p2")
    rag_g2, rag_a2 = eval_arm(eval_p2, lambda it: build_msgs(it, retrieve(it, RAG_K)), "rag/p2")
    # old policy (eval_p1) for interference table
    zs_a1 = accuracy(eval_p1, generate(base, tok, eval_p1, lambda it: build_msgs(it), label="zs/p1"))
    icl_a1 = accuracy(eval_p1, generate(base, tok, eval_p1, lambda it: build_msgs(it, icl_demos), label="icl/p1"))
    rag_a1 = accuracy(eval_p1, generate(base, tok, eval_p1, lambda it: build_msgs(it, retrieve(it, RAG_K)), label="rag/p1"))

    tok_zs = sum(prompt_tokens(tok, build_msgs(it)) for it in eval_p2) / len(eval_p2)
    tok_icl = sum(prompt_tokens(tok, build_msgs(it, icl_demos)) for it in eval_p2) / len(eval_p2)
    tok_rag = sum(prompt_tokens(tok, build_msgs(it, retrieve(it, RAG_K))) for it in eval_p2) / len(eval_p2)

    # Qualitative on one drifted (monitoring) item — capture baselines on the
    # CLEAN base now, before get_peft_model wraps it with the online adapter.
    qual_item = next(it for it in eval_p2 if it["category"] == "monitoring")
    qual = {
        "prompt": render_prompt(qual_item), "gold": qual_item["action"],
        "zs": generate(base, tok, [qual_item], lambda it: build_msgs(it), label="q/zs")[0],
        "icl": generate(base, tok, [qual_item], lambda it: build_msgs(it, icl_demos), label="q/icl")[0],
        "rag": generate(base, tok, [qual_item], lambda it: build_msgs(it, retrieve(it, RAG_K)), label="q/rag")[0],
    }

    # ---- the self-distillation / implicit-feedback loop --------------------
    # For each item the model first makes its OWN decision (conditioned on a
    # couple of recent decisions). Your behaviour is the only supervision:
    #   guess == what-you-did  -> REINFORCE the model's own on-policy decision
    #   guess != what-you-did  -> CORRECT toward what-you-did
    # Either way the target is a bare action — never a hand-written gold answer.
    print("\n== self-distillation loop: model guesses, your behaviour confirms/corrects ==",
          flush=True)
    teacher_msgs = []
    for i, it in enumerate(stream):
        shots = [(stream[j], stream[j]["action"]) for j in range(max(0, i - TEACHER_SHOTS), i)]
        teacher_msgs.append((it, shots))
    guesses = generate(base, tok, list(range(len(stream))),
                       demos_fn=lambda idx: build_msgs(teacher_msgs[idx][0], teacher_msgs[idx][1]),
                       label="self-guess")
    sdft_rows, n_reinforce = [], 0
    for it, guess in zip(stream, guesses):
        pred = parse_action(guess)
        reinforced = pred == it["action"]
        n_reinforce += int(reinforced)
        sdft_rows.append({"prompt": render_prompt(it), "target": it["action"],
                          "action": it["action"], "pred": pred,
                          "feedback": "reinforce" if reinforced else "correct"})
    self_acc = n_reinforce / len(sdft_rows)
    with (OUT_DIR / "sdft_targets.jsonl").open("w") as fh:
        for r in sdft_rows:
            fh.write(json.dumps(r) + "\n")
    print(f"  teacher self-agreement with observed action: {self_acc:.2f}", flush=True)

    # ---- online loop: attach LoRA once, per-item bs=1 replay updates ---------
    print("\n== online SDFT: per-item bs=1 updates w/ replay + checkpoint eval ==", flush=True)
    peft_cfg = PeftLoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                              target_modules=LORA_TARGET, task_type="CAUSAL_LM")
    model = get_peft_model(base, peft_cfg)
    update = make_updater(model, tok)
    curve = {"pos": [], "acc_p1": [], "acc_p2": []}
    buffer: list[dict] = []
    rrng = random.Random(SEED)
    ckpts = set(CHECKPOINTS)
    for i, row in enumerate(sdft_rows):
        buffer.append(row)
        buffer = buffer[-REPLAY:]
        # Balanced replay: pair the fresh item with a recent one of a DIFFERENT
        # action so every bs=1 update sees ≥2 classes (kills majority collapse).
        others = [b for b in buffer[:-1] if b["action"] != row["action"]] or buffer[:-1]
        batch = [row] + (rrng.sample(others, 1) if others else [])
        update(batch, STEPS_PER_ITEM)
        pos = i + 1
        if pos in ckpts:
            a1 = accuracy(eval_p1, generate(model, tok, eval_p1, lambda it: build_msgs(it), label=f"sdft@{pos}/p1"))
            a2 = accuracy(eval_p2, generate(model, tok, eval_p2, lambda it: build_msgs(it), label=f"sdft@{pos}/p2"))
            curve["pos"].append(pos)
            curve["acc_p1"].append(a1)
            curve["acc_p2"].append(a2)
            print(f"  checkpoint {pos}: acc_p1={a1:.2f} acc_p2={a2:.2f}", flush=True)

    model.save_pretrained(str(ADAPTER_DIR))
    sdft_a2 = curve["acc_p2"][-1]
    sdft_a1 = curve["acc_p1"][-1]
    tok_sdft = tok_zs  # served bare
    adapter_bytes = (ADAPTER_DIR / "adapter_model.safetensors").stat().st_size

    # ---- qualitative: fill in the online-SDFT reply on the same drifted item
    qual["sdft"] = generate(model, tok, [qual_item], lambda it: build_msgs(it), label="q/sdft")[0]

    results["arms"] = {
        "ZS":          {"acc_cur": zs_a2,  "acc_old": zs_a1,  "tok_per_query": tok_zs,   "labels_needed": 0},
        "ICL k=%d" % ICL_K: {"acc_cur": icl_a2, "acc_old": icl_a1, "tok_per_query": tok_icl, "labels_needed": ICL_K},
        "RAG k=%d" % RAG_K: {"acc_cur": rag_a2, "acc_old": rag_a1, "tok_per_query": tok_rag, "labels_needed": STREAM_LEN},
        "Online-SDFT": {"acc_cur": sdft_a2, "acc_old": sdft_a1, "tok_per_query": tok_sdft, "labels_needed": 0},
    }
    results["curve"] = curve
    results["qualitative"] = qual
    results["adapter_bytes"] = adapter_bytes
    results["teacher_self_acc"] = self_acc
    results["reinforce_frac"] = self_acc  # fraction of stream where model's own guess was confirmed
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2))
    print("\nwrote", OUT_DIR / "results.json", flush=True)

    make_figure(results)
    print("DONE", flush=True)


def make_figure(results: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = results["arms"]
    curve = results["curve"]
    drift_at = results["config"]["drift_at"]
    colors = {"ZS": "#9aa0a6", "ICL": "#e8710a", "RAG": "#d93025", "Online-SDFT": "#1a73e8"}

    def col(name):
        for k, c in colors.items():
            if name.startswith(k):
                return c
        return "#5f6368"

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.4, 4.9))

    # Panel A: accuracy on the CURRENT policy vs recurring prompt-token cost
    for name, d in arms.items():
        x, y = d["tok_per_query"], d["acc_cur"] * 100
        axA.scatter(x, y, s=170, color=col(name), zorder=3, edgecolor="white", linewidth=1.5)
        dy = 3.2 if not name.startswith("RAG") else -5.0
        axA.annotate(name, (x, y), textcoords="offset points", xytext=(8, dy),
                     fontsize=10.5, fontweight="bold", color=col(name))
    axA.set_xlabel("Recurring prompt tokens / query  (on-device cost, every notification)")
    axA.set_ylabel("Accuracy on current policy  (%)")
    axA.set_title("A.  Same accuracy, a fraction of the cost", fontsize=12, fontweight="bold")
    axA.grid(True, alpha=0.25)
    axA.set_ylim(0, 105)
    axA.axvspan(0, 70, color="#1a73e8", alpha=0.05)
    axA.text(35, 6, "bare-prompt zone\n(weights carry the policy)", ha="center",
             fontsize=8.5, color="#1a73e8", style="italic")

    # Panel B: continual adaptation across the drift
    pos, a1, a2 = curve["pos"], [v * 100 for v in curve["acc_p1"]], [v * 100 for v in curve["acc_p2"]]
    ref = max(arms["ICL k=%d" % results["config"]["icl_k"]]["acc_cur"],
              arms["RAG k=%d" % results["config"]["rag_k"]]["acc_cur"]) * 100
    axB.axhline(ref, color="#e8710a", ls=":", lw=1.6,
                label="ICL / RAG on new policy\n(but +5× tokens every call)")
    axB.axvline(drift_at, color="#5f6368", ls="--", lw=1.3)
    axB.text(drift_at - 0.6, 6, "policy drift\n(you go on-call)", fontsize=8.5,
             color="#5f6368", ha="right")
    axB.plot(pos, a1, "-o", color="#7b3fa0", lw=2.2, label="old policy (focus week)")
    axB.plot(pos, a2, "-o", color="#1a73e8", lw=2.4, label="new policy (on-call)")
    axB.set_xlabel("Items streamed  (one batch_size=1 update each)")
    axB.set_ylabel("Held-out accuracy  (%)")
    axB.set_title("B.  It learns the new policy from the stream — free at serve time",
                  fontsize=12, fontweight="bold")
    axB.set_ylim(0, 105)
    axB.grid(True, alpha=0.25)
    axB.legend(fontsize=8.0, loc="lower right", framealpha=0.95)

    adapter_mb = results["adapter_bytes"] / 1e6
    fig.suptitle(
        f"On-device continual triage · LFM2.5-230M · policy lives in a {adapter_mb:.1f} MB LoRA adapter, no gold labels",
        fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "online_sdft_triage.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
