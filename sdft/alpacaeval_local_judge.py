"""Local pairwise AlpacaEval-style judge for Colab T4 (no OpenAI).

Uses the same protocol *shape* as AE2 (instruction + reference vs model
output → preference → ``win_rate``), but a 4-bit open instruct model instead
of GPT-4-Turbo. This is **not** leaderboard-equivalent:

- Judge quality / agreement with humans differs from ``weighted_alpaca_eval_gpt4_turbo``
- Length-controlled win-rate (``length_controlled_winrate``) needs the official
  GPT-fitted GLM pipeline — not available here; we report raw ``win_rate`` +
  ``avg_length`` instead

Default judge: ``Qwen/Qwen3.5-9B`` in bitsandbytes 4-bit (~5–7 GB
weights; fits Colab T4 ~15 GB at batch size 1 alongside a unloaded 230M
policy). Override with env ``ALPACA_EVAL_LOCAL_JUDGE`` (e.g.
``Qwen/Qwen2.5-7B-Instruct`` if OOM or older transformers).

Requires a recent ``transformers`` with Qwen3.5 support. Thinking mode is
disabled in the chat template so short pairwise verdicts fit
``max_new_tokens``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Sequence

# Default T4-safe open judge (9B NF4 ≈ 5–7 GB). Post-trained Qwen3.5 checkpoint
# (no Instruct suffix). Alternatives: Qwen/Qwen2.5-7B-Instruct (safer / older
# transformers), meta-llama/Llama-3.1-8B-Instruct, google/gemma-2-9b-it.
DEFAULT_LOCAL_JUDGE = "Qwen/Qwen3.5-9B"
LOCAL_JUDGE_ENV = "ALPACA_EVAL_LOCAL_JUDGE"

ALPACA_EVAL_HF_REPO = "tatsu-lab/alpaca_eval"
# AE2 official reference (gpt-4-1106-preview), not text-davinci-003.
GPT4_TURBO_REFERENCE_JSON = "alpaca_eval_gpt4_baseline.json"

# AE-style Output (a)/(b) schema — works better for open chat models than
# the GPT-4 logprob "m"/"M" classifier prompt.
_PAIRWISE_USER = """\
Select the output (a) or (b) that best matches the given instruction. Choose \
your preferred output, which can be subjective. Your answer should ONLY \
contain: Output (a) or Output (b). Do not explain.

## Instruction:
{instruction}

## Output (a):
{output_1}

## Output (b):
{output_2}

## Which is best, Output (a) or Output (b)?
"""

_EXPLICIT_A = re.compile(r"output\s*\(\s*a\s*\)", re.IGNORECASE)
_EXPLICIT_B = re.compile(r"output\s*\(\s*b\s*\)", re.IGNORECASE)
_LETTER_A = re.compile(r"(?:^|[^a-z0-9])\(?\s*a\s*\)?(?:[^a-z0-9]|$)", re.IGNORECASE)
_LETTER_B = re.compile(r"(?:^|[^a-z0-9])\(?\s*b\s*\)?(?:[^a-z0-9]|$)", re.IGNORECASE)


def default_local_judge_model() -> str:
    """Return ``ALPACA_EVAL_LOCAL_JUDGE`` or the T4 default HF id."""
    return (os.environ.get(LOCAL_JUDGE_ENV) or "").strip() or DEFAULT_LOCAL_JUDGE


def parse_pairwise_verdict(text: str) -> int | None:
    """Parse a judge completion into preference ``1`` (a) or ``2`` (b).

    Returns ``None`` if the verdict cannot be determined. Does not load a model.
    """
    if text is None:
        return None
    raw = str(text).strip()
    if not raw:
        return None

    # First line often holds the label for instruct models.
    first = raw.splitlines()[0].strip()
    candidates = (first, raw)

    for chunk in candidates:
        ea = _EXPLICIT_A.search(chunk)
        eb = _EXPLICIT_B.search(chunk)
        if ea and not eb:
            return 1
        if eb and not ea:
            return 2
        if ea and eb:
            return 1 if ea.start() < eb.start() else 2

    # Terse answers: "a", "(b)", "Output a", AE-clf "m"/"M" (case-sensitive)
    compact = re.sub(r"\s+", " ", first).strip().strip("\"'`.,;:")
    if compact == "m":
        return 1
    if compact == "M":
        return 2
    low = compact.lower()
    if low in {"a", "(a)", "a)", "output a", "output (a)"}:
        return 1
    if low in {"b", "(b)", "b)", "output b", "output (b)"}:
        return 2

    for chunk in candidates:
        ma = _LETTER_A.search(chunk)
        mb = _LETTER_B.search(chunk)
        if ma and not mb:
            return 1
        if mb and not ma:
            return 2
        if ma and mb:
            return 1 if ma.start() < mb.start() else 2
    return None


def load_gpt4_turbo_reference(
    *,
    num_examples: int | None = None,
    json_path: str | Path | None = None,
) -> list[dict[str, str]]:
    """Load AE2 ``gpt4_turbo`` / gpt-4-1106-preview reference rows."""
    if json_path is None:
        from huggingface_hub import hf_hub_download

        path = Path(
            hf_hub_download(
                ALPACA_EVAL_HF_REPO,
                GPT4_TURBO_REFERENCE_JSON,
                repo_type="dataset",
            )
        )
    else:
        path = Path(json_path)

    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"expected a JSON list in {path}")

    out: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        instruction = str(row.get("instruction", "")).strip()
        output = str(row.get("output", "")).strip()
        if not instruction:
            continue
        out.append(
            {
                "instruction": instruction,
                "output": output,
                "generator": str(row.get("generator") or "gpt4_turbo"),
            }
        )
    if num_examples is not None:
        if num_examples < 0:
            raise ValueError("num_examples must be non-negative")
        out = out[:num_examples]
    return out


def _should_swap(instruction: str) -> bool:
    """Deterministic position-bias mitigation (same idea as AE random switch)."""
    digest = hashlib.md5(instruction.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 2 == 1


def _winrate_from_preferences(preferences: Sequence[float]) -> dict[str, Any]:
    """Compute AE-compatible win-rate fields without importing alpaca_eval."""
    prefs = [float(p) for p in preferences if p == p]  # drop NaN
    n = len(prefs)
    if n == 0:
        return {
            "win_rate": None,
            "standard_error": None,
            "n_wins": 0,
            "n_draws": 0,
            "n_total": 0,
            "discrete_win_rate": None,
        }
    # preference: 1 = reference wins, 2 = model wins, 1.5 = draw
    wins = sum(1 for p in prefs if p > 1.5)
    draws = sum(1 for p in prefs if abs(p - 1.5) < 1e-9)
    # Absolute scoring: model share of (wins + 0.5 * draws) / n * 100
    score_sum = sum((p - 1.0) for p in prefs)  # maps 1→0, 1.5→0.5, 2→1
    win_rate = (score_sum / n) * 100.0
    # Bernoulli SE on [0,1] then *100
    p = score_sum / n
    se = ((p * (1.0 - p) / n) ** 0.5) * 100.0 if n > 1 else 0.0
    discrete = (wins / n) * 100.0
    return {
        "win_rate": win_rate,
        "standard_error": se,
        "n_wins": wins,
        "n_draws": draws,
        "n_total": n,
        "discrete_win_rate": discrete,
    }


def _load_judge_model(model_id: str, *, load_in_4bit: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    kwargs: dict[str, Any] = {"trust_remote_code": True}
    if torch.cuda.is_available() and load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = "auto"
        kwargs["torch_dtype"] = torch.float16
    elif torch.cuda.is_available():
        kwargs["device_map"] = "auto"
        kwargs["torch_dtype"] = torch.float16
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        kwargs["torch_dtype"] = torch.float16
    else:
        kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if "device_map" not in kwargs:
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            model = model.to("mps")
        else:
            model = model.to("cpu")
    model.eval()
    return model, tokenizer


def _generate_verdict(model, tokenizer, instruction: str, output_1: str, output_2: str) -> str:
    import torch

    user = _PAIRWISE_USER.format(
        instruction=instruction,
        output_1=output_1,
        output_2=output_2,
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are an impartial evaluator comparing two assistant outputs. "
                "Reply with only Output (a) or Output (b)."
            ),
        },
        {"role": "user", "content": user},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        # Qwen3.5 thinks by default; without enable_thinking=False, a short
        # max_new_tokens budget is consumed by <think> and never yields a/b.
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    else:
        prompt = user

    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens=16,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    new = out[:, enc["input_ids"].shape[1] :]
    return tokenizer.decode(new[0], skip_special_tokens=True).strip()


def evaluate_with_local_judge(
    model_outputs: Sequence[dict[str, str]] | str | Path,
    *,
    name: str | None = None,
    output_dir: str | Path | None = None,
    max_instances: int | None = None,
    judge_model: str | None = None,
    reference_outputs: Sequence[dict[str, str]] | None = None,
    load_in_4bit: bool = True,
    randomize_order: bool = True,
) -> dict[str, Any]:
    """Pairwise-compare ``model_outputs`` vs gpt4_turbo reference with a local HF judge.

    Returns a summary shaped like ``evaluate_model_outputs`` (metrics include
    ``win_rate``, ``avg_length``; ``length_controlled_winrate`` is always
    ``None`` for the local path).
    """
    from .alpacaeval_score import (
        _generator_from_rows,
        extract_model_metrics,
        write_model_outputs,
    )

    if isinstance(model_outputs, (str, Path)):
        path = Path(model_outputs)
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"expected a JSON list in {path}")
        model_name = name or _generator_from_rows(rows) or path.stem
    else:
        rows = list(model_outputs)
        model_name = name or _generator_from_rows(rows) or "Current model"

    if max_instances is not None:
        rows = rows[:max_instances]

    refs = list(reference_outputs) if reference_outputs is not None else load_gpt4_turbo_reference()
    ref_by_instruction = {
        str(r["instruction"]).strip(): str(r["output"]) for r in refs if r.get("instruction")
    }

    paired: list[tuple[str, str, str]] = []
    missing = 0
    for row in rows:
        instruction = str(row.get("instruction", "")).strip()
        model_out = str(row.get("output", ""))
        ref = ref_by_instruction.get(instruction)
        if ref is None:
            missing += 1
            continue
        paired.append((instruction, ref, model_out))

    if not paired:
        raise ValueError(
            "no overlapping instructions between model_outputs and gpt4_turbo "
            f"reference (missing={missing}, n_model={len(rows)}, n_ref={len(refs)})"
        )

    judge_id = (judge_model or "").strip() or default_local_judge_model()
    model, tokenizer = _load_judge_model(judge_id, load_in_4bit=load_in_4bit)

    annotations: list[dict[str, Any]] = []
    preferences: list[float] = []
    try:
        for instruction, ref_out, model_out in paired:
            if model_out == ref_out:
                pref = 1.5
                raw = ""
                swapped = False
            else:
                out_1, out_2 = ref_out, model_out
                swapped = randomize_order and _should_swap(instruction)
                if swapped:
                    out_1, out_2 = model_out, ref_out
                raw = _generate_verdict(model, tokenizer, instruction, out_1, out_2)
                verdict = parse_pairwise_verdict(raw)
                if verdict is None:
                    pref = float("nan")
                else:
                    # verdict 1 = output_1 wins, 2 = output_2 wins in the *prompt* order
                    pref = float(verdict)
                    if swapped:
                        pref = 3.0 - pref
            preferences.append(pref)
            annotations.append(
                {
                    "instruction": instruction,
                    "output_1": ref_out,
                    "output_2": model_out,
                    "generator_1": "gpt4_turbo",
                    "generator_2": model_name,
                    "preference": pref if pref == pref else None,
                    "raw_completion": raw,
                    "is_switched_outputs": swapped,
                    "annotator": judge_id,
                }
            )
    finally:
        del model
        from .utils import release_cuda_memory

        release_cuda_memory()

    # Prefer alpaca_eval's metric helper when importable.
    try:
        from alpaca_eval.metrics import get_winrate

        valid = [a for a in annotations if a.get("preference") is not None]
        wr = get_winrate(valid) if valid else _winrate_from_preferences([])
    except Exception:
        wr = _winrate_from_preferences(
            [p for p in preferences if p == p]
        )

    avg_length = (
        int(sum(len(str(r.get("output", ""))) for r in rows) / len(rows)) if rows else 0
    )
    leaderboard = {
        model_name: {
            **wr,
            "avg_length": avg_length,
            "length_controlled_winrate": None,
            "mode": "local_judge",
            "judge_model": judge_id,
        }
    }
    metrics = extract_model_metrics(leaderboard, model_name)
    # Ensure LC is explicitly null for local path.
    metrics["length_controlled_winrate"] = None
    metrics["judge_model"] = judge_id
    metrics["judge"] = "local"
    metrics["lc_note"] = (
        "length_controlled_winrate requires the official GPT-4-Turbo GLM "
        "pipeline; local judge reports raw win_rate + avg_length only"
    )

    out_dir = Path(output_dir) if output_dir is not None else Path("outputs/alpacaeval") / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    write_model_outputs(out_dir / "model_outputs.json", rows)
    (out_dir / "annotations.json").write_text(
        json.dumps(annotations, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "annotators_config": f"local:{judge_id}",
        "judge": "local",
        "judge_model": judge_id,
        "max_instances": max_instances,
        "n_model_outputs": len(rows),
        "n_annotated": len(annotations),
        "n_missing_reference": missing,
        "output_dir": str(out_dir),
        "metrics": metrics,
        "caveat": (
            "Local open-model pairwise judge ≈ AE2 protocol shape; not "
            "official AlpacaEval 2 GPT-4-Turbo LC leaderboard-equivalent"
        ),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
