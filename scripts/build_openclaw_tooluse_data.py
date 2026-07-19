#!/usr/bin/env python3
"""Build curated OpenClaw-style tool-use SDFT data and a held-out eval set.

Produces:
  data/openclaw_tooluse.jsonl         — Alpaca rows for optional generate
  data/openclaw_tooluse_sdft.jsonl    — eval-aligned two-turn SDFT pairs
  data/openclaw_eval_heldout.jsonl    — canonical held-out eval (ids + answers)
  data/openclaw_demo.jsonl            — mirror of held-out (question/answer)
  data/openclaw_split_manifest.json   — split documentation / overlap=0

Training targets match the eval tool loop:
  turn 1 — emit <tool_call> only (no fake interpreter in the generation)
  turn 2 — after observation, emit Answer: \\boxed{...}

Usage:
  uv run python scripts/build_openclaw_tooluse_data.py --write-sdft
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sdft.toolcall.format import (  # noqa: E402
    DEFAULT_LFM_JSON_SYSTEM,
    LFM_TOOL_CALL_END,
    LFM_TOOL_CALL_START,
)
from sdft.toolcall.sandbox import CODE_INTERPRETER_TOOL  # noqa: E402
from sdft.toolcall.split_guard import (  # noqa: E402
    assert_no_question_overlap,
    normalize_question,
    question_id,
    write_heldout_jsonl,
)
from sdft.utils import load_tokenizer  # noqa: E402
from sdft.config import ModelConfig  # noqa: E402

INSTRUCTION = (
    "Solve the math problem. When you need to compute, call the code_interpreter tool "
    "with Python code. After using tool results, give the final answer as "
    "Answer: \\boxed{number}."
)

DATA_DIR = ROOT / "data"
FORBIDDEN_FEW_SHOT = ["What is 3 + 5?"]
MODEL_NAME = "LiquidAI/LFM2.5-230M"


def _tool_call_block(code: str, *, fmt: str = "lfm") -> str:
    if fmt == "lfm":
        payload = json.dumps([{"name": "code_interpreter", "arguments": {"code": code}}])
        return f"{LFM_TOOL_CALL_START}{payload}{LFM_TOOL_CALL_END}"
    return (
        "<tool_call>\n"
        f'{{"name": "code_interpreter", "arguments": {{"code": {json.dumps(code)}}}}}\n'
        "</tool_call>"
    )


def _answer_block(answer: str) -> str:
    return f"Answer: \\boxed{{{answer}}}"


def _trajectory(*, code: str, interpreter_out: str, answer: str, lead_in: str = "") -> str:
    """Legacy single-turn Alpaca completion (OpenClaw tags; for generate inspection)."""
    prefix = f"{lead_in.strip()}\n\n" if lead_in.strip() else ""
    return (
        f"{prefix}{_tool_call_block(code, fmt='openclaw')}\n\n"
        f"<interpreter>\n{interpreter_out.strip()}\n</interpreter>\n\n"
        f"{_answer_block(answer)}"
    )


def _sdft_turn_pairs(row: dict[str, str], tokenizer) -> list[dict[str, str]]:
    """LFM-native two-turn pairs matching eval ToolLoopConfig(format='lfm')."""
    question = row["question"]
    code = row["code"]
    interp = row["interpreter_out"]
    answer = row["answer"]
    lead = row.get("lead_in", "")

    turn1 = (
        f"{lead.strip()}\n{_tool_call_block(code, fmt='lfm')}".strip()
        if lead.strip()
        else _tool_call_block(code, fmt="lfm")
    )
    msgs1 = [
        {"role": "system", "content": DEFAULT_LFM_JSON_SYSTEM},
        {"role": "user", "content": question},
    ]
    prefix1 = tokenizer.apply_chat_template(
        msgs1,
        tools=[CODE_INTERPRETER_TOOL],
        tokenize=False,
        add_generation_prompt=True,
    )
    msgs2 = [
        {"role": "system", "content": DEFAULT_LFM_JSON_SYSTEM},
        {"role": "user", "content": question},
        {"role": "assistant", "content": turn1},
        {"role": "tool", "content": interp},
    ]
    prefix2 = tokenizer.apply_chat_template(
        msgs2,
        tools=[CODE_INTERPRETER_TOOL],
        tokenize=False,
        add_generation_prompt=True,
    )
    return [
        {"prompt": prefix1, "sdft_response": turn1},
        {"prompt": prefix2, "sdft_response": _answer_block(answer)},
    ]


def _row(question: str, code: str, answer: str, lead_in: str = "") -> dict[str, str]:
    return {
        "question": question,
        "code": code,
        "interpreter_out": answer,
        "answer": answer,
        "lead_in": lead_in,
    }


def _build_train_problems(rng: random.Random) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    fixed = [
        ("What is 12 + 19?", "print(12 + 19)", "31", "I'll add the two numbers."),
        ("Compute 45 - 17.", "print(45 - 17)", "28", "Subtraction with Python."),
        ("What is 7 times 8?", "print(7 * 8)", "56", "Multiplication."),
        ("Divide 144 by 12.", "print(144 // 12)", "12", "Integer division."),
        ("What is 2 to the power of 10?", "print(2 ** 10)", "1024", "Exponentiation."),
        ("Find 15 % 4.", "print(15 % 4)", "3", "Remainder."),
        ("Sum 1 through 10.", "print(sum(range(1, 11)))", "55", "Use range and sum."),
        ("What is sqrt(81)?", "import math\nprint(int(math.isqrt(81)))", "9", "Square root."),
        ("Evaluate (3 + 5) * 2.", "print((3 + 5) * 2)", "16", "Order of operations."),
        ("What is 100 - 37?", "print(100 - 37)", "63", ""),
        ("What is 5 factorial?", "import math\nprint(math.factorial(5))", "120", "Factorial."),
        ("What is 2 to the 12th power?", "print(2 ** 12)", "4096", "Power of two."),
    ]
    for q, code, ans, lead in fixed:
        rows.append(_row(q, code, ans, lead))

    for _ in range(50):
        a, b = rng.randint(2, 99), rng.randint(2, 99)
        op = rng.choice(["add", "sub", "mul"])
        if op == "add":
            rows.append(_row(f"What is {a} + {b}?", f"print({a} + {b})", str(a + b), "I'll use the code interpreter."))
        elif op == "sub":
            hi, lo = max(a, b), min(a, b)
            rows.append(_row(f"Compute {hi} - {lo}.", f"print({hi} - {lo})", str(hi - lo), "I'll use the code interpreter."))
        else:
            rows.append(_row(f"What is {a} times {b}?", f"print({a} * {b})", str(a * b), "I'll use the code interpreter."))

    for _ in range(25):
        n = rng.randint(3, 11)
        rows.append(
            _row(
                f"What is {n} factorial?",
                f"import math\nprint(math.factorial({n}))",
                str(math.factorial(n)),
                "Factorial via math.factorial.",
            )
        )

    for _ in range(20):
        base, exp = rng.randint(2, 12), rng.randint(2, 4)
        rows.append(_row(f"Evaluate {base}^{exp}.", f"print({base} ** {exp})", str(base**exp), ""))

    for _ in range(20):
        a, pct = rng.randint(10, 90), rng.choice([10, 20, 25, 50])
        val = a * pct / 100
        ans = str(int(val) if val == int(val) else val)
        rows.append(_row(f"What is {pct}% of {a}?", f"print({a} * {pct} / 100)", ans, "Percentage calculation."))

    return rows


def _build_heldout_problems() -> list[dict[str, str]]:
    """Fixed medium/easy held-out bank — different numbers than the train fixed set."""
    return [
        _row("What is 17 times 23?", "print(17 * 23)", "391", "I'll multiply with Python."),
        _row(
            "What is the sum of squares from 1 to 10?",
            "print(sum(i*i for i in range(1, 11)))",
            "385",
            "Sum of squares via a loop.",
        ),
        _row(
            "How many ways to choose 3 items from 10?",
            "import math\nprint(math.comb(10, 3))",
            "120",
            "Binomial coefficient.",
        ),
        _row("What is 999 divided by 7?", "print(999 // 7)", "142", "Integer division."),
        _row("Compute the product 13 * 17 * 19.", "print(13 * 17 * 19)", "4199", "Triple product."),
        _row("Sum all even numbers from 2 to 50.", "print(sum(range(2, 51, 2)))", "650", "Even sum with range step 2."),
        _row("What is 1234 + 5678?", "print(1234 + 5678)", "6912", "Large addition."),
        _row(
            "How many divisors does 36 have?",
            "print(sum(1 for i in range(1, 37) if 36 % i == 0))",
            "9",
            "Count divisors.",
        ),
        _row("Evaluate 9^3.", "print(9 ** 3)", "729", ""),
        _row("Evaluate 20^2.", "print(20 ** 2)", "400", ""),
        _row("What is 31 + 47?", "print(31 + 47)", "78", "I'll use the code interpreter."),
        _row("What is 53 times 61?", "print(53 * 61)", "3233", "I'll use the code interpreter."),
        _row("What is 25% of 84?", "print(84 * 25 / 100)", "21", "Percentage calculation."),
        _row("What is 8 factorial?", "import math\nprint(math.factorial(8))", "40320", "Factorial via math.factorial."),
        _row("Compute 88 - 29.", "print(88 - 29)", "59", "I'll use the code interpreter."),
        _row("What is 11 to the power of 3?", "print(11 ** 3)", "1331", "Exponentiation."),
        _row("Sum 1 through 20.", "print(sum(range(1, 21)))", "210", "Use range and sum."),
        _row("Find 29 % 7.", "print(29 % 7)", "1", "Remainder."),
        _row("Divide 512 by 8.", "print(512 // 8)", "64", "Integer division."),
        _row("What is sqrt(144)?", "import math\nprint(int(math.isqrt(144)))", "12", "Square root."),
        _row("Evaluate (6 + 9) * 4.", "print((6 + 9) * 4)", "60", "Order of operations."),
        _row("What is 41 times 19?", "print(41 * 19)", "779", "I'll use the code interpreter."),
        _row("How many ways to choose 2 items from 12?", "import math\nprint(math.comb(12, 2))", "66", "Binomial coefficient."),
        _row("What is 777 + 888?", "print(777 + 888)", "1665", "Large addition."),
        _row("Sum all odd numbers from 1 to 25.", "print(sum(range(1, 26, 2)))", "169", "Odd sum."),
        _row("What is 14 factorial?", "import math\nprint(math.factorial(14))", str(math.factorial(14)), "Factorial."),
        _row("Compute the product 7 * 11 * 13.", "print(7 * 11 * 13)", "1001", "Triple product."),
        _row("What is 50% of 96?", "print(96 * 50 / 100)", "48", "Percentage calculation."),
        _row("Evaluate 15^2.", "print(15 ** 2)", "225", ""),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--write-sdft",
        action="store_true",
        help="also write data/openclaw_tooluse_sdft.jsonl (eval-aligned SDFT pairs)",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    heldout = _build_heldout_problems()
    heldout_norms = {normalize_question(r["question"]) for r in heldout}
    forbidden_norms = {normalize_question(q) for q in FORBIDDEN_FEW_SHOT}

    train_raw = _build_train_problems(rng)
    train: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in train_raw:
        nq = normalize_question(row["question"])
        if nq in heldout_norms or nq in forbidden_norms or nq in seen:
            continue
        seen.add(nq)
        train.append(row)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    tooluse_path = DATA_DIR / "openclaw_tooluse.jsonl"
    with tooluse_path.open("w") as fh:
        for row in train:
            fh.write(
                json.dumps(
                    {
                        "instruction": INSTRUCTION,
                        "input": row["question"],
                        "output": _trajectory(
                            code=row["code"],
                            interpreter_out=row["interpreter_out"],
                            answer=row["answer"],
                            lead_in=row["lead_in"],
                        ),
                    }
                )
                + "\n"
            )
    print(f"wrote {len(train)} training rows to {tooluse_path}")

    heldout_path = DATA_DIR / "openclaw_eval_heldout.jsonl"
    held = write_heldout_jsonl(heldout, heldout_path)
    demo_path = DATA_DIR / "openclaw_demo.jsonl"
    with demo_path.open("w") as fh:
        for row in held:
            fh.write(json.dumps({"question": row["question"], "answer": row["answer"]}) + "\n")

    assert_no_question_overlap(
        eval_questions=[r["question"] for r in held],
        train_questions=[p["question"] for p in train],
        forbidden=FORBIDDEN_FEW_SHOT,
        label="openclaw_eval_heldout",
    )
    manifest = {
        "train_file": "data/openclaw_tooluse.jsonl",
        "train_n": len(train),
        "heldout_file": "data/openclaw_eval_heldout.jsonl",
        "heldout_n": len(held),
        "heldout_ids": [question_id(r["question"]) for r in held],
        "overlap": 0,
        "forbidden_few_shot": FORBIDDEN_FEW_SHOT,
        "notes": (
            "Held-out bank is a fixed disjoint problem list (different numbers). "
            "Train filtered against held-out + reserved few-shot. Deduped by normalized question."
        ),
    }
    manifest_path = DATA_DIR / "openclaw_split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(held)} held-out eval rows to {heldout_path}")
    print(f"wrote {len(held)} demo mirror rows to {demo_path}")
    print(f"wrote split manifest to {manifest_path}")

    if args.write_sdft:
        tokenizer = load_tokenizer(ModelConfig(name=MODEL_NAME))
        sdft_path = DATA_DIR / "openclaw_tooluse_sdft.jsonl"
        n_pairs = 0
        with sdft_path.open("w") as fh:
            for row in train:
                for pair in _sdft_turn_pairs(row, tokenizer):
                    fh.write(json.dumps(pair) + "\n")
                    n_pairs += 1
        print(f"wrote {n_pairs} LFM-aligned SDFT turn pairs ({len(train)} problems) to {sdft_path}")


if __name__ == "__main__":
    main()
