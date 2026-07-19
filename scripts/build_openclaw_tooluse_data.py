#!/usr/bin/env python3
"""Build curated OpenClaw-style tool-use SDFT data and an easy eval set.

Produces:
  data/openclaw_tooluse.jsonl       — Alpaca rows (instruction/input/output) for sdft.generate
  data/openclaw_tooluse_sdft.jsonl  — identity SDFT pairs (sdft_response == gold output)
  data/openclaw_demo.jsonl          — easy eval rows (question + numeric answer)

Each training completion teaches the ReTool protocol:
  1. Optional brief reasoning
  2. <tool_call>{"name": "code_interpreter", ...}</tool_call>
  3. <interpreter>...</interpreter> with the sandbox result
  4. Answer: \\boxed{...}

Usage:
  uv run python scripts/build_openclaw_tooluse_data.py
  uv run python scripts/build_openclaw_tooluse_data.py --write-sdft
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

INSTRUCTION = (
    "Solve the math problem. When you need to compute, call the code_interpreter tool "
    "with Python code. After using tool results, give the final answer as "
    "Answer: \\boxed{number}."
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def _trajectory(*, code: str, interpreter_out: str, answer: str, lead_in: str = "") -> str:
    """Render a single-turn OpenClaw ReTool assistant completion."""
    prefix = f"{lead_in.strip()}\n\n" if lead_in.strip() else ""
    return (
        f"{prefix}"
        f"<tool_call>\n"
        f'{{"name": "code_interpreter", "arguments": {{"code": {json.dumps(code)}}}}}\n'
        f"</tool_call>\n\n"
        f"<interpreter>\n{interpreter_out.strip()}\n</interpreter>\n\n"
        f"Answer: \\boxed{{{answer}}}"
    )


def _build_problems(rng: random.Random) -> list[dict[str, str]]:
    """Return rows with keys: question, code, interpreter_out, answer, lead_in."""
    rows: list[dict[str, str]] = []

    # Fixed easy arithmetic (deterministic demos).
    fixed = [
        ("What is 12 + 19?", "print(12 + 19)", "31", "31", "I'll add the two numbers."),
        ("Compute 45 - 17.", "print(45 - 17)", "28", "28", "Subtraction with Python."),
        ("What is 7 times 8?", "print(7 * 8)", "56", "56", "Multiplication."),
        ("Divide 144 by 12.", "print(144 // 12)", "12", "12", "Integer division."),
        ("What is 2 to the power of 10?", "print(2 ** 10)", "1024", "1024", "Exponentiation."),
        ("Find 15 % 4.", "print(15 % 4)", "3", "3", "Remainder."),
        ("Sum 1 through 10.", "print(sum(range(1, 11)))", "55", "55", "Use range and sum."),
        ("What is sqrt(81)?", "import math\nprint(int(math.isqrt(81)))", "9", "9", "Square root."),
        ("Evaluate (3 + 5) * 2.", "print((3 + 5) * 2)", "16", "16", "Order of operations."),
        ("What is 100 - 37?", "print(100 - 37)", "63", "63", ""),
    ]
    for question, code, interp, answer, lead in fixed:
        rows.append(
            {
                "question": question,
                "code": code,
                "interpreter_out": interp,
                "answer": answer,
                "lead_in": lead,
            }
        )

    # Synthetic variations.
    for _ in range(40):
        a, b = rng.randint(2, 99), rng.randint(2, 99)
        op = rng.choice(["add", "sub", "mul"])
        if op == "add":
            question = f"What is {a} + {b}?"
            code = f"print({a} + {b})"
            answer = str(a + b)
        elif op == "sub":
            hi, lo = max(a, b), min(a, b)
            question = f"Compute {hi} - {lo}."
            code = f"print({hi} - {lo})"
            answer = str(hi - lo)
        else:
            question = f"What is {a} times {b}?"
            code = f"print({a} * {b})"
            answer = str(a * b)
        rows.append(
            {
                "question": question,
                "code": code,
                "interpreter_out": answer,
                "answer": answer,
                "lead_in": "I'll use the code interpreter.",
            }
        )

    for _ in range(20):
        n = rng.randint(3, 12)
        question = f"What is {n} factorial?"
        code = (
            "import math\n"
            f"print(math.factorial({n}))"
        )
        import math

        answer = str(math.factorial(n))
        rows.append(
            {
                "question": question,
                "code": code,
                "interpreter_out": answer,
                "answer": answer,
                "lead_in": "Factorial via math.factorial.",
            }
        )

    for _ in range(15):
        base = rng.randint(2, 20)
        exp = rng.randint(2, 4)
        question = f"Evaluate {base}^{exp}."
        code = f"print({base} ** {exp})"
        answer = str(base**exp)
        rows.append(
            {
                "question": question,
                "code": code,
                "interpreter_out": answer,
                "answer": answer,
                "lead_in": "",
            }
        )

    for _ in range(15):
        a = rng.randint(10, 99)
        pct = rng.choice([10, 20, 25, 50])
        question = f"What is {pct}% of {a}?"
        code = f"print({a} * {pct} / 100)"
        val = a * pct / 100
        answer = str(int(val) if val == int(val) else val)
        interp = str(int(val)) if val == int(val) else f"{val:g}"
        rows.append(
            {
                "question": question,
                "code": code,
                "interpreter_out": interp,
                "answer": answer,
                "lead_in": "Percentage calculation.",
            }
        )

    return rows


def _demo_subset(problems: list[dict[str, str]], rng: random.Random, n: int = 20) -> list[dict[str, str]]:
    """Hold out easy eval examples not overlapping training prompts."""
    picks = rng.sample(problems, min(n, len(problems)))
    return picks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--write-sdft",
        action="store_true",
        help="also write data/openclaw_tooluse_sdft.jsonl (identity SDFT pairs)",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    all_problems = _build_problems(rng)
    rng.shuffle(all_problems)

    demo = _demo_subset(all_problems, rng, n=20)
    demo_questions = {d["question"] for d in demo}
    train = [p for p in all_problems if p["question"] not in demo_questions]

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    tooluse_path = DATA_DIR / "openclaw_tooluse.jsonl"
    with tooluse_path.open("w") as fh:
        for row in train:
            output = _trajectory(
                code=row["code"],
                interpreter_out=row["interpreter_out"],
                answer=row["answer"],
                lead_in=row["lead_in"],
            )
            fh.write(
                json.dumps(
                    {
                        "instruction": INSTRUCTION,
                        "input": row["question"],
                        "output": output,
                    }
                )
                + "\n"
            )
    print(f"wrote {len(train)} training rows to {tooluse_path}")

    demo_path = DATA_DIR / "openclaw_demo.jsonl"
    with demo_path.open("w") as fh:
        for row in demo:
            fh.write(
                json.dumps({"question": row["question"], "answer": row["answer"]})
                + "\n"
            )
    print(f"wrote {len(demo)} demo eval rows to {demo_path}")

    if args.write_sdft:
        sdft_path = DATA_DIR / "openclaw_tooluse_sdft.jsonl"
        with sdft_path.open("w") as fh:
            for row in train:
                prompt = f"{INSTRUCTION}\n\n{row['question']}".strip()
                completion = _trajectory(
                    code=row["code"],
                    interpreter_out=row["interpreter_out"],
                    answer=row["answer"],
                    lead_in=row["lead_in"],
                )
                fh.write(
                    json.dumps(
                        {
                            "prompt": prompt,
                            "response": completion,
                            "sdft_response": completion,
                        }
                    )
                    + "\n"
                )
        print(f"wrote {len(train)} identity SDFT pairs to {sdft_path}")


if __name__ == "__main__":
    main()
