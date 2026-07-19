"""Train/eval overlap guards for OpenClaw tool-use splits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def normalize_question(q: str) -> str:
    return " ".join(q.strip().lower().split())


def question_id(q: str) -> str:
    return hashlib.sha256(normalize_question(q).encode("utf-8")).hexdigest()[:16]


def load_questions_from_jsonl(path: Path | str, *, field_candidates: tuple[str, ...] = ()) -> list[str]:
    path = Path(path)
    questions: list[str] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        q = _extract_question(row, field_candidates)
        if q is None:
            raise ValueError(f"Could not extract question from {path}: keys={sorted(row)}")
        questions.append(q)
    return questions


def _extract_question(row: dict[str, Any], field_candidates: tuple[str, ...]) -> str | None:
    keys = field_candidates or ("question", "input", "prompt")
    for key in keys:
        if key not in row or row[key] is None:
            continue
        val = row[key]
        if isinstance(val, str) and val.strip():
            if key == "prompt" and "<|im_start|>user" in val:
                # Eval-aligned train prompt: take last user turn content.
                part = val.rsplit("<|im_start|>user\n", 1)[-1]
                return part.split("<|im_end|>", 1)[0].strip()
            return val.strip()
        if isinstance(val, list):
            for msg in val:
                if isinstance(msg, dict) and msg.get("role") == "user" and msg.get("content"):
                    return str(msg["content"]).strip()
    return None


def assert_no_question_overlap(
    *,
    eval_questions: list[str],
    train_questions: list[str],
    forbidden: list[str] | None = None,
    label: str = "eval",
) -> None:
    """Raise ValueError if any eval question overlaps train or forbidden prompts."""
    train_ids = {question_id(q) for q in train_questions}
    train_norm = {normalize_question(q) for q in train_questions}
    eval_seen: set[str] = set()
    dups: list[str] = []
    overlaps: list[str] = []

    for q in eval_questions:
        nq = normalize_question(q)
        qid = question_id(q)
        if nq in eval_seen:
            dups.append(q)
        eval_seen.add(nq)
        if qid in train_ids or nq in train_norm:
            overlaps.append(q)

    forbidden = forbidden or []
    forbidden_hits = [
        q
        for q in eval_questions
        if normalize_question(q) in {normalize_question(f) for f in forbidden}
        or question_id(q) in {question_id(f) for f in forbidden}
    ]

    errors: list[str] = []
    if dups:
        errors.append(f"{label} has duplicate questions: {dups!r}")
    if overlaps:
        errors.append(f"{label} overlaps train set ({len(overlaps)}): {overlaps!r}")
    if forbidden_hits:
        errors.append(f"{label} contains reserved few-shot prompts: {forbidden_hits!r}")
    if errors:
        raise ValueError(" ; ".join(errors))


def write_heldout_jsonl(
    rows: list[dict[str, str]],
    path: Path | str,
) -> list[dict[str, Any]]:
    """Write held-out eval rows with stable ids; drop duplicate questions."""
    path = Path(path)
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        q = row["question"].strip()
        nq = normalize_question(q)
        if nq in seen:
            continue
        seen.add(nq)
        item = {
            "id": question_id(q),
            "question": q,
            "answer": str(row["answer"]).strip(),
        }
        out.append(item)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for item in out:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    return out
