"""Parse chat transcript messages into typed segments for web rendering."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Literal

from markupsafe import Markup, escape

from sdft.toolcall.format import (
    DEFAULT_COT_LINE,
    LFM_TOOL_CALL_END,
    LFM_TOOL_CALL_START,
    _match_answer_boxed,
)

SegmentKind = Literal[
    "reasoning", "tool_call", "tool_result", "answer", "prose", "refusal"
]

EMPTY_ASSISTANT_FALLBACK = "I'm sorry, but I can't assist with that."

_THINK_LINE = re.compile(r"^Think\s*:", re.IGNORECASE | re.MULTILINE)
_REFUSAL_START = re.compile(
    r"I'm sorry(?:,|\.)?\s+but I can't\b",
    re.IGNORECASE,
)
_REFUSAL_NOISE_MARKERS = re.compile(
    r"can't perform|cannot perform|can't actually|do not have access|don't have access"
    r"|only allow|sandbox|safe environment|available tools"
    r"|Let me know if you'd like help|Let me know how else I can assist",
    re.IGNORECASE,
)

_LFM_TOOL_CALL = re.compile(
    rf"{re.escape(LFM_TOOL_CALL_START)}.*?{re.escape(LFM_TOOL_CALL_END)}",
    re.DOTALL,
)
_OPENCLAW_TOOL_CALL = re.compile(r"<tool_call>\s*.*?\s*</tool_call>", re.DOTALL)
_INTERPRETER = re.compile(r"<interpreter>\s*(.*?)\s*</interpreter>", re.DOTALL)
_PYTHON_FENCE = re.compile(r"```python\s*.*?\s*```", re.DOTALL)
_CODE_TAG = re.compile(r"<code>\s*.*?\s*</code>", re.DOTALL)

_SCANNERS: list[tuple[SegmentKind, str, re.Pattern[str], bool]] = [
    ("tool_call", "Tool call", _LFM_TOOL_CALL, False),
    ("tool_call", "Tool call", _OPENCLAW_TOOL_CALL, False),
    ("tool_result", "Interpreter", _INTERPRETER, True),
    ("tool_call", "Code", _PYTHON_FENCE, False),
    ("tool_call", "Code", _CODE_TAG, False),
]


@dataclass(frozen=True)
class TranscriptSegment:
    kind: SegmentKind
    content: str
    label: str = ""
    boxed: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _is_reasoning(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if _THINK_LINE.search(stripped):
        return True
    cot = DEFAULT_COT_LINE.strip()
    if cot and cot in stripped:
        return True
    first = stripped.split("\n", 1)[0].strip()
    return first.lower().startswith("think:")


def _is_refusal_noise(text: str) -> bool:
    """True for repeated capability/tool sandbox apologies, not real answers."""
    stripped = text.strip()
    if not stripped:
        return False
    if _match_answer_boxed(stripped) is not None:
        return False
    if not _REFUSAL_START.search(stripped):
        return False
    return bool(_REFUSAL_NOISE_MARKERS.search(stripped))


def _split_refusal_loops(text: str) -> tuple[str, list[str]]:
    """Split helpful text from trailing repeated refusal/apology loops."""
    stripped = text.strip()
    if not stripped:
        return "", []

    match = _REFUSAL_START.search(stripped)
    if not match:
        return stripped, []

    clean = stripped[: match.start()].rstrip()
    rest = stripped[match.start() :]
    chunks = re.split(
        r"(?=I'm sorry(?:,|\.)?\s+but I can't\b)",
        rest,
        flags=re.IGNORECASE,
    )
    refusals = [chunk.strip() for chunk in chunks if chunk.strip() and _is_refusal_noise(chunk)]
    return clean, refusals


def _split_reasoning_prefix(text: str) -> tuple[str | None, str]:
    """Return a leading ``Think:`` line and the remainder when present."""
    stripped = text.strip()
    if not stripped:
        return None, ""
    first_line, _, rest = stripped.partition("\n")
    if not first_line.strip().lower().startswith("think:"):
        return None, stripped
    if not rest.strip():
        return first_line.strip(), ""
    return first_line.strip(), rest.strip()


def _refusal_segment(refusals: list[str]) -> TranscriptSegment | None:
    if not refusals:
        return None
    sample = refusals[0]
    count = len(refusals)
    label = f"Skipped refusals ({count})" if count > 1 else "Skipped refusal"
    return TranscriptSegment(kind="refusal", content=sample, label=label)


def _kind_for_clean_text(
    text: str,
    *,
    saw_tool: bool,
    is_final: bool,
    default: SegmentKind = "prose",
) -> SegmentKind:
    if _match_answer_boxed(text) is not None:
        return "answer"
    gap_kind = _classify_gap(text, saw_tool=saw_tool, is_final=is_final)
    if gap_kind == "answer":
        return "answer"
    if gap_kind == "reasoning":
        return default
    return default


def _expand_text_block(
    text: str,
    *,
    saw_tool: bool = False,
    is_final: bool = False,
    split_reasoning: bool = False,
    default_kind: SegmentKind = "prose",
) -> list[TranscriptSegment]:
    """Split one text block into reasoning, answer/prose, and folded refusals."""
    stripped = text.strip()
    if not stripped:
        return []

    segments: list[TranscriptSegment] = []
    work = stripped
    if split_reasoning and default_kind != "reasoning":
        reasoning, rest = _split_reasoning_prefix(stripped)
        if reasoning:
            segments.append(TranscriptSegment(kind="reasoning", content=reasoning))
            work = rest

    if not work:
        return segments

    clean, refusals = _split_refusal_loops(work)
    if clean:
        kind = _kind_for_clean_text(
            clean,
            saw_tool=saw_tool,
            is_final=is_final,
            default=default_kind,
        )
        segments.append(
            TranscriptSegment(
                kind=kind,
                content=clean,
                boxed=_match_answer_boxed(clean) if kind == "answer" else None,
            )
        )

    refusal = _refusal_segment(refusals)
    if refusal:
        segments.append(refusal)

    if segments:
        return segments

    refusal_only = _refusal_segment(refusals)
    return [refusal_only] if refusal_only else []


def _classify_gap(text: str, *, saw_tool: bool, is_final: bool) -> SegmentKind | None:
    stripped = text.strip()
    if not stripped:
        return None
    if _match_answer_boxed(stripped) is not None:
        return "answer"
    if _is_reasoning(stripped):
        return "reasoning"
    if saw_tool and is_final:
        return "answer"
    if is_final:
        return "prose"
    if saw_tool:
        return "answer"
    return "prose"


def _scan_special_spans(text: str) -> list[tuple[int, int, SegmentKind, str, str]]:
    """Return ordered non-overlapping (start, end, kind, label, content) spans."""
    spans: list[tuple[int, int, SegmentKind, str, str]] = []
    pos = 0
    while pos < len(text):
        best: tuple[int, int, SegmentKind, str, str] | None = None
        for kind, label, pattern, inner in _SCANNERS:
            match = pattern.search(text, pos)
            if not match:
                continue
            content = match.group(1) if inner else match.group(0)
            candidate = (match.start(), match.end(), kind, label, content)
            if best is None or candidate[0] < best[0]:
                best = candidate
        if best is None:
            break
        spans.append(best)
        pos = best[1]
    return spans


def display_assistant_content(content: str | None) -> str:
    """Return assistant text for UI, substituting the standard refusal when empty."""
    if content is None or not str(content).strip():
        return EMPTY_ASSISTANT_FALLBACK
    return str(content)


def parse_message_content(role: str, content: str) -> list[TranscriptSegment]:
    """Split one transcript message into renderable segments."""
    text = content or ""
    if role == "assistant" and not text.strip():
        return [TranscriptSegment(kind="prose", content=EMPTY_ASSISTANT_FALLBACK)]
    if role == "tool":
        match = _INTERPRETER.search(text)
        if match:
            inner = match.group(1).strip()
            return [
                TranscriptSegment(
                    kind="tool_result",
                    content=inner or text.strip(),
                    label="Interpreter",
                )
            ]
        stripped = text.strip()
        if not stripped:
            return []
        return [
            TranscriptSegment(
                kind="tool_result",
                content=stripped,
                label="Tool output",
            )
        ]

    if role not in ("assistant", "system"):
        if not text.strip():
            return []
        return [TranscriptSegment(kind="prose", content=text)]

    spans = _scan_special_spans(text)
    if not spans:
        stripped = text.strip()
        if not stripped:
            return []
        return _expand_text_block(
            stripped,
            split_reasoning=True,
            is_final=True,
        )

    segments: list[TranscriptSegment] = []
    cursor = 0
    saw_tool = False
    for start, end, kind, label, span_content in spans:
        gap = text[cursor:start]
        if gap.strip():
            segments.extend(
                _expand_text_block(
                    gap,
                    saw_tool=saw_tool,
                    is_final=False,
                    split_reasoning=not saw_tool,
                    default_kind="reasoning",
                )
            )

        segments.append(
            TranscriptSegment(
                kind=kind,
                content=span_content.strip(),
                label=label,
            )
        )
        if kind in ("tool_call", "tool_result"):
            saw_tool = True
        cursor = end

    tail = text[cursor:]
    if tail.strip():
        segments.extend(
            _expand_text_block(
                tail,
                saw_tool=saw_tool,
                is_final=True,
                split_reasoning=False,
            )
        )

    return segments


def highlight_boxed(text: str, boxed: str | None = None) -> Markup:
    """Render answer text with a highlighted ``\\boxed{...}`` span when present."""
    if not text:
        return Markup("")
    if boxed is None:
        boxed = _match_answer_boxed(text)
    if not boxed:
        return Markup(escape(text))

    needle = f"\\boxed{{{boxed}}}"
    idx = text.find(needle)
    if idx < 0:
        return Markup(escape(text))

    before = escape(text[:idx])
    after = escape(text[idx + len(needle) :])
    inner = escape(boxed)
    return Markup(
        f'{before}<span class="chat-boxed">\\boxed{{{inner}}}</span>{after}'
    )
