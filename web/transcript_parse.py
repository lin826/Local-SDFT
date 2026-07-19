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

SegmentKind = Literal["reasoning", "tool_call", "tool_result", "answer", "prose"]

_THINK_LINE = re.compile(r"^Think\s*:", re.IGNORECASE | re.MULTILINE)

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


def parse_message_content(role: str, content: str) -> list[TranscriptSegment]:
    """Split one transcript message into renderable segments."""
    text = content or ""
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
        kind: SegmentKind = "answer" if _match_answer_boxed(stripped) else "prose"
        return [
            TranscriptSegment(
                kind=kind,
                content=stripped,
                boxed=_match_answer_boxed(stripped) if kind == "answer" else None,
            )
        ]

    segments: list[TranscriptSegment] = []
    cursor = 0
    saw_tool = False
    for start, end, kind, label, span_content in spans:
        gap = text[cursor:start]
        gap_kind = _classify_gap(gap, saw_tool=saw_tool, is_final=False)
        if gap_kind:
            gap_text = gap.strip()
            segments.append(
                TranscriptSegment(
                    kind=gap_kind,
                    content=gap_text,
                    boxed=_match_answer_boxed(gap_text) if gap_kind == "answer" else None,
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
    tail_kind = _classify_gap(tail, saw_tool=saw_tool, is_final=True)
    if tail_kind:
        tail_text = tail.strip()
        segments.append(
            TranscriptSegment(
                kind=tail_kind,
                content=tail_text,
                boxed=_match_answer_boxed(tail_text) if tail_kind == "answer" else None,
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
