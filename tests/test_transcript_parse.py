"""Unit tests for chat transcript segment parsing."""

from __future__ import annotations

from sdft.toolcall.format import (
    DEFAULT_COT_LINE,
    LFM_TOOL_CALL_END,
    LFM_TOOL_CALL_START,
)
from web.transcript_parse import highlight_boxed, parse_message_content


def _kinds(role: str, content: str) -> list[str]:
    return [s.kind for s in parse_message_content(role, content)]


def test_plain_prose_unchanged():
    text = "Hello! Here is a geek joke about Emacs."
    segments = parse_message_content("assistant", text)
    assert len(segments) == 1
    assert segments[0].kind == "prose"
    assert segments[0].content == text


def test_user_message_single_prose():
    segments = parse_message_content("user", "What is 2+2?")
    assert len(segments) == 1
    assert segments[0].kind == "prose"


def test_openclaw_tool_loop_response():
    text = (
        "Think: sum the first ten integers.\n\n"
        "I'll run code.\n\n"
        "<tool_call>\n"
        '{"name": "code_interpreter", "arguments": {"code": "print(sum(range(1, 11)))"}}\n'
        "</tool_call>\n\n"
        "<interpreter>\n55\n</interpreter>\n\n"
        "Answer: \\boxed{55}"
    )
    segments = parse_message_content("assistant", text)
    assert _kinds("assistant", text) == [
        "reasoning",
        "tool_call",
        "tool_result",
        "answer",
    ]
    assert segments[-1].boxed == "55"
    assert "code_interpreter" in segments[1].content
    assert segments[2].content == "55"


def test_lfm_tool_call_block():
    payload = (
        f'{LFM_TOOL_CALL_START}[{{"name": "code_interpreter", '
        f'"arguments": {{"code": "print(8)"}}}}]{LFM_TOOL_CALL_END}'
    )
    text = f"{DEFAULT_COT_LINE}\n\n{payload}\n\nAnswer: \\boxed{{8}}"
    segments = parse_message_content("assistant", text)
    assert segments[0].kind == "reasoning"
    assert segments[1].kind == "tool_call"
    assert segments[2].kind == "answer"
    assert segments[2].boxed == "8"


def test_tool_role_interpreter_tag():
    text = "\n\n<interpreter>\n42\n</interpreter>\n\n"
    segments = parse_message_content("tool", text)
    assert len(segments) == 1
    assert segments[0].kind == "tool_result"
    assert segments[0].content == "42"
    assert segments[0].label == "Interpreter"


def test_tool_role_plain_output():
    segments = parse_message_content("tool", "8")
    assert segments[0].kind == "tool_result"
    assert segments[0].label == "Tool output"


def test_highlight_boxed_renders_span():
    html = highlight_boxed("Answer: \\boxed{55}", "55")
    assert 'class="chat-boxed"' in str(html)
    assert "\\boxed{55}" in str(html)


def test_boxed_only_answer():
    segments = parse_message_content("assistant", "Final: \\boxed{hello}")
    assert segments[0].kind == "answer"
    assert segments[0].boxed == "hello"
