"""Unit tests for chat transcript segment parsing."""

from __future__ import annotations

from sdft.toolcall.format import (
    DEFAULT_COT_LINE,
    LFM_TOOL_CALL_END,
    LFM_TOOL_CALL_START,
)
from web.transcript_parse import (
    EMPTY_ASSISTANT_FALLBACK,
    display_assistant_content,
    highlight_boxed,
    parse_message_content,
)


def _kinds(role: str, content: str) -> list[str]:
    return [s.kind for s in parse_message_content(role, content)]


def test_empty_assistant_content_shows_refusal_fallback():
    for content in ("", "   ", "\n\t"):
        segments = parse_message_content("assistant", content)
        assert len(segments) == 1
        assert segments[0].kind == "prose"
        assert segments[0].content == EMPTY_ASSISTANT_FALLBACK


def test_display_assistant_content_fallback():
    assert display_assistant_content("") == EMPTY_ASSISTANT_FALLBACK
    assert display_assistant_content("  ") == EMPTY_ASSISTANT_FALLBACK
    assert display_assistant_content(None) == EMPTY_ASSISTANT_FALLBACK
    assert display_assistant_content("Hello") == "Hello"


def test_empty_user_message_stays_empty():
    assert parse_message_content("user", "") == []


def test_plain_prose_unchanged():
    text = "Hello! Here is a short note about Emacs."
    segments = parse_message_content("assistant", text)
    assert len(segments) == 1
    assert segments[0].kind == "prose"
    assert segments[0].content == text


def test_refusal_opener_without_noise_kept_as_prose():
    text = (
        "I'm sorry, but I can't assist with that. Making apple juice is a recipe "
        "that requires specific ingredients and equipment, which are not accessible "
        "through my current capabilities."
    )
    segments = parse_message_content("assistant", text)
    assert len(segments) == 1
    assert segments[0].kind == "prose"
    assert segments[0].content == text


def test_exact_refusal_fallback_string_renders():
    segments = parse_message_content("assistant", EMPTY_ASSISTANT_FALLBACK)
    assert len(segments) == 1
    assert segments[0].kind == "prose"
    assert segments[0].content == EMPTY_ASSISTANT_FALLBACK


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


_REFUSAL_LOOP = (
    "I'm sorry, but I can't perform actual sewing actions. However, I can provide "
    "general sewing tips or help you find resources if needed! Let me know how "
    "else I can assist."
)

_SANDBOX_REFUSAL = (
    "I'm sorry, but I can't perform that action. The available tools only allow me "
    "to execute Python code in a safe sandbox environment. Let me know if you'd "
    "like help with something else!"
)


def test_sewing_bench_hides_repeated_refusals():
    helpful = (
        "Think: call the sewing tool, then describe the steps you'd take.\n\n"
        "For example:\n"
        "- Choose a fabric (cotton, linen, etc.)\n"
        "- Sew the button in place\n\n"
        "Would you like more specific instructions?"
    )
    text = helpful + _REFUSAL_LOOP * 7
    segments = parse_message_content("assistant", text)
    assert [s.kind for s in segments] == ["reasoning", "prose", "refusal"]
    assert segments[0].content.startswith("Think:")
    assert "Choose a fabric" in segments[1].content
    assert segments[2].label == "Skipped refusals (7)"
    assert segments[2].content == _REFUSAL_LOOP
    assert _REFUSAL_LOOP not in segments[1].content


def test_math_bench_keeps_answer_folds_sandbox_refusals():
    helpful = (
        "Think: call the interpreter, then compute the product.\n\n"
        "First, multiply 13 and 17:\n13 × 17 = 221\n\n"
        "So, the product is **4279**."
    )
    text = helpful + _SANDBOX_REFUSAL * 5
    segments = parse_message_content("assistant", text)
    assert [s.kind for s in segments] == ["reasoning", "prose", "refusal"]
    assert "**4279**" in segments[1].content
    assert segments[2].label == "Skipped refusals (5)"


def test_boxed_answer_not_classified_as_refusal():
    text = (
        "I'm sorry for the confusion earlier. The result is \\boxed{42}."
    )
    segments = parse_message_content("assistant", text)
    assert len(segments) == 1
    assert segments[0].kind == "answer"
    assert segments[0].boxed == "42"


def test_tool_loop_tail_refusals_folded():
    text = (
        "Think: sum values.\n\n"
        "<tool_call>\n"
        '{"name": "code_interpreter", "arguments": {"code": "print(1)"}}\n'
        "</tool_call>\n\n"
        "<interpreter>\n1\n</interpreter>\n\n"
        "Answer: \\boxed{1}"
        + _SANDBOX_REFUSAL * 3
    )
    segments = parse_message_content("assistant", text)
    assert segments[-1].kind == "refusal"
    assert segments[-2].kind == "answer"
    assert segments[-2].boxed == "1"
