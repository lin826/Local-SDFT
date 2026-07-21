"""Demonstration-conditioned teacher prompt construction (the SDFT trick).

The teacher is the *same model* as the student; it only sees a richer context:
the conversation, the golden response presented as its own prior reply, and a
re-instruction turn asking it to answer again. Conditioning on the golden
response lets in-context learning produce an improved target distribution —
no separate teacher model or reward function required.
"""

from __future__ import annotations

DEFAULT_REINSTRUCT = (
    "Now answer with a response of your own, including the thinking process."
)


def build_student_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """The student sees the plain conversation context."""
    return [dict(m) for m in messages]


def build_teacher_messages(
    messages: list[dict[str, str]],
    demonstration: str,
    reinstruct: str = DEFAULT_REINSTRUCT,
) -> list[dict[str, str]]:
    """The teacher sees the context plus the golden response, then re-answers."""
    return [
        *[dict(m) for m in messages],
        {"role": "assistant", "content": demonstration},
        {"role": "user", "content": reinstruct},
    ]
