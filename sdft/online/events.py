"""Event and data schemas for the online learning loop."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant"]
DemoSource = Literal["correction", "accepted", "topic_content"]


def new_id() -> str:
    return uuid.uuid4().hex


def now() -> float:
    return time.time()


@dataclass
class Message:
    conversation_id: str
    role: Role
    content: str
    id: str = field(default_factory=new_id)
    ts: float = field(default_factory=now)
    reply_to: str | None = None  # id of the message this responds to

    def to_chat(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class Correction:
    """A user edit/rewrite of an assistant reply — the highest-value SDFT signal."""

    conversation_id: str
    message_id: str  # the assistant message being corrected
    original: str
    corrected: str
    id: str = field(default_factory=new_id)
    ts: float = field(default_factory=now)


@dataclass
class Demonstration:
    """A unit of SDFT training data: conversation context + golden response text.

    The teacher model is conditioned on ``demonstration`` (see sdft/teacher.py);
    the student sees only ``messages`` and must match the teacher on its own
    on-policy continuation.
    """

    source: DemoSource
    conversation_id: str
    messages: list[dict[str, str]]  # chat-formatted context, normally ending in a user turn
    demonstration: str  # golden response text used to condition the teacher
    topic: str = "general"
    weight: float = 1.0
    id: str = field(default_factory=new_id)
    created_at: float = field(default_factory=now)
    times_trained: int = 0

    def to_row(self) -> dict[str, Any]:
        import json

        return {
            "id": self.id,
            "created_at": self.created_at,
            "source": self.source,
            "conversation_id": self.conversation_id,
            "topic": self.topic,
            "messages_json": json.dumps(self.messages),
            "demonstration": self.demonstration,
            "weight": self.weight,
            "times_trained": self.times_trained,
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "Demonstration":
        import json

        return cls(
            id=row["id"],
            created_at=row["created_at"],
            source=row["source"],
            conversation_id=row["conversation_id"],
            topic=row["topic"],
            messages=json.loads(row["messages_json"]),
            demonstration=row["demonstration"],
            weight=row["weight"],
            times_trained=row["times_trained"],
        )


@dataclass
class TrainingRun:
    steps: int
    demo_ids: list[str]
    metrics: dict[str, float]
    adapter_version: int | None = None
    id: str = field(default_factory=new_id)
    started_at: float = field(default_factory=now)
    finished_at: float | None = None


@dataclass
class AdapterVersion:
    version: int
    path: str
    training_run_id: str | None = None
    note: str = ""
    created_at: float = field(default_factory=now)
    active: bool = False
