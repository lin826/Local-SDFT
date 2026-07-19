"""Turn interaction events into SDFT demonstrations.

Signal rules (v1):
- correction: user edits/rewrites an assistant reply -> golden demonstration.
- accepted: a conversation closes with uncorrected assistant replies -> the
  model's own replies become self-demonstrations (lower weight). "Closed" means
  the user moved on: started a new conversation, jumped subject, or ended the
  session without correcting.
- topic jumps tag the resulting demonstrations so the replay buffer can keep
  domain diversity.
"""

from __future__ import annotations

import re

from ..config import OnlineConfig
from .events import Correction, Demonstration, Message
from .store import SQLiteStore

MIN_REPLY_CHARS = 16  # skip trivially short replies ("OK", "Sure!") as self-demos


def auto_topic(messages: list[Message]) -> str:
    """Deterministic topic tag from the first user message."""
    for m in messages:
        if m.role == "user" and m.content.strip():
            words = re.findall(r"[a-z0-9]+", m.content.lower())[:6]
            return "-".join(words) if words else "general"
    return "general"


class SignalExtractor:
    def __init__(self, store: SQLiteStore, config: OnlineConfig):
        self.store = store
        self.config = config

    def on_correction(self, correction: Correction, topic: str | None = None) -> Demonstration | None:
        """Build a demonstration from a user-corrected assistant reply."""
        msgs = self.store.conversation_messages(correction.conversation_id)
        idx = next((i for i, m in enumerate(msgs) if m.id == correction.message_id), None)
        if idx is None:
            return None
        context = [m.to_chat() for m in msgs[:idx]]
        if not context:
            return None
        demo = Demonstration(
            source="correction",
            conversation_id=correction.conversation_id,
            messages=context,
            demonstration=correction.corrected,
            topic=topic or auto_topic(msgs),
            weight=self.config.correction_weight,
        )
        return self.store.add_demonstration(demo)

    def close_conversation(self, conversation_id: str, topic: str | None = None) -> list[Demonstration]:
        """Harvest self-demonstrations from uncorrected assistant replies."""
        msgs = self.store.conversation_messages(conversation_id)
        if not msgs:
            return []
        corrected = self.store.corrected_message_ids(conversation_id)
        topic = topic or auto_topic(msgs)

        demos: list[Demonstration] = []
        # Walk newest-first so the cap keeps the freshest replies.
        for idx in range(len(msgs) - 1, -1, -1):
            m = msgs[idx]
            if m.role != "assistant" or m.id in corrected:
                continue
            if len(m.content.strip()) < MIN_REPLY_CHARS:
                continue
            context = [mm.to_chat() for mm in msgs[:idx]]
            if not context:
                continue
            demo = Demonstration(
                source="accepted",
                conversation_id=conversation_id,
                messages=context,
                demonstration=m.content,
                topic=topic,
                weight=self.config.accepted_weight,
            )
            demos.append(self.store.add_demonstration(demo))
            if len(demos) >= self.config.max_accepted_per_conversation:
                break
        return demos
