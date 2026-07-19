"""Incremental LoRA online learning loop for the /data web UI."""

from .feedback import build_train_examples
from .loop import run_online_turn
from .paths import (
    online_session_dir,
    online_session_path,
    session_adapter_dir,
)
from .schema import OnlineLearningSession, OnlineTurn, TurnLatency
from .session import (
    adapter_ready,
    create_session,
    list_sessions,
    list_sessions_with_adapter,
    load_session,
    save_session,
)
from .stats import aggregate_tone_counts, aggregate_turn_latencies, build_design_summary
from .tone import classify_tone, resolve_tone

__all__ = [
    "OnlineLearningSession",
    "OnlineTurn",
    "TurnLatency",
    "aggregate_tone_counts",
    "aggregate_turn_latencies",
    "build_design_summary",
    "build_train_examples",
    "classify_tone",
    "adapter_ready",
    "create_session",
    "list_sessions",
    "list_sessions_with_adapter",
    "load_session",
    "online_session_dir",
    "online_session_path",
    "resolve_tone",
    "run_online_turn",
    "save_session",
    "session_adapter_dir",
]
