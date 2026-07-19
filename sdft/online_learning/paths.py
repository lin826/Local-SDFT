"""Filesystem layout for online-learning sessions."""

from __future__ import annotations

from pathlib import Path

from sdft.records.paths import new_run_id, project_root


def online_sessions_root(root: Path | None = None) -> Path:
    return project_root(root) / "outputs" / "online-learning"


def online_session_dir(session_id: str, root: Path | None = None) -> Path:
    return online_sessions_root(root) / session_id


def online_session_path(session_id: str, root: Path | None = None) -> Path:
    return online_session_dir(session_id, root) / "session.json"


def session_adapter_dir(session_id: str, root: Path | None = None) -> Path:
    return online_session_dir(session_id, root) / "adapter"


def new_session_id() -> str:
    return new_run_id("ol")
