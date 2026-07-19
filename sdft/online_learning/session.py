"""Load/save online-learning session JSON."""

from __future__ import annotations

from pathlib import Path

from sdft.config import load_config
from sdft.records.paths import utc_now_iso
from sdft.records.store import read_json, write_json

from .paths import (
    new_session_id,
    online_session_dir,
    online_session_path,
    session_adapter_dir,
)
from .schema import OnlineLearningSession
from .stats import aggregate_turn_latencies, build_design_summary


def build_session(
    config_path: str = "configs/online_learning.yaml",
    *,
    session_id: str | None = None,
    root: Path | None = None,
) -> OnlineLearningSession:
    """Build an in-memory session; nothing is written until ``save_session``."""
    cfg = load_config(config_path)
    sid = session_id or new_session_id()
    adapter = str(session_adapter_dir(sid, root))
    now = utc_now_iso()
    return OnlineLearningSession(
        id=sid,
        created_at=now,
        updated_at=now,
        config_path=config_path,
        adapter_dir=adapter,
        model=cfg.model.name,
        design_summary=build_design_summary(
            session_id=sid,
            config_path=config_path,
            model=cfg.model.name,
            turn_count=0,
            adapter_dir=adapter,
        ),
    )


def create_session(
    config_path: str = "configs/online_learning.yaml",
    *,
    session_id: str | None = None,
    root: Path | None = None,
) -> OnlineLearningSession:
    session = build_session(config_path, session_id=session_id, root=root)
    online_session_dir(session.id, root).mkdir(parents=True, exist_ok=True)
    save_session(session, root=root)
    return session


def session_persisted(session_id: str, *, root: Path | None = None) -> bool:
    return online_session_path(session_id, root).is_file()


def resolve_session(
    session_id: str,
    *,
    config_path: str = "configs/online_learning.yaml",
    root: Path | None = None,
) -> OnlineLearningSession:
    """Load a saved session or return an ephemeral in-memory session."""
    if session_persisted(session_id, root=root):
        return load_session(session_id, root=root)
    return build_session(config_path, session_id=session_id, root=root)


def load_session(session_id: str, *, root: Path | None = None) -> OnlineLearningSession:
    path = online_session_path(session_id, root)
    if not path.is_file():
        raise FileNotFoundError(f"online session {session_id!r} not found")
    return OnlineLearningSession.from_dict(read_json(path))


def save_session(session: OnlineLearningSession, *, root: Path | None = None) -> Path:
    session.latency_summary = aggregate_turn_latencies(session.turns)
    session.design_summary = build_design_summary(
        session_id=session.id,
        config_path=session.config_path,
        model=session.model,
        turn_count=session.turn_count,
        adapter_dir=session.adapter_dir,
    )
    path = online_session_path(session.id, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, session.to_dict())
    return path


def adapter_ready(adapter_dir: Path | str) -> bool:
    """True when a PEFT LoRA adapter has been saved under ``adapter_dir``."""
    path = Path(adapter_dir)
    return path.is_dir() and (path / "adapter_config.json").is_file()


def list_sessions_with_adapter(
    *, root: Path | None = None, limit: int = 10
) -> list[OnlineLearningSession]:
    """Online sessions with a saved adapter, newest first (cap ``limit``)."""
    ready: list[OnlineLearningSession] = []
    for session in list_sessions(root=root, limit=max(limit * 5, 50)):
        if adapter_ready(session.adapter_dir):
            ready.append(session)
        if len(ready) >= limit:
            break
    return ready


def list_sessions(*, root: Path | None = None, limit: int = 20) -> list[OnlineLearningSession]:
    from .paths import online_sessions_root

    base = online_sessions_root(root)
    if not base.is_dir():
        return []
    sessions: list[OnlineLearningSession] = []
    for child in sorted(base.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not child.is_dir():
            continue
        path = child / "session.json"
        if not path.is_file():
            continue
        try:
            sessions.append(OnlineLearningSession.from_dict(read_json(path)))
        except (KeyError, TypeError, ValueError):
            continue
        if len(sessions) >= limit:
            break
    return sessions
