"""SQLite-backed persistence for interactions, demonstrations, and training runs."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from .events import AdapterVersion, Correction, Demonstration, Message, TrainingRun

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    ts REAL NOT NULL,
    reply_to TEXT
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, ts);

CREATE TABLE IF NOT EXISTS corrections (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    original TEXT NOT NULL,
    corrected TEXT NOT NULL,
    ts REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS demonstrations (
    id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    source TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    topic TEXT NOT NULL,
    messages_json TEXT NOT NULL,
    demonstration TEXT NOT NULL,
    weight REAL NOT NULL,
    times_trained INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS training_runs (
    id TEXT PRIMARY KEY,
    started_at REAL NOT NULL,
    finished_at REAL,
    steps INTEGER NOT NULL,
    demo_ids_json TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    adapter_version INTEGER
);

CREATE TABLE IF NOT EXISTS adapter_versions (
    version INTEGER PRIMARY KEY,
    created_at REAL NOT NULL,
    path TEXT NOT NULL,
    training_run_id TEXT,
    note TEXT NOT NULL DEFAULT '',
    active INTEGER NOT NULL DEFAULT 0
);
"""


class SQLiteStore:
    """Thread-safe-enough store: one connection guarded by a lock.

    The controller serializes serving and training, so contention is minimal;
    the lock only protects against concurrent writers from the web/CLI layer.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock, self._conn:
            self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ---- messages -------------------------------------------------------

    def add_message(self, msg: Message) -> Message:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO messages (id, conversation_id, role, content, ts, reply_to)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (msg.id, msg.conversation_id, msg.role, msg.content, msg.ts, msg.reply_to),
            )
        return msg

    def get_message(self, message_id: str) -> Message | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM messages WHERE id = ?", (message_id,)
            ).fetchone()
        return self._msg_from_row(row) if row else None

    def conversation_messages(self, conversation_id: str) -> list[Message]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY ts, rowid",
                (conversation_id,),
            ).fetchall()
        return [self._msg_from_row(r) for r in rows]

    def list_conversations(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT conversation_id FROM messages ORDER BY conversation_id"
            ).fetchall()
        return [r[0] for r in rows]

    @staticmethod
    def _msg_from_row(row: sqlite3.Row) -> Message:
        return Message(
            id=row["id"],
            conversation_id=row["conversation_id"],
            role=row["role"],
            content=row["content"],
            ts=row["ts"],
            reply_to=row["reply_to"],
        )

    # ---- corrections ----------------------------------------------------

    def add_correction(self, corr: Correction) -> Correction:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO corrections (id, conversation_id, message_id, original,"
                " corrected, ts) VALUES (?, ?, ?, ?, ?, ?)",
                (corr.id, corr.conversation_id, corr.message_id,
                 corr.original, corr.corrected, corr.ts),
            )
        return corr

    def corrected_message_ids(self, conversation_id: str) -> set[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT message_id FROM corrections WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchall()
        return {r[0] for r in rows}

    # ---- demonstrations -------------------------------------------------

    def add_demonstration(self, demo: Demonstration) -> Demonstration:
        row = demo.to_row()
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO demonstrations (id, created_at, source, conversation_id,"
                " topic, messages_json, demonstration, weight, times_trained)"
                " VALUES (:id, :created_at, :source, :conversation_id, :topic,"
                " :messages_json, :demonstration, :weight, :times_trained)",
                row,
            )
        return demo

    def _fetch_demos(self, where: str = "", params: tuple = ()) -> list[Demonstration]:
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM demonstrations {where} ORDER BY created_at", params
            ).fetchall()
        return [Demonstration.from_row(dict(r)) for r in rows]

    def all_demonstrations(self) -> list[Demonstration]:
        return self._fetch_demos()

    def untrained_demonstrations(self) -> list[Demonstration]:
        return self._fetch_demos("WHERE times_trained = 0")

    def trained_demonstrations(self) -> list[Demonstration]:
        return self._fetch_demos("WHERE times_trained > 0")

    def mark_trained(self, demo_ids: list[str]) -> None:
        with self._lock, self._conn:
            self._conn.executemany(
                "UPDATE demonstrations SET times_trained = times_trained + 1 WHERE id = ?",
                [(d,) for d in demo_ids],
            )

    # ---- training runs & adapter versions --------------------------------

    def record_training_run(self, run: TrainingRun) -> TrainingRun:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO training_runs (id, started_at, finished_at, steps,"
                " demo_ids_json, metrics_json, adapter_version)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (run.id, run.started_at, run.finished_at, run.steps,
                 json.dumps(run.demo_ids), json.dumps(run.metrics), run.adapter_version),
            )
        return run

    def recent_training_runs(self, limit: int = 20) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM training_runs ORDER BY started_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def count_training_runs(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM training_runs").fetchone()
        return int(row[0])

    def add_adapter_version(self, av: AdapterVersion) -> AdapterVersion:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO adapter_versions (version, created_at, path,"
                " training_run_id, note, active) VALUES (?, ?, ?, ?, ?, ?)",
                (av.version, av.created_at, av.path, av.training_run_id,
                 av.note, int(av.active)),
            )
        return av

    def set_active_adapter(self, version: int) -> None:
        with self._lock, self._conn:
            self._conn.execute("UPDATE adapter_versions SET active = 0")
            self._conn.execute(
                "UPDATE adapter_versions SET active = 1 WHERE version = ?", (version,)
            )

    def get_active_adapter(self) -> AdapterVersion | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM adapter_versions WHERE active = 1"
            ).fetchone()
        return self._av_from_row(row) if row else None

    def list_adapter_versions(self) -> list[AdapterVersion]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM adapter_versions ORDER BY version"
            ).fetchall()
        return [self._av_from_row(r) for r in rows]

    @staticmethod
    def _av_from_row(row: sqlite3.Row) -> AdapterVersion:
        return AdapterVersion(
            version=row["version"],
            created_at=row["created_at"],
            path=row["path"],
            training_run_id=row["training_run_id"],
            note=row["note"],
            active=bool(row["active"]),
        )
