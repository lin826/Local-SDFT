"""One online-learning turn: collect, preview, train, persist."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sdft.config import load_config
from sdft.records.benchmark import LatencyPhases
from sdft.records.collect import collect_record
from sdft.records.paths import utc_now_iso

from .inference import generate_preview
from .paths import session_adapter_dir
from .schema import OnlineTurn
from .session import load_session, save_session
from .stats import turn_latency_from_phases
from .train_step import run_train_step


def _replay_examples(session, new_row: dict[str, str], buffer_size: int) -> list[dict[str, str]]:
    prior = [
        {"instruction": t.instruction, "input": t.input, "output": t.output}
        for t in session.turns
    ]
    combined = [*prior, new_row]
    if buffer_size <= 0:
        return [new_row]
    return combined[-buffer_size:]


def run_online_turn(
    session_id: str,
    *,
    instruction: str,
    input: str = "",
    output: str = "",
    tags: list[str] | None = None,
    preview: bool = True,
    root: Path | None = None,
) -> OnlineTurn:
    """Append one example, optionally preview, run a tiny LoRA update, persist session."""
    session = load_session(session_id, root=root)
    cfg = load_config(session.config_path)
    adapter_dir = session_adapter_dir(session_id, root)
    phases = LatencyPhases()
    turn_index = session.turn_count + 1
    tag_list = list(tags or [])
    tag_list.append(f"online:{session_id}")

    row = {
        "instruction": instruction.strip(),
        "input": input.strip(),
        "output": output.strip(),
    }

    input_tokens: int | None = None
    output_tokens: int | None = None
    preview_text = ""

    ol = cfg.online_learning
    if preview and ol.preview_before_train:
        with phases.span("inference_preview"):
            preview_text, input_tokens, output_tokens = generate_preview(
                cfg,
                adapter_dir,
                row["instruction"],
                row["input"],
                max_new_tokens=ol.preview_max_new_tokens,
            )

    replay = _replay_examples(session, row, ol.replay_buffer_size)
    with phases.span("train_update"):
        run_train_step(cfg, adapter_dir, replay)

    latency = turn_latency_from_phases(
        phases.to_list(),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    with phases.span("record_collect"):
        record = collect_record(
            row["instruction"],
            input=row["input"],
            output=row["output"],
            source="web",
            tags=tag_list,
            metadata={
                "online_session_id": session_id,
                "turn_index": turn_index,
                "latency": latency.to_dict(),
            },
        )

    turn = OnlineTurn(
        turn_index=turn_index,
        instruction=row["instruction"],
        input=row["input"],
        output=row["output"],
        preview=preview_text,
        record_id=record.id,
        created_at=utc_now_iso(),
        latency=latency,
        latency_phases=phases.to_list(),
        tags=tag_list,
    )
    session.turns.append(turn)
    session.updated_at = turn.created_at
    save_session(session, root=root)
    return turn


def turn_result_context(turn: OnlineTurn, session_id: str) -> dict[str, Any]:
    """Template-friendly dict for one completed turn."""
    return {
        "session_id": session_id,
        "turn": turn,
        "latency": turn.latency.to_dict(),
        "latency_phases": turn.latency_phases,
    }
