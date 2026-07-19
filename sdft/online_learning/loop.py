"""One online-learning turn: SDFT generate, LoRA update, optional preview, persist."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sdft.config import load_config
from sdft.records.benchmark import LatencyPhases
from sdft.records.collect import collect_record
from sdft.records.paths import utc_now_iso

from .generate_step import generate_sdft_response, turns_to_fewshot_examples
from .inference import generate_preview
from .paths import session_adapter_dir
from .schema import OnlineTurn
from .session import load_session, save_session
from .stats import turn_latency_from_phases
from .train_step import run_train_step


def _replay_examples(
    session,
    new_row: dict[str, str],
    buffer_size: int,
) -> list[dict[str, str]]:
    prior = [
        {
            "instruction": t.instruction,
            "input": t.input,
            "sdft_response": t.sdft_response,
        }
        for t in session.turns
        if t.sdft_response.strip()
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
    """SDFT-generate a target, LoRA-update on it, optionally preview, persist session."""
    session = load_session(session_id, root=root)
    cfg = load_config(session.config_path)
    adapter_dir = session_adapter_dir(session_id, root)
    phases = LatencyPhases()
    turn_index = session.turn_count + 1
    tag_list = list(tags or [])
    tag_list.append(f"online:{session_id}")

    instruction = instruction.strip()
    user_input = input.strip()
    gold_output = output.strip()

    fewshots = turns_to_fewshot_examples(session.turns)
    generate_input_tokens: int | None = None
    generate_output_tokens: int | None = None

    with phases.span("generate_sdft"):
        sdft_response, generate_input_tokens, generate_output_tokens = generate_sdft_response(
            cfg,
            instruction=instruction,
            user_input=user_input,
            fewshot_examples=fewshots,
        )

    train_row = {
        "instruction": instruction,
        "input": user_input,
        "sdft_response": sdft_response,
    }
    replay = _replay_examples(session, train_row, cfg.online_learning.replay_buffer_size)
    with phases.span("train_update"):
        run_train_step(cfg, adapter_dir, replay)

    preview_text = ""
    preview_input_tokens: int | None = None
    preview_output_tokens: int | None = None
    ol = cfg.online_learning
    if preview and ol.preview_before_train:
        with phases.span("inference_preview"):
            preview_text, preview_input_tokens, preview_output_tokens = generate_preview(
                cfg,
                adapter_dir,
                instruction,
                user_input,
                max_new_tokens=ol.preview_max_new_tokens,
            )

    latency = turn_latency_from_phases(
        phases.to_list(),
        input_tokens=generate_input_tokens,
        output_tokens=generate_output_tokens,
        preview_input_tokens=preview_input_tokens,
        preview_output_tokens=preview_output_tokens,
    )

    with phases.span("record_collect"):
        record = collect_record(
            instruction,
            input=user_input,
            output=gold_output,
            source="web",
            tags=tag_list,
            metadata={
                "online_session_id": session_id,
                "turn_index": turn_index,
                "sdft_response": sdft_response,
                "gold_output_for_collection_only": bool(gold_output),
                "latency": latency.to_dict(),
            },
        )

    turn = OnlineTurn(
        turn_index=turn_index,
        instruction=instruction,
        input=user_input,
        output=gold_output,
        sdft_response=sdft_response,
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
