"""One online-learning turn: tone feedback, SDFT update, then inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sdft.config import load_config
from sdft.records.benchmark import LatencyPhases
from sdft.records.collect import collect_record
from sdft.records.paths import utc_now_iso

from .feedback import build_train_examples
from .generate_step import generate_sdft_response, turns_to_fewshot_examples
from .inference import generate_preview
from .paths import session_adapter_dir
from .schema import OnlineTurn
from .session import resolve_session, save_session
from .stats import turn_latency_from_phases
from .tone import resolve_tone
from .train_step import run_train_step


def run_online_turn(
    session_id: str,
    *,
    instruction: str,
    input: str = "",
    output: str = "",
    tags: list[str] | None = None,
    preview: bool = True,
    tone_override: str | None = None,
    config_path: str | None = None,
    root: Path | None = None,
) -> OnlineTurn:
    """Classify feedback tone, LoRA-update, then infer the assistant reply."""
    session = resolve_session(
        session_id,
        config_path=config_path or "configs/online_learning.yaml",
        root=root,
    )
    cfg = load_config(session.config_path)
    adapter_dir = session_adapter_dir(session_id, root)
    phases = LatencyPhases()
    turn_index = session.turn_count + 1
    tag_list = list(tags or [])
    tag_list.append(f"online:{session_id}")

    instruction = instruction.strip()
    user_input = input.strip()
    gold_output = output.strip()
    prior_turns = list(session.turns)

    feedback_tone: str | None = None
    feedback_reward: int | None = None
    feedback_source = "none"
    if prior_turns:
        with phases.span("tone_classify"):
            feedback_tone, feedback_reward, feedback_source = resolve_tone(
                instruction,
                override=tone_override,
            )

    fewshots = turns_to_fewshot_examples(prior_turns)
    generate_input_tokens: int | None = None
    generate_output_tokens: int | None = None
    prev_rewrite: str | None = None

    if feedback_tone == "negative" and prior_turns:
        with phases.span("generate_prev_rewrite"):
            prev = prior_turns[-1]
            prev_fewshots = turns_to_fewshot_examples(prior_turns[:-1])
            prev_rewrite, _, _ = generate_sdft_response(
                cfg,
                instruction=prev.instruction,
                user_input=prev.input,
                fewshot_examples=prev_fewshots,
            )

    with phases.span("generate_sdft"):
        sdft_response, generate_input_tokens, generate_output_tokens = generate_sdft_response(
            cfg,
            instruction=instruction,
            user_input=user_input,
            fewshot_examples=fewshots,
        )

    replay, preference_action, trained_on = build_train_examples(
        cfg,
        prior_turns=prior_turns,
        instruction=instruction,
        user_input=user_input,
        sdft_response=sdft_response,
        feedback_tone=feedback_tone,
        feedback_reward=feedback_reward,
        prev_rewrite=prev_rewrite,
    )
    with phases.span("train_update"):
        run_train_step(cfg, adapter_dir, replay)

    assistant_reply = ""
    preview_input_tokens: int | None = None
    preview_output_tokens: int | None = None
    ol = cfg.online_learning
    if preview:
        with phases.span("inference_reply"):
            assistant_reply, preview_input_tokens, preview_output_tokens = generate_preview(
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
                "assistant_reply": assistant_reply,
                "feedback_tone": feedback_tone,
                "feedback_reward": feedback_reward,
                "feedback_source": feedback_source,
                "preference_action": preference_action,
                "trained_on": trained_on,
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
        assistant_reply=assistant_reply,
        preview=assistant_reply,
        record_id=record.id,
        created_at=utc_now_iso(),
        latency=latency,
        latency_phases=phases.to_list(),
        tags=tag_list,
        feedback_tone=feedback_tone,
        feedback_reward=feedback_reward,
        feedback_source=feedback_source,
        preference_action=preference_action,
        trained_on=trained_on,
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
