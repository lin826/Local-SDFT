"""Preference-aware training example selection for chat-style online learning."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sdft.config import Config

if TYPE_CHECKING:
    from .schema import OnlineTurn


def _turn_train_row(turn: OnlineTurn, *, sdft_response: str | None = None) -> dict[str, str]:
    return {
        "instruction": turn.instruction,
        "input": turn.input,
        "sdft_response": sdft_response or turn.sdft_response,
    }


def build_train_examples(
    cfg: Config,
    *,
    prior_turns: list[OnlineTurn],
    instruction: str,
    user_input: str,
    sdft_response: str,
    feedback_tone: str | None,
    feedback_reward: int | None,
    prev_rewrite: str | None = None,
) -> tuple[list[dict[str, str]], str, list[dict[str, Any]]]:
    """Build replay rows for this update step and describe the preference action."""
    current_row = {
        "instruction": instruction,
        "input": user_input,
        "sdft_response": sdft_response,
    }
    preference_action = "first_turn"
    trained_on: list[dict[str, Any]] = [
        {"role": "current_sdft", "instruction": instruction, "input": user_input},
    ]

    if not prior_turns:
        buffer_size = cfg.online_learning.replay_buffer_size
        rows = [current_row] if buffer_size <= 0 else [current_row]
        return rows, preference_action, trained_on

    prev = prior_turns[-1]
    prior_rows = [
        _turn_train_row(t)
        for t in prior_turns
        if t.sdft_response.strip()
    ]
    prior_rows = prior_rows[:-1] if prior_rows else []

    tone = feedback_tone or "neutral"
    reward = feedback_reward if feedback_reward is not None else 0

    if tone == "positive" and reward > 0:
        preferred = (prev.assistant_reply or prev.preview or prev.sdft_response).strip()
        if preferred:
            prev_row = {
                "instruction": prev.instruction,
                "input": prev.input,
                "sdft_response": preferred,
            }
            prior_rows.append(prev_row)
            prior_rows.append(dict(prev_row))
            preference_action = "reinforce_prev"
            trained_on.append(
                {
                    "role": "reinforce_prev",
                    "instruction": prev.instruction,
                    "input": prev.input,
                    "target": preferred,
                }
            )
        else:
            preference_action = "reinforce_prev_skipped"
    elif tone == "negative" and reward < 0:
        rewrite = (prev_rewrite or "").strip()
        if rewrite:
            prev_row = {
                "instruction": prev.instruction,
                "input": prev.input,
                "sdft_response": rewrite,
            }
            prior_rows.append(prev_row)
            preference_action = "rewrite_prev"
            trained_on.append(
                {
                    "role": "rewrite_prev",
                    "instruction": prev.instruction,
                    "input": prev.input,
                    "target": rewrite,
                }
            )
        else:
            preference_action = "rewrite_prev_skipped"
    else:
        preference_action = "neutral_skip_prev"
        trained_on.append({"role": "neutral_skip_prev"})

    combined = [*prior_rows, current_row]
    buffer_size = cfg.online_learning.replay_buffer_size
    if buffer_size <= 0:
        return [current_row], preference_action, trained_on
    return combined[-buffer_size:], preference_action, trained_on
