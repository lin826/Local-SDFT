"""Unit tests for local GRPO reward helpers (no model download)."""

from __future__ import annotations

from sdft.rewards import boxed_reward, instruction_reward, resolve_reward


def test_instruction_reward_prefers_helpful_non_refusal():
    gold = ["Mix flour and water, then bake at 350F for 30 minutes."]
    good = instruction_reward(
        ["Mix flour with water and bake for about half an hour at 350 degrees."],
        gold=gold,
    )
    bad = instruction_reward(
        ["I'm sorry, but I can't assist with that."],
        gold=gold,
    )
    empty = instruction_reward([""], gold=gold)
    assert good[0] > bad[0]
    assert empty[0] < 0


def test_resolve_reward_names():
    assert resolve_reward("instruction") is instruction_reward
    assert resolve_reward("boxed") is boxed_reward


def test_boxed_reward_uses_strict_box():
    scores = boxed_reward(
        [r"The answer is \boxed{42}", "no box here"],
        ground_truth=["42", "42"],
    )
    assert scores[0] == 1.0
    assert scores[1] == -1.0
