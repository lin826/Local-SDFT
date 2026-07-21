"""Reward + shaper checks for the lifelong skill-accumulation demo.

The contract that makes catastrophic forgetting *measurable*: each skill's
shaper produces a target that passes its own reward, and each skill's reward
FAILS on another skill's output (so applying the wrong skill is detectable).
"""

from sdft.online.demo import SKILLS
from sdft.online.reward import SIGNOFF, get_reward_fn, get_shaper


def test_every_skill_has_reward_and_shaper():
    for name, rf, _, _, _ in SKILLS:
        assert get_reward_fn(rf) is not None
        assert get_shaper(rf) is not None, f"{name} needs a shaper for reliable SFT targets"


def test_shaper_output_passes_its_own_reward():
    # A messy model reply, reshaped, must earn full marks for that skill.
    messy = "Well, um, here are\nsome thoughts. Maybe consider this, and that too."
    for name, rf, _, _, _ in SKILLS:
        if rf == "calc_tool":
            prompt = "What is 12 + 8?"
        else:
            prompt = {"skill_summary": "Summarize: a long paragraph about budgets and hiring.",
                      "skill_bullets": "List ways to save money.",
                      "skill_signoff": "Reply to: can we meet Thursday?"}[rf]
        target = get_shaper(rf)(prompt, messy)
        assert get_reward_fn(rf)(prompt, target) >= 0.99, f"{name} shaper -> failing target"


def test_skills_are_mutually_exclusive():
    # summary (one line) must fail bullets/signoff; bullets must fail summary; etc.
    summ = "The budget rose and hiring slowed this quarter."
    bullets = "- first\n- second\n- third"
    signed = f"Sure, Thursday works for me.\n\n{SIGNOFF}"
    assert get_reward_fn("skill_summary")("Summarize: x", summ) >= 0.99
    assert get_reward_fn("skill_summary")("Summarize: x", bullets) == 0.0
    assert get_reward_fn("skill_bullets")("List x", bullets) >= 0.99
    assert get_reward_fn("skill_bullets")("List x", summ) == 0.0
    assert get_reward_fn("skill_signoff")("Reply to: x", signed) >= 0.99
    assert get_reward_fn("skill_signoff")("Reply to: x", summ) == 0.0


def test_summary_rejects_tool_call_and_bullets():
    r = get_reward_fn("skill_summary")
    assert r("Summarize: x", '<tool>calc("1+1")</tool>') == 0.0
    assert r("Summarize: x", "- a bulleted line") == 0.0
