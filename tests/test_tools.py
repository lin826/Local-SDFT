import pytest

from sdft.online.tools import (
    extract_arithmetic,
    parse_calc_call,
    run_calc_call,
    safe_eval,
)
from sdft.online.reward import get_reward_fn, get_shaper


class TestSafeEval:
    @pytest.mark.parametrize("expr,val", [
        ("3 + 4", 7), ("128 * 47", 6016), ("913 - 476", 437),
        ("10 / 4", 2.5), ("2 ** 5", 32), ("7 % 3", 1), ("-5 + 2", -3),
        ("6 x 7", 42), ("6 × 7", 42),
    ])
    def test_valid(self, expr, val):
        assert safe_eval(expr) == val

    @pytest.mark.parametrize("expr", ["__import__('os')", "a + b", "", None, "print(1)"])
    def test_invalid_returns_none(self, expr):
        assert safe_eval(expr) is None


class TestParse:
    def test_parse_calc_call_variants(self):
        assert parse_calc_call('<tool>calc("3 + 4")</tool>') == "3 + 4"
        assert parse_calc_call("calc(128*47)") == "128*47"
        assert parse_calc_call("calc('9 - 2')") == "9 - 2"
        assert parse_calc_call("no tool here") is None

    def test_extract_arithmetic_from_prompt(self):
        assert extract_arithmetic("What is 347 + 288?") == "347 + 288"
        assert extract_arithmetic("no math") is None

    def test_run_calc_call(self):
        assert run_calc_call('<tool>calc("347 + 288")</tool>') == 635
        assert run_calc_call("I think it is 600") is None


class TestCalcReward:
    def setup_method(self):
        self.rfn = get_reward_fn("calc_tool")

    def test_correct_tool_call_full_reward(self):
        assert self.rfn("What is 347 + 288?", '<tool>calc("347 + 288")</tool>') == 1.0

    def test_tool_call_wrong_value_partial(self):
        assert self.rfn("What is 347 + 288?", '<tool>calc("347 + 289")</tool>') == 0.4

    def test_no_tool_call_zero_even_if_right(self):
        # rewards tool USE, not freehand arithmetic
        assert self.rfn("What is 347 + 288?", "The answer is 635.") == 0.0

    def test_shaper_yields_correct_call(self):
        shp = get_shaper("calc_tool")
        shaped = shp("What is 913 - 476?", "who knows")
        assert self.rfn("What is 913 - 476?", shaped) == 1.0
        assert run_calc_call(shaped) == 437


class TestDemoPromptSets:
    def test_calc_heldout_numbers_disjoint_from_coach(self):
        import re
        from sdft.online.demo import prompts_for
        coach, held = prompts_for("calc_tool")
        coach_nums = {n for p in coach for n in re.findall(r"\d+", p)}
        held_nums = {n for p in held for n in re.findall(r"\d+", p)}
        # memorization-impossibility premise of the demo
        assert coach_nums.isdisjoint(held_nums)
