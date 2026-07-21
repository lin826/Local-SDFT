from sdft.config import Config, OnlineConfig, load_config


def test_online_defaults():
    cfg = Config()
    assert cfg.online.backend == "torch"
    assert cfg.online.min_new_demos == 4
    assert cfg.online.lr == 1e-4
    # the offline pipeline defaults are untouched
    assert cfg.model.name == "LiquidAI/LFM2.5-230M"


def test_online_yaml_overlay(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "model:\n  name: LiquidAI/LFM2-1.2B\n"
        "online:\n  backend: echo\n  min_new_demos: 2\n  reward_fn: house_style\n"
    )
    cfg = load_config(p)
    assert cfg.model.name == "LiquidAI/LFM2-1.2B"
    assert cfg.online.backend == "echo"
    assert cfg.online.min_new_demos == 2
    assert cfg.online.reward_fn == "house_style"
    # untouched defaults survive
    assert cfg.online.replay_ratio == 0.5


def test_unknown_online_key_rejected(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("online:\n  nonsense: 1\n")
    import pytest

    with pytest.raises(ValueError):
        load_config(p)
