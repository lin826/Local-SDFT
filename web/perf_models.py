"""Perf chat model / adapter selection for ``/perf``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sdft.config import load_config
from sdft.online_learning.session import (
    adapter_ready,
    list_sessions_with_adapter,
    load_session,
)

from web.demo_conditions import DEFAULT_CONFIG, SDFT_ALPACA_CONFIG

ONLINE_CONFIG_PREFIX = "online:"
PERF_BASE_CONFIGS: tuple[str, ...] = (DEFAULT_CONFIG, SDFT_ALPACA_CONFIG)
ONLINE_PERF_LIMIT = 10


@dataclass(frozen=True)
class PerfModelOption:
    """One entry in the /perf Config dropdown."""

    value: str
    label: str
    kind: str  # "base" | "online"


@dataclass(frozen=True)
class PerfModelSelection:
    """Resolved model load plan for one /perf chat turn."""

    config_path: str
    yaml_config_path: str
    base_model: str | None
    adapter_dir: Path | None = None
    online_session_id: str | None = None
    label: str = ""

    @property
    def model_path(self) -> str:
        if self.adapter_dir is not None:
            return str(self.adapter_dir)
        cfg = load_config(self.yaml_config_path)
        return cfg.model.name

    @property
    def ablation_config_path(self) -> str:
        """Config path used for AlpacaEval prompt-strategy ablation behavior."""
        if self.online_session_id:
            return DEFAULT_CONFIG
        return self.config_path


def online_config_value(session_id: str) -> str:
    return f"{ONLINE_CONFIG_PREFIX}{session_id}"


def is_online_config_path(config_path: str) -> bool:
    return config_path.startswith(ONLINE_CONFIG_PREFIX)


def online_session_id_from_config(config_path: str) -> str | None:
    if not is_online_config_path(config_path):
        return None
    session_id = config_path[len(ONLINE_CONFIG_PREFIX) :].strip()
    return session_id or None


def perf_infer_url_for_session(session_id: str) -> str:
    return f"/perf?config_path={online_config_value(session_id)}"


def list_online_perf_options(*, limit: int = ONLINE_PERF_LIMIT) -> list[PerfModelOption]:
    options: list[PerfModelOption] = []
    for session in list_sessions_with_adapter(limit=limit):
        options.append(
            PerfModelOption(
                value=online_config_value(session.id),
                label=f"Online: {session.id}",
                kind="online",
            )
        )
    return options


def perf_model_options(*, online_limit: int = ONLINE_PERF_LIMIT) -> list[PerfModelOption]:
    options = [
        PerfModelOption(value=path, label=path, kind="base") for path in PERF_BASE_CONFIGS
    ]
    options.extend(list_online_perf_options(limit=online_limit))
    return options


def known_perf_config_values(*, online_limit: int = ONLINE_PERF_LIMIT) -> frozenset[str]:
    return frozenset(opt.value for opt in perf_model_options(online_limit=online_limit))


def resolve_perf_config(config_path: str) -> PerfModelSelection:
    """Resolve a /perf Config value to load settings and optional LoRA adapter."""
    if is_online_config_path(config_path):
        session_id = online_session_id_from_config(config_path)
        if not session_id:
            raise ValueError("invalid online config path")
        session = load_session(session_id)
        adapter = Path(session.adapter_dir)
        if not adapter_ready(adapter):
            raise ValueError(
                f"online session {session_id!r} has no saved adapter at {adapter}"
            )
        return PerfModelSelection(
            config_path=config_path,
            yaml_config_path=session.config_path,
            base_model=session.model,
            adapter_dir=adapter,
            online_session_id=session_id,
            label=f"Online: {session_id}",
        )

    if config_path not in PERF_BASE_CONFIGS:
        raise ValueError(f"unknown perf config {config_path!r}")

    return PerfModelSelection(
        config_path=config_path,
        yaml_config_path=config_path,
        base_model=None,
        label=config_path,
    )


def resolve_perf_config_from_adapter(adapter_path: str) -> str | None:
    """Map an adapter directory path to ``online:{session_id}`` when possible."""
    path = Path(adapter_path).resolve()
    if adapter_ready(path):
        session_id = path.parent.name
        session_path = path.parent / "session.json"
        if session_path.is_file():
            return online_config_value(session_id)
    return None
