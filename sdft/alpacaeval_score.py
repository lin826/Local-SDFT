"""Official AlpacaEval 2 scoring via ``alpaca_eval.evaluate``.

Builds ``instruction`` / ``output`` / ``generator`` annotations and calls the
library's pairwise GPT-4 judge (default: ``weighted_alpaca_eval_gpt4_turbo``
vs ``gpt4_turbo`` reference) to obtain ``win_rate`` and
``length_controlled_winrate``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Sequence

# AE2 default annotator (AlpacaEval 2.0). Override via evaluate_model_outputs(...).
DEFAULT_ANNOTATORS_CONFIG = "weighted_alpaca_eval_gpt4_turbo"


def require_openai_api_key() -> str:
    """Return ``OPENAI_API_KEY`` or raise with a clear setup message."""
    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not key:
        raise EnvironmentError(
            "OPENAI_API_KEY is required for official AlpacaEval judging "
            f"(annotator `{DEFAULT_ANNOTATORS_CONFIG}`). "
            "Set it in the environment (or Colab Secrets) before scoring. "
            "Judging uses paid OpenAI API calls — expect real $ cost on full AE2."
        )
    return key


def to_model_outputs(
    instructions: Sequence[str],
    outputs: Sequence[str],
    *,
    generator: str,
) -> list[dict[str, str]]:
    """Build AlpacaEval annotation rows (``instruction``, ``output``, ``generator``)."""
    if len(instructions) != len(outputs):
        raise ValueError(
            f"instructions/outputs length mismatch: "
            f"{len(instructions)} vs {len(outputs)}"
        )
    gen = (generator or "").strip() or "unnamed"
    rows: list[dict[str, str]] = []
    for instruction, output in zip(instructions, outputs):
        rows.append(
            {
                "instruction": str(instruction),
                "output": str(output),
                "generator": gen,
            }
        )
    return rows


def write_model_outputs(path: str | Path, rows: Sequence[dict[str, str]]) -> Path:
    """Write model outputs JSON for ``alpaca_eval`` / CLI reuse."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(list(rows), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return out


def extract_model_metrics(
    leaderboard: Any,
    name: str,
) -> dict[str, Any]:
    """Pull win-rate fields for ``name`` from an ``evaluate`` leaderboard frame."""
    if leaderboard is None:
        raise ValueError("leaderboard is None")

    # DataFrame (orient=index → model names as index) or dict-of-dicts.
    if hasattr(leaderboard, "loc"):
        if name not in leaderboard.index:
            raise KeyError(
                f"model {name!r} missing from leaderboard "
                f"(have {list(leaderboard.index)!r})"
            )
        row = leaderboard.loc[name]
        data = {str(k): _jsonable(v) for k, v in row.items()}
    elif isinstance(leaderboard, dict):
        if name not in leaderboard:
            raise KeyError(
                f"model {name!r} missing from leaderboard "
                f"(have {list(leaderboard)!r})"
            )
        data = {str(k): _jsonable(v) for k, v in leaderboard[name].items()}
    else:
        raise TypeError(f"unexpected leaderboard type: {type(leaderboard)!r}")

    return {
        "name": name,
        "win_rate": data.get("win_rate"),
        "standard_error": data.get("standard_error"),
        "length_controlled_winrate": data.get("length_controlled_winrate"),
        "n_total": data.get("n_total"),
        "avg_length": data.get("avg_length"),
        "raw": data,
    }


def evaluate_model_outputs(
    model_outputs: Sequence[dict[str, str]] | str | Path,
    *,
    name: str | None = None,
    output_dir: str | Path | None = None,
    max_instances: int | None = None,
    annotators_config: str = DEFAULT_ANNOTATORS_CONFIG,
    reference_outputs: Any = None,
    is_overwrite_leaderboard: bool = True,
) -> dict[str, Any]:
    """Run official ``alpaca_eval.evaluate`` and return a JSON-serializable summary.

    Requires ``OPENAI_API_KEY`` and the optional ``alpacaeval`` extra
    (``pip install alpaca-eval`` / ``uv sync --extra alpacaeval``).
    """
    require_openai_api_key()
    try:
        from alpaca_eval import evaluate as alpaca_evaluate
    except ImportError as err:
        raise ImportError(
            "alpaca_eval is not installed. Install with: "
            'uv sync --extra alpacaeval   # or: pip install "alpaca-eval"'
        ) from err

    if isinstance(model_outputs, (str, Path)):
        path = Path(model_outputs)
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"expected a JSON list in {path}")
        model_name = name or _generator_from_rows(rows) or path.stem
        outputs_arg: Any = str(path)
    else:
        rows = list(model_outputs)
        model_name = name or _generator_from_rows(rows) or "Current model"
        outputs_arg = rows

    out_dir = Path(output_dir) if output_dir is not None else Path("outputs/alpacaeval") / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Persist inputs for reproducibility even if the library writes elsewhere.
    write_model_outputs(out_dir / "model_outputs.json", rows)

    eval_kwargs: dict[str, Any] = {
        "model_outputs": outputs_arg,
        "name": model_name,
        "output_path": str(out_dir),
        "annotators_config": annotators_config,
        "is_return_instead_of_print": True,
        "is_overwrite_leaderboard": is_overwrite_leaderboard,
        "max_instances": max_instances,
        # Avoid rewriting the package's precomputed leaderboard CSV.
        "precomputed_leaderboard": None,
        "is_cache_leaderboard": False,
    }
    if reference_outputs is not None:
        eval_kwargs["reference_outputs"] = reference_outputs

    leaderboard, annotations = alpaca_evaluate(**eval_kwargs)
    metrics = extract_model_metrics(leaderboard, model_name)

    summary = {
        "annotators_config": annotators_config,
        "max_instances": max_instances,
        "n_model_outputs": len(rows),
        "output_dir": str(out_dir),
        "metrics": metrics,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if annotations is not None and not (out_dir / "annotations.json").exists():
        # Library usually writes this; keep a fallback dump.
        try:
            import pandas as pd

            if isinstance(annotations, pd.DataFrame):
                annotations.to_json(
                    out_dir / "annotations.json", orient="records", indent=2
                )
            else:
                (out_dir / "annotations.json").write_text(
                    json.dumps(annotations, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
        except Exception:
            pass

    return summary


def _generator_from_rows(rows: Sequence[dict[str, Any]]) -> str | None:
    for row in rows:
        gen = row.get("generator")
        if gen:
            return str(gen)
    return None


def _jsonable(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
