"""YAML-backed configuration for the SDFT pipeline."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    name: str = "LiquidAI/LFM2.5-230M"  # post-trained chat checkpoint
    dtype: str = "float32"  # 230M params: full fp32 is cheap and stable on MPS
    attn_implementation: str | None = None  # None -> transformers default (sdpa)


@dataclass
class DataConfig:
    dataset: str = "yahma/alpaca-cleaned"  # HF dataset id, or "json"/"csv" with data_files
    data_files: str | None = None  # local file path when dataset is "json"/"csv"
    split: str = "train"
    prompt_fields: list[str] = field(default_factory=lambda: ["instruction", "input"])
    response_field: str = "output"
    num_examples: int = 500
    seed: int = 0


@dataclass
class GenerateConfig:
    num_shots: int = 2  # in-context demonstrations sampled from the dataset
    batch_size: int = 8
    max_new_tokens: int = 256
    temperature: float = 0.0  # 0 -> greedy decoding
    top_p: float = 1.0
    min_response_chars: int = 1  # filter degenerate/empty generations
    out_path: str = "data/sdft_data.jsonl"


@dataclass
class LoraConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    # LFM2 hybrid layout: 6 attention blocks (self_attn.{q,k,v,out}_proj),
    # 8 conv blocks (conv.{in_proj,out_proj,conv}), SwiGLU MLPs (feed_forward.{w1,w2,w3}).
    # A regex string matches full dotted paths (PEFT re.fullmatch); a list matches
    # leaf-name suffixes. Note the leaf "out_proj" exists in BOTH block types.
    target_modules: list[str] | str = field(
        default_factory=lambda: r".*self_attn\.(q|k|v|out)_proj"
    )


@dataclass
class TrainConfig:
    output_dir: str = "outputs/sdft-lfm25-230m"
    epochs: float = 1.0
    lr: float = 2e-4
    batch_size: int = 4
    grad_accum: int = 4
    max_length: int = 1024
    warmup_steps: int = 10
    logging_steps: int = 10
    save_strategy: str = "epoch"
    seed: int = 0
    # Which jsonl field to train on: "sdft_response" (SDFT) or "response" (gold SFT).
    target_field: str = "sdft_response"


@dataclass
class GrpoConfig:
    """Group Relative Policy Optimization (TRL GRPOTrainer) knobs."""

    output_dir: str = "outputs/grpo-lfm25-230m"
    epochs: float = 1.0
    lr: float = 5e-5
    # GRPO: per_device_train_batch_size must be divisible by num_generations.
    batch_size: int = 2
    grad_accum: int = 1
    num_generations: int = 2
    max_prompt_length: int = 512
    max_completion_length: int = 256
    temperature: float = 0.7
    warmup_steps: int = 0
    logging_steps: int = 1
    save_strategy: str = "epoch"
    seed: int = 0
    # Reward: "instruction" (refusal + length + gold overlap) or "boxed" (tool/math).
    reward: str = "instruction"


@dataclass
class ToolCallConfig:
    max_rounds: int = 16
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    # auto | openclaw (ReTool JSON + <interpreter>) | lfm (native apply_chat_template)
    format: str = "auto"
    system_prompt: str | None = None
    # Fixed one-line CoT cue appended to system prompt (ablation flag).
    cot_line: str | None = None
    max_context_chars: int = 12000
    max_obs_chars: int = 1024
    sandbox_timeout_s: int = 30


@dataclass
class OnlineLearningConfig:
    train_steps: int = 2
    replay_buffer_size: int = 8
    preview_before_train: bool = False
    preview_max_new_tokens: int = 128
    session_root: str = "outputs/online-learning"


@dataclass
class OpenClawEvalConfig:
    dataset: str = "zhuzilin/aime-2024"
    data_file: str | None = None
    split: str = "train"
    num_examples: int | None = 2
    n_samples: int = 1
    # Prepend k high-quality tool-use demos before each test question (not pass@k).
    few_shot_k: int = 0
    strict_box_verify: bool = True
    out_dir: str = "outputs/benchmarks/openclaw-rl"
    seed: int = 0


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    generation: GenerateConfig = field(default_factory=GenerateConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    training: TrainConfig = field(default_factory=TrainConfig)
    grpo: GrpoConfig = field(default_factory=GrpoConfig)
    toolcall: ToolCallConfig = field(default_factory=ToolCallConfig)
    online_learning: OnlineLearningConfig = field(default_factory=OnlineLearningConfig)
    openclaw_eval: OpenClawEvalConfig = field(default_factory=OpenClawEvalConfig)


def _apply(section: Any, values: dict[str, Any], path: str) -> None:
    valid = {f.name: f for f in dataclasses.fields(section)}
    for key, value in values.items():
        if key not in valid:
            raise ValueError(f"Unknown config key '{path}.{key}' (valid: {sorted(valid)})")
        setattr(section, key, value)


def load_config(path: str | Path) -> Config:
    cfg = Config()
    raw = yaml.safe_load(Path(path).read_text()) or {}
    for section_name, values in raw.items():
        if not hasattr(cfg, section_name):
            raise ValueError(f"Unknown config section '{section_name}'")
        _apply(getattr(cfg, section_name), values or {}, section_name)
    return cfg
