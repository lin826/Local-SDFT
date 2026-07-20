"""SDFT step 1 — teacher generation.

The model rewrites each training target in its own distribution: for every
example we show it a few in-context demonstrations (gold pairs from the same
dataset) and let it answer the target prompt itself. The generated answers
replace the original targets for fine-tuning.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from .config import Config, load_config
from .data import build_teacher_messages, load_examples, sample_fewshots
from .utils import load_model, load_tokenizer, pick_device, to_model_device


@torch.inference_mode()
def generate_responses(cfg: Config, examples: list[dict], device: str) -> list[str | None]:
    tokenizer = load_tokenizer(cfg.model)
    tokenizer.padding_side = "left"  # batched generation needs left padding
    model = load_model(cfg.model, device)
    model.eval()

    gen = cfg.generation
    do_sample = gen.temperature > 0
    rng = random.Random(cfg.data.seed)
    results: list[str | None] = [None] * len(examples)

    for start in range(0, len(examples), gen.batch_size):
        batch = examples[start : start + gen.batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                build_teacher_messages(
                    sample_fewshots(examples, start + i, gen.num_shots, rng),
                    example["prompt"],
                ),
                tokenize=False,
                add_generation_prompt=True,
            )
            for i, example in enumerate(batch)
        ]
        # The chat template already emits BOS; don't add another.
        enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = to_model_device(enc, model)
        out = model.generate(
            **enc,
            max_new_tokens=gen.max_new_tokens,
            do_sample=do_sample,
            temperature=gen.temperature if do_sample else None,
            top_p=gen.top_p if do_sample else None,
            pad_token_id=tokenizer.pad_token_id,
        )
        new_tokens = out[:, enc["input_ids"].shape[1] :]
        for i, text in enumerate(tokenizer.batch_decode(new_tokens, skip_special_tokens=True)):
            text = text.strip()
            results[start + i] = text if len(text) >= gen.min_response_chars else None
        done = min(start + gen.batch_size, len(examples))
        print(f"  generated {done}/{len(examples)}", flush=True)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--num-examples", type=int, default=None)
    parser.add_argument("--out", default=None, help="output jsonl path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.num_examples is not None:
        cfg.data.num_examples = args.num_examples
    if args.out is not None:
        cfg.generation.out_path = args.out

    device = pick_device()
    print(f"device: {device}")
    examples = load_examples(cfg.data)
    print(f"loaded {len(examples)} examples from {cfg.data.dataset}")

    responses = generate_responses(cfg, examples, device)

    out_path = Path(cfg.generation.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with out_path.open("w") as fh:
        for example, sdft_response in zip(examples, responses):
            if sdft_response is None:
                continue
            fh.write(
                json.dumps(
                    {
                        "prompt": example["prompt"],
                        "response": example["response"],  # original gold target
                        "sdft_response": sdft_response,  # model's own rewrite (used for SFT)
                    }
                )
                + "\n"
            )
            kept += 1
    print(f"wrote {kept}/{len(examples)} SDFT pairs to {out_path}")


if __name__ == "__main__":
    main()
