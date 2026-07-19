# OpenClaw-RL evaluation with LFM2.5-230M

This repo includes a **thin adapter** for [OpenClaw-RL](https://github.com/Gen-Verse/OpenClaw-RL)'s
`toolcall-rl/` ReTool benchmark protocol. It runs locally with HuggingFace
`transformers` — no SLIME, Megatron, or SGLang required for eval.

## How OpenClaw-RL expects tool calling

OpenClaw-RL's ReTool path (`toolcall-rl/`) trains/evaluates models on math
problems with a **code interpreter** tool:

1. **System prompt** lists tools inside `<tools>...</tools>`.
2. **Model** emits a tool call:

   ```text
   <tool_call>
   {"name": "code_interpreter", "arguments": {"code": "print(2+2)"}}
   </tool_call>
   ```

3. **Sandbox** executes Python; result is injected as:

   ```text
   <interpreter>
   4
   </interpreter>
   ```

4. **Model** continues until it answers:

   ```text
   Answer: \boxed{42}
   ```

5. **Scoring** uses strict `\boxed{}` extraction (Math-DAPO style); reward is
   `+1` correct / `-1` incorrect. Eval on AIME-2024 uses `pass@k` over multiple
   samples.

See upstream `toolcall-rl/README.md` and `generate_with_retool.py` for the full
RL training loop (SGLang router, PRM mode, etc.).

## LFM2.5-230M native tool calling

[LFM2.5-230M](https://huggingface.co/LiquidAI/LFM2.5-230M) has **native** tool
support via `tokenizer.apply_chat_template(..., tools=...)`:

- Default: Pythonic calls between `<|tool_call_start|>` and
  `<|tool_call_end|>`, e.g.
  `[code_interpreter(code="print(2+2)")]`
- Tool results use the **`tool` role** (not `<interpreter>` tags)
- JSON tool calls can be requested in the system prompt

Our adapter supports both:

| `toolcall.format` | Protocol |
|---|---|
| `auto` | Detect from chat template (LFM → native, else OpenClaw) |
| `openclaw` | ReTool JSON + `<interpreter>` (matches OpenClaw-RL eval) |
| `lfm` | Native LFM chat template + `tool` role |

**Gap:** LFM2.5-230M is tuned for lightweight agentic tasks, not heavy math
reasoning. Expect low AIME scores out of the box; the value is having a wired
eval path before/after SDFT fine-tuning on tool-use data.

## Quick start (smoke eval)

```bash
uv sync --extra toolcall

# 2 AIME-2024 problems, auto-detected format (~ downloads 230M weights)
bash scripts/run_openclaw_rl_eval.sh

# Or directly:
uv run python -m sdft.toolcall.openclaw_eval \
  --config configs/openclaw_rl_eval.yaml \
  --num-examples 2 \
  --format auto
```

Results land in `outputs/benchmarks/openclaw-rl/latest.json`.

## Full AIME-2024 eval

```bash
uv run python -m sdft.toolcall.openclaw_eval \
  --config configs/openclaw_rl_eval.yaml \
  --num-examples 30 \
  --n-samples 16 \
  --format openclaw
```

`--n-samples 16` mirrors upstream `retool_qwen3_4b_rl.sh` (`pass@16`).

## Trained / merged checkpoint

Point `model.name` at your merged output or set in YAML:

```yaml
model:
  name: outputs/smoke-merged   # or outputs/sdft-lfm25-230m after merge
```

```bash
uv run python -m sdft.toolcall.openclaw_eval \
  --config configs/openclaw_rl_eval.yaml
```

## Unit tests (no model download)

```bash
uv sync --extra dev --extra toolcall
uv run pytest tests/test_toolcall_format.py -q
```

## API surface

| Module | Purpose |
|---|---|
| `sdft/toolcall/format.py` | Parse/format OpenClaw + LFM tool calls |
| `sdft/toolcall/sandbox.py` | Sync `code_interpreter` sandbox |
| `sdft/toolcall/loop.py` | `run_tool_loop()` multi-turn inference |
| `sdft/toolcall/scoring.py` | OpenClaw-style boxed answer scoring |
| `sdft/toolcall/openclaw_eval.py` | CLI eval harness |

## Upstream OpenClaw-RL (RL training)

For the full distributed RL pipeline, clone upstream into `third_party/` (see
`third_party/openclaw-rl/README.md`) and follow `toolcall-rl/README.md`.
Replace `HF_CKPT` with your LFM checkpoint **only after** verifying tool-call
format compatibility — upstream scripts target Qwen3 and assume SGLang.

## Known blockers

- **No SGLang/vLLM adapter yet** — local eval uses HF generate (slow on CPU).
- **230M math capability** — model card warns against reasoning-heavy math.
- **Format alignment** — `openclaw` mode uses ReTool prompts on a non-Qwen model;
  use `lfm` or fine-tune on ReTool-SFT for best native behavior.
- **Full upstream eval** requires CUDA cluster + SLIME setup from `instructions/README.md`.
