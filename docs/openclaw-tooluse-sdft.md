# OpenClaw-style tool-use SDFT

Curated synthetic math trajectories teach LFM2.5-230M the ReTool protocol:

1. `<tool_call>{"name": "code_interpreter", ...}</tool_call>`
2. `<interpreter>...</interpreter>` (injected during eval; included in training targets)
3. `Answer: \boxed{...}`

## Data files

| File | Purpose |
|---|---|
| `data/openclaw_tooluse.jsonl` | Alpaca rows (`instruction`, `input`, `output`) — generate input |
| `data/openclaw_tooluse_sdft.jsonl` | Identity SDFT pairs (`prompt`, `response`, `sdft_response`) — train input |
| `data/openclaw_demo.jsonl` | Easy held-out eval (`question`, `answer`) |

Build (from repo root):

```bash
uv run python scripts/build_openclaw_tooluse_data.py --write-sdft
```

## Why identity SDFT?

The base 230M model does not emit reliable tool calls. Running `sdft.generate` would
rewrite gold trajectories into plain text and erase the tool format. For the bootstrap
run we train on **gold completions** (`sdft_response == response`). Re-enable generate
after the model learns basic tool syntax.

## Pipeline

```bash
uv sync --extra toolcall

# 1. Data
uv run python scripts/build_openclaw_tooluse_data.py --write-sdft

# 2. Train (identity SDFT jsonl)
uv run python -m sdft.train --config configs/openclaw_tooluse_sdft.yaml \
  --data data/openclaw_tooluse_sdft.jsonl

# 3. Merge
uv run python -m sdft.merge --config configs/openclaw_tooluse_sdft.yaml \
  --out outputs/openclaw-tooluse-merged

# 4. Three-way demo-set comparison (format=openclaw matches training targets)
# Zero-shot base
uv run python -m sdft.toolcall.openclaw_eval \
  --config configs/openclaw_demo_eval.yaml --format openclaw \
  --out-dir outputs/benchmarks/openclaw-rl/demo-zero-shot

# One-shot base (prepends one tool_call→interpreter→boxed demo; not pass@k)
uv run python -m sdft.toolcall.openclaw_eval \
  --config configs/openclaw_demo_eval.yaml --format openclaw --one-shot \
  --out-dir outputs/benchmarks/openclaw-rl/demo-one-shot

# Post-SDFT: point model at merged checkpoint
# (see scripts/run_openclaw_tooluse_sdft.sh)

# 5. Optional AIME slice (low scores expected on 230M)
uv run python -m sdft.toolcall.openclaw_eval \
  --config configs/openclaw_rl_eval.yaml --num-examples 3 --format openclaw
```

Or run the orchestration script:

```bash
bash scripts/run_openclaw_tooluse_sdft.sh
```

## Training row format

Alpaca (`openclaw_tooluse.jsonl`):

```json
{
  "instruction": "Solve the math problem. When you need to compute, call the code_interpreter tool ...",
  "input": "What is 12 + 19?",
  "output": "<tool_call>\n{\"name\": \"code_interpreter\", \"arguments\": {\"code\": \"print(12 + 19)\"}}\n</tool_call>\n\n<interpreter>\n31\n</interpreter>\n\nAnswer: \\boxed{31}"
}
```

SDFT train jsonl (`openclaw_tooluse_sdft.jsonl`):

```json
{
  "prompt": "Solve the math problem...\n\nWhat is 12 + 19?",
  "response": "<tool_call>...",
  "sdft_response": "<tool_call>..."
}
```

Demo eval (`openclaw_demo.jsonl`):

```json
{"question": "What is 7 times 8?", "answer": "56"}
```
