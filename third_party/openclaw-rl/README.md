# OpenClaw-RL (vendored reference)

This directory is intentionally **not** a full clone of [OpenClaw-RL](https://github.com/Gen-Verse/OpenClaw-RL).

Local-SDFT implements a **thin adapter** in `sdft/toolcall/` that mirrors the
`toolcall-rl/` ReTool protocol (tool format, sandbox, scoring) without pulling
in the SLIME/Megatron/SGLang training stack.

To use the upstream project for RL training or cluster eval:

```bash
git clone --depth 1 https://github.com/Gen-Verse/OpenClaw-RL.git third_party/OpenClaw-RL
cd third_party/OpenClaw-RL
pip install -r toolcall-rl/requirements.txt
# See toolcall-rl/README.md and instructions/README.md for SLIME setup.
```

Relevant upstream files:

- `toolcall-rl/generate_with_retool.py` — multi-turn generation + tool loop
- `toolcall-rl/tool_sandbox.py` — Python sandbox
- `toolcall-rl/README.md` — training scripts and tool XML/JSON format

Local adapter docs: [docs/openclaw-rl-eval.md](../docs/openclaw-rl-eval.md)
