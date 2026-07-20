"""Local BFCL (Berkeley Function-Calling Leaderboard) subset evaluation.

Official BFCL uses ``bfcl-eval`` + vLLM/sglang. This package is a **local,
Apple-Silicon-friendly subset** that:

- loads BFCL-v3 single-turn categories from the public HF dataset
- generates with our transformers/MPS stack (LFM chat template + tools)
- scores AST accuracy (simple / multiple / parallel) and irrelevance

Not a full leaderboard run (no live / multi-turn / executable / web-search).
"""

from .ast_score import score_bfcl_example
from .parse import parse_function_calls

__all__ = ["parse_function_calls", "score_bfcl_example"]
