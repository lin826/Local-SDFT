"""Safe synchronous Python sandbox for code_interpreter tool calls."""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

CODE_INTERPRETER_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "code_interpreter",
        "description": (
            "A tool for executing Python code in a safe sandbox environment. "
            "Use this to perform calculations."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute.",
                }
            },
            "required": ["code"],
        },
    },
}

ALLOWED_MODULES = {
    "math",
    "random",
    "datetime",
    "collections",
    "itertools",
    "functools",
    "operator",
    "statistics",
    "decimal",
    "fractions",
}

DANGEROUS_PATTERNS = [
    r"import\s+os",
    r"import\s+sys",
    r"import\s+subprocess",
    r"import\s+shutil",
    r"import\s+glob",
    r"import\s+socket",
    r"import\s+requests",
    r"from\s+os\s+import",
    r"from\s+sys\s+import",
    r"from\s+subprocess\s+import",
    r"eval\s*\(",
    r"exec\s*\(",
    r"open\s*\(",
    r"__import__\s*\(",
    r"compile\s*\(",
]

DEFAULT_TIMEOUT_S = 30
DEFAULT_MAX_OUTPUT_CHARS = 4096


def _check_code_safety(code: str) -> tuple[bool, str]:
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, code):
            return False, f"Blocked dangerous pattern: {pattern}"
    return True, ""


def execute_code_interpreter(
    code: str,
    *,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
) -> str:
    """Execute Python code in an isolated subprocess."""
    code = code.strip()
    if not code:
        return "Error: No Python code found"

    ok, reason = _check_code_safety(code)
    if not ok:
        return f"Error: {reason}"

    preamble = "\n".join(f"import {mod}" for mod in sorted(ALLOWED_MODULES))
    wrapped = f"{preamble}\n\n{code}\n"

    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "snippet.py"
        script.write_text(wrapped, encoding="utf-8")
        try:
            proc = subprocess.run(
                ["python3", str(script)],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return f"Error: Execution timed out after {timeout_s}s"

    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    if proc.returncode != 0:
        detail = stderr or stdout or f"exit code {proc.returncode}"
        return f"Error: {detail}"

    output = stdout or "(no output)"
    if len(output) > max_output_chars:
        output = output[:max_output_chars] + f"\n... [truncated {len(output) - max_output_chars} chars]"
    return output
