"""Run the scripts in scripts/ as subprocesses, from the repository root."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run(script: str, *args) -> str:
    """Run ``scripts/<script>`` and return its stdout; fail on a nonzero exit."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *map(str, args)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout
