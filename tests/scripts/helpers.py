"""Run the scripts in scripts/ as subprocesses, from the repository root, or import one."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]


def load(script: str) -> ModuleType:
    """Import ``scripts/<script>`` as a module (it may not import sibling scripts)."""
    path = ROOT / "scripts" / script
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = sys.modules[spec.name] = importlib.util.module_from_spec(spec)  # for dataclasses
    spec.loader.exec_module(module)
    return module


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
