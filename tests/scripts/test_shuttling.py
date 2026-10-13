"""The shuttling scripts reproduce Table S2, the gadgets, the space counts and Fig. 3."""

from __future__ import annotations

import shutil

import pytest

from tests.scripts.helpers import run

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")


@needs_node
def test_table_s2():
    out = run("shuttling/extract_timings.py")
    assert "All 48 memory/fibre schedules pass verify.cjs." in out
    assert "33&Algebraic fibre&25.48&29.36&54.84&30.430&16.34&18.85&34.60&19.728" in out
    assert "matches table_s2.csv" in out


def test_graph_gadgets_and_space():
    assert run("shuttling/build_graph_gadgets.py").count("matches the animation") == 6
    out = run("shuttling/compute_space_overhead.py")
    assert "l=33 algebraic        4785 physical qubits" in out
    assert "l=33 gauging          8151 physical qubits" in out
    assert "matches space_overhead.json" in out


def test_fig3(tmp_path):
    run("shuttling/plot_fig3.py", "--out", tmp_path / "fig3.pdf")
    assert (tmp_path / "fig3.pdf").stat().st_size > 0
