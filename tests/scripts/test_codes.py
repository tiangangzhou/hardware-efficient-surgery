"""The code-construction scripts run and reproduce the paper's numbers (they assert them)."""

from __future__ import annotations

import shutil

import pytest

from tests.scripts.helpers import run


def test_verify_instances():
    out = run("codes/verify_instances.py")
    assert "l=33: [[2442,113]], ranks (1017, 1312)" in out and "All checks passed." in out


def test_worked_examples():
    out = run("codes/worked_examples.py")
    assert "adapter [[1206,91]]" in out and "adapter [[2010,133]]" in out


def test_crt_packets():
    assert "All CRT-packet checks passed." in run("codes/crt_packets.py")


def test_qdistrnd_table_and_inputs(tmp_path):
    assert "agree with the published table" in run("codes/qdistrnd.py", "summarize")
    assert run("codes/qdistrnd.py", "export", "--output", tmp_path).count("hashes match") == 3


@pytest.mark.slow
def test_adapters():
    assert "agree with the SM" in run("codes/adapters.py")


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("c++") is None, reason="needs a C++17 compiler")
def test_exhaustive_distances():
    out = run("codes/verify_instances.py", "--exhaustive")
    assert "l=9: exact (d_X, d_Z) = (9, 8)" in out and "l=15: exact (d_X, d_Z) = (11, 10)" in out
