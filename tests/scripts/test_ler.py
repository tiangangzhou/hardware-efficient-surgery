"""The LER scripts reproduce the paper's numbers from the published data."""

from __future__ import annotations

import csv
import json

import pytest

from qldpc_surgery.decoding import LSD_KWARGS, RELAY_KWARGS
from tests.scripts.helpers import ROOT, run

DATA = ROOT / "data" / "ler"


def test_fig4(tmp_path):
    out = run("ler/l33/plot_fig4.py", "--out", tmp_path / "fig.pdf")
    assert "Surgery Z p=0.003    231/500000    per cycle 2.311e-05 [2.026e-05, 2.622e-05]" in out
    assert " Idling X p=0.0016     2/5000000   per cycle 2.000e-08 [3.326e-09, 6.176e-08]" in out
    assert (tmp_path / "fig.pdf").stat().st_size > 0


def test_effective_distances():
    out = run("ler/l33/fit_effective_distance.py")
    assert "Idling Average   8.45   15.9 [ 13.1,  18.8] (+2.9/-2.8)   0.424" in out
    assert "Surgery Average  11.88   22.8 [ 21.1,  24.4] (+1.7/-1.6)   0.008" in out


def test_circuit_error_table():
    out = run("ler/l33/circuit_error_table.py")
    assert "0.0025  Surgery X   2,775,000   16    8    24           5             16    3" in out
    assert "p=0.003: T1 (0, 8, 0)  T2 (2, 0, 7)" in out and "16 rows" in out


def test_code_capacity_table():
    out = run("ler/l33/summarize_code_capacity.py")
    assert "348 shards (388 collected), 90,336,000 shots, 1,273 failures" in out
    assert " 0.015 Merged   20,000,000      0  <1.5e-07  20,000,000      1     5e-08" in out
    assert "0.05 Merged       20,000     11   0.00055       4,000    437     0.109" in out


def test_l9_figure(tmp_path):
    out = run("ler/l9/plot_figure.py", "--out", tmp_path / "fig.pdf")
    assert "packet X p=0.00100   48/2825536  reported LER 1.699e-05 (any, 55 observables)" in out
    assert "40 tasks, 9,436,691 shots, 3,178 failures" in out


def test_published_decoder_settings():
    """The published runs used the decoder settings of ``qldpc_surgery.decoding``."""
    relay, lsd = json.loads(json.dumps(RELAY_KWARGS)), LSD_KWARGS  # tuples as JSON lists
    counts = json.loads((DATA / "l33" / "counts.json").read_text())["decoder"]
    campaigns = json.loads((DATA / "l33" / "code_capacity_campaigns.json").read_text())
    with (DATA / "l9" / "parent.csv").open() as stream:
        l9 = json.loads(next(csv.DictReader(stream, skipinitialspace=True))["json_metadata"])
    for used_relay, used_lsd in [
        (counts["relay_bp"], counts["bp_lsd"]),
        (l9["relay_kwargs"], l9["lsd_kwargs"]),
    ]:
        assert {k: used_relay[k] for k in relay} == relay
        assert {k: used_lsd[k] for k in lsd} == lsd
    for campaign in campaigns["campaigns"].values():  # code capacity: BP-LSD only
        assert {k: campaign["decoder"][k] for k in lsd} == lsd


def test_l9_circuits_and_tiny_run(tmp_path):
    out = run(
        "ler/l9/run_circuits.py", "--dry-run", "--modes", "fibre,packet,idling", "--p-list", "0.001"
    )
    for mode, x_obs, z_obs in (("fibre", 61, 41), ("packet", 55, 47), ("idling", 52, 52)):
        assert f"{mode} X p=0.001: {x_obs} observables" in out
        assert f"{mode} Z p=0.001: {z_obs} observables" in out
    stats = tmp_path / "l9.csv"
    run(
        "ler/l9/run_circuits.py",
        *("--modes", "fibre", "--basis", "X", "--p-list", "0.001"),
        *("--shots", "10", "--workers", "1", "--out", stats),
    )
    assert "1 tasks, 10 shots" in run(
        "ler/l9/plot_figure.py", "--stats", stats, "--out", tmp_path / "fig.pdf"
    )


@pytest.mark.slow
def test_l9_joint_observables():
    out = run("ler/l9/run_circuits.py", "--dry-run", "--modes", "joint", "--p-list", "0.001")
    assert "joint X p=0.001: 113 observables" in out and "joint Z p=0.001: 91 observables" in out


@pytest.mark.slow
def test_l33_circuits():
    out = run("ler/l33/run_circuits.py", "--dry-run", "--p-list", "0.002")
    assert "surgery X p=0.002: 181 observables" in out  # 148 + 33
    assert "surgery Z p=0.002: 113 observables" in out  # 148 - (33 + 2)
    assert "idling X p=0.002: 148 observables" in out


@pytest.mark.slow
def test_code_capacity_run(tmp_path):
    out = run(
        "ler/l33/run_code_capacity.py",
        *("--codes", "merged", "--basis", "Z", "--p-list", "0.05"),
        *("--shots", "2000", "--out", tmp_path / "cc.csv"),
    )
    assert "merged: [[2442, 113]]" in out and "merged Z p=0.05: " in out
    assert len((tmp_path / "cc.csv").read_text().splitlines()) == 2


@pytest.mark.slow
def test_l33_batched_campaign(tmp_path):
    runs, masks = tmp_path / "runs", tmp_path / "masks.json"
    run(
        "ler/l33/run_batched.py",
        *("collect", "--task", "idling,X,0.003", "--shard", "1", "--shots", "20", "--out", runs),
    )
    run("ler/l33/run_batched.py", "finalize", "--out", runs)
    out = run("ler/l33/run_batched.py", "summary", "--out", runs, "--masks", masks)
    assert "/20 failures, 1 shards" in out
    assert "Idling X" in run("ler/l33/circuit_error_table.py", "--masks", masks)
