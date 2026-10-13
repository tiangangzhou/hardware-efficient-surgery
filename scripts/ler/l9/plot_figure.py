"""SM figure "Logical error rates under circuit noise for l = 9" (``figures/l9_circuit_ler.pdf``).

Idling vs fibre, packet and joint X-type surgery, ten noisy syndrome rounds, separate X
and Z experiments.  A shot fails if any recorded observable is wrong (sinter's
``errors``): in panel (a), X, any data logical or canonical surgery readout (52 / 61 /
55 / 113 observables for idling / fibre / packet / joint); in panel (b), Z, any surviving
data logical (52 / 41 / 47 / 91).  Each point is failures / shots per ten-round
experiment with a nominal two-sided 95% Wilson interval (not adjusted for adaptive
stopping); after zero failures only the upper limit is drawn.

Reads the published sinter statistics in ``data/ler/l9/`` (40 tasks; the p = 0.001
top-up rows replace the parent rows of the same tasks) after checking their hashes,
settings, stopping rules and observable counts, or with ``--stats`` the CSVs written by
``run_circuits.py``.

    uv run python scripts/ler/l9/plot_figure.py [--stats l9.csv ...] [--out fig.pdf]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator, LogLocator, NullLocator

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "ler" / "l9"
FIGURE = ROOT / "figures" / "l9_circuit_ler.pdf"
MASK = "obs_mistake_mask="
# mode: (label, colour, marker, line style, observables in the X and Z experiments)
MODES = {
    "idling": ("Idling", "#547AA5", "o", "-", {"X": 52, "Z": 52}),
    "fibre": ("Fibre", "#D9824B", "s", "--", {"X": 61, "Z": 41}),
    "packet": ("Packet", "#3F8F6B", "^", "-.", {"X": 55, "Z": 47}),
    "joint": ("Joint", "#7956A8", "D", ":", {"X": 113, "Z": 91}),
}


def wilson_interval(errors: int, shots: int) -> tuple[float, float]:
    """Two-sided 95% Wilson score interval of errors / shots."""
    z = 1.959963984540054
    fraction = errors / shots
    denominator = 1 + z * z / shots
    centre = (fraction + z * z / (2 * shots)) / denominator
    radius = z * math.sqrt(fraction * (1 - fraction) / shots + z * z / (4 * shots**2)) / denominator
    return (
        0.0 if errors == 0 else max(0.0, centre - radius),
        1.0 if errors == shots else min(1.0, centre + radius),
    )


def read_tasks(*paths: Path) -> dict[tuple, dict]:
    """{(mode, basis, p): task} of sinter CSVs, adding up the rows of each task."""
    tasks: dict[tuple, dict] = {}
    for path in paths:
        with Path(path).open() as stream:
            records = list(csv.DictReader(stream, skipinitialspace=True))
        for record in records:
            meta = json.loads(record["json_metadata"])
            observables = MODES[meta["mode"]][4][meta["basis"]]
            assert meta["block_observables"] + meta["time_like_observables"] == observables
            counts = json.loads(record["custom_counts"])
            masks = {k: n for k, n in counts.items() if k.startswith(MASK)}  # failures per mask
            assert all(len(mask) == len(MASK) + observables for mask in masks)
            assert sum(masks.values()) == int(record["errors"]), "missing failure masks"
            key = meta["mode"], meta["basis"], meta["p"]
            task = tasks.setdefault(
                key, dict(meta=meta, strong_id=record["strong_id"], shots=0, errors=0)
            )
            assert task["strong_id"] == record["strong_id"], f"two tasks with key {key}"
            task["shots"] += int(record["shots"]) - int(record["discards"])
            task["errors"] += int(record["errors"])
    return tasks


def load_archive() -> dict[tuple, dict]:
    """The 40 published tasks: the parent campaign with its p = 0.001 rows topped up."""
    provenance = json.loads((DATA / "provenance.json").read_text())
    tasks: dict[tuple, dict] = {}
    for name in ("parent", "topup"):
        info = provenance["campaigns"][name]
        path = DATA / info["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == info["sha256"], path
        campaign = read_tasks(path)
        assert len(campaign) == info["tasks"]
        assert sum(task["shots"] for task in campaign.values()) == info["shots"]
        for key, task in campaign.items():
            meta = task["meta"]
            assert meta["rounds"] == 10 and meta["noise_model"] == "DepolarizingNoiseModel"
            assert meta["lsd_kwargs"]["ms_scaling_factor"] == 0.625
            assert meta["surgery_basis"] == (None if meta["mode"] == "idling" else "X")
            assert task["shots"] >= info["max_shots"] or task["errors"] >= info["max_errors"]
            if name == "topup":  # its counts include the parent's 200,000 shots
                assert key[2] == 0.001 and task["strong_id"] == tasks[key]["strong_id"]
                assert task["shots"] >= tasks[key]["shots"] == 200_000
            tasks[key] = task
    totals = provenance["totals"]
    assert len(tasks) == totals["tasks"]
    assert sum(t["shots"] for t in tasks.values()) == totals["shots"]
    assert sum(t["errors"] for t in tasks.values()) == totals["errors"]
    return tasks


def plot(rows: list[dict], output: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "axes.linewidth": 0.7,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.25), sharex=True, sharey=True)
    titles = {"X": "(a) $X$: data + surgery outputs", "Z": "(b) $Z$: surviving data logicals"}
    for ax, (basis, title) in zip(axes, titles.items(), strict=True):
        for mode, (_, color, marker, line, _) in MODES.items():
            points = [r for r in rows if r["mode"] == mode and r["basis"] == basis]
            if not points:
                continue
            ax.plot(
                [r["p"] for r in points],
                [r["ler"] if r["errors"] else np.nan for r in points],
                color=color,
                linestyle=line,
                linewidth=1.0,
            )
            positive = [r for r in points if r["errors"]]
            if positive:
                ax.errorbar(
                    [r["p"] for r in positive],
                    [r["ler"] for r in positive],
                    yerr=[
                        [r["ler"] - r["low"] for r in positive],
                        [r["high"] - r["ler"] for r in positive],
                    ],
                    fmt=marker,
                    color=color,
                    markersize=3.8,
                    markerfacecolor="white",
                    markeredgewidth=0.8,
                    capsize=2,
                    elinewidth=0.7,
                )
            for r in points:
                if r["errors"] == 0:  # upper limit only
                    ax.plot(
                        r["p"],
                        r["high"],
                        marker="v",
                        markerfacecolor="none",
                        color=color,
                        linestyle="none",
                    )
        ax.set(
            xscale="log",
            yscale="log",
            xlim=(0.00091, 0.0055),
            ylim=(5e-6, 1.0),
            title=title,
            xlabel=r"Physical error probability, $p$",
        )
        ax.xaxis.set_major_locator(FixedLocator([0.001, 0.002, 0.003, 0.005]))
        ax.xaxis.set_major_formatter(FixedFormatter(["0.001", "0.002", "0.003", "0.005"]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=7))
        ax.grid(which="major", color="#e4e4e4", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=8)
    axes[0].set_ylabel("LER per 10-round experiment")
    handles = [
        Line2D(
            [],
            [],
            color=color,
            marker=marker,
            linestyle=line,
            linewidth=1,
            markersize=4,
            markerfacecolor="white",
            label=label,
        )
        for label, color, marker, line, _ in MODES.values()
    ]
    fig.legend(
        handles=handles,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.53, 1.02),
        frameon=False,
        handlelength=2.5,
        columnspacing=1.8,
    )
    fig.subplots_adjust(left=0.092, right=0.992, bottom=0.185, top=0.825, wspace=0.12)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=350, facecolor="white", metadata={"CreationDate": None})
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stats", type=Path, nargs="+", help="sinter CSV(s) from run_circuits.py")
    parser.add_argument("--out", type=Path, default=FIGURE)
    args = parser.parse_args()
    tasks = load_archive() if args.stats is None else read_tasks(*args.stats)
    rows = []
    for (mode, basis, p), task in sorted(tasks.items()):
        shots, errors = task["shots"], task["errors"]
        low, high = wilson_interval(errors, shots)
        rows.append(
            dict(mode=mode, basis=basis, p=p, errors=errors, ler=errors / shots, low=low, high=high)
        )
        print(
            f"{mode:>6} {basis} p={p:.5f} {errors:>4}/{shots:<8} reported LER "
            f"{errors / shots:.3e} ({'any' if basis == 'X' else 'block'}, "
            f"{MODES[mode][4][basis]} observables)"
        )
    plot(rows, args.out)
    print(
        f"{len(rows)} tasks, {sum(t['shots'] for t in tasks.values()):,} shots, "
        f"{sum(t['errors'] for t in tasks.values()):,} failures"
    )


if __name__ == "__main__":
    main()
