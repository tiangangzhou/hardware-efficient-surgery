"""Fig. 3: syndrome-extraction time per round and physical qubits of the three instances.

Top: X- and Z-check round times (Table S2, ``data/shuttling/table_s2.csv``) with one or
two AOD pairs per direction.  Bottom: data, auxiliary and syndrome atoms when all l
canonical logicals of a fibre are measured in parallel (``space_overhead.json``).

    uv run python scripts/shuttling/plot_fig3.py   # ~2 s
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.hatch
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "shuttling"
LIFTS = (9, 15, 33)
INSTANCES = ("I", "II", "III")
# method: (label, face colour, hatch colour, hatch)
METHODS = {
    "memory": ("Memory", "#AEB7BF", "#E3E6E9", "\\"),
    "algebraic": ("Algebraic fibre", "#175387", "#AEC3D5", "/"),
    "gauging": ("Graph-WY", "#B49A4C", "#E5DCC0", "--"),
    "edge_expansion": ("Graph-IGND", "#A65B6E", "#E0C6CC", "o"),
}
TEXT, GRID, SPINE = "#30302E", "#E9EBED", "#677078"
WIDTH, STEP = 0.145, 1 / 6  # bar width and spacing within an instance (one unit apart)
TIME_LIMIT = {"Z": 50, "X": 75}  # ms


def style() -> None:
    times = "Times New Roman" in {font.name for font in font_manager.fontManager.ttflist}
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "STIXGeneral"],
            "mathtext.fontset": "custom" if times else "stix",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.cal": "Times New Roman",
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "hatch.linewidth": 0.35,
            "pdf.fonttype": 42,
        }
    )
    matplotlib.hatch.SmallCircles.size = 0.1  # circles of radius 1.2 pt, as published


def draw_bars(ax, values: dict[str, list[float]], limit: float, ticks, labels=None) -> list:
    """Grouped bars (one group per instance) on a clean axis; returns the bar containers."""
    bars = []
    for i, (method, (label, face, hatch_colour, hatch)) in enumerate(METHODS.items()):
        x = [j + (i - 1.5) * STEP for j in range(len(LIFTS))]
        bars.append(
            ax.bar(
                x,
                values[method],
                WIDTH,
                color=face,
                edgecolor=hatch_colour,
                linewidth=0,
                hatch=hatch,
                label=label,
                zorder=2,
            )
        )
    ax.set(xlim=(-0.52, 2.52), ylim=(0, limit), yticks=ticks)
    ax.set_xticks(range(len(LIFTS)), labels or [])
    ax.tick_params(axis="x", labelsize=10.5, pad=3.5)
    ax.tick_params(axis="y", labelsize=9.5, pad=4)
    ax.grid(axis="y", color=GRID, linewidth=0.45, zorder=0)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set(color=SPINE, linewidth=0.65, zorder=3)
    return bars


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--out", type=Path, default=ROOT / "figures" / "timing_and_physical_space_compact.pdf"
    )
    args = parser.parse_args()
    with (DATA / "table_s2.csv").open() as handle:
        times = {(r["method"], int(r["lift"]), int(r["pairs"])): r for r in csv.DictReader(handle)}
    space = json.loads((DATA / "space_overhead.json").read_text())["rows"]
    qubits = {(r["method"], r["lift"]): r["total_physical_qubits"] for r in space}

    style()
    fig = plt.figure(figsize=(6.4, 5.1))
    width, height = 460.8, 367.2  # figure size in points; axes boxes below in points
    box = lambda x, y, w, h: fig.add_axes((x / width, y / height, w / width, h / height))
    for column, pairs in enumerate((1, 2)):
        x0 = (55.296, 265.772)[column]
        for y0, sector in ((269.187, "Z"), (175.459, "X")):
            ax = box(x0, y0, 191.342, 70.473)
            values = {
                m: [float(times[m, lift, pairs][f"{sector}_ms"]) for lift in LIFTS] for m in METHODS
            }
            ticks = range(0, TIME_LIMIT[sector], 20)
            draw_bars(ax, values, TIME_LIMIT[sector], ticks, INSTANCES if sector == "X" else None)
            ax.tick_params(axis="y", labelleft=column == 0)
            ax.text(
                0.012,
                0.833,
                f"${sector}$-check syndrome extraction",
                transform=ax.transAxes,
                fontsize=13,
                va="baseline",
            )
        fig.text(
            (x0 + 95.671) / width,
            0.9565,
            f"${pairs}+{pairs}$ AOD pairs",
            fontsize=13,
            ha="center",
            va="baseline",
        )

    ax = box(55.296, 60.588, 401.818, 91.615)
    values = {m: [qubits[m, lift] for lift in LIFTS] for m in METHODS}
    labels = [f"{name} ({lift})" for name, lift in zip(INSTANCES, LIFTS)]
    bars = draw_bars(ax, values, 10_000, (0, 4000, 8000), labels)
    for container in bars:
        ax.bar_label(container, padding=3, fontsize=9, color=TEXT)
    ax.text(
        0.012,
        0.877,
        "Measuring all logical operators of a cyclic basis in parallel",
        transform=ax.transAxes,
        fontsize=13,
        va="baseline",
    )
    ax.set_xlabel("Instance (lift)", fontsize=13, labelpad=5)
    fig.text(
        0.023, 0.7014, "Time per round (ms)", fontsize=13, rotation=90, ha="center", va="center"
    )
    fig.text(
        0.023, 0.2897, "Total physical qubits", fontsize=13, rotation=90, ha="center", va="center"
    )
    fig.legend(
        handles=bars,
        loc="lower center",
        bbox_to_anchor=(255.75 / width, 0.016),
        ncols=4,
        frameon=False,
        fontsize=12.5,
        handlelength=1.05,
        handleheight=0.79,
        handletextpad=0.45,
        columnspacing=1.2,
        borderaxespad=0,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, metadata={"CreationDate": None})  # reproducible bytes
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
