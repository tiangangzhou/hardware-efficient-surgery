"""Fig. 4: per-cycle LER of memory and X-type fibre surgery at l = 33, by readout basis.

Draws ``figures/bplsd_by_basis_per_cycle.pdf`` from the published counts
``data/ler/l33/counts.json``, or from a sinter CSV of ``run_circuits.py`` (``--stats``).

A shot fails if any recorded logical observable is wrong after the whole r = 20-round
experiment; each basis has its own failure probability P = failures / shots.  A point is
the effective per-cycle rate [1 - (1 - 2P)^(1/r)] / 2, the flip probability per round
that compounds to P over r independent rounds.  Error bars: the 95% binomial
likelihood-ratio interval of P (deviance <= chi2_1(0.95) = 3.84), mapped through the
same increasing conversion.

    uv run python scripts/ler/l33/plot_fig4.py [--stats l33.csv] [--out fig4.pdf]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedFormatter, FixedLocator, LogLocator, NullLocator
from scipy.optimize import brentq
from scipy.special import xlogy
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[3]
COUNTS = ROOT / "data" / "ler" / "l33" / "counts.json"
FIGURE = ROOT / "figures" / "bplsd_by_basis_per_cycle.pdf"
CUTOFF = chi2.ppf(0.95, 1)
STYLES = {"Idling": ("#994455", "o", "-"), "Surgery": ("#004488", "s", "--")}


def load_counts(stats: Path | None = None) -> tuple[int, list[dict]]:
    """(rounds, rows) with rows {experiment, basis, p, num_shots, errors}."""
    if stats is None:
        counts = json.loads(COUNTS.read_text())
        return counts["rounds"], counts["results"]
    import sinter

    rows, rounds = [], set()
    for stat in sinter.read_stats_from_csv_files(stats):
        meta = stat.json_metadata
        rounds.add(meta["rounds"])
        rows.append(
            dict(
                experiment=meta["experiment"].capitalize(),
                basis=meta["basis"],
                p=meta["p"],
                num_shots=stat.shots - stat.discards,
                errors=stat.errors,
            )
        )
    if len(rounds) != 1:
        raise ValueError(f"expected one round count in {stats}, found {sorted(rounds)}")
    return rounds.pop(), rows


def deviance(errors: int, shots: int, q: float) -> float:
    """Binomial deviance 2 [log L(f) - log L(q)] of failure probability q, f = errors / shots.

    The no-failure term uses log1p, which keeps it accurate at rates ~1e-8.
    """
    if not 0 < q < 1:
        # At q = 0 or 1 the likelihood vanishes unless every shot agrees with q (then q = f).
        return 0.0 if q in (0, 1) and errors == q * shots else float("inf")
    f = errors / shots
    rest = (shots - errors) * (np.log1p(-f) - np.log1p(-q)) if errors < shots else 0.0
    return float(2 * (xlogy(errors, f) - xlogy(errors, q) + rest))


def likelihood_interval(errors: int, shots: int) -> tuple[float, float]:
    """95% likelihood-ratio interval {q : deviance(q) <= chi2_1(0.95)}."""
    f = errors / shots
    excess = lambda q: deviance(errors, shots, q) - CUTOFF
    low = 0.0 if errors == 0 else brentq(excess, 0.0, f, xtol=1e-16)
    high = 1.0 if errors == shots else brentq(excess, f, 1.0, xtol=1e-16)
    return low, high


def per_cycle(probability: float, rounds: int) -> float:
    """[1 - (1 - 2P)^(1/r)] / 2, computed without cancellation."""
    return float(-np.expm1(np.log1p(-2 * probability) / rounds) / 2)


def plot(rows: list[dict], output: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral"],
            "mathtext.fontset": "stix",
            "font.size": 12.5,
            "axes.labelsize": 14,
            "legend.fontsize": 12.5,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.linewidth": 0.75,
            "axes.edgecolor": "#333333",
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "ytick.minor.size": 2,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "ytick.minor.width": 0.55,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,  # also needed for the STIX minus sign in the PDF tick labels
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.55), sharex=True, sharey=True)
    for ax, basis, panel in zip(axes, ("Z", "X"), ("a", "b"), strict=True):
        for experiment, (color, marker, linestyle) in STYLES.items():
            points = [r for r in rows if r["experiment"] == experiment and r["basis"] == basis]
            rate, low, high = (np.array([r[k] for r in points]) for k in ("rate", "low", "high"))
            ax.errorbar(
                np.array([r["p"] for r in points]),
                rate,
                yerr=[rate - low, high - rate],
                color=color,
                marker=marker,
                markersize=4.8,
                markeredgewidth=0.85,
                markerfacecolor="white",
                linestyle=linestyle,
                linewidth=1.35,
                elinewidth=0.85,
                capsize=2.6,
                capthick=0.85,
                label="Memory" if experiment == "Idling" else experiment,
                zorder=3,
            )
        ax.set(
            xscale="log",
            yscale="log",
            xlim=(0.00149, 0.0032),
            ylim=(4e-10, 4e-5),
            xlabel=r"Physical error rate, $p$",
        )
        ax.text(
            0.045,
            0.96,
            rf"({panel}) ${{{basis}}}$ basis",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=14,
        )
        ax.xaxis.set_major_locator(FixedLocator([0.0016, 0.002, 0.0025, 0.003]))
        ax.xaxis.set_major_formatter(FixedFormatter(["0.0016", "0.002", "0.0025", "0.003"]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=[2, 5], numticks=20))
        ax.grid(axis="y", which="major", color="#E6E6E6", linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(0.015, 0.87),
            frameon=False,
            handlelength=1.8,
            handletextpad=0.5,
            borderpad=0.15,
            labelspacing=0.3,
        )
        ax.xaxis.labelpad = 5
    axes[0].set_ylabel("LER per cycle", labelpad=4)
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.24, top=0.97, wspace=0.13)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=400, facecolor="white", metadata={"CreationDate": None})
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stats", type=Path, help="sinter CSV from run_circuits.py")
    parser.add_argument("--out", type=Path, default=FIGURE)
    args = parser.parse_args()
    rounds, rows = load_counts(args.stats)
    rows = sorted(rows, key=lambda r: (r["basis"], r["experiment"], r["p"]))
    for r in rows:
        low, high = likelihood_interval(r["errors"], r["num_shots"])
        r["rate"] = per_cycle(r["errors"] / r["num_shots"], rounds)
        r["low"], r["high"] = per_cycle(low, rounds), per_cycle(high, rounds)
        print(
            f"{r['experiment']:>7} {r['basis']} p={r['p']:<7g} {r['errors']:>4}/"
            f"{r['num_shots']:<9} per cycle {r['rate']:.3e} [{r['low']:.3e}, {r['high']:.3e}]"
        )
    plot(rows, args.out)


if __name__ == "__main__":
    main()
