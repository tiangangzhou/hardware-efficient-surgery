"""Effective distances of idling and surgery at l = 33 (SM, l = 33 "Logical error rates").

Model: the full-shot (20-round) failure probability is P(p) = C (p / 0.0025)^b over
p = 0.002, 0.0025, 0.003, and d_eff = 2b - 1 (failures of weight (d + 1) / 2 give
P ~ p^((d+1)/2)).  C and b are fitted by binomial maximum likelihood to the counts of
``data/ler/l33/counts.json`` (or a sinter CSV with ``--stats``):

* X or Z: one binomial per p (3 counts, 2 parameters);
* Average: the SM's P_bar = (P_X + P_Z) / 2 obeys the power law, and at each p the split
  (P_X, P_Z) = 2 P_bar (f, 1 - f) is a free nuisance parameter that is profiled out
  (6 counts, 2 + 3 parameters).

The 95% interval of b is the profile-likelihood interval
{b : min_C D(C, b) <= D_min + chi2_1(0.95)}, mapped to d_eff.  "GOF p" is the asymptotic
chi2 tail of the residual deviance D_min with 3 - 2 = 6 - 5 = 1 degree of freedom.  The
paper quotes the Average rows: d_eff = 15.9 (+2.9/-2.8) for idling and 22.8 (+1.7/-1.6)
for surgery, whose fit has p = 0.008.

    uv run python scripts/ler/l33/fit_effective_distance.py [--stats l33.csv]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from plot_fig4 import CUTOFF, deviance, load_counts
from scipy.optimize import brentq, minimize, minimize_scalar
from scipy.stats import chi2

P0 = 0.0025
FIT_P = (0.002, 0.0025, 0.003)


def mean_deviance(m: float, pair: list[dict]) -> float:
    """Deviance of the (X, Z) counts at mean failure probability m, minimised over the split."""
    if not 0 < m < 0.1:
        return float("inf")

    def split(f: float) -> float:
        weights = (f, 1 - f)
        return sum(deviance(r["errors"], r["num_shots"], 2 * m * w) for r, w in zip(pair, weights))

    result = minimize_scalar(
        split, bounds=(1e-10, 1 - 1e-10), method="bounded", options={"xatol": 1e-12}
    )
    assert result.success
    return float(result.fun)


def fit(cost, start: float) -> tuple[float, float, float, float]:
    """Maximum-likelihood b of cost(log C, b), its 95% profile interval and the deviance."""
    candidates = [
        minimize(
            cost,
            [math.log(start), b],
            method="Nelder-Mead",
            options={"xatol": 1e-9, "fatol": 1e-9, "maxiter": 2000},
        )
        for b in (4.0, 10.0, 16.0)
    ]
    best = min(candidates, key=lambda r: r.fun)
    assert all(r.success for r in candidates) and max(r.fun for r in candidates) - best.fun < 1e-5
    b = float(best.x[1])

    def excess(slope: float) -> float:
        """Profile deviance of b = slope (minimised over log C) above the cutoff."""
        result = minimize_scalar(
            lambda log_c: cost([log_c, slope]),
            bounds=(-30, -3),
            method="bounded",
            options={"xatol": 1e-10},
        )
        assert result.success
        return result.fun - best.fun - CUTOFF

    assert abs(excess(b) + CUTOFF) < 1e-5
    return b, brentq(excess, b - 15, b), brentq(excess, b, b + 15), float(best.fun)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stats", type=Path, help="sinter CSV from run_circuits.py")
    _, rows = load_counts(parser.parse_args().stats)
    print(
        f"{'experiment':>10} {'basis':>7} {'b':>6} {'d_eff':>6} {'95% interval':>17} {'GOF p':>7}"
    )
    for experiment in ("Idling", "Surgery"):
        pairs = []  # [X row, Z row] at each fitted p
        for p in FIT_P:
            pair = [r for r in rows if r["experiment"] == experiment and r["p"] == p]
            pair.sort(key=lambda r: r["basis"])
            assert [r["basis"] for r in pair] == ["X", "Z"], (experiment, p)
            pairs.append(pair)
        for basis in ("X", "Z", "Average"):

            def cost(theta) -> float:
                log_c, b = theta
                total = 0.0
                for pair in pairs:
                    q = math.exp(log_c + b * math.log(pair[0]["p"] / P0))
                    if basis == "Average":
                        total += mean_deviance(q, pair)
                    else:
                        r = pair["XZ".index(basis)]
                        total += deviance(r["errors"], r["num_shots"], q)
                return total

            middle = [r for r in pairs[1] if basis in ("Average", r["basis"])]  # p = P0
            start = sum(r["errors"] / r["num_shots"] for r in middle) / len(middle)
            b, low, high, residual = fit(cost, start)
            d, lo, hi = 2 * b - 1, 2 * low - 1, 2 * high - 1
            print(
                f"{experiment:>10} {basis:>7} {b:6.2f} {d:6.1f} "
                f"[{lo:5.1f}, {hi:5.1f}] (+{hi - d:.1f}/-{d - lo:.1f}) "
                f"{chi2.sf(residual, 1):7.3f}"
            )


if __name__ == "__main__":
    main()
