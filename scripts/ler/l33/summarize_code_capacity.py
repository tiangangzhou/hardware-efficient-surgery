"""SM table "Code-capacity shot counts and logical error rates" (l = 33, p = 0.015-0.05).

Adds the published cluster shards ``data/ler/l33/code_capacity_shards.csv`` (settings in
``code_capacity_campaigns.json``) into one entry per (code, basis, p).  The p = 0.01
shards lie outside the table and are left out; at p = 0.015 the two fixed batches of
10^7 shots per code and basis are combined.  The LER is failures / shots; after zero
failures in N shots the table gives the one-sided 95% upper bound 1 - 0.05^(1/N).
Before adding, every campaign is checked to use one BP-LSD setting (min-sum scaling
0.625), to have its recorded number of shards with distinct seeds, and every shard to
have reached its stopping rule (shot cap, or failure cap for adaptive shards).

    uv run python scripts/ler/l33/summarize_code_capacity.py
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parents[3] / "data" / "ler" / "l33"
EXCLUDED_P = 0.01
LABELS = {"mem": "Bare", "merged": "Merged"}


def check(campaigns: dict, shards: list[dict]) -> None:
    decoders = [c["decoder"] for c in campaigns.values()]
    assert all(d == decoders[0] for d in decoders) and decoders[0]["ms_scaling_factor"] == 0.625
    assert len({s["seed"] for s in shards}) == len(shards), "repeated shard seed"
    for name, campaign in campaigns.items():
        mine = [s for s in shards if s["campaign"] == name]
        assert len(mine) == campaign["shards"], name
        for s in mine:
            full = int(s["shots"]) == int(s["shot_cap"])
            assert full or (not campaign["fixed_shot_count"] and s["errors"] >= s["max_errors"])


def ler(shots: int, errors: int) -> str:
    if errors == 0:
        return f"<{-math.expm1(math.log(0.05) / shots):.2g}"
    return f"{errors / shots:.3g}"


def main() -> None:
    campaigns = json.loads((DATA / "code_capacity_campaigns.json").read_text())["campaigns"]
    with (DATA / "code_capacity_shards.csv").open() as stream:
        shards = list(csv.DictReader(stream))
    for s in shards:
        s.update(p=float(s["p"]), errors=int(s["errors"]), max_errors=int(s["max_errors"]))
    check(campaigns, shards)
    used = [s for s in shards if s["p"] != EXCLUDED_P]
    totals: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])  # shots, errors
    for s in used:
        totals[s["code"], s["basis"], s["p"]][0] += int(s["shots"])
        totals[s["code"], s["basis"], s["p"]][1] += s["errors"]
    assert len(totals) == 20, "2 codes x 2 bases x 5 p"

    print(
        f"{'p':>6} {'code':<7}{'X shots':>12} {'errors':>6} {'LER':>9}{'Z shots':>12} "
        f"{'errors':>6} {'LER':>9}"
    )
    for p in sorted({key[2] for key in totals}):
        for code, label in LABELS.items():
            cells = ""
            for basis in "XZ":
                shots, errors = totals[code, basis, p]
                cells += f"{shots:>12,} {errors:>6} {ler(shots, errors):>9}"
            print(f"{p:>6g} {label:<7}{cells}")
    for name in sorted({s["campaign"] for s in shards if s["p"] == 0.015}):
        errors = defaultdict(int)
        for s in shards:
            if s["campaign"] == name:
                errors[s["code"], s["basis"]] += s["errors"]
        listed = ", ".join(f"{LABELS[c]} {b}: {errors[c, b]}" for c in LABELS for b in "XZ")
        print(f"p = 0.015, {name}: {listed}")
    print(
        f"{len(used)} shards ({len(shards)} collected), "
        f"{sum(t[0] for t in totals.values()):,} shots, "
        f"{sum(t[1] for t in totals.values()):,} failures"
    )


if __name__ == "__main__":
    main()
