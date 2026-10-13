"""SM table "Circuit-level sample sizes and logical failures" (l = 33, Fig. 4 counts).

Splits every logical failure of Fig. 4 in two independent ways:

* by decoder stage: T1 if Relay-BP (including its initial plain-BP iterations) converged
  to a wrong correction, T2 if it did not converge and the BP-LSD fallback was wrong;
* by type, from the observable mistake mask ('E' = wrong observable): "block only" if
  only block observables (the first 148, the data logicals) are wrong, "time-like only"
  if only time-like fibre readouts (the last 33, surgery X only) are wrong, else "both".
  Idling and surgery Z record block observables only.

The failures of the cluster points p = 0.002-0.003 are read from
``data/ler/l33/failure_masks.json`` (written by ``run_batched.py summary``) and checked
against the counts of Fig. 4; the p = 0.0016 rows were recorded by hand without masks
and are taken from ``data/ler/l33/counts.json``.

    uv run python scripts/ler/l33/circuit_error_table.py [--masks masks.json]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parents[3] / "data" / "ler" / "l33"
TYPES = ("block_only", "time_only", "both")


def classify(masks: dict[str, int], block: int) -> Counter:
    """Count failures by type from their observable mistake masks."""
    types = Counter(dict.fromkeys(TYPES, 0))
    for mask, count in masks.items():
        wrong_block, wrong_time = "E" in mask[:block], "E" in mask[block:]
        assert wrong_block or wrong_time, "a failure has a wrong observable"
        kind = "both" if wrong_block and wrong_time else "time_only" if wrong_time else "block_only"
        types[kind] += count
    return types


def rows(masks_file: Path) -> list[dict]:
    masks = json.loads(masks_file.read_text())
    block = masks["block_observables"]
    out = []
    for r in masks["results"]:
        t1, t2 = classify(r["relay_failures"], block), classify(r["lsd_failures"], block)
        out.append(dict(r, errors=t1.total() + t2.total(), t1=t1, t2=t2))
    if masks_file.resolve() == DATA / "failure_masks.json":
        counts = json.loads((DATA / "counts.json").read_text())["results"]
        published = {(r["p"], r["experiment"], r["basis"]): r for r in counts}
        for r in out:
            reference = published[r["p"], r["experiment"], r["basis"]]
            assert (r["num_shots"], r["errors"]) == (reference["num_shots"], reference["errors"])
        for r in counts:
            if "failure_types" in r:  # p = 0.0016: no T1 failures, so all types are T2
                assert r["relay_errors"] == 0 and sum(r["failure_types"].values()) == r["errors"]
                none = Counter(dict.fromkeys(TYPES, 0))
                out.append(dict(r, t1=none, t2=Counter(r["failure_types"])))
    for r in out:
        if r["experiment"] == "Idling" or r["basis"] == "Z":  # no time-like observables
            assert r["t1"]["block_only"] + r["t2"]["block_only"] == r["errors"]
    return sorted(out, key=lambda r: (r["p"], r["experiment"], r["basis"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--masks", type=Path, default=DATA / "failure_masks.json")
    table = rows(parser.parse_args().masks)
    print(
        f"{'p':>6}  {'experiment':<11} {'shots':>9} {'T1':>4} {'T2':>4} {'total':>5}  "
        f"{'block only':>10} {'time-like only':>14} {'both':>4}"
    )
    for r in table:
        types = r["t1"] + r["t2"]
        print(
            f"{r['p']:>6}  {r['experiment'] + ' ' + r['basis']:<11} {r['num_shots']:>9,} "
            f"{r['t1'].total():>4} {r['t2'].total():>4} {r['errors']:>5}  "
            f"{types['block_only']:>10} {types['time_only']:>14} {types['both']:>4}"
        )
    print(f"{len(table)} rows")
    print("Surgery X errors by stage and type (block only, time-like only, both):")
    for r in table:
        if (r["experiment"], r["basis"]) == ("Surgery", "X"):
            t1, t2 = (tuple(r[stage][k] for k in TYPES) for stage in ("t1", "t2"))
            print(f"  p={r['p']}: T1 {t1}  T2 {t2}")


if __name__ == "__main__":
    main()
