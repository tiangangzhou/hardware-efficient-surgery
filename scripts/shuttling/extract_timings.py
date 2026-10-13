"""Table S2 (shuttling and gate times) and the Fig. 3 timings, from the published animation.

Replays with Node (``extract.cjs``) the schedulers embedded in
``lp-surgery-aod-polynomials.html`` on its embedded models: X, Z and nonparallel X+Z
check rounds of memory, algebraic fibre surgery, Graph-WY and Graph-IGND at l = 9, 15, 33
with one or two AOD pairs, and the parallel X/Z schedules of memory and fibre surgery over
r = 8, 10, 20 rounds (average time per round, T_XZ^(r) / r).  ``verify.cjs`` independently
checks all 48 memory and fibre schedules.  Prints the LaTeX rows of Table S2 and compares
the times with ``data/shuttling/table_s2.csv`` (``--write`` replaces it).  Needs ``node``.

    uv run python scripts/shuttling/extract_timings.py   # ~15 s
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
from pathlib import Path

from animation import compiler_inputs

HERE = Path(__file__).resolve().parent
TABLE = HERE.parents[1] / "data" / "shuttling" / "table_s2.csv"
LIFTS = (9, 15, 33)
LABELS = {
    "memory": "Memory",
    "algebraic": "Algebraic fibre",
    "gauging": "Graph-WY",
    "edge_expansion": "Graph-IGND",
}
FIELDS = ("lift", "method", "pairs", "X_ms", "Z_ms", "XZ_ms", "XZ_parallel_ms")


def extract() -> list[dict]:
    """Rows of Table S2 (times in ms), sorted by lift, method and AOD pairs."""
    result = subprocess.run(
        ["node", str(HERE / "extract.cjs")],
        input=json.dumps(compiler_inputs()),
        text=True,
        capture_output=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr)
    output = json.loads(result.stdout)
    assert output["verified"] == 48, "schedule check failed"
    order = list(LABELS)
    return sorted(output["rows"], key=lambda r: (r["lift"], order.index(r["method"]), r["pairs"]))


def latex(rows: list[dict]) -> str:
    """Table S2 rows: l & protocol & (X & Z & X+Z & parallel X+Z) for 1 and 2 AOD pairs."""
    lines = []
    for lift in LIFTS:
        for method, label in LABELS.items():
            cells = [str(lift), label]
            for row in (r for r in rows if (r["lift"], r["method"]) == (lift, method)):
                cells += [f"{row[k]:.2f}" for k in ("X_ms", "Z_ms", "XZ_ms")]
                parallel = row["XZ_parallel_ms"]
                cells.append(r"\textemdash" if parallel == "" else f"{parallel:.3f}")
            lines.append("&".join(cells) + r"\\")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--write", action="store_true", help=f"replace {TABLE.name}")
    args = parser.parse_args()
    rows = extract()
    print("All 48 memory/fibre schedules pass verify.cjs.\n")
    print(latex(rows))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    if args.write:
        TABLE.write_text(buffer.getvalue())
    elif TABLE.read_text() != buffer.getvalue():
        raise SystemExit(f"\ndiffers from {TABLE}")
    else:
        print(f"\nmatches {TABLE.name}")


if __name__ == "__main__":
    main()
