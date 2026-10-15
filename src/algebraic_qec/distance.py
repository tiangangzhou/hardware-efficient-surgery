"""Exhaustive minimum-weight logical search for CSS codes with cyclic symmetry.

``exhaustive.cpp`` (compiled with ``c++ -std=c++17``) decides whether one sector of a CSS code
has a nontrivial logical of weight at most W.  Let C be the opposite-sector checks, S the
same-sector stabilizers and L a basis of ker(S)/rowspan(C).  A logical is a word e in ker(C);
it is nontrivial iff e is not in rowspan(S) = ker(S)^perp, iff <e, v> = 1 for some
v in ker(S) = span(L) + rowspan(C), iff e pairs oddly with some vector of L.

Soundness of "no nontrivial logical of weight <= W".  Let e be a nontrivial logical of
minimum weight and suppose |e| <= W.

1. |e & s| <= floor(|s|/2) for every stabilizer s, otherwise e + s is lighter.  A branch
   violating this is pruned; once a row attains it, the rest of that row is excluded.
2. No nonempty proper subset f of e has zero syndrome: f and e + f would both be lighter
   and one of them nontrivial.  So a branch inside e first reaches zero syndrome at e.
3. If rowspan(C) and rowspan(S) are invariant under the simultaneous cyclic shift of every
   block of ``period`` qubits (checked by ``write_input``), some shift of e has qubit b*period
   as its first qubit, b its first occupied block; the search runs over these anchors b.

For a partial support p inside e, every unsatisfied check meets e \\ p in an odd number of
qubits that are still available.  The search branches over the available qubits of one
unsatisfied check, each later sibling excluding the earlier ones, so exactly one branch
stays inside e.  Unsatisfied checks with pairwise disjoint available qubits each need a
distinct further qubit, which lower-bounds |e \\ p|.  Hence a complete run over all anchors
finds e or another nontrivial logical of weight <= W.  Proving W = d - 1 and exhibiting a
logical of weight d gives the exact sector distance d of the static code.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from algebraic_qec.bitmask import Rowspace, logical_quotient_basis, row_masks, set_bits, shift_mask

SOURCE = Path(__file__).with_name("exhaustive.cpp")


def sector_rows(hx, hz, sector: str) -> tuple[list[int], list[int], list[int]]:
    """(C, S, L) as bitmasks for the X sector (C = H_Z, S = H_X) or the Z sector."""
    if sector not in ("X", "Z"):
        raise ValueError(f"sector must be 'X' or 'Z', got {sector!r}")
    checks, stabilizers = (hz, hx) if sector == "X" else (hx, hz)
    checks, stabilizers = row_masks(checks), row_masks(stabilizers)
    return checks, stabilizers, logical_quotient_basis(stabilizers, checks, hx.shape[1])


def write_input(path: Path, hx, hz, sector: str, period: int, max_weight: int) -> None:
    """Write ``n #C #S #L period max_weight``, then one line ``count index...`` per row of C,
    S and L.  Raises if C or S is not invariant under the cyclic shift of the blocks."""
    n = hx.shape[1]
    rows = sector_rows(hx, hz, sector)
    for group in rows[:2]:
        span = Rowspace(group)
        if not all(span.contains(shift_mask(r, 1, period, n // period)) for r in group):
            raise ValueError(f"{sector} sector is not invariant under the cyclic shift")
    with path.open("w") as stream:
        print(n, *map(len, rows), period, max_weight, file=stream)
        for row in (r for group in rows for r in group):
            bits = set_bits(row)
            print(len(bits), *bits, file=stream)


def compile_search(out_dir: Path) -> Path:
    """Compile ``exhaustive.cpp`` into ``out_dir``."""
    binary = out_dir / "exhaustive"
    subprocess.run(["c++", "-O3", "-std=c++17", str(SOURCE), "-o", str(binary)], check=True)
    return binary


def run_search(binary: Path, input_path: Path, time_limit: float = 600) -> dict:
    """One search: status "proved" (with its summary line), "witness" or "timeout"."""
    run = subprocess.run(
        [str(binary), str(input_path), str(time_limit)], capture_output=True, text=True
    )
    last = run.stdout.splitlines()[-1] if run.stdout else ""
    if run.returncode == 0 and last.startswith("PROVED_NO_LOGICAL_WEIGHT_LE"):
        return dict(status="proved", summary=last)
    if run.returncode == 1 and last.startswith("WITNESS"):
        return dict(status="witness", witness=[int(q) for q in last.split()[1:]])
    if run.returncode == 3:
        return dict(status="timeout")
    raise RuntimeError(f"exhaustive search failed (exit {run.returncode}): {run.stdout[-500:]}")


def sector_distance(hx, hz, sector: str, period: int, bounds: range, out_dir: Path, binary):
    """Exact sector distance: search with W = bounds[0], bounds[0] + 1, ... until a witness.

    The first bound must be proved; then the first witness has weight exactly W, because
    W - 1 was proved (hence ``bounds`` must be consecutive).
    """
    if bounds.step != 1:
        raise ValueError(f"bounds must be consecutive, got {bounds}")
    n = hx.shape[1]
    for bound in bounds:
        path = out_dir / f"{sector}_w{bound}.dat"
        write_input(path, hx, hz, sector, period, bound)
        outcome = run_search(binary, path)
        if outcome["status"] == "witness":
            if bound == bounds[0]:
                raise RuntimeError(f"{sector}: a logical of weight <= {bound}; start lower")
            if len(outcome["witness"]) != bound:
                raise RuntimeError(
                    f"{sector}: witness of weight {len(outcome['witness'])} != {bound}"
                )
            return dict(distance=bound, witness=sorted(outcome["witness"]))
        complete = f"PROVED_NO_LOGICAL_WEIGHT_LE {bound} first 0 last {n // period}"
        if outcome.get("summary") != complete:
            raise RuntimeError(f"{sector} bound {bound}: incomplete search")
    raise RuntimeError(f"no {sector} logical up to weight {bounds[-1]}")
