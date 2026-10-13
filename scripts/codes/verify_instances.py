"""Merged codes of algebraic fibre surgery for instances I-III (lifts 9, 15, 33).

SM "Auxiliary LP code construction" and "Three code examples".  For each instance of
:mod:`algebraic_qec.examples.surgery_instances` this checks:

* the tube D has monomial entries, zeros and uniform row weight, and the cyclic-diagonal
  gate schedule (target orders 123, 234, 341, counting from one) has no conflicts, also
  after lifting;
* ker D = R t with t_q = 1: kernel dimension one in every CRT component and a unit minor;
* H_X f1 = f0 d1 and d0 d1 = 0, so the merged checks commute;
* the readout f1 ker(d1) is the carved fibre (l canonical plus two residual operators),
  with no readout relations, independent modulo the data Z stabilizers;
* merged check ranks (237, 307), (399, 517), (1017, 1312), so k = 41, 59, 113;
* the logical operators in ``data/codes/witnesses.json`` are nontrivial: d_X <= 9, 11, 21
  and d_Z <= 8, 10, 20.

``--exhaustive`` runs the search of :mod:`algebraic_qec.distance` (needs ``c++``): exact
sector distances (d_X, d_Z) = (9, 8) and (11, 10) at l = 9, 15, hence merged distances 8
and 10.  With ``--ell 33`` it proves only that no logical of weight < 20 exists (about two
minutes; the paper states d <= 20).

    uv run python scripts/codes/verify_instances.py               # ~5 s
    uv run python scripts/codes/verify_instances.py --exhaustive  # ~10 s more
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from common import DATA, readout_images

from algebraic_qec import CyclicPolynomial, PolynomialMatrix, RingDecomposition
from algebraic_qec.bitmask import Rowspace, mask, nullspace_basis, row_masks, set_bits
from algebraic_qec.distance import compile_search, run_search, sector_distance, write_input
from algebraic_qec.examples.surgery_instances import SURGERY_INSTANCES, SurgeryInstance
from algebraic_qec.tube import component_ranks, maximal_minors, tube_kernel_generator

MERGED_RANKS = {9: (237, 307), 15: (399, 517), 33: (1017, 1312)}
ROW_WEIGHT = {9: 3, 15: 3, 33: 4}
AUXILIARY_QUBITS = {9: 279, 15: 465, 33: 1320}
EXACT_DISTANCES = {9: dict(X=9, Z=8), 15: dict(X=11, Z=10)}


def check_tube(case: SurgeryInstance, dec: RingDecomposition) -> None:
    """Sparse monomial D with a conflict-free schedule, and ker D = R t with t_q = 1."""
    ell, d, q = case.ell, case.factor_d, case.couple_column
    m, n = d.shape
    entries = dict(d.nonzero_entries())
    assert n == m + 1 and 0 < len(entries) < m * n
    assert all(p.weight == 1 for p in entries.values()), "every nonzero entry is a monomial"
    assert [sum(d[i, j].weight for j in range(n)) for i in range(m)] == [ROW_WEIGHT[ell]] * m
    # Round s of check row i targets tube column (i + s) mod n: each row visits its support,
    # and the gate endpoints of one round (x^e maps sheet a to a + e) are distinct qubits.
    schedule = [[(i + s) % n for s in range(n - 1)] for i in range(m)]
    assert all(sorted(row) == [j for j in range(n) if d[i, j]] for i, row in enumerate(schedule))
    for layer in zip(*schedule):
        shifts = [next(iter(d[i, j].exponents)) for i, j in enumerate(layer)]
        ends = {j * ell + (a + e) % ell for j, e in zip(layer, shifts) for a in range(ell)}
        assert len(set(layer)) == m and len(ends) == m * ell

    assert component_ranks(d, dec) == [m] * len(dec.factors), "kernel dimension one per component"
    minor = maximal_minors(d)[q, 0]
    assert all(dec.project(minor, c) for c in range(len(dec.factors))), "Delta_q is a unit"
    t = tube_kernel_generator(d, q, dec)
    assert d @ t == PolynomialMatrix.zero(ell, m, 1) and t[q, 0] == CyclicPolynomial.one(ell)


def check_readout(case: SurgeryInstance, data, maps) -> None:
    """f1 ker(d1) equals the carved fibre ker A' placed on cokernel coordinate j."""
    ell, (_, j) = case.ell, case.fibre
    images = readout_images(maps["d1"].lift(), maps["f1"].lift())
    fibre = nullspace_basis(row_masks(case.selected_factor().lift()), 5 * ell)
    # Bit (r, a) of ker A' sits on data qubit (5 r + j, a).
    intended = [mask(((b // ell) * 5 + j) * ell + b % ell for b in set_bits(v)) for v in fibre]
    image = Rowspace(images)
    assert len(intended) == image.rank == len(images) == ell + 2, "no readout relations"
    assert all(image.contains(v) for v in intended), "the image is the intended fibre"
    data_z = row_masks(data.binary_matrix_z)
    assert Rowspace(data_z + images).rank - Rowspace(data_z).rank == ell + 2


def check_witnesses(case: SurgeryInstance, data, hx, hz, witnesses: dict) -> None:
    """Nontrivial logical operators of the merged code; the Z one also of the data code."""
    for sector, (checks, stabilizers) in dict(X=(hz, hx), Z=(hx, hz)).items():
        word = mask(witnesses[sector])
        assert all((word & r).bit_count() % 2 == 0 for r in row_masks(checks)), sector
        assert not Rowspace(row_masks(stabilizers)).contains(word), sector
    z = mask(witnesses["Z"])
    assert max(witnesses["Z"]) < data.n and len(witnesses["Z"]) == case.merged_parameters[2]
    assert all((z & r).bit_count() % 2 == 0 for r in row_masks(data.binary_matrix_x))
    assert not Rowspace(row_masks(data.binary_matrix_z)).contains(z)


def exhaustive(case: SurgeryInstance, hx, hz, out: Path, binary: Path) -> str:
    """Exact sector distances (l = 9, 15) or no logical of weight < d (l = 33)."""
    ell, (n, _, d) = case.ell, case.merged_parameters
    if ell in EXACT_DISTANCES:
        bounds = range(d - 1, d + 2)
        found = {s: sector_distance(hx, hz, s, ell, bounds, out, binary)["distance"] for s in "XZ"}
        assert found == EXACT_DISTANCES[ell]
        return f"exact (d_X, d_Z) = ({found['X']}, {found['Z']})"
    for sector in "XZ":
        path = out / f"l{ell}_{sector}.dat"
        write_input(path, hx, hz, sector, ell, d - 1)
        proved = f"PROVED_NO_LOGICAL_WEIGHT_LE {d - 1} first 0 last {n // ell}"
        assert run_search(binary, path, time_limit=3600).get("summary") == proved
    return f"no logical of weight < {d} in either sector"


def verify(case: SurgeryInstance, witnesses: dict) -> tuple:
    ell, dec = case.ell, RingDecomposition(case.ell)
    check_tube(case, dec)
    data, code, maps = case.data_code(), case.merged_code(), case.surgery_maps()
    d1, d0, f1, f0 = maps["d1"], maps["d0"], maps["f1"], maps["f0"]
    assert data.matrix_x @ f1 == f0 @ d1, "attachment is a chain map"
    assert d0 @ d1 == PolynomialMatrix.zero(ell, d0.shape[0], d1.shape[1])
    hx, hz = code.binary_matrix_x, code.binary_matrix_z
    assert not ((hx @ hz.T).toarray() % 2).any(), "merged checks commute"
    check_readout(case, data, maps)
    ranks = (Rowspace(row_masks(hx)).rank, Rowspace(row_masks(hz)).rank)
    assert ranks == MERGED_RANKS[ell] and (code.n, code.k) == case.merged_parameters[:2]
    assert code.k == code.n - sum(ranks) == data.k - (ell + 2), "no extra auxiliary logicals"
    assert code.n - data.n == AUXILIARY_QUBITS[ell]
    check_witnesses(case, data, hx, hz, witnesses)
    weights = {s: len(w) for s, w in witnesses.items()}
    print(f"l={ell}: [[{code.n},{code.k}]], ranks {ranks}, readout {ell}+2, witnesses {weights}")
    return hx, hz


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ell", type=int, choices=(9, 15, 33), help="verify one instance")
    parser.add_argument("--exhaustive", action="store_true", help="run the exhaustive search")
    args = parser.parse_args()
    witnesses = json.loads((DATA / "witnesses.json").read_text())["witnesses"]
    out = Path(tempfile.mkdtemp(prefix="exhaustive-")) if args.exhaustive else None
    binary = compile_search(out) if out else None
    for case in SURGERY_INSTANCES:
        if args.ell not in (None, case.ell):
            continue
        hx, hz = verify(case, witnesses[str(case.ell)])
        if binary and (case.ell in EXACT_DISTANCES or args.ell == 33):
            print(f"l={case.ell}: {exhaustive(case, hx, hz, out, binary)}", flush=True)
    print("All checks passed.")


if __name__ == "__main__":
    main()
