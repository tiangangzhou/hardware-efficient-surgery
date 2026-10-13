"""CRT-packet selection at l = 33 (SM "Fibre and CRT-packet selection", Table "Selected CRT
unions for l = 33").

For the seed A of instance III over R_33 this checks:

* the CRT factors g_0, ..., g_4 of x^33 + 1 (degrees 1, 2, 10, 10, 10) and, for the table,
  d_S = sum_{c in S} deg g_c and the weight of g_S = prod_{c in S} g_c;
* divisor unions: S_q = {c : g_c | x^q + 1} has g_{S_q} = x^q + 1 and its complement has
  g = (x^33 + 1)/(x^q + 1) = sum_a x^{aq}, for every proper divisor q of 33;
* the selectors H_{L,i}^fib and H_{L,i,S}^CRT for both canonical fibres i = 3, 4 and all 31
  nonempty S: ker A' = R e_S u_i + e_S K_res, of binary dimension d_S + 2 [0 in S], and
  l + 2 = 35 for the full fibre;
* the parallel selector (A; g_{S_3} e_3^T; g_{S_4} e_4^T) of SM "Parallel measurements with
  one auxiliary code", Eq. (high-rate-general-selector), for all 1023 nonempty packet
  selections T of both fibres: ker = K_T + K_res, of dimension k_T + 2.

Prints all 31 unions; rows of the SM table are marked with '*'.

    uv run python scripts/codes/crt_packets.py   # ~3 s
"""

from __future__ import annotations

import itertools

from common import column, e_S, g_S, kernel_space, residuals

from algebraic_qec import CyclicPolynomial, PolynomialMatrix, RingDecomposition
from algebraic_qec.bitmask import Rowspace, orbit_masks, set_bits
from algebraic_qec.canonical import systematic_kernel_generators
from algebraic_qec.examples.surgery_instances import instance

ELL, J = 33, (3, 4)
SUBSETS = [s for size in range(6) for s in itertools.combinations(range(5), size)]
# Table "Selected CRT unions for l = 33": S -> (d_S, exponents of g_S, wt(g_S)).
SM_TABLE = {
    (0,): (1, [0, 1], 2),
    (1,): (2, [0, 1, 2], 3),
    (2,): (10, [0, 3, 5, 7, 10], 5),
    (3,): (10, [0, 1, 5, 9, 10], 5),
    (4,): (10, list(range(11)), 11),
    (0, 1): (3, [0, 3], 2),
    (0, 4): (11, [0, 11], 2),
    (1, 2, 3): (22, [0, 11, 22], 3),
}


def check_table(dec: RingDecomposition) -> None:
    factors = [[0, 1], [0, 1, 2], [0, 3, 5, 7, 10], [0, 1, 5, 9, 10], list(range(11))]
    assert [set_bits(f) for f in dec.factors] == factors
    print(f"{'S':16s} {'d_S':>4s} {'wt(g_S)':>8s}  g_S")
    for subset in SUBSETS[1:]:
        g, d_s = g_S(dec, subset), sum(dec.component_degrees[c] for c in subset)
        if subset in SM_TABLE:
            assert SM_TABLE[subset] == (d_s, sorted(g.exponents), g.weight), subset
        terms = " + ".join(f"x^{e}" for e in sorted(g.exponents)) or "0"
        mark = "*" if subset in SM_TABLE else " "
        print(f"{mark}{str(set(subset)):15s} {d_s:4d} {g.weight:8d}  {terms}")
    for q in (1, 3, 11):
        # g_c divides x^q + 1 exactly when x^q + 1 vanishes in component c.
        divisor = CyclicPolynomial(ELL, [0, q])
        union = [c for c in range(5) if not dec.project(divisor, c)]
        assert g_S(dec, union) == divisor
        complement = [c for c in range(5) if c not in union]
        assert g_S(dec, complement) == CyclicPolynomial(ELL, range(0, ELL, q))
    print("Divisor unions: g_{S_q} = 1 + x^q and (x^33+1)/(x^q+1) = sum_a x^(aq) for q = 1, 3, 11.")


def selected_kernel(a, rows: dict, targets: list) -> int:
    """Check ker(A; rows) = span of the target vectors' orbits; rows maps (row, col) -> poly."""
    height = 1 + max((i for i, _ in rows), default=-1)
    space = kernel_space(a.vstack(PolynomialMatrix(ELL, (height, 5), rows)))
    words = [w for vector in targets for w in orbit_masks(vector)]
    assert space.rank == Rowspace(words).rank and all(space.contains(w) for w in words)
    return space.rank


def main() -> None:
    a, dec = instance(ELL).factor_a, RingDecomposition(ELL)
    assert dec.component_degrees == [1, 2, 10, 10, 10]
    check_table(dec)
    generators = systematic_kernel_generators(a, J, dec)
    scaled = lambda e, u: column([e * u[j, 0] for j in range(5)])
    one = CyclicPolynomial.one(ELL)
    for i, u in zip(J, generators):
        other = J[1] if i == J[0] else J[0]
        fibre, coordinates = {(0, other): one}, [r for r in range(5) if r != other]
        assert selected_kernel(a, fibre, [u, *residuals(ELL)]) == ELL + 2
        for subset in SUBSETS[1:]:
            rows = fibre | {(1 + k, r): g_S(dec, subset) for k, r in enumerate(coordinates)}
            e = e_S(dec, subset)
            expected = sum(dec.component_degrees[c] for c in subset) + 2 * (0 in subset)
            assert selected_kernel(a, rows, [scaled(e, u), *residuals(ELL, e)]) == expected
    print("Single-fibre selectors: ker A' = R e_S u_i + e_S K_res for i = 3, 4 and all 31 S.")

    count = 0
    for s3, s4 in itertools.product(SUBSETS, repeat=2):
        k = sum(dec.component_degrees[c] for c in s3 + s4)
        if k == 0:
            continue
        # The row g_S e_i^T is zero (omitted) for a complete selection S.
        rows = {(n, i): g_S(dec, s) for n, (i, s) in enumerate(zip(J, (s3, s4))) if len(s) < 5}
        targets = [scaled(e_S(dec, s), u) for u, s in zip(generators, (s3, s4)) if s]
        assert selected_kernel(a, rows, targets + residuals(ELL)) == k + 2
        count += 1
    assert count == 1023
    print(f"Parallel selectors: ker(A; H_T) = K_T + K_res for all {count} nonempty selections T.")
    print("All CRT-packet checks passed.")


if __name__ == "__main__":
    main()
