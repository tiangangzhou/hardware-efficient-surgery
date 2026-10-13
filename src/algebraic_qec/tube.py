"""Tube factors D and selected factors A' of the auxiliary code LP(A', D).

SM "Auxiliary LP code construction": the surgery needs ker D = R_l t with a unit coordinate
t_q, normalised to t_q = 1; then ker d1 = ker A' (x) t and evaluation at q recovers ker A'
(Eq. (sm-tube-kernel)).  For an m x (m+1) factor D, ker D is free of rank one iff D has rank
m in every CRT component.  The maximal minors Delta_j (D with column j deleted; the Cramer
signs vanish over F2) then span ker D, t_q is a unit iff Delta_q is, and t = Delta / Delta_q.
SM "Fibre and CRT-packet selection": A' = [A; H_L] appends logical-selection rows to A.
"""

from __future__ import annotations

import itertools

from algebraic_qec.examples.surgery_instances import instance
from algebraic_qec.polymatrix import PolynomialMatrix
from algebraic_qec.polynomial import CyclicPolynomial
from algebraic_qec.ring_kernel import RingDecomposition


def maximal_minors(factor: PolynomialMatrix) -> PolynomialMatrix:
    """The column vector (Delta_0, ..., Delta_m) of an m x (m+1) matrix D; D Delta = 0."""
    rows, cols = factor.shape
    if cols != rows + 1:
        raise ValueError(f"maximal minors need an m x (m+1) matrix, got {factor.shape}")
    minors = {}
    for deleted in range(cols):
        kept = [j for j in range(cols) if j != deleted]
        total = CyclicPolynomial.zero(factor.period)
        for permutation in itertools.permutations(kept):
            term = CyclicPolynomial.one(factor.period)
            for i, j in enumerate(permutation):
                term = term * factor[i, j]
            total = total + term
        minors[deleted, 0] = total
    return PolynomialMatrix(factor.period, (cols, 1), minors)


def component_ranks(
    factor: PolynomialMatrix, decomposition: RingDecomposition | None = None
) -> list[int]:
    """Rank of D mod g_c over E_c for every CRT component c."""
    dec = decomposition or RingDecomposition(factor.period)
    return [factor.shape[1] - len(dec.component_kernel(factor, c)) for c in range(len(dec.factors))]


def unit_columns(
    vector: PolynomialMatrix, decomposition: RingDecomposition | None = None
) -> list[int]:
    """Coordinates j whose entry is a unit of R_l, i.e. nonzero in every CRT component."""
    dec = decomposition or RingDecomposition(vector.period)
    return sorted(
        j
        for (j, _), poly in vector.nonzero_entries()
        if all(dec.project(poly, c) for c in range(len(dec.factors)))
    )


def tube_kernel_generator(
    factor: PolynomialMatrix, couple_column: int, decomposition: RingDecomposition | None = None
) -> PolynomialMatrix:
    """The generator t of ker D = R_l t with t_q = 1 (q = couple_column): Delta / Delta_q."""
    dec = decomposition or RingDecomposition(factor.period)
    if component_ranks(factor, dec) != [factor.shape[0]] * len(dec.factors):
        raise ValueError("ker D is not free of rank one: D drops rank in a CRT component")
    minors = maximal_minors(factor)
    scale = dec.inverse(minors[couple_column, 0])
    return minors @ PolynomialMatrix(factor.period, (1, 1), {(0, 0): scale})


def lp33_tube_factor() -> PolynomialMatrix:
    """The l = 33 tube used by the circuit benchmark: the 4 x 5 D of instance III (SM Eq.
    (s-lp33-seed-tube)) with columns reordered as (2, 0, 4, 3, 1), so that it couples at its
    column 0 (= SM column q = 2).  This only relabels auxiliary qubits."""
    d = instance(33).factor_d
    order = (2, 0, 4, 3, 1)
    entries = {(i, k): d[i, j] for i in range(d.shape[0]) for k, j in enumerate(order)}
    return PolynomialMatrix(33, d.shape, entries)


def extend_factor(base: PolynomialMatrix, rows: list[tuple[int, ...]]) -> PolynomialMatrix:
    """A' = [A; H_L] with H_L given row by row as F2[x] polynomials stored as ints."""
    selection = {
        (i, j): CyclicPolynomial.from_bits(base.period, bits)
        for i, row in enumerate(rows)
        for j, bits in enumerate(row)
    }
    return base.vstack(PolynomialMatrix(base.period, (len(rows), base.shape[1]), selection))
