"""Aligned information sets and canonical kernel generators u_i of a factor A.

SM "Extended LP construction and logical bases": in each CRT component c, J_c is the set of
non-pivot columns of RREF(A_c), A_c = A mod g_c (columns scanned left to right), and u_i^(c)
is the systematic kernel vector with (u_i^(c))_i' = delta_ii' for i' in J_c.  On the common
set J = cap_c J_c the glued vectors

    u_i = sum_c eps_c u_i^(c) in ker A,   i in J,

satisfy (u_i)_i' = delta_ii' for i, i' in J.  With the coordinate vectors e_j of coker B
they give the canonical Z classes q_ij = [(u_i (x) e_j, 0)] of LP(A, B), Eq. (logical-fibres),
and the dual X representatives (e_i (x) u_j, 0) when B = A^dagger.
"""

from __future__ import annotations

from algebraic_qec.polymatrix import PolynomialMatrix
from algebraic_qec.ring_kernel import RingDecomposition


def aligned_information_set(
    matrix: PolynomialMatrix, size: int, decomposition: RingDecomposition
) -> tuple[int, ...]:
    """The first ``size`` columns of J = cap_c J_c (for the paper's seeds J = (3, 4))."""
    common = set(range(matrix.shape[1]))
    for c in range(len(decomposition.factors)):
        common &= decomposition.component_kernel(matrix, c).keys()
    if len(common) < size:
        raise ValueError(f"no aligned information set of size {size}: J = {sorted(common)}")
    return tuple(sorted(common)[:size])


def systematic_kernel_generators(
    matrix: PolynomialMatrix, information_set: tuple[int, ...], decomposition: RingDecomposition
) -> tuple[PolynomialMatrix, ...]:
    """The column vectors u_i = sum_c eps_c u_i^(c) for i in the information set."""
    dec = decomposition
    kernels = [dec.component_kernel(matrix, c) for c in range(len(dec.factors))]
    generators = []
    for i in information_set:
        u = PolynomialMatrix.zero(matrix.period, matrix.shape[1], 1)
        for c, basis in enumerate(kernels):
            if i not in basis:
                raise ValueError(f"column {i} is a pivot column in CRT component {c}")
            entries = {(j, 0): dec.lift(v, c) for j, v in enumerate(basis[i]) if v}
            u += PolynomialMatrix(matrix.period, u.shape, entries)
        generators.append(u)
    return tuple(generators)
