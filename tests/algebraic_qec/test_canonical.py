"""Aligned information sets, canonical generators u_i and the canonical logical pairs."""

import numpy as np
import pytest
import scipy.sparse

from algebraic_qec import CyclicPolynomial as P
from algebraic_qec import PolynomialMatrix, RingDecomposition, gf2_rank
from algebraic_qec.canonical import aligned_information_set, systematic_kernel_generators
from algebraic_qec.examples.surgery_instances import SURGERY_INSTANCES, instance


@pytest.mark.parametrize("case", SURGERY_INSTANCES, ids=lambda c: f"l{c.ell}")
def test_generators_are_normalized_kernel_vectors(case):
    a, dec = case.factor_a, RingDecomposition(case.ell)
    assert aligned_information_set(a, 2, dec) == (3, 4)
    for i, u in zip((3, 4), systematic_kernel_generators(a, (3, 4), dec)):
        assert not dict((a @ u).nonzero_entries())
        assert [u[j, 0] for j in (3, 4)] == [P(case.ell, [0] if i == j else []) for j in (3, 4)]


def test_l9_generators_match_the_sm():
    # SM Eq. (s-lp9-generators).
    a, dec = instance(9).factor_a, RingDecomposition(9)
    u3, u4 = systematic_kernel_generators(a, (3, 4), dec)
    assert [sorted(u3[j, 0].exponents) for j in range(5)] == [
        [2, 6, 8],
        [0, 3, 6, 8],
        [2, 3],
        [0],
        [],
    ]
    assert [sorted(u4[j, 0].exponents) for j in range(3)] == [
        [0, 1, 3, 7, 8],
        [1, 2, 3, 4, 7, 8],
        [2, 4],
    ]


def test_canonical_pairs_of_the_l9_code():
    """Z: x^a (u_i (x) e_j, 0), X: x^a (e_i (x) u_j, 0), i, j in J: 4 * 9 = 36 conjugate pairs
    (SM "Three code examples", l = 9), independent modulo the stabilizers."""
    case = instance(9)
    code, ell = case.data_code(), case.ell
    u = dict(zip((3, 4), systematic_kernel_generators(case.factor_a, (3, 4), RingDecomposition(9))))
    unit = {r: PolynomialMatrix(ell, (5, 1), {(r, 0): P.one(ell)}) for r in (3, 4)}

    def lifted(vectors):  # each 25-block vector placed in the left sector, all l shifts
        padded = [PolynomialMatrix(ell, (34, 1), dict(v.nonzero_entries())) for v in vectors]
        return scipy.sparse.hstack([v.lift() for v in padded]).toarray().astype(int)

    pairs = [(i, j) for i in (3, 4) for j in (3, 4)]
    z = lifted([u[i].kron(unit[j]) for i, j in pairs])
    x = lifted([unit[i].kron(u[j]) for i, j in pairs])
    assert not (code.binary_matrix_x @ z % 2).any() and not (code.binary_matrix_z @ x % 2).any()
    assert np.array_equal(x.T @ z % 2, np.eye(36))
    for checks, logicals in ((code.binary_matrix_z, z), (code.binary_matrix_x, x)):
        assert gf2_rank(np.vstack([checks.toarray(), logicals.T])) - gf2_rank(checks) == 36


def test_misaligned_components_raise():
    # Over R_3 = F2 x F4 the row (eps_0, eps_1) has its pivot in column 0 over F2 and in
    # column 1 over F4, so no column is free in both components.
    dec = RingDecomposition(3)
    matrix = PolynomialMatrix(3, (1, 2), {(0, 0): dec.lift(1, 0), (0, 1): dec.lift(1, 1)})
    with pytest.raises(ValueError, match="aligned information set"):
        aligned_information_set(matrix, 1, dec)
