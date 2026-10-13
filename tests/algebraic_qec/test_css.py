"""CSS codes over R_l: the commutation condition and k."""

import pytest

from algebraic_qec import CPMCSSCode
from algebraic_qec import CyclicPolynomial as P
from algebraic_qec import PolynomialMatrix as M


def bicycle(period: int, a: list[int], b: list[int]) -> CPMCSSCode:
    """H_X = (a | b), H_Z = (b^dagger | a^dagger): H_X H_Z^dagger = ab + ba = 0."""
    pa, pb = (M(period, (1, 1), {(0, 0): P(period, e)}) for e in (a, b))
    return CPMCSSCode(pa.hstack(pb), pb.adjoint().hstack(pa.adjoint()))


def test_lifted_checks_commute_and_k_counts_dependent_checks():
    code = bicycle(5, [0, 1], [2])
    assert code.n == 10
    assert not ((code.binary_matrix_x @ code.binary_matrix_z.T).toarray() % 2).any()
    assert code.k == 0
    # 1 + x + x^2 lifts to the all-ones 3 x 3 matrix, of rank one: k = 6 - 1 - 1.
    assert bicycle(3, [0, 1, 2], [0, 1, 2]).k == 4


def test_non_commuting_checks_raise():
    # (1 + x)(1 + x^2)^dagger = (1 + x)^2 = 1 + x^2 != 0 over R_3.
    hx, hz = M(3, (1, 1), {(0, 0): P(3, [0, 1])}), M(3, (1, 1), {(0, 0): P(3, [0, 2])})
    with pytest.raises(ValueError, match="commute"):
        CPMCSSCode(hx, hz)
