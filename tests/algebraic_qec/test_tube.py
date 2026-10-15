"""Tube factors D: ker D = R_l t with t_q = 1, and selected factors A'."""

import pytest

from algebraic_qec import CyclicPolynomial as P
from algebraic_qec import PolynomialMatrix, RingDecomposition, gf2_rank
from algebraic_qec.examples.surgery_instances import SURGERY_INSTANCES
from algebraic_qec.tube import (
    component_ranks,
    extend_factor,
    maximal_minors,
    tube_kernel_generator,
    unit_columns,
)

TUBES = [(c.factor_d, c.couple_column) for c in SURGERY_INSTANCES]


@pytest.mark.parametrize("d, q", TUBES, ids=["l9", "l15", "l33"])
def test_tube_kernel_is_free_of_rank_one_with_unit_coordinate(d, q):
    dec, (m, n) = RingDecomposition(d.period), d.shape
    assert component_ranks(d, dec) == [m] * len(dec.factors)
    minors = maximal_minors(d)
    assert not dict((d @ minors).nonzero_entries()) and q in unit_columns(minors, dec)
    t = tube_kernel_generator(d, q, dec)
    assert not dict((d @ t).nonzero_entries()) and t[q, 0] == P.one(d.period)
    assert n * d.period - gf2_rank(d.lift()) == d.period  # binary kernel: one fibre R_l t


def test_tube_dropping_rank_is_rejected():
    # A dense monomial D has rank one in the component 1 + x (every monomial is 1 there).
    d = PolynomialMatrix.from_monomials(9, [[0, 1, 2], [0, 2, 4]])
    assert component_ranks(d)[0] == 1
    with pytest.raises(ValueError, match="free of rank one"):
        tube_kernel_generator(d, 0)


def test_extend_factor_appends_selection_rows():
    a = PolynomialMatrix.from_monomials(9, [[0, 0, 0], [0, 1, 2]])
    extended = extend_factor(a, [(0, 0, 1), (0b11, 0, 0)])
    assert (
        extended.shape == (4, 3) and extended[2, 2] == P.one(9) and extended[3, 0] == P(9, [0, 1])
    )
