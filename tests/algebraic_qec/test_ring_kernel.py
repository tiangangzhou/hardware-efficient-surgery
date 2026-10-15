"""The CRT decomposition of R_l and kernels over R_l."""

import random

import numpy as np
import pytest

from algebraic_qec import CyclicPolynomial as P
from algebraic_qec import PolynomialMatrix, RingDecomposition, gf2_rank, kernel
from algebraic_qec.examples.surgery_instances import lifted_product_code
from tests.algebraic_qec.fixtures import LP_CODES


@pytest.mark.parametrize("ell", [3, 9, 15, 33])
def test_idempotents(ell):
    dec = RingDecomposition(ell)
    eps = [dec.lift(1, c) for c in range(len(dec.factors))]
    assert sum(dec.component_degrees) == ell
    assert sum(eps, P.zero(ell)) == P.one(ell)
    for c, e in enumerate(eps):
        assert e * e == e
        assert [dec.project(e, d) for d in range(len(eps))] == [
            int(c == d) for d in range(len(eps))
        ]


def test_crt_round_trip_and_inverse():
    dec = RingDecomposition(15)
    p = P(15, [0, 1, 3])  # 1 + x + x^3 is coprime to x^15 + 1, hence a unit
    components = [dec.project(p, c) for c in range(len(dec.factors))]
    assert sum((dec.lift(v, c) for c, v in enumerate(components)), P.zero(15)) == p
    assert p * dec.inverse(p) == P.one(15) and dec.inverse(P(15, [4])) == P(15, [11])
    with pytest.raises(ZeroDivisionError):
        dec.inverse(P(15, [0, 1]))  # 1 + x vanishes in the component 1 + x


def test_even_lift_is_rejected():
    with pytest.raises(ValueError, match="squarefree"):
        RingDecomposition(4)


def test_annihilator_of_a_zero_divisor():
    # 1 + x vanishes only in the component F2[x]/(1 + x), whose idempotent is 1 + x + x^2.
    (generator,) = kernel(PolynomialMatrix(3, (1, 1), {(0, 0): P(3, [0, 1])}))
    assert generator[0, 0] == P(3, [0, 1, 2])


def assert_kernel_matches_binary(matrix: PolynomialMatrix) -> None:
    """The cyclic shifts of the generators span the kernel of the binary lift."""
    generators = kernel(matrix)
    binary = matrix.lift()
    assert all(not ((binary @ g.lift()).toarray() % 2).any() for g in generators)
    shifts = [g.lift().toarray().T for g in generators]  # rows: binary forms of x^-a g
    span = gf2_rank(np.vstack(shifts)) if shifts else 0
    assert span == matrix.shape[1] * matrix.period - gf2_rank(binary)


@pytest.mark.parametrize("name", ["l9", "l15", "l33", "lp^{3,7}_{16}"])
def test_kernel_of_lp_checks(name):
    ell, seed, _ = LP_CODES[name]
    assert_kernel_matches_binary(lifted_product_code(ell, seed).matrix_x)


def test_kernel_of_random_matrices():
    rng = random.Random(42)
    for _ in range(20):
        ell = rng.choice([3, 5, 7, 9, 11, 15])
        rows, cols = rng.randint(1, 3), rng.randint(1, 5)
        entries = {
            (i, j): P(ell, [e for e in range(ell) if rng.random() < 0.3])
            for i in range(rows)
            for j in range(cols)
            if rng.random() < 0.5
        }
        assert_kernel_matches_binary(PolynomialMatrix(ell, (rows, cols), entries))
