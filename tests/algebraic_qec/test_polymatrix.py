"""Matrices over R_l and the lift convention P(s)[a, a + s] = 1."""

import numpy as np
import pytest

from algebraic_qec import CyclicPolynomial as P
from algebraic_qec import PolynomialMatrix as M


def dense(matrix: M) -> np.ndarray:
    return matrix.lift().toarray()


def test_entries_and_shape():
    m = M.from_monomials(5, [[0, None], [3, 1]])
    assert m.shape == (2, 2) and m[0, 0] == P.one(5) and m[1, 0] == P(5, [3])
    assert not m[0, 1] and dict(M(5, (1, 1), {(0, 0): P.zero(5)}).nonzero_entries()) == {}
    with pytest.raises(IndexError):
        m[2, 0]


def test_products_cancel_mod_two():
    assert M.from_monomials(5, [[1, 0]]) @ M.from_monomials(5, [[0], [2]]) == M(
        5, (1, 1), {(0, 0): P(5, [1, 2])}
    )
    assert not dict(
        (M.from_monomials(5, [[1, 1]]) @ M.from_monomials(5, [[0], [0]])).nonzero_entries()
    )


def test_stacking_and_kron_index_order():
    a, b = M.from_monomials(5, [[0, 1]]), M.from_monomials(5, [[2], [3]])
    assert a.hstack(a).shape == (1, 4) and a.vstack(a).shape == (2, 2)
    assert a.block_diag(b)[1, 2] == P(5, [2])
    k = a.kron(b)  # entry (i1 * 2 + i2, j1 * 1 + j2) = a[i1, j1] b[i2, j2]
    assert k.shape == (2, 2) and k[1, 1] == P(5, [4]) and k[0, 1] == P(5, [3])
    with pytest.raises(ValueError):
        a.hstack(b)


def test_lift_convention():
    lifted = dense(M.from_monomials(4, [[1, None], [None, 0]]))
    expected = np.zeros((8, 8), dtype=np.uint8)
    for a in range(4):
        expected[a, (a + 1) % 4] = expected[4 + a, 4 + a] = 1
    assert np.array_equal(lifted, expected)


def test_lift_is_a_ring_homomorphism_and_adjoint_is_transpose():
    a = M(6, (1, 2), {(0, 0): P(6, [0, 2]), (0, 1): P(6, [1])})
    b = M(6, (2, 1), {(0, 0): P(6, [3]), (1, 0): P(6, [0, 4])})
    assert np.array_equal(dense(a @ b), dense(a) @ dense(b) % 2)
    assert np.array_equal(dense(a + a), np.zeros((6, 12)))
    assert np.array_equal(dense(a.adjoint()), dense(a).T)
