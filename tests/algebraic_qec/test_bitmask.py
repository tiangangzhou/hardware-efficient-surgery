"""GF(2) linear algebra on bit-packed vectors."""

import random

import numpy as np
import scipy.sparse

from algebraic_qec import CyclicPolynomial, PolynomialMatrix, gf2_rank
from algebraic_qec.bitmask import (
    Rowspace,
    logical_quotient_basis,
    nullspace_basis,
    orbit_masks,
    row_masks,
    set_bits,
    shift_mask,
    span_words,
)


def test_gf2_rank_reduces_entries_mod_two():
    assert gf2_rank(np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])) == 2
    assert gf2_rank(np.array([[2, 0], [0, 3]])) == 1
    assert gf2_rank(scipy.sparse.coo_array(([1, 1, 1], ([0, 0, 0], [0, 0, 1])), shape=(2, 3))) == 1
    matrix = scipy.sparse.csr_array(np.array([[2, 1]], dtype=np.uint8))
    assert gf2_rank(matrix) == 1 and matrix.data.tolist() == [2, 1]  # input not mutated


def test_rank_and_nullspace_on_random_matrices():
    rng = np.random.default_rng(1)
    for _ in range(10):
        matrix = rng.integers(0, 2, size=(7, 12))
        rows = row_masks(matrix)
        basis = nullspace_basis(rows, 12)
        assert Rowspace(rows).rank == gf2_rank(matrix) == 12 - len(basis)
        assert all((v & r).bit_count() % 2 == 0 for v in basis for r in rows)
        assert Rowspace(basis).rank == len(basis)


def test_logical_quotient_dimension():
    rng = np.random.default_rng(2)
    stabilizers = row_masks(rng.integers(0, 2, size=(3, 10)))
    checks = nullspace_basis(stabilizers, 10)[:4]  # orthogonal to the stabilizers
    logicals = logical_quotient_basis(checks, stabilizers, 10)
    assert len(logicals) == 10 - len(checks) - Rowspace(stabilizers).rank


def test_orbit_columns_are_cyclic_shifts():
    vector = PolynomialMatrix(11, (2, 1), {(0, 0): CyclicPolynomial(11, [0, 3])})
    masks = orbit_masks(vector)
    assert set_bits(masks[0]) == [0, 8]  # x^s sits on sheet -s in column 0 of the lift
    for s in random.Random(0).sample(range(11), 5):
        assert shift_mask(masks[0], s, 11, 2) == masks[s]


def test_span_words_enumerates_the_span_once():
    words = list(span_words([0b0011, 0b0110, 0b1000]))
    assert len(words) == len(set(words)) == 7 and 0 not in words
