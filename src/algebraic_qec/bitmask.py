"""GF(2) linear algebra with binary vectors stored as Python ints (bit i = coordinate i).

For lifted matrices bit block*l + sheet is the binary column of :meth:`PolynomialMatrix.lift`.
Rank, kernel and logical-quotient computations are Gaussian elimination by XOR of ints.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np
import scipy.sparse

from algebraic_qec.polymatrix import PolynomialMatrix


def set_bits(word: int) -> list[int]:
    """Indices of the set bits, ascending."""
    out = []
    while word:
        low = word & -word
        out.append(low.bit_length() - 1)
        word ^= low
    return out


def mask(support: Iterable[int]) -> int:
    """The vector with the given support."""
    word = 0
    for i in support:
        word |= 1 << int(i)
    return word


def row_masks(matrix) -> list[int]:
    """Rows of a dense or sparse integer matrix as vectors over GF(2) (entries taken mod 2)."""
    csr = scipy.sparse.csr_array(matrix, dtype=np.int64, copy=True)
    csr.sum_duplicates()
    odd = csr.data % 2 == 1
    return [
        mask(csr.indices[start:stop][odd[start:stop]])
        for start, stop in zip(csr.indptr[:-1], csr.indptr[1:])
    ]


class Rowspace:
    """Row space over GF(2), kept in echelon form {leading bit: row}."""

    def __init__(self, rows: Iterable[int] = ()) -> None:
        self.pivots: dict[int, int] = {}
        for row in rows:
            self.add(row)

    @property
    def rank(self) -> int:
        return len(self.pivots)

    def reduce(self, row: int) -> int:
        """Remainder of a row modulo the space."""
        while row and (pivot := self.pivots.get(row.bit_length() - 1)):
            row ^= pivot
        return row

    def add(self, row: int) -> None:
        if row := self.reduce(row):
            self.pivots[row.bit_length() - 1] = row

    def contains(self, row: int) -> bool:
        return not self.reduce(row)


def gf2_rank(matrix) -> int:
    """Rank over GF(2) of a dense or sparse integer matrix (entries taken mod 2)."""
    return Rowspace(row_masks(matrix)).rank


def nullspace_basis(rows: list[int], nbits: int) -> list[int]:
    """A basis of {v in GF(2)^nbits : |v & r| even for every row r}, one vector per non-pivot
    column of the reduced row echelon form."""
    pivots: dict[int, int] = {}  # fully reduced: no pivot row has another pivot's lead bit
    for row in rows:
        for lead, pivot in pivots.items():
            if row >> lead & 1:
                row ^= pivot
        if row:
            lead = row.bit_length() - 1
            for other in pivots:
                if pivots[other] >> lead & 1:
                    pivots[other] ^= row
            pivots[lead] = row
    basis = []
    for free in (c for c in range(nbits) if c not in pivots):
        basis.append(1 << free | mask(lead for lead, row in pivots.items() if row >> free & 1))
    return basis


def logical_quotient_basis(
    cycle_checks: list[int], stabilizers: list[int], nbits: int
) -> list[int]:
    """A basis of ker(cycle_checks) / rowspan(stabilizers), e.g. Z logicals ker H_X / im H_Z^T."""
    span = Rowspace(stabilizers)
    logicals = []
    for vector in nullspace_basis(cycle_checks, nbits):
        if not span.contains(vector):
            logicals.append(vector)
            span.add(vector)
    return logicals


def span_words(basis: list[int]) -> Iterator[int]:
    """Every nonzero vector of span(basis) once, in Gray-code order."""
    word = previous = 0
    for index in range(1, 1 << len(basis)):
        gray = index ^ (index >> 1)
        word ^= basis[(gray ^ previous).bit_length() - 1]
        previous = gray
        yield word


def shift_mask(word: int, shift: int, period: int, blocks: int) -> int:
    """Rotate the sheet index of every block of ``period`` bits by ``shift`` (sheet a -> a +
    shift): the binary form of x^-shift acting on a lifted vector."""
    shift %= period
    full = (1 << period) - 1
    out = 0
    for block in range(blocks):
        chunk = word >> (block * period) & full
        out |= ((chunk << shift | chunk >> (period - shift)) & full) << (block * period)
    return out


def orbit_masks(vector: PolynomialMatrix) -> list[int]:
    """The l binary columns of the lift of a column vector v; column a is the binary form of
    x^-a v (SM: "x^a times a polynomial column is represented by column -a of its lift")."""
    period = vector.period
    masks = [0] * period
    for (block, _), poly in vector.nonzero_entries():
        for s in poly.exponents:
            for sheet in range(period):
                masks[sheet] ^= 1 << (block * period + (sheet - s) % period)
    return masks
