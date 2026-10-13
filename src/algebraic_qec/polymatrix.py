"""Sparse matrices over R_l = F2[x]/(x^l + 1) and their binary lifts.

``lift`` replaces each entry p = sum_s x^s by the l x l block sum_s P(s), P(s)[a, a + s] = 1
(see :mod:`algebraic_qec.polynomial`): binary row i*l + a and column j*l + b belong to
block (i, j) and sheets a, b.  Lifting preserves sums and products and sends ``adjoint``
(transpose with x -> x^-1, the SM's dagger) to the binary transpose.  ``kron`` is the
Kronecker product over R_l with the usual index order (i1 * rows2 + i2, j1 * cols2 + j2),
the tensor-product ordering of the SM's LP checks.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence

import numpy as np
import scipy.sparse

from algebraic_qec.polynomial import CyclicPolynomial


class PolynomialMatrix:
    """A rows x cols matrix over F2[x]/(x^period + 1), storing only nonzero entries."""

    def __init__(
        self,
        period: int,
        shape: tuple[int, int],
        entries: Mapping[tuple[int, int], CyclicPolynomial] | None = None,
    ) -> None:
        rows, cols = shape
        self.period = period
        self.shape = (rows, cols)
        self._entries: dict[tuple[int, int], CyclicPolynomial] = {}
        for (i, j), poly in (entries or {}).items():
            self._check_index(i, j)
            if poly.period != period:
                raise ValueError(f"entry period {poly.period} != matrix period {period}")
            if poly:
                self._entries[i, j] = poly

    @classmethod
    def zero(cls, period: int, rows: int, cols: int) -> PolynomialMatrix:
        return cls(period, (rows, cols))

    @classmethod
    def identity(cls, period: int, size: int) -> PolynomialMatrix:
        one = CyclicPolynomial.one(period)
        return cls(period, (size, size), {(i, i): one for i in range(size)})

    @classmethod
    def from_monomials(
        cls, period: int, exponents: Sequence[Sequence[int | None]]
    ) -> PolynomialMatrix:
        """Matrix with entry x^e for each exponent e, and 0 for None."""
        shape = (len(exponents), len(exponents[0]) if exponents else 0)
        entries = {
            (i, j): CyclicPolynomial.monomial(period, e)
            for i, row in enumerate(exponents)
            for j, e in enumerate(row)
            if e is not None
        }
        return cls(period, shape, entries)

    def _check_index(self, i: int, j: int) -> None:
        if not (0 <= i < self.shape[0] and 0 <= j < self.shape[1]):
            raise IndexError(f"index ({i}, {j}) out of range for shape {self.shape}")

    def __getitem__(self, key: tuple[int, int]) -> CyclicPolynomial:
        self._check_index(*key)
        return self._entries.get(key, CyclicPolynomial.zero(self.period))

    def nonzero_entries(self) -> Iterator[tuple[tuple[int, int], CyclicPolynomial]]:
        """Pairs ((i, j), entry) over the nonzero entries."""
        return iter(self._entries.items())

    def __add__(self, other: PolynomialMatrix) -> PolynomialMatrix:
        if (self.period, self.shape) != (other.period, other.shape):
            raise ValueError(f"cannot add {self} and {other}")
        entries = dict(self._entries)
        for key, poly in other._entries.items():
            entries[key] = entries[key] + poly if key in entries else poly
        return PolynomialMatrix(self.period, self.shape, entries)

    def __matmul__(self, other: PolynomialMatrix) -> PolynomialMatrix:
        if self.period != other.period or self.shape[1] != other.shape[0]:
            raise ValueError(f"cannot multiply {self} and {other}")
        rows_of_other: dict[int, list[tuple[int, CyclicPolynomial]]] = {}
        for (k, j), poly in other._entries.items():
            rows_of_other.setdefault(k, []).append((j, poly))
        entries: dict[tuple[int, int], CyclicPolynomial] = {}
        for (i, k), left in self._entries.items():
            for j, right in rows_of_other.get(k, []):
                term = left * right
                entries[i, j] = entries[i, j] + term if (i, j) in entries else term
        return PolynomialMatrix(self.period, (self.shape[0], other.shape[1]), entries)

    def adjoint(self) -> PolynomialMatrix:
        """Transpose with entrywise x -> x^-1; lifts to the binary transpose."""
        entries = {(j, i): poly.adjoint() for (i, j), poly in self._entries.items()}
        return PolynomialMatrix(self.period, self.shape[::-1], entries)

    def _with(self, other: PolynomialMatrix, shape, offset) -> PolynomialMatrix:
        """``self`` and ``other`` in a matrix of the given shape, ``other`` shifted by offset."""
        if self.period != other.period:
            raise ValueError(f"period mismatch: {self.period} != {other.period}")
        entries = dict(self._entries)
        for (i, j), poly in other._entries.items():
            entries[i + offset[0], j + offset[1]] = poly
        return PolynomialMatrix(self.period, shape, entries)

    def hstack(self, other: PolynomialMatrix) -> PolynomialMatrix:
        """[self | other]."""
        (rows, cols), (rows2, cols2) = self.shape, other.shape
        if rows != rows2:
            raise ValueError(f"cannot hstack {self} and {other}")
        return self._with(other, (rows, cols + cols2), (0, cols))

    def vstack(self, other: PolynomialMatrix) -> PolynomialMatrix:
        """[self; other]."""
        (rows, cols), (rows2, cols2) = self.shape, other.shape
        if cols != cols2:
            raise ValueError(f"cannot vstack {self} and {other}")
        return self._with(other, (rows + rows2, cols), (rows, 0))

    def block_diag(self, other: PolynomialMatrix) -> PolynomialMatrix:
        """[[self, 0], [0, other]]."""
        (rows, cols), (rows2, cols2) = self.shape, other.shape
        return self._with(other, (rows + rows2, cols + cols2), (rows, cols))

    def kron(self, other: PolynomialMatrix) -> PolynomialMatrix:
        """Kronecker product self (x) other over R_l."""
        if self.period != other.period:
            raise ValueError(f"period mismatch: {self.period} != {other.period}")
        (rows, cols), (rows2, cols2) = self.shape, other.shape
        entries = {
            (i1 * rows2 + i2, j1 * cols2 + j2): p1 * p2
            for (i1, j1), p1 in self._entries.items()
            for (i2, j2), p2 in other._entries.items()
        }
        return PolynomialMatrix(self.period, (rows * rows2, cols * cols2), entries)

    def lift(self) -> scipy.sparse.csr_array:
        """Binary (rows*l) x (cols*l) lift: entry x^s of block (i, j) puts a 1 at
        (i*l + a, j*l + (a + s) mod l) for every sheet a."""
        period, sheets = self.period, np.arange(self.period)
        rows, cols = [], []
        for (i, j), poly in self._entries.items():
            for s in poly.exponents:
                rows.append(i * period + sheets)
                cols.append(j * period + (sheets + s) % period)
        shape = (self.shape[0] * period, self.shape[1] * period)
        if not rows:
            return scipy.sparse.csr_array(shape, dtype=np.uint8)
        rows, cols = np.concatenate(rows), np.concatenate(cols)
        return scipy.sparse.csr_array((np.ones(len(rows), np.uint8), (rows, cols)), shape=shape)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PolynomialMatrix):
            return NotImplemented
        return (self.period, self.shape, self._entries) == (
            other.period,
            other.shape,
            other._entries,
        )

    def __repr__(self) -> str:
        return f"PolynomialMatrix(period={self.period}, shape={self.shape})"
