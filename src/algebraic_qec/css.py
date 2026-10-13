"""CSS codes with check matrices over R_l = F2[x]/(x^l + 1) (SM "CSS surgery as a mapping cone").

H_X and H_Z are matrices over R_l with the same number of block columns.  The CSS condition
H_X H_Z^dagger = 0 over R_l is equivalent to H_X H_Z^T = 0 for the binary lifts, since
lifting preserves products and sends the adjoint to the transpose.
"""

from __future__ import annotations

from functools import cached_property

import scipy.sparse

from algebraic_qec.bitmask import gf2_rank
from algebraic_qec.polymatrix import PolynomialMatrix


class CPMCSSCode:
    """The CSS code with checks H_X = matrix_x, H_Z = matrix_z, lifted to n = l * columns qubits."""

    def __init__(self, matrix_x: PolynomialMatrix, matrix_z: PolynomialMatrix) -> None:
        if matrix_x.period != matrix_z.period or matrix_x.shape[1] != matrix_z.shape[1]:
            raise ValueError(f"incompatible check matrices {matrix_x} and {matrix_z}")
        if any((matrix_x @ matrix_z.adjoint()).nonzero_entries()):
            raise ValueError("check matrices do not commute: H_X H_Z^dagger != 0")
        self.matrix_x = matrix_x
        self.matrix_z = matrix_z

    @property
    def period(self) -> int:
        return self.matrix_x.period

    @property
    def n(self) -> int:
        return self.matrix_x.shape[1] * self.period

    @cached_property
    def binary_matrix_x(self) -> scipy.sparse.csr_array:
        return self.matrix_x.lift()

    @cached_property
    def binary_matrix_z(self) -> scipy.sparse.csr_array:
        return self.matrix_z.lift()

    @cached_property
    def k(self) -> int:
        """n - rank H_X - rank H_Z over GF(2)."""
        return self.n - gf2_rank(self.binary_matrix_x) - gf2_rank(self.binary_matrix_z)
