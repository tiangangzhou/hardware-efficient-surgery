"""Lifted-product codes LP(A, A^dagger) over R_l (SM Eq. (lp-checks) with B = A^dagger).

For an r x n seed A over R_l,

    H_X = (A (x) I_n | I_r (x) A^dagger),   H_Z = (I_n (x) A | A^dagger (x) I_r),

with qubits ordered as (A_1 (x) B_0) + (A_0 (x) B_1) and H_X H_Z^dagger = A (x) A^dagger +
A (x) A^dagger = 0.  The code has l (n^2 + r^2) qubits, and k = (n - r)^2 l when A has rank
r in every CRT component (more otherwise, by the Kunneth formula, Eq. (kunneth)).

``CAIN_LP_CODES`` are the four LP codes of Cain et al., arXiv:2603.28627, Appendix A; their
``distance_bound`` values are upper bounds quoted there.  The first is the data code of
instance III.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from algebraic_qec.css import CPMCSSCode
from algebraic_qec.polymatrix import PolynomialMatrix


def lifted_product_code(period: int, seed: Sequence[Sequence[int | None]]) -> CPMCSSCode:
    """LP(A, A^dagger) for the monomial seed A with entries x^e (None for 0)."""
    a = PolynomialMatrix.from_monomials(period, seed)
    rows, cols = a.shape
    eye_r, eye_n = PolynomialMatrix.identity(period, rows), PolynomialMatrix.identity(period, cols)
    a_dag = a.adjoint()
    matrix_x = a.kron(eye_n).hstack(eye_r.kron(a_dag))
    matrix_z = eye_n.kron(a).hstack(a_dag.kron(eye_r))
    return CPMCSSCode(matrix_x, matrix_z)


@dataclass(frozen=True)
class LPCodeSpec:
    """A named LP(A, A^dagger) code."""

    name: str
    period: int
    seed: tuple[tuple[int, ...], ...]
    distance_bound: int

    def build(self) -> CPMCSSCode:
        return lifted_product_code(self.period, self.seed)


CAIN_LP_CODES: tuple[LPCodeSpec, ...] = (
    LPCodeSpec(
        "lp^{3,5}_{20}",
        33,
        ((0, 0, 0, 0, 0), (0, 14, 19, 11, 26), (0, 13, 2, 15, 21)),
        20,
    ),
    LPCodeSpec(
        "lp^{3,7}_{16}",
        45,
        (
            (29, 21, 31, 15, 37, 25, 27),
            (13, 25, 19, 26, 11, 18, 29),
            (31, 2, 27, 32, 41, 41, 18),
        ),
        16,
    ),
    LPCodeSpec(
        "lp^{3,7}_{20}",
        75,
        ((0, 71, 73, 68, 33, 50, 47), (38, 39, 60, 26, 18, 1, 23), (73, 6, 5, 42, 20, 22, 73)),
        20,
    ),
    LPCodeSpec(
        "lp^{3,7}_{24}",
        91,
        ((57, 75, 42, 80, 7, 67, 27), (57, 73, 34, 12, 27, 50, 87), (21, 53, 70, 18, 1, 3, 18)),
        24,
    ),
)
