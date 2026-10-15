"""Helpers shared by the scripts in this directory.

Binary vectors are ints (bit i = qubit i), as in :mod:`algebraic_qec.bitmask`.  Notation
follows SM "Fibre and CRT-packet selection": g_S = prod_{c in S} g_c, e_S = sum_{c in S} eps_c,
and K_res = {z in ker A : z_r = 0 for r in J}.
"""

from __future__ import annotations

from pathlib import Path

from algebraic_qec import CyclicPolynomial, PolynomialMatrix, RingDecomposition
from algebraic_qec.bitmask import Rowspace, nullspace_basis, row_masks

DATA = Path(__file__).resolve().parents[2] / "data" / "codes"

if not __debug__:  # the scripts here check their results with assert statements
    raise SystemExit("scripts/codes verify with assert statements: run them without -O")


def column(polys: list[CyclicPolynomial]) -> PolynomialMatrix:
    """The column vector with these entries."""
    entries = {(i, 0): p for i, p in enumerate(polys)}
    return PolynomialMatrix(polys[0].period, (len(polys), 1), entries)


def kernel_space(matrix: PolynomialMatrix) -> Rowspace:
    """The binary kernel of the lift of a matrix over R_l."""
    return Rowspace(nullspace_basis(row_masks(matrix.lift()), matrix.shape[1] * matrix.period))


def readout_images(d1, f1) -> list[int]:
    """f1 ker(d1) for binary d1, f1: the data Z operators measured by the surgery, Eq.
    (css-surgery-readout), one per kernel basis vector."""
    rows = row_masks(f1)
    cycles = nullspace_basis(row_masks(d1), d1.shape[1])
    return [sum(((r & v).bit_count() & 1) << i for i, r in enumerate(rows)) for v in cycles]


def g_S(dec: RingDecomposition, subset) -> CyclicPolynomial:
    """g_S in R_l (zero when S contains every component)."""
    g = CyclicPolynomial.one(dec.period)
    for c in subset:
        g = g * CyclicPolynomial.from_bits(dec.period, dec.factors[c])
    return g


def e_S(dec: RingDecomposition, subset) -> CyclicPolynomial:
    """e_S = sum_{c in S} eps_c."""
    return sum((dec.lift(1, c) for c in subset), CyclicPolynomial.zero(dec.period))


def residuals(ell: int, scale: CyclicPolynomial | None = None) -> list[PolynomialMatrix]:
    """Generators scale * eps_0 (e_0 + e_r), r = 1, 2, of K_res for the paper's seeds (SM Eq.
    (s-lp9-carving)); eps_0 = sum_a x^a."""
    eps0 = CyclicPolynomial(ell, range(ell))
    eps0 = eps0 if scale is None else eps0 * scale
    zero = CyclicPolynomial.zero(ell)
    return [column([eps0, *(eps0 if c == r else zero for c in range(1, 5))]) for r in (1, 2)]
