"""Algebraic adapters between LP blocks with unequal lifts (SM "Algebraic adapters").

Equal-lift adapters are checked in ``worked_examples.py``.  This script checks the SM
example l_1 = 33, l_2 = 11, e = gcd(l_1, l_2) = 11, coupling fibre (3,3) of block 1 to
fibre (4,4) of block 2 with offset s:

* block 1 is the l = 33 seed of instance III, block 2 an l = 11 seed with the same aligned
  information set J = {3, 4} (``SEED_11``); each block carries its fibre selector and the
  period row (x^e + 1) e_i^T of Eq. (adapter-period-checks), and the binary link of
  Eq. (adapter-link-binary) couples the first e coefficients with offset s;
* the binary kernel of the joint factor is exactly Eq. (adapter-kernel-explicit),
  {(h_1 p u_{i_1}, h_2 x^s p u_{i_2}) : p in R_e} + K_res,1 + K_res,2 with h_mu = sum_r x^{re},
  of dimension e + 4;
* thickening with a binary repetition tube and attaching at the cokernel coordinates gives
  a commuting merged code whose readout is exactly this kernel on the two data fibres;
* for l_1 = l_2 = 33 (e = l, no period rows) the binary factor is the lift of the ring
  factor of Eq. (adapter-factor).

Beyond the SM: the lifted period row (x^11 + 1) e_3^T of the l = 33 block has 33 binary
rows of rank 22, and each of the 11 dependent rows adds one auxiliary logical qubit after
thickening; keeping a row basis (its first 22 rows) gives k = k_1 + k_2 - (e + 4).

    uv run python scripts/codes/adapters.py   # ~7 s
"""

from __future__ import annotations

import math

import numpy as np
import scipy.linalg
from common import column, readout_images, residuals

from algebraic_qec import CyclicPolynomial, PolynomialMatrix, RingDecomposition
from algebraic_qec.bitmask import Rowspace, mask, nullspace_basis, orbit_masks, row_masks, set_bits
from algebraic_qec.canonical import aligned_information_set, systematic_kernel_generators
from algebraic_qec.examples.surgery_instances import instance, lifted_product_code

J = (3, 4)
SEED_11 = ((0, 0, 0, 0, 0), (0, 6, 6, 0, 4), (0, 8, 7, 6, 4))  # aligned information set J
SEEDS = {33: instance(33).seed, 11: SEED_11}


class Block:
    """LP(A, A^dagger) over R_l with selected fibre (i, i), i in J."""

    def __init__(self, ell: int, i: int):
        self.ell, self.i = ell, i
        self.a = PolynomialMatrix.from_monomials(ell, SEEDS[ell])
        dec = RingDecomposition(ell)
        assert aligned_information_set(self.a, 2, dec) == J
        self.u = systematic_kernel_generators(self.a, J, dec)[J.index(i)]
        self.code = lifted_product_code(ell, SEEDS[ell])

    def rows(self, e: int) -> PolynomialMatrix:
        """A, the fibre selector e_r^T (r in J, r != i) and, if e < l, (x^e + 1) e_i^T."""
        one = CyclicPolynomial.one(self.ell)
        rows = self.a.vstack(PolynomialMatrix(self.ell, (1, 5), {(0, 7 - self.i): one}))
        if e == self.ell:
            return rows
        period = one + CyclicPolynomial.monomial(self.ell, e)
        return rows.vstack(PolynomialMatrix(self.ell, (1, 5), {(0, self.i): period}))


def joint_factor(blocks: tuple[Block, Block], shift: int, row_basis: bool = False):
    """Binary A'_ad (block rows, period rows, e link rows), e, and each block's row offset.

    With ``row_basis`` only the first l - e binary rows of each lifted period row are kept.
    """
    first, second = blocks
    e = math.gcd(first.ell, second.ell)
    local = [b.rows(e).lift().toarray() for b in blocks]
    if row_basis:
        local = [rows[: 5 * b.ell - e] if b.ell != e else rows for b, rows in zip(blocks, local)]
    # L_1 = e_{i_1}^T (I_e | 0),  L_2 = e_{i_2}^T (P_e(-s) | 0).
    link = np.zeros((e, 5 * (first.ell + second.ell)), np.uint8)
    for a in range(e):
        link[a, first.i * first.ell + a] = 1
        link[a, 5 * first.ell + second.i * second.ell + (a - shift) % e] = 1
    return np.vstack([scipy.linalg.block_diag(*local), link]), e, [0, len(local[0])]


def expected_kernel(blocks: tuple[Block, Block], shift: int, e: int) -> list[int]:
    """Eq. (adapter-kernel-explicit): joint period-e coefficients plus both residuals."""
    first, second = blocks

    def coefficient(b: Block, a: int) -> int:
        """Binary vector of x^a h u_i in block b, with h = sum_r x^{re}."""
        eta = CyclicPolynomial(b.ell, range(a, a + b.ell, e))
        return orbit_masks(column([eta * b.u[r, 0] for r in range(5)]))[0]

    out = [
        coefficient(first, a) | coefficient(second, a + shift) << 5 * first.ell for a in range(e)
    ]
    for b, offset in ((first, 0), (second, 5 * first.ell)):
        out += [orbit_masks(v)[0] << offset for v in residuals(b.ell)]
    return out


def check_pair(l1: int, l2: int, shift: int, row_basis: bool) -> tuple[int, int, int]:
    """Kernel, thickened merged code and readout; returns n, k and the extra logicals."""
    blocks = (Block(l1, 3), Block(l2, 4))
    factor, e, row_offsets = joint_factor(blocks, shift, row_basis)
    rows, cols = factor.shape
    kernel = Rowspace(nullspace_basis(row_masks(factor), cols))
    expected = expected_kernel(blocks, shift, e)
    assert kernel.rank == e + 4 == Rowspace(expected).rank
    assert all(kernel.contains(v) for v in expected)

    # Repetition tube D = (1 1), coupled at q = 0: ker D is spanned by (1, 1).
    eye = lambda size: np.eye(size, dtype=np.uint8)
    d = np.array([[1, 1]], np.uint8)
    d1 = np.vstack([np.kron(factor, eye(2)), np.kron(eye(cols), d)])
    d0 = np.hstack([np.kron(eye(rows), d), factor])
    assert not (d0.astype(int) @ d1 % 2).any()
    data_x = scipy.linalg.block_diag(*(b.code.binary_matrix_x.toarray() for b in blocks))
    data_z = scipy.linalg.block_diag(*(b.code.binary_matrix_z.toarray() for b in blocks))
    f1 = np.zeros((data_x.shape[1], d1.shape[1]), np.uint8)
    f0 = np.zeros((data_x.shape[0], d1.shape[0]), np.uint8)
    qubits = checks = col = 0
    for b, row in zip(blocks, row_offsets):
        # Factor column (r, a) -> data qubit (5r + i, a); A row (r, a) -> X check (5r + i, a).
        r, a = np.divmod(np.arange(5 * b.ell), b.ell)
        f1[qubits + (5 * r + b.i) * b.ell + a, 2 * (col + r * b.ell + a)] = 1
        r, a = np.divmod(np.arange(3 * b.ell), b.ell)
        f0[checks + (5 * r + b.i) * b.ell + a, 2 * (row + r * b.ell + a)] = 1
        qubits += b.code.n
        checks += b.code.binary_matrix_x.shape[0]
        col += 5 * b.ell
    assert (data_x.astype(int) @ f1 % 2 == f0.astype(int) @ d1 % 2).all()
    merged_x = np.block([[data_x, f0], [np.zeros((len(d0), data_x.shape[1]), np.uint8), d0]])
    merged_z = np.block([[data_z, np.zeros((len(data_z), len(d1)), np.uint8)], [f1.T, d1.T]])
    assert not (merged_x.astype(int) @ merged_z.T % 2).any(), "merged checks commute"

    images = readout_images(d1, f1)
    target = f1[:, ::2].argmax(axis=0)  # data qubit attached to each factor column
    image = Rowspace(images)
    assert image.rank == len(images) == e + 4, "readout equals the joint kernel, no relations"
    assert all(image.contains(mask(target[c] for c in set_bits(v))) for v in expected)
    stabilizers = row_masks(data_z)
    assert Rowspace(stabilizers + images).rank - Rowspace(stabilizers).rank == e + 4
    n = merged_x.shape[1]
    k = n - Rowspace(row_masks(merged_x)).rank - Rowspace(row_masks(merged_z)).rank
    return n, k, k - (blocks[0].code.k + blocks[1].code.k - (e + 4))


def check_equal_lift_reduction(shift: int) -> None:
    """At l_1 = l_2 = l the binary link is the lift of e_{i_1}^T + x^{-s} e_{i_2}^T."""
    blocks = (Block(33, 3), Block(33, 4))
    factor, e, _ = joint_factor(blocks, shift)
    link = {(0, 3): CyclicPolynomial.one(33), (0, 9): CyclicPolynomial.monomial(33, -shift)}
    ring = blocks[0].rows(e).block_diag(blocks[1].rows(e))
    ring = ring.vstack(PolynomialMatrix(33, (1, 10), link))
    assert e == 33 and row_masks(factor) == row_masks(ring.lift())


def main() -> None:
    for shift in (0, 1, 5):
        check_equal_lift_reduction(shift)
    print("Equal lifts: the binary link is the lifted ring link e_i1^T + x^-s e_i2^T.")
    for shift in (0, 1):
        for row_basis in (False, True):
            n, k, extra = check_pair(33, 11, shift, row_basis)
            assert extra == (0 if row_basis else 33 - 22)
            basis = "row basis of" if row_basis else "all"
            print(
                f"(33, 11), s = {shift}, {basis} period rows: [[{n},{k}]], {extra} extra logicals"
            )
    print("Unequal lifts (33, 11): joint kernel and readout agree with the SM.")


if __name__ == "__main__":
    main()
