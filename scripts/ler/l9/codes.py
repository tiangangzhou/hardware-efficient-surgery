"""Codes and X-type surgery gadgets of the l = 9 circuit simulations (instance I, SM)."""

from __future__ import annotations

import numpy as np
from qldpc.codes.common import CSSCode
from qldpc.objects import Pauli

from algebraic_qec import CyclicPolynomial, PolynomialMatrix, RingDecomposition, gf2_rank
from algebraic_qec.canonical import aligned_information_set, systematic_kernel_generators
from algebraic_qec.examples.surgery_instances import instance
from algebraic_qec.tube import component_ranks, tube_kernel_generator
from qldpc_surgery.hmatrix.gadget import GadgetSpec

MODES = ("fibre", "packet", "joint", "idling")
CANONICAL = {"fibre": 9, "packet": 3, "joint": 9, "idling": 0}
RESIDUAL = {"fibre": 2, "packet": 2, "joint": 4, "idling": 0}


def css_code(hx: np.ndarray, hz: np.ndarray) -> CSSCode:
    return CSSCode(hx.astype(np.int_), hz.astype(np.int_), is_subsystem_code=False)


def lift(matrix: PolynomialMatrix) -> np.ndarray:
    return matrix.lift().toarray().astype(np.uint8)


def code_and_gadget(mode: str) -> tuple[CSSCode, GadgetSpec | None]:
    """Data code and X-type surgery gadget of instance I for one mode.

    * ``fibre``: A' = [A; e_4^T], nine canonical readouts, two residual directions;
    * ``packet``: rows (1 + x^3) e_i (i < 4) select the CRT union S = {0, 1} (three
      readouts); the generator is projected by the idempotent 1 + x^3 + x^6;
    * ``joint``: fibre (3, 3) of two data blocks with zero offset, nine product
      readouts, four residual directions;
    * ``idling``: the data code alone.
    """
    case = instance(9)
    a, d, q = case.factor_a, case.factor_d, case.couple_column
    hx, hz = case.data_code().matrix_x, case.data_code().matrix_z
    decomposition = RingDecomposition(9)
    assert component_ranks(d, decomposition) == [3, 3, 3]
    t = tube_kernel_generator(d, q, decomposition)
    if mode == "idling":
        return css_code(lift(hx), lift(hz)), None

    unit = CyclicPolynomial.one(9)
    aligned = aligned_information_set(a, 2, decomposition)
    assert tuple(aligned) == (3, 4)
    u = systematic_kernel_generators(a, aligned, decomposition)[0]
    if mode == "joint":
        link = PolynomialMatrix(
            9, (3, 10), {(0, 4): unit, (1, 9): unit, (2, 3): unit, (2, 8): unit}
        )
        a_sel = a.block_diag(a).vstack(link)
        u = u.vstack(u)
        hx, hz = hx.block_diag(hx), hz.block_diag(hz)
    else:
        a_sel = a.vstack(PolynomialMatrix(9, (1, 5), {(0, 4): unit}))
        if mode == "packet":
            g = CyclicPolynomial(9, [0, 3])
            a_sel = a_sel.vstack(PolynomialMatrix(9, (4, 5), {(i, i): g for i in range(4)}))
            projector = CyclicPolynomial(9, [0, 3, 6])
            u = PolynomialMatrix(9, u.shape, {k: v * projector for k, v in u.nonzero_entries()})
    m, n = a_sel.shape
    eye = lambda size: PolynomialMatrix.identity(9, size)
    d1 = a_sel.kron(eye(4)).vstack(eye(n).kron(d))
    d0 = eye(m).kron(d).hstack(a_sel.kron(eye(3)))
    blocks = 2 if mode == "joint" else 1
    # Tube coordinate q <-> data fibre 3 of each block (34 qubit, 15 Z-check columns).
    gamma1 = PolynomialMatrix(
        9,
        (n * 4, hx.shape[1]),
        {((b * 5 + j) * 4 + q, b * 34 + 15 + j): unit for b in range(blocks) for j in range(5)},
    )
    gamma0 = PolynomialMatrix(
        9,
        (hz.shape[0], d1.shape[0]),
        {(b * 15 + 9 + r, (b * 3 + r) * 4 + q): unit for b in range(blocks) for r in range(3)},
    )
    readout = u.kron(t)
    assert not any((d1 @ readout).nonzero_entries()) and not any((d0 @ d1).nonzero_entries())
    assert gamma0 @ d1 == hz @ gamma1.adjoint()
    f1 = lift(gamma1).T
    readout_vectors = lift(readout).T[: CANONICAL[mode]]
    batch_x = readout_vectors @ f1.T % 2
    assert gf2_rank(batch_x) == CANONICAL[mode]
    code = css_code(lift(hx), lift(hz))
    gadget = GadgetSpec.from_maps(  # X type: the attachment enters the merged H_X
        code,
        batch_x[0],
        basis=Pauli.X,
        validate=True,
        f1=f1,
        incidence=lift(d1),
        f0=lift(gamma0),
        partial_0=lift(d0),
        readout_vectors=readout_vectors,
        batch_x=batch_x,
    )
    return code, gadget
