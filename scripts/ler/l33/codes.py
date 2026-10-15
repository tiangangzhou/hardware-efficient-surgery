"""The l = 33 codes of Fig. 4 and the code-capacity table.

The data code is the [[1122, 148]] LP code of instance III; the merged [[2442, 113]]
code is X-type fibre surgery on its fibre (3, 3), measuring all 33 fibre operators.
"""

from __future__ import annotations

import numpy as np
from qldpc.codes.common import CSSCode
from qldpc.objects import Pauli

from algebraic_qec import CyclicPolynomial, PolynomialMatrix, RingDecomposition, gf2_rank
from algebraic_qec.canonical import aligned_information_set, systematic_kernel_generators
from algebraic_qec.examples.surgery_instances import instance
from algebraic_qec.tube import component_ranks, extend_factor, maximal_minors, unit_columns
from qldpc_surgery.hmatrix.gadget import GadgetSpec


def css_code(hx: np.ndarray, hz: np.ndarray) -> CSSCode:
    return CSSCode(hx.astype(np.int_), hz.astype(np.int_), is_subsystem_code=False)


def lift(matrix: PolynomialMatrix) -> np.ndarray:
    return matrix.lift().toarray().astype(np.uint8)


def data_code() -> CSSCode:
    """The [[1122, 148]] LP code of instance III (the idling baseline)."""
    code = instance(33).data_code()
    return css_code(lift(code.matrix_x), lift(code.matrix_z))


def tube_factor() -> PolynomialMatrix:
    """The 4 x 5 D of instance III (SM Eq. (s-lp33-seed-tube)) with columns reordered as
    (2, 0, 4, 3, 1), so that it couples at its column 0 (= SM column q = 2).  This only
    relabels auxiliary qubits."""
    d = instance(33).factor_d
    order = (2, 0, 4, 3, 1)
    entries = {(i, k): d[i, j] for i in range(d.shape[0]) for k, j in enumerate(order)}
    return PolynomialMatrix(33, d.shape, entries)


def surgery_maps() -> dict[str, np.ndarray]:
    """Fibre surgery on fibre (3, 3) of the l = 33 code, H_L = e_4^T.

    D is ``tube_factor()``, the D printed in the Supplemental Material with permuted
    columns (the same merged code up to a relabelling of auxiliary qubits; this ordering
    generated the published data).  The coupled coordinate q, the first unit entry of the
    kernel generator t of D (its maximal minors), is q = 0 here and the SM's q = 2.  The
    readout vectors are the 33 cyclic shifts of u_3 (x) t.  Unlike the SM, t is not
    normalised to t_q = 1: the 33 time-like observables are those of the SM times the unit
    t_q, a basis of the same measured space, so no failure event changes (a shot fails iff
    any of them is wrong).
    """
    alpha = beta = 3
    code = instance(33).data_code()
    ell = code.period
    a = instance(33).factor_a
    decomposition = RingDecomposition(ell)
    m_a, n_a = a.shape
    aligned = aligned_information_set(a, n_a - m_a, decomposition)
    other = next(j for j in aligned if j != alpha)
    a_sel = extend_factor(a, [tuple(int(j == other) for j in range(n_a))])  # A' = [A; e_4]
    d = tube_factor()
    m_d, n_d = d.shape
    assert component_ranks(d, decomposition) == [m_d] * len(decomposition.factors)
    q = unit_columns(maximal_minors(d), decomposition)[0]
    rows, cols = a_sel.shape
    unit = CyclicPolynomial.one(ell)
    eye = lambda size: PolynomialMatrix.identity(ell, size)

    d1 = a_sel.kron(eye(n_d)).vstack(eye(cols).kron(d))
    d0 = eye(rows).kron(d).hstack(a_sel.kron(eye(m_d)))
    num_data, num_aux = code.matrix_x.shape[1], d1.shape[0]
    gamma1 = PolynomialMatrix(
        ell, (cols * n_d, num_data), {(j * n_d + q, beta * n_a + j): unit for j in range(n_a)}
    )
    gamma0 = PolynomialMatrix(
        ell,
        (code.matrix_z.shape[0], num_aux),
        {(beta * m_a + r, r * n_d + q): unit for r in range(m_a)},
    )
    hx = code.matrix_x.hstack(PolynomialMatrix.zero(ell, code.matrix_x.shape[0], num_aux))
    hx = hx.vstack(gamma1.hstack(d1.adjoint()))
    hz = code.matrix_z.hstack(gamma0).vstack(
        PolynomialMatrix.zero(ell, d0.shape[0], num_data).hstack(d0)
    )

    u = systematic_kernel_generators(a, aligned, decomposition)[aligned.index(alpha)]
    readout = u.kron(maximal_minors(d))
    assert not any((d1 @ readout).nonzero_entries())
    readout_vectors = lift(readout).T  # (33, auxiliary qubits)
    f1 = lift(gamma1).T
    batch_x = readout_vectors @ f1.T % 2  # (33, data qubits)
    assert gf2_rank(batch_x) == ell
    return dict(
        h_x=lift(hx),
        h_z=lift(hz),
        f1=f1,
        incidence=lift(d1),
        f0=lift(gamma0),
        partial_0=lift(d0),
        readout_vectors=readout_vectors,
        batch_x=batch_x,
    )


def surgery_gadget() -> GadgetSpec:
    """X-type gadget (the attachment enters the merged H_X) measuring the 33 fibre operators.

    Validated as a mapping cone; its merged checks equal those of ``surgery_maps``.
    """
    maps = surgery_maps()
    keys = ("f1", "incidence", "f0", "partial_0", "batch_x", "readout_vectors")
    gadget = GadgetSpec.from_maps(
        data_code(), maps["batch_x"][0], basis=Pauli.X, validate=True, **{k: maps[k] for k in keys}
    )
    assert np.array_equal(gadget.HX_merged.astype(np.uint8), maps["h_x"])
    assert np.array_equal(gadget.HZ_merged.astype(np.uint8), maps["h_z"])
    return gadget


def merged_checks() -> tuple[np.ndarray, np.ndarray]:
    """Binary check matrices (H_X, H_Z) of the merged [[2442, 113]] code."""
    maps = surgery_maps()
    return maps["h_x"], maps["h_z"]
