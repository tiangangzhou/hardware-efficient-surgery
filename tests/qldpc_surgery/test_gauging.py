"""Gauging measurement of Williamson and Yoder arXiv:2410.02213 (Graph-WY) and
``parallel_gadgets``."""

from __future__ import annotations

import numpy as np
import pytest
from qldpc import codes
from qldpc.codes.common import CSSCode
from qldpc.objects import Pauli

from qldpc_surgery import build_gadget, parallel_gadgets
from qldpc_surgery.hmatrix.gadget import GadgetSpec, gf2_rank
from tests.qldpc_surgery.fixtures import logical


def dimension(g: GadgetSpec) -> int:
    return CSSCode(g.HX_merged.astype(int), g.HZ_merged.astype(int)).dimension


def test_triangle_graph_on_a_steane_logical() -> None:
    code = codes.SteaneCode()
    x = logical(code, Pauli.X)
    g = build_gadget(
        code, x, basis=Pauli.X, method="gauging", edges=[(0, 1), (1, 2), (0, 2)], expand=False
    )
    assert g.incidence.shape == (3, 3) and np.array_equal(g.partial_0, [[1, 1, 1]])  # one flux
    assert np.array_equal(g.f0 @ g.incidence % 2, np.asarray(code.matrix_z) @ g.f1 % 2)
    assert (g.HX_merged.shape[1], dimension(g)) == (10, 0)


def test_deformed_checks_follow_paths_between_their_qubits() -> None:
    code = CSSCode(np.zeros((0, 4), int), np.array([[1, 0, 0, 1]]))  # Z0 Z3 on a path 0-1-2-3
    edges = [(0, 1), (1, 2), (2, 3)]
    g = build_gadget(
        code, np.ones(4, np.uint8), basis=Pauli.X, method="gauging", edges=edges, expand=False
    )
    assert np.array_equal(g.f0, [[1, 1, 1]]) and g.partial_0.shape == (0, 3)


def test_rejects_a_disconnected_graph() -> None:
    code = codes.SteaneCode()
    with pytest.raises(ValueError, match="connected"):
        build_gadget(
            code,
            logical(code, Pauli.X),
            basis=Pauli.X,
            method="gauging",
            edges=[(0, 1)],
            expand=False,
        )


def test_cellulated_flux_checks_span_all_cycles() -> None:
    code = CSSCode(np.zeros((0, 8), int), np.zeros((0, 8), int))
    ring = [(i, (i + 1) % 8) for i in range(8)]
    g = build_gadget(
        code,
        np.ones(8, np.uint8),
        basis=Pauli.Z,
        method="gauging",
        edges=ring,
        expand=False,
        cellulate_to=3,
        validate=True,
    )
    assert g.partial_0.sum(axis=1).max() == 3
    assert gf2_rank(g.partial_0) == len(g.incidence) - 7  # |E| - |V| + 1 independent cycles
    assert dimension(g) == 7


def test_parallel_gadgets_measure_overlapping_logicals() -> None:
    code = CSSCode(np.zeros((0, 3), int), np.zeros((0, 3), int))
    logicals = np.array([[1, 1, 0], [0, 1, 1]], np.uint8)
    gs = [
        build_gadget(code, x, basis=Pauli.X, method="gauging", edges=[(0, 1)], expand=False)
        for x in logicals
    ]
    g = parallel_gadgets(gs)
    assert np.array_equal(g.effective_batch_x, logicals)
    assert np.array_equal(g.effective_readout_vectors, [[1, 1, 0, 0], [0, 0, 1, 1]])
    assert dimension(g) == 1
    with pytest.raises(ValueError, match="independent"):
        parallel_gadgets([gs[0], gs[0]])


def test_parallel_gadgets_reject_undeclared_measurements() -> None:
    code = CSSCode(np.zeros((0, 3), int), np.zeros((0, 3), int))
    x = np.array([1, 1, 0], np.uint8)  # the vertex 2 is isolated, so X2 is measured too
    g = GadgetSpec.from_maps(
        code,
        x,
        basis=Pauli.X,
        f1=np.eye(3, dtype=np.uint8),
        incidence=np.array([[1, 1, 0]]),
        f0=np.zeros((0, 1)),
        partial_0=np.zeros((0, 1)),
        batch_x=x[None],
        readout_vectors=x[None],
    )
    with pytest.raises(ValueError, match="undeclared"):
        parallel_gadgets([g])


def test_parallel_gadgets_restore_cycles_implied_only_by_shared_checks() -> None:
    # Each gadget closes a cycle with the product of the two equal data checks; in
    # parallel that product only gives the sum of both cycles.
    code = CSSCode(np.zeros((0, 3), int), np.array([[1, 1, 0], [1, 1, 0]]))
    g1 = GadgetSpec.from_maps(
        code,
        np.array([1, 1, 0]),
        basis=Pauli.X,
        f1=np.eye(3)[:, :2],
        incidence=np.array([[1, 1], [1, 1]]),
        f0=np.eye(2),
        partial_0=np.zeros((0, 2)),
    )
    g2 = GadgetSpec.from_maps(
        code,
        np.ones(3),
        basis=Pauli.X,
        f1=np.eye(3),
        incidence=np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]]),
        f0=np.array([[1, 0, 0], [0, 1, 1]]),
        partial_0=np.zeros((0, 3)),
    )
    assert dimension(g1) == dimension(g2) == 1
    g = parallel_gadgets([g1, g2])
    assert g.partial_0.shape == (1, 5) and dimension(g) == 0
