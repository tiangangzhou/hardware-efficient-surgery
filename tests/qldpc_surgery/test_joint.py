"""The bridge of Swaroop, Jochym-O'Connor, Yoder arXiv:2410.03628 and its merged code."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
from qldpc import codes
from qldpc.objects import Pauli

from qldpc_surgery import build_bridge, build_gadget
from qldpc_surgery.hmatrix.gadget import gf2_rank
from qldpc_surgery.hmatrix.joint import JointSpec, _graph, joint_merged_code
from tests.qldpc_surgery.fixtures import bb36, gb_code, logical


def joint(c_l, c_r, basis, i=0, j=0, **kw):
    g_l = build_gadget(c_l, logical(c_l, basis, i), basis=basis)
    g_r = build_gadget(c_r, logical(c_r, basis, j), basis=basis)
    return g_l, g_r, build_bridge(g_l, g_r, **kw)


def assert_skiptree_identity(bridge: JointSpec) -> None:
    """T_s d1_s = H_R Pi_s^T on both sides, with (3, 2)-sparse T_s."""
    for T, g, labels in [
        (bridge.T_l, bridge.g_l_aug, bridge.label_l),
        (bridge.T_r, bridge.g_r_aug, bridge.label_r),
    ]:
        Pi = np.zeros((len(labels), bridge.width), int)
        for v, label in enumerate(labels):
            if label >= 0:
                Pi[v, label] = 1
        assert np.array_equal(T @ g.incidence % 2, bridge.H_R @ Pi.T)
        assert T.sum(axis=1).max() <= 3 and T.sum(axis=0).max() <= 2  # Theorem 7


CASES = {
    "Steane x Steane": lambda b: joint(codes.SteaneCode(), codes.SteaneCode(), b),
    "BB36, two logicals": lambda b: (lambda c: joint(c, c, b, 0, 1))(bb36()),
    "GB62 x Steane": lambda b: joint(gb_code(62), codes.SteaneCode(), b),
}


@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
@pytest.mark.parametrize("case", list(CASES))
def test_joint_code_measures_only_the_product(case, basis) -> None:
    g_l, g_r, bridge = CASES[case](basis)
    assert_skiptree_identity(bridge)
    merged = joint_merged_code(g_l, g_r, bridge)
    hx, hz = np.asarray(merged.matrix_x, int), np.asarray(merged.matrix_z, int)
    assert not (hx @ hz.T % 2).any()
    measured = hx if basis is Pauli.X else hz
    n = merged.num_qudits
    if g_l.code is g_r.code:
        product, singles = g_l.x ^ g_r.x, [g_l.x]
        k = g_l.code.dimension - 1
    else:
        n_l = g_l.code.num_qudits
        product = np.r_[g_l.x, g_r.x]
        singles = [np.r_[g_l.x, 0 * g_r.x], np.r_[0 * g_l.x, g_r.x]]
        k = g_l.code.dimension + g_r.code.dimension - 1
        assert len(product) == n_l + g_r.code.num_qudits

    def stabilizer(v):
        row = np.r_[v, np.zeros(n - len(v), int)]
        return gf2_rank(np.vstack([measured, row])) == gf2_rank(measured)

    assert stabilizer(product) and not any(stabilizer(v) for v in singles)
    assert merged.dimension == k


def test_adapter_checks_are_sparse_and_port_cycles_are_cellulated() -> None:
    code = gb_code(62)
    g_l, g_r, bridge = joint(code, code, Pauli.Z, 0, 0, cellulate_max_len=6)
    hx = np.asarray(joint_merged_code(g_l, g_r, bridge).matrix_x, int)
    assert hx[-(bridge.width - 1) :].sum(axis=1).max() <= 8  # [T_l T_r H_R]: 3 + 3 + 2
    for g, ports in [(bridge.g_l_aug, bridge.port_l), (bridge.g_r_aug, bridge.port_r)]:
        cycles = nx.cycle_basis(_graph(g.incidence).subgraph(ports))
        assert max(map(len, cycles), default=0) <= 6


def test_skiptree_ignores_repeated_edges() -> None:
    """In BB36 two edges of the Z gadget join the same vertices; only one may carry T."""
    code = bb36()
    g_l, g_r, bridge = joint(code, code, Pauli.Z)
    assert len(np.unique(g_l.incidence, axis=0)) < len(g_l.incidence)
    assert_skiptree_identity(bridge)


def test_bridge_requires_two_gadgets_of_one_basis() -> None:
    code = codes.SteaneCode()
    g_x = build_gadget(code, logical(code, Pauli.X), basis=Pauli.X)
    g_z = build_gadget(code, logical(code, Pauli.Z), basis=Pauli.Z)
    with pytest.raises(ValueError, match="same Pauli basis"):
        build_bridge(g_x, g_z)
    with pytest.raises(ValueError, match="width"):
        build_bridge(g_x, g_x, port_subset_l=(0,))
