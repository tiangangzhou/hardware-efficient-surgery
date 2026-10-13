"""Algorithms 1-3 of Ide, Gowda, Nadkarni, Dauphinais arXiv:2410.02753 (Graph-IGND)."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import sympy
from qldpc import codes
from qldpc.codes.common import CSSCode
from qldpc.objects import Pauli

from qldpc_surgery.hmatrix.edge_expansion import (
    _fiedler_sweep_cuts,
    algorithm_1,
    algorithm_2,
    algorithm_3,
    cellulate,
    cheeger_constant,
    edge_expansion_maps,
    expand_hyperedges,
    restrict_maps,
    sparsest_cut,
)
from qldpc_surgery.hmatrix.gadget import GadgetSpec, gf2_rank

HAMMING = np.array([[(j >> i) & 1 for j in range(1, 16)] for i in range(4)], np.uint8)
STEANE = HAMMING[:3, :7]  # the [7, 4] Hamming checks: H_X = H_Z of the Steane code


def graph(edges, n_v) -> np.ndarray:
    incidence = np.zeros((len(edges), n_v), np.uint8)
    for i, edge in enumerate(edges):
        incidence[i, list(edge)] = 1
    return incidence


def grid(n: int) -> np.ndarray:
    edges = [(r * n + c, r * n + c + 1) for r in range(n) for c in range(n - 1)]
    return graph(
        edges + [(r * n + c, (r + 1) * n + c) for r in range(n - 1) for c in range(n)], n * n
    )


def no_checks(incidence):
    """(H, f0) without complementary checks: every cycle of the graph is new."""
    return np.zeros((0, 1), np.uint8), np.zeros((0, len(incidence)), np.uint8)


def test_example_4_two_paths_need_three_edges() -> None:
    incidence = graph([(0, 1), (1, 2), (3, 4), (4, 5)], 6)
    assert cheeger_constant(incidence) == 0
    expanded = algorithm_1(incidence)
    assert len(expanded) - len(incidence) == 3 and cheeger_constant(expanded) >= 1
    assert np.array_equal(expanded[: len(incidence)], incidence)


def test_example_5_eight_cycle_needs_two_edges_then_cellulation() -> None:
    incidence = graph([(i, (i + 1) % 8) for i in range(8)], 8)
    assert cheeger_constant(incidence) == 0.5
    expanded = algorithm_1(incidence)
    assert len(expanded) - 8 == 2 and cheeger_constant(expanded) >= 1
    H, f0 = no_checks(expanded)
    d0, d1, _ = cellulate(algorithm_2(expanded, H, f0), expanded, f0, target_weight=4)
    assert d0.sum(axis=1).max() <= 4 and not (d0.astype(int) @ d1 % 2).any()


def test_example_6_steane_weight_three_logical_gives_a_9_0_code() -> None:
    code = CSSCode(STEANE, STEANE, is_subsystem_code=False)
    x = np.array([1, 1, 1, 0, 0, 0, 0], np.uint8)
    f1, d1, f0, d0 = edge_expansion_maps(code, x, basis=Pauli.X)
    assert d0.shape[0] == 0 and cheeger_constant(d1) >= 1
    g = GadgetSpec.from_maps(code, x, basis=Pauli.X, f1=f1, incidence=d1, f0=f0, partial_0=d0)
    merged = CSSCode(g.HX_merged.astype(int), g.HZ_merged.astype(int))
    assert (merged.num_qudits, merged.dimension) == (9, 0)


def test_example_7_hamming_support_needs_two_edges() -> None:
    x = np.zeros(15, np.uint8)
    x[[2, 3, 4, 11, 13]] = 1  # X3 X4 X5 X12 X14
    f1, d1_star, f0_star = restrict_maps(HAMMING, x)
    assert np.array_equal(f0_star @ d1_star % 2, HAMMING @ f1 % 2)  # Eq. (36)
    assert cheeger_constant(d1_star) == 0.5
    expanded = algorithm_1(d1_star)
    assert len(expanded) - len(d1_star) == 2 and cheeger_constant(expanded) >= 1


def test_cheeger_constant_and_sparsest_cut_are_exact() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        n_v = int(rng.integers(2, 9))
        incidence = graph([rng.choice(n_v, 2, replace=False) for _ in range(n_v + 2)], n_v)
        best = min(
            (incidence[:, list(S)].sum(axis=1) % 2).sum() / len(S)
            for size in range(1, n_v // 2 + 1)
            for S in itertools.combinations(range(n_v), size)
        )
        h, S = sparsest_cut(incidence)
        assert h == pytest.approx(best) == (incidence @ S % 2).sum() / S.sum()


def test_fiedler_sweep_overestimates_and_algorithm_1_reaches_one_above_20_vertices() -> None:
    small = grid(4)
    sweep = _fiedler_sweep_cuts(small)
    assert ((small @ sweep.T % 2).sum(axis=0) / sweep.sum(axis=1)).min() >= cheeger_constant(small)
    large = grid(6)  # 36 vertices: Fiedler sweep estimate
    assert cheeger_constant(algorithm_1(large)) >= 1


def test_algorithm_2_complements_the_cycles_of_the_data_checks() -> None:
    # Three parallel edges: two independent cycles, one of which ([0, 1, 1]) is the
    # product of the two (redundant) deformed checks and so must not reappear in d0.
    incidence = graph([(0, 1)] * 3, 2)
    H = np.array([[1, 1], [1, 1]], np.uint8)
    f0 = np.array([[0, 1, 0], [0, 0, 1]], np.uint8)
    d0 = algorithm_2(incidence, H, f0)
    assert not (d0.astype(int) @ incidence % 2).any()
    assert gf2_rank(d0) == 1 and gf2_rank(np.vstack([d0, [[0, 1, 1]]])) == 2


def test_algorithm_2_returns_a_minimum_cycle_basis() -> None:
    incidence = grid(4)  # the minimum basis: the 9 unit squares
    d0 = algorithm_2(incidence, *no_checks(incidence))
    assert not (d0.astype(int) @ incidence % 2).any()
    assert gf2_rank(d0) == 9 and set(d0.sum(axis=1)) == {4}


def test_expand_hyperedges_preserves_the_chain_map_and_joins_components() -> None:
    incidence = np.array([[1, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 1], [1, 1, 0, 0, 0, 0]], np.uint8)
    f0 = np.eye(3, dtype=np.uint8)
    d1, f0_new = expand_hyperedges(incidence, f0)
    assert set(d1.sum(axis=1)) == {2}
    assert np.array_equal(f0_new @ d1 % 2, f0 @ incidence % 2)  # f0 d1 is unchanged
    assert gf2_rank(d1) == 5  # connected (the in-order matching would leave {4, 5} apart)


def test_cellulate_splits_unions_and_long_cycles() -> None:
    incidence = graph([(i, (i + 1) % 16) for i in range(16)] + [(16, 17), (17, 18), (16, 18)], 19)
    long_cycle = np.r_[np.ones(16), np.zeros(3)].astype(np.uint8)
    rows = np.array([long_cycle, np.ones(19, np.uint8)])  # the 16-cycle, and it plus a triangle
    f0 = np.zeros((1, 19), np.uint8)
    d0, d1, f0_new = cellulate(rows, incidence, f0, target_weight=4)
    assert d0.sum(axis=1).max() <= 4 and not (d0.astype(int) @ d1 % 2).any()
    padded = np.hstack([rows, np.zeros((2, d0.shape[1] - 19), np.uint8)])
    assert gf2_rank(np.vstack([d0, padded])) == gf2_rank(d0)  # the old rows are spanned
    assert np.array_equal(d1[:19], incidence) and not f0_new[:, 19:].any()


def test_algorithm_3_cellulates_when_d0_is_too_heavy() -> None:
    # Two disjoint checks on supp(x): Algorithm 1 closes them into a 4-cycle.
    H = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], np.uint8)
    x = np.ones(4, np.uint8)
    main = algorithm_3(H, x)
    assert main[3].sum(axis=1).max() == 4
    f1, d1, f0, d0 = algorithm_3(H, x, cellulate_to=3)
    assert d0.sum(axis=1).max() <= 3 and len(d1) > len(main[1])
    assert not ((H @ f1 + f0 @ d1) % 2).any() and not (d0.astype(int) @ d1 % 2).any()


def test_gross_code_cycle_weights() -> None:
    """Regression of Algorithm 2 on the gross code (weight profile and 4-cycles)."""
    x, y = sympy.symbols("x y")
    code = codes.BBCode((12, 6), x**3 + y + y**2, y**3 + x + x**2)
    f1, d1, f0, d0 = algorithm_3(
        np.asarray(code.matrix_z, np.uint8), np.asarray(code.get_logical_ops(Pauli.X)[0], np.uint8)
    )
    assert sorted(d0.sum(axis=1)) == [3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 6, 6]
    rows = [set(np.flatnonzero(r)) for r in np.vstack([f0, d0])]
    assert sum(len(a & b) >= 2 for a, b in itertools.combinations(rows, 2)) == 2
