"""Bridge (repetition-code adapter) joining two gadgets for a joint measurement.

Swaroop, Jochym-O'Connor, Yoder, "Universal adapters between quantum LDPC codes",
arXiv:2410.03628. To measure X_l X_r (or Z_l Z_r) with two gadgets of the same Pauli
type, w = ``width`` adapter qubits are added, the i-th one joining the vertices with
SkipTree label i of the two ports. SkipTree (Sec. III, Algorithm 1) labels the port
vertices of each side so that the tree paths between labels i and i + 1, the rows
of T, have length <= 3 and share no edge more than twice (Theorem 7). A path's
boundary is its two ends, so

    T d1 = H_R Pi^T    (hence T d1 Pi = H_R, the identity of the paper),

with d1 the side's edge-vertex incidence, Pi[v, i] = 1 iff port vertex v has label i,
and H_R the (w-1) x w repetition-code check matrix (rows e_i + e_{i+1}). The adapter
cycles [T_l | T_r | H_R] therefore commute with the new vertex checks, which gain
the adapter columns Pi (Sec. IV, Lemma 10). The product of all new vertex checks is
the data operator X(x_l + x_r), as the adapter columns cancel pairwise; with both
graphs connected, no other product of them is free of auxiliary qubits, so only
X_l X_r is measured.

Each side's port subgraph is first made connected and its basis cycles are cut to at
most ``cellulate_max_len`` by chords between far-apart cycle vertices; the added edges
are appended to the gadget, whose cycle checks are then completed by the lightest
independent cycles to a basis of all cycles. Paths run through
weight-two edges only, so hyperedge rows of d1 get zero columns in T. No distance
guarantee (Lemma 9 needs w >= d and relative expansion) is checked here.
"""

from __future__ import annotations

import dataclasses

import networkx as nx
import numpy as np
from qldpc.codes.common import CSSCode
from qldpc.objects import Pauli, PauliXZ
from scipy.linalg import block_diag

from .edge_expansion import algorithm_2
from .gadget import GadgetSpec, XorBasis, basis_matrices, bitmask


@dataclasses.dataclass(frozen=True, eq=False)
class JointSpec:
    """Adapter between two gadgets of the same ``basis``.

    ``port_l[i]``/``port_r[i]`` are the joined vertices (positions in each gadget's
    vertex set); ``label_l[v]`` is the SkipTree label of vertex v (-1 off the port).
    ``extra_ancilla_s`` are the weight-two edges added on side s, and ``g_s_aug`` is
    gadget s with them appended. ``T_s`` (w-1 x edges of ``g_s_aug``) and ``H_R``
    satisfy T_s d1_s = H_R Pi_s^T.
    """

    width: int
    basis: PauliXZ
    port_l: tuple[int, ...]
    port_r: tuple[int, ...]
    label_l: tuple[int, ...]
    label_r: tuple[int, ...]
    extra_ancilla_l: np.ndarray
    extra_ancilla_r: np.ndarray
    T_l: np.ndarray
    T_r: np.ndarray
    H_R: np.ndarray
    g_l_aug: GadgetSpec
    g_r_aug: GadgetSpec


def _graph(incidence: np.ndarray) -> nx.Graph:
    """Graph on the vertices with one edge per distinct weight-two row of d1."""
    graph = nx.Graph()
    graph.add_nodes_from(range(incidence.shape[1]))
    graph.add_edges_from(tuple(np.flatnonzero(row).tolist()) for row in incidence if row.sum() == 2)
    return graph


def _connect(graph: nx.Graph, ports: tuple[int, ...]) -> list[tuple[int, int]]:
    """Join the components of the port subgraph by edges between their smallest ports."""
    added = []
    while len(components := list(nx.connected_components(graph.subgraph(ports)))) > 1:
        edge = tuple(sorted((min(components[0]), min(components[1]))))
        graph.add_edge(*edge)
        added.append(edge)
    return added


def _cellulate(graph: nx.Graph, ports: tuple[int, ...], max_len: int) -> list[tuple[int, int]]:
    """Add chords until every basis cycle of the port subgraph has length <= max_len.

    A long cycle is cut by a missing chord between vertices as far apart along it as
    possible (opposite ones first), so the chords do not pile up on one vertex.
    """
    added = []
    while long := [c for c in nx.cycle_basis(graph.subgraph(ports)) if len(c) > max_len]:
        cycle, k = long[0], len(long[0])
        chord = next(
            (
                tuple(sorted((cycle[i], cycle[(i + gap) % k])))
                for gap in range(k // 2, 1, -1)
                for i in range(k)
                if not graph.has_edge(cycle[i], cycle[(i + gap) % k])
            ),
            None,
        )
        if chord is None:
            raise RuntimeError(f"no chord left to cellulate the port cycle {cycle}")
        graph.add_edge(*chord)
        added.append(chord)
    return added


def _skiptree_order(tree: nx.Graph, root: int) -> list[int]:
    """SkipTree vertex order (arXiv:2410.03628 Algorithm 1) of a spanning tree of ``tree``."""
    span = nx.minimum_spanning_tree(tree)
    order: list[int] = []
    visited: set[int] = set()

    def label_first(v: int, skip: bool) -> None:
        visited.add(v)
        order.append(v)
        children = [c for c in span.neighbors(v) if c not in visited]
        for i, child in enumerate(children):
            if i == len(children) - 1 and not skip:
                label_first(child, skip=False)
            else:
                label_last(child)

    def label_last(v: int) -> None:
        visited.add(v)
        for child in span.neighbors(v):
            if child not in visited:
                label_first(child, skip=True)
        order.append(v)

    label_first(root, skip=False)
    return order


def _skiptree(
    graph: nx.Graph, ports: tuple[int, ...], root: int, incidence: np.ndarray
) -> tuple[np.ndarray, list[int]]:
    """(T, labels) on a spanning tree of the port subgraph, T indexed by rows of d1.

    Row i of T is the tree path between the vertices labelled i and i + 1. Of several
    rows of d1 with the same two ends, only the first carries the tree edge (repeated
    columns would cancel in T d1).
    """
    relabel = {v: i for i, v in enumerate(sorted(ports))}
    tree = nx.minimum_spanning_tree(nx.relabel_nodes(graph.subgraph(ports).copy(), relabel))
    tree_edges = {e: i for i, e in enumerate(sorted(tuple(sorted(e)) for e in tree.edges()))}
    order = _skiptree_order(tree, relabel[ports[root]])
    T_tree = np.zeros((len(order) - 1, len(tree_edges)), dtype=np.int_)
    for i in range(len(order) - 1):
        path = nx.shortest_path(tree, order[i], order[i + 1])
        for e in zip(path[:-1], path[1:]):
            T_tree[i, tree_edges[tuple(sorted(e))]] ^= 1
    labels = [-1] * incidence.shape[1]
    for i, v in enumerate(order):
        labels[sorted(ports)[v]] = i
    T = np.zeros((len(T_tree), len(incidence)), dtype=np.int_)
    assigned = set()
    for r, row in enumerate(incidence):
        ends = np.flatnonzero(row).tolist()
        if len(ends) != 2 or not set(ends) <= relabel.keys():
            continue
        e = tuple(sorted(relabel[v] for v in ends))
        if e in tree_edges and e not in assigned:
            T[:, r] = T_tree[:, tree_edges[e]]
            assigned.add(e)
    return T, labels


def _append_edges(g: GadgetSpec, edges: np.ndarray) -> GadgetSpec:
    """Gadget with weight-two edges appended to d1 (zero f0 columns) and its cycle checks
    completed to a basis of all cycles, ker d1^T, by the lightest independent cycles.

    The cycles that the data checks generate in the single gadget are needed once the
    gadget shares data with a second one, as in ``gauging._restore_flux_checks``.
    """
    incidence = np.vstack([g.incidence, edges]).astype(np.uint8)
    pad = lambda m: np.hstack([m, np.zeros((len(m), len(edges)), np.uint8)])
    span, cycles = XorBasis(), list(pad(g.partial_0))
    for row in cycles:
        span.add(bitmask(row))
    no_checks = np.zeros((0, 0), np.uint8), np.zeros((0, len(incidence)), np.uint8)
    cycles += [row for row in algorithm_2(incidence, *no_checks) if span.add(bitmask(row))]
    partial_0 = np.array(cycles, np.uint8).reshape(-1, len(incidence))
    return dataclasses.replace(g, incidence=incidence, f0=pad(g.f0), partial_0=partial_0)


def build_bridge(
    g_l: GadgetSpec,
    g_r: GadgetSpec,
    *,
    port_subset_l: tuple[int, ...] | None = None,
    port_subset_r: tuple[int, ...] | None = None,
    spanning_tree_root_l: int = 0,
    spanning_tree_root_r: int = 0,
    cellulate_max_len: int | None = None,
) -> JointSpec:
    """Adapter of width min(|port_l|, |port_r|) joining two gadgets of the same basis.

    The ports default to all vertices; the first ``width`` of each are used.
    ``spanning_tree_root_s`` indexes the port tuple. ``cellulate_max_len`` defaults to
    the largest measured-type stabiliser weight of the two codes.
    """
    if g_l.basis is not g_r.basis:
        raise ValueError("a CSS bridge joins two gadgets of the same Pauli basis")
    if cellulate_max_len is None:
        cellulate_max_len = max(
            int(basis_matrices(g.code, g.basis)[0].sum(axis=1).max()) for g in (g_l, g_r)
        )
    sides = []
    for g, subset in ((g_l, port_subset_l), (g_r, port_subset_r)):
        sides.append((g, tuple(range(g.f1.shape[1])) if subset is None else tuple(subset)))
    width = min(len(ports) for _, ports in sides)
    if width < 2:
        raise ValueError(f"bridge width must be >= 2, got {width}")
    out = {}
    for s, (g, ports), root in zip("lr", sides, (spanning_tree_root_l, spanning_tree_root_r)):
        ports = ports[:width]
        graph = _graph(g.incidence)
        added = _connect(graph, ports) + _cellulate(graph, ports, cellulate_max_len)
        extra = np.zeros((len(added), g.f1.shape[1]), np.uint8)
        for i, edge in enumerate(added):
            extra[i, list(edge)] = 1
        aug = _append_edges(g, extra)
        T, labels = _skiptree(graph, ports, root, aug.incidence)
        out |= {
            f"port_{s}": ports,
            f"label_{s}": tuple(labels),
            f"extra_ancilla_{s}": extra,
            f"T_{s}": T,
            f"g_{s}_aug": aug,
        }
    H_R = np.eye(width - 1, width, dtype=np.int_) + np.eye(width - 1, width, 1, dtype=np.int_)
    return JointSpec(width=width, basis=g_l.basis, H_R=H_R, **out)


def joint_merged_code(g_l: GadgetSpec, g_r: GadgetSpec, bridge: JointSpec) -> CSSCode:
    """Merged code of the joint measurement, columns (data | Q'_l | Q'_r | adapter).

    The data block is shared when ``g_l.code is g_r.code`` and is block-diagonal over
    the two codes otherwise. With measured/complementary checks (H, H') and
    D = diag(d1_l, d1_r):

        measured:       [[H, 0, 0], [F1^T, D^T, Pi]]
        complementary:  [[H', F0, 0], [0, diag(d0_l, d0_r), 0], [0, [T_l T_r], H_R]]
    """
    a, b = bridge.g_l_aug, bridge.g_r_aug
    (H_l, Hc_l), (H_r, Hc_r) = basis_matrices(a.code, a.basis), basis_matrices(b.code, b.basis)
    if g_l.code is g_r.code:
        H, Hc, F1T, F0 = H_l, Hc_l, np.vstack([a.f1.T, b.f1.T]), np.hstack([a.f0, b.f0])
    else:
        H, Hc = block_diag(H_l, H_r), block_diag(Hc_l, Hc_r)
        F1T, F0 = block_diag(a.f1.T, b.f1.T), block_diag(a.f0, b.f0)
    w = bridge.width
    Pi = np.zeros((len(bridge.label_l) + len(bridge.label_r), w), dtype=np.int_)
    for v, label in enumerate(bridge.label_l + bridge.label_r):
        if label >= 0:
            Pi[v, label] = 1
    D0 = block_diag(a.partial_0, b.partial_0)
    zeros = lambda rows, cols: np.zeros((rows, cols), dtype=np.int_)
    M_meas = np.block(
        [[H, zeros(len(H), F0.shape[1] + w)], [F1T, block_diag(a.incidence.T, b.incidence.T), Pi]]
    )
    M_comp = np.block(
        [
            [Hc, F0, zeros(len(Hc), w)],
            [zeros(len(D0), Hc.shape[1]), D0, zeros(len(D0), w)],
            [zeros(w - 1, Hc.shape[1]), np.hstack([bridge.T_l, bridge.T_r]), bridge.H_R],
        ]
    ).astype(np.int_)
    M_meas = M_meas.astype(np.int_)
    hx, hz = (M_meas, M_comp) if bridge.basis is Pauli.X else (M_comp, M_meas)
    return CSSCode(hx, hz, is_subsystem_code=False)
