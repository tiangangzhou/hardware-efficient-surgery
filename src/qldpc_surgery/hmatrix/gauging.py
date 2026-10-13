"""Gauging logical measurement ("Graph-WY") and parallel composition of gadgets.

Williamson and Yoder, "Low-overhead fault-tolerant quantum computation by gauging
logical operators", arXiv:2410.02213. The vertices are the qubits of supp(x) and the
edges are auxiliary qubits; the vertex checks A_v = X_v prod_{e at v} X_e are the
new measured checks, a generating set of cycles gives the flux checks
B_p = prod_{e in p} Z_e, and each complementary check s becomes s prod_{e in mu} Z_e
with boundary(mu) = supp(s) & supp(x) (Lemma 1; written here for an X measurement).
The graph follows steps 1-2 of Theorem 3: pair up the qubits of every check
intersection, then add edges until the Cheeger constant h is 1 (here with Algorithm 1
of arXiv:2410.02753). With h >= 1 the deformed code keeps the distance (Lemma 2);
h is computed exactly only for |V| <= 20. Steps 3-4 (layers, decongestion) are not
implemented: long flux cycles are cut by chords instead (``edge_expansion.cellulate``),
which cannot lower h.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import pairwise

import networkx as nx
import numpy as np
from qldpc.codes.common import CSSCode
from qldpc.objects import PauliXZ
from scipy.linalg import block_diag

from .edge_expansion import _horton_candidates, algorithm_1, algorithm_2, cellulate
from .gadget import (
    GF2,
    GadgetSpec,
    XorBasis,
    basis_matrices,
    bitmask,
    complementary_checks,
    gf2_rank,
)


def gauging_maps(
    code: CSSCode,
    x: np.ndarray,
    *,
    basis: PauliXZ,
    edges: Iterable[tuple[int, int]] | None = None,
    expand: bool = True,
    cellulate_to: int | str | None = "native",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(f1, d1, f0, d0) of Graph-WY for the ``basis``-type logical on ``x``.

    ``edges`` (pairs of positions in supp(x), ascending) replaces the pairing of
    step 1; ``expand=False`` skips step 2, and the graph must then be connected. Each
    deformed check uses shortest paths between consecutive qubits of its intersection
    (the step-1 edges themselves). ``cellulate_to="native"`` caps flux checks at the
    largest complementary check weight (at least 3).
    """
    x = np.asarray(x, np.uint8)
    H = complementary_checks(code, x, basis)
    support = np.flatnonzero(x)
    w = len(support)
    restriction = H[:, support]
    if edges is None:
        pairs = set()
        for row in restriction:
            vertices = np.flatnonzero(row).tolist()
            pairs.update(zip(vertices[::2], vertices[1::2]))
    else:
        pairs = {tuple(sorted((int(a), int(b)))) for a, b in edges}
    incidence = np.zeros((len(pairs), w), np.uint8)
    for i, pair in enumerate(sorted(pairs)):
        incidence[i, list(pair)] = 1
    if expand and w > 1:
        incidence = algorithm_1(incidence)
    edge_list = [tuple(np.flatnonzero(row).tolist()) for row in incidence]
    graph = nx.Graph()
    graph.add_nodes_from(range(w))
    graph.add_edges_from(edge_list)
    if not nx.is_connected(graph):
        raise ValueError("gauging requires a connected auxiliary graph")

    first_index = {edge: i for i, edge in reversed(list(enumerate(edge_list)))}
    f0 = np.zeros((len(H), len(edge_list)), np.uint8)
    for i, row in enumerate(restriction):
        vertices = np.flatnonzero(row).tolist()
        for a, b in zip(vertices[::2], vertices[1::2]):
            for u, v in pairwise(nx.shortest_path(graph, a, b)):
                f0[i, first_index[tuple(sorted((u, v)))]] ^= 1

    # Flux checks: an independent set of Horton cycles, lightest first, spanning all
    # |E| - |V| + 1 cycles (including two-edge cycles of parallel edges).
    span, cycles = XorBasis(), []
    for cycle in _horton_candidates(incidence):
        if span.add(sum(1 << e for e in cycle)):
            cycles.append(np.isin(np.arange(len(edge_list)), list(cycle)).astype(np.uint8))
            if len(cycles) == len(edge_list) - w + 1:
                break
    partial_0 = np.array(cycles, np.uint8).reshape(-1, len(edge_list))
    if cellulate_to == "native":
        cellulate_to = max(3, int(H.sum(axis=1).max(initial=0)))
    if cellulate_to is not None:
        partial_0, incidence, f0 = cellulate(partial_0, incidence, f0, target_weight=cellulate_to)
    f1 = np.zeros((len(x), w), np.uint8)
    f1[support, np.arange(w)] = 1
    return f1, incidence, f0, partial_0


def _restore_flux_checks(
    gadgets: Sequence[GadgetSpec], H_comp: np.ndarray, f0: np.ndarray, partial_0: np.ndarray
) -> np.ndarray:
    """Append the cycles of each auxiliary graph that the parallel code no longer generates.

    A gadget may omit a cycle that equals a product of deformed data checks. With several
    gadgets on shared data, that product is the sum of the cycles of all gadgets, so
    each cycle is no longer generated on its own. Cycles independent of the
    complementary checks and of ``partial_0`` are added from a minimum cycle basis of
    each graph.
    """
    n = H_comp.shape[1]
    span = XorBasis()
    for data_row, aux_row in zip(H_comp, f0):
        span.add(bitmask(data_row) | bitmask(aux_row) << n)
    for row in partial_0:
        span.add(bitmask(row) << n)
    extra, offset, cycle_bases = [], 0, {}
    for g in gadgets:
        key = g.incidence.shape, g.incidence.tobytes()
        if key not in cycle_bases:
            no_checks = np.zeros((0, n), np.uint8), np.zeros((0, len(g.incidence)), np.uint8)
            cycle_bases[key] = algorithm_2(g.incidence, *no_checks)
        for row in cycle_bases[key]:
            if span.add(bitmask(row) << (n + offset)):
                extra.append(np.zeros(f0.shape[1], np.uint8))
                extra[-1][offset : offset + len(row)] = row
        offset += len(g.incidence)
    return np.vstack([partial_0, *extra]) if extra else partial_0


def parallel_gadgets(gadgets: Sequence[GadgetSpec], *, validate: bool = True) -> GadgetSpec:
    """One gadget measuring the logicals of all ``gadgets`` (same code and Pauli type).

    The auxiliary systems are disjoint and attach to shared data, so the measured
    supports may overlap; the data checks appear once, extended by every f0. The
    readouts must be independent modulo the measured-type stabilisers, and cycles that
    sharing leaves ungenerated are restored as flux checks. With ``validate``, the
    result is also checked to measure exactly the declared logicals and to add no
    logical qubits.
    """
    first = gadgets[0]
    for g in gadgets:
        same_code = all(
            np.array_equal(a, b)
            for a, b in [
                (g.code.matrix_x, first.code.matrix_x),
                (g.code.matrix_z, first.code.matrix_z),
            ]
        )
        if g.basis is not first.basis or not same_code:
            raise ValueError("parallel gadgets must use the same data code and Pauli basis")
    H_meas, H_comp = basis_matrices(first.code, first.basis)
    batch = np.vstack([g.effective_batch_x for g in gadgets])
    span = XorBasis()
    for row in H_meas:
        span.add(bitmask(row))
    rank_stabilizers = len(span)
    for row in batch:
        span.add(bitmask(row))
    if len(span) - rank_stabilizers != len(batch):
        raise ValueError("batch readouts must be independent modulo data stabilizers")
    f0 = np.hstack([g.f0 for g in gadgets])
    result = GadgetSpec.from_maps(
        first.code,
        batch[0],
        basis=first.basis,
        f1=np.hstack([g.f1 for g in gadgets]),
        incidence=block_diag(*(g.incidence for g in gadgets)),
        f0=f0,
        partial_0=_restore_flux_checks(
            gadgets, H_comp, f0, block_diag(*(g.partial_0 for g in gadgets))
        ),
        batch_x=batch,
        readout_vectors=block_diag(*(g.effective_readout_vectors for g in gadgets)),
        validate=validate,
    )
    if validate:
        # Every product of new checks without auxiliary support, f1(ker d1), must be a
        # declared readout up to stabilisers.
        measured = np.asarray(GF2(result.incidence).null_space() @ GF2(result.f1.T))
        if any(span.reduce(bitmask(row)) for row in measured):
            raise ValueError("gadget has undeclared logical measurements")
        data_k = H_meas.shape[1] - gf2_rank(H_meas) - gf2_rank(H_comp)
        hx, hz = result.HX_merged, result.HZ_merged
        if hx.shape[1] - gf2_rank(hx) - gf2_rank(hz) != data_k - len(batch):
            raise ValueError("parallel gadget introduces ancillary logical degrees of freedom")
    return result
