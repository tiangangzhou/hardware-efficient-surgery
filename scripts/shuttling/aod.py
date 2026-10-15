"""Gate layers and check-atom destinations of one syndrome round of a graph gadget.

For the graph-based baselines the check atoms sit in a one-dimensional register and
are rearranged between gate layers (SM, "Shuttling depth and timing", physical
implementation). ``compile_check_round`` splits the X and then the Z Tanner graph
into Delta matchings (Delta = maximum check or qubit degree, Koenig's edge colouring)
and gives, for each matching, the slot of every check atom: a matched check moves to
its qubit's slot, an unmatched one is parked beyond the data register, and vacancies
fill the remaining slots. ``build_graph_gadgets.py`` checks these layers against the
graph models embedded in the shuttling animation, which times the rearrangements.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import maximum_bipartite_matching


def matching_layers(matrix: np.ndarray) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Split the edges (check, qubit) of a binary matrix into Delta matchings.

    The bipartite graph is completed with dummy edges to a Delta-regular multigraph,
    from which perfect matchings are removed one at a time (Hall's theorem guarantees
    them); dummy edges are dropped from the output.
    """
    rows, cols = np.nonzero(matrix)
    if not len(rows):
        return ()
    n = max(matrix.shape)
    real = set(zip(rows.tolist(), cols.tolist()))
    counts: list[dict[int, int]] = [{} for _ in range(n)]
    for r, c in real:
        counts[r][c] = 1
    row_degree = np.bincount(rows, minlength=n)
    col_degree = np.bincount(cols, minlength=n)
    delta = int(max(row_degree.max(), col_degree.max()))
    row_deficit, col_deficit = delta - row_degree, delta - col_degree
    r = c = 0
    while r < n and c < n:  # dummy edges, filling row and column deficits in order
        if not row_deficit[r]:
            r += 1
        elif not col_deficit[c]:
            c += 1
        else:
            amount = int(min(row_deficit[r], col_deficit[c]))
            counts[r][c] = counts[r].get(c, 0) + amount
            row_deficit[r] -= amount
            col_deficit[c] -= amount
    layers = []
    for _ in range(delta):
        edges = [(r, c) for r, entries in enumerate(counts) for c in sorted(entries)]
        graph = sparse.csr_matrix((np.ones(len(edges)), tuple(zip(*edges))), shape=(n, n))
        layer = []
        for r, c in enumerate(maximum_bipartite_matching(graph, perm_type="column").tolist()):
            counts[r][c] -= 1
            if not counts[r][c]:
                del counts[r][c]
            if (r, c) in real:
                layer.append((r, c))
                real.remove((r, c))
        layers.append(tuple(layer))
    return tuple(layers)


@dataclass(frozen=True)
class Route:
    """Slot ``destinations[i]`` of the atom in slot i; the first ``active_count``
    slots hold check atoms, the others are vacancies."""

    destinations: tuple[int, ...]
    active_count: int


@dataclass(frozen=True)
class MatchingLayer:
    basis: str
    pairs: tuple[tuple[int, int], ...]
    route: Route


@dataclass(frozen=True)
class CheckSchedule:
    layers: tuple[MatchingLayer, ...]


def compile_check_round(hx: np.ndarray, hz: np.ndarray) -> CheckSchedule:
    """Matching layers of the X checks, then of the Z checks, with their routes.

    Slots 0..n-1 are the data qubits; check i of a layer goes to the slot of its
    matched qubit, and the unmatched checks to slots n, n + 1, ... in order.
    """
    layers = []
    for basis, h in [("X", hx), ("Z", hz)]:
        m, n = h.shape
        for pairs in matching_layers(h):
            destinations = np.full(m + n, -1)
            for r, c in pairs:
                destinations[r] = c
            idle = np.flatnonzero(destinations[:m] < 0)
            destinations[idle] = n + np.arange(len(idle))
            free = np.setdiff1d(np.arange(m + n), destinations)
            destinations[destinations < 0] = free
            layers.append(MatchingLayer(basis, pairs, Route(tuple(destinations.tolist()), m)))
    return CheckSchedule(tuple(layers))
