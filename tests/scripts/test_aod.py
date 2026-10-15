"""Gate layers and check-atom destinations of the graph gadgets (``aod.py``)."""

from __future__ import annotations

import numpy as np

from tests.scripts.helpers import load

aod = load("shuttling/aod.py")
compile_check_round, matching_layers = aod.compile_check_round, aod.matching_layers


def test_matching_layers_cover_every_edge_once_in_delta_layers() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        h = (rng.random((int(rng.integers(1, 8)), 9)) < 0.4).astype(np.uint8)
        layers = matching_layers(h)
        assert len(layers) == max(h.sum(axis=0).max(), h.sum(axis=1).max())  # Koenig
        for layer in layers:
            assert len({r for r, _ in layer}) == len({c for _, c in layer}) == len(layer)
        assert sorted(p for layer in layers for p in layer) == list(zip(*np.nonzero(h)))
    assert matching_layers(np.zeros((2, 5), np.uint8)) == ()


def test_check_atoms_go_to_their_qubit_or_park_beyond_the_data() -> None:
    hx = np.array([[1, 1, 1], [0, 1, 0], [0, 0, 1]], np.uint8)
    hz = np.array([[1, 0, 1]], np.uint8)
    schedule = compile_check_round(hx, hz)
    assert [layer.basis for layer in schedule.layers] == ["X"] * 3 + ["Z"] * 2
    for layer in schedule.layers:
        h = hx if layer.basis == "X" else hz
        route = layer.route
        assert sorted(route.destinations) == list(range(sum(h.shape)))
        assert route.active_count == len(h)
        matched = dict(layer.pairs)
        for check, slot in enumerate(route.destinations[: len(h)]):
            assert slot == matched[check] if check in matched else slot >= h.shape[1]
