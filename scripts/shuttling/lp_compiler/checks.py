"""Independent checks of compiled models: measured matrices and physical constraints.

``expected`` rebuilds the merged check matrices over R_l from the instance (not from the
compiler's layout), and ``check_models`` replays every model's moves and pulses: each
gate measures exactly the expected matrix entries, data never moves, pickups are full
grids, no atom moves during its pulse or sits in two moves, pairs and banks are never
double-booked, every view contains its atoms, and zero cells leave checks at home.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from scipy.spatial import cKDTree

from algebraic_qec import CyclicPolynomial as Poly
from algebraic_qec import PolynomialMatrix as PM
from algebraic_qec.examples.surgery_instances import instance

from .layout import CHECK_FIBRES, DATA_FIBRES, MODES
from .models import LIFTS, PAIRS


def expected(p, mode):
    """(H_X, H_Z) of the merged code of ``mode`` (the data code alone for memory)."""
    source = instance(p)
    A, D = source.factor_a, source.factor_d
    d, n = D.shape
    q = source.couple_column
    identity = lambda n: PM.identity(p, n)
    e = lambda j: PM(p, (1, 5), {(0, j): Poly.one(p)})
    C = A.vstack(e(4))
    copies = 1
    if mode == "packet":
        for j in range(4):
            C = C.vstack(PM(p, (1, 5), {(0, j): Poly(p, [0, 1])}))
    if mode == "adapter":
        C2 = A.vstack(e(3))
        C = C.hstack(PM.zero(p, 4, 5)).vstack(PM.zero(p, 4, 5).hstack(C2))
        C = C.vstack(PM(p, (1, 10), {(0, 3): Poly.one(p), (0, 9): Poly(p, [1])}))
        copies = 2
    m, r = C.shape
    nd = DATA_FIBRES * copies
    nc = CHECK_FIBRES * copies
    hx = A.kron(identity(5)).hstack(identity(3).kron(A.adjoint()))
    hz = identity(5).kron(A).hstack(A.adjoint().kron(identity(3)))
    if copies == 2:
        zero = PM.zero(p, CHECK_FIBRES, DATA_FIBRES)
        hx = hx.hstack(zero).vstack(zero.hstack(hx))
        hz = hz.hstack(zero).vstack(zero.hstack(hz))
    if mode == "memory":
        return hx, hz
    d1 = C.kron(identity(n)).vstack(identity(r).kron(D))
    d0 = identity(m).kron(D).hstack(C.kron(identity(d)))
    f1 = PM(
        p,
        (nd, r * n),
        {
            (DATA_FIBRES * b + 5 * i + 3 + b, (5 * b + i) * n + q): Poly.one(p)
            for b in range(copies)
            for i in range(5)
        },
    )
    f0 = PM(
        p,
        (nc, m * n + r * d),
        {
            (CHECK_FIBRES * b + 5 * a + 3 + b, (4 * b + a) * n + q): Poly.one(p)
            for b in range(copies)
            for a in range(3)
        },
    )
    merged_x = hx.hstack(f0).vstack(PM.zero(p, m * d, nd).hstack(d0))
    merged_z = hz.hstack(PM.zero(p, nc, m * n + r * d)).vstack(f1.adjoint().hstack(d1.adjoint()))
    return merged_x, merged_z


def check_models(payload):
    """Check every model of ``payload``; one report per model."""
    families = {*MODES, "memory"}
    assert {(m.get("mode"), m["lift"], m["pairsPerCode"]) for m in payload["models"]} == {
        (s, p, b) for s in families for p in LIFTS for b in PAIRS
    }, "Need all lift/budget schedules for all four families"
    reports = []
    for model in payload["models"]:
        p, mode = model["lift"], model["mode"]
        atoms = model["atoms"]
        xy = np.array([a[:2] for a in atoms], float)
        fixed = model["fixed"]
        start = xy.copy()
        banks = {z["id"]: z for z in model["banks"]}
        owner = {m["id"]: m["bank"] for m in model["modules"]}

        def visible(points, bounds):
            return (
                points[:, 0].min() >= bounds[0] - 1e-5
                and points[:, 0].max() <= bounds[1] + 1e-5
                and points[:, 1].min() >= bounds[2] - 1e-5
                and points[:, 1].max() <= bounds[3] + 1e-5
            )

        module_bounds = {m["id"]: m.get("transportBounds", m["bounds"]) for m in model["modules"]}
        assert visible(xy, model["layout"]["bounds"]), ("initial view clips atoms", mode, p)
        for basis, M in zip(("X", "Z"), expected(p, mode), strict=False):
            if model.get("measurement", "all") not in ("all", basis):
                continue
            actual = Counter(
                (atoms[a][4] * p + atoms[a][5], atoms[b][4] * p + atoms[b][5])
                for g in model["gates"]
                if g["basis"] == basis
                for a, b in g["edges"]
            )
            coo = M.lift().tocoo()
            assert actual == Counter(zip(coo.row, coo.col, strict=False)), (
                mode,
                p,
                basis,
                "wrong measured matrix",
            )
        hx, hz = expected(p, mode)
        assert hx @ hz.adjoint() == PM.zero(p, hx.shape[0], hz.shape[0])
        events = []
        for mi, mv in enumerate(model["moves"]):
            t, d, dx, dy, pair, ids, task = mv
            assert abs(d - 2e6 * np.sqrt(np.hypot(dx, dy) * 1e-6 / 5500)) < 3e-6
            assert all(i >= fixed for i in ids), "Data moved"
            assert all(pair in banks[owner[atoms[i][3]]]["pairs"] for i in ids), (
                "Cross-bank movement"
            )
            events.append((round(t, 5), 1, mi))
            events.append((round(t + d, 5), 0, mi))
        for gi, g in enumerate(model["gates"]):
            events.append((g["t"] + 0.5, 2, gi))
        # Cartesian pickup coordinates at every move start, exact pulse endpoints,
        # and zero-carve fibres at home are checked independently of the compiler.
        running = {}
        active_atoms = set()
        for _t, kind, i in sorted(events):
            if kind == 1:
                mv = model["moves"][i]
                ids = mv[5]
                assert not active_atoms.intersection(ids), ("atom scheduled twice", mode, p, i)
                pts = np.round(xy[ids], 6)
                assert len(ids) == len(set(pts[:, 0])) * len(set(pts[:, 1])), (
                    "non-Cartesian pickup",
                    mode,
                    p,
                    i,
                )
                running[i] = ids
                active_atoms.update(ids)
            elif kind == 0:
                mv = model["moves"][i]
                ids = running.pop(i)
                xy[ids] += mv[2:4]
                active_atoms.difference_update(ids)
                assert visible(xy[ids], model["layout"]["bounds"]), (
                    "full view clips transport",
                    mode,
                    p,
                    i,
                )
                for module in {atoms[a][3] for a in ids}:
                    selected = [a for a in ids if atoms[a][3] == module]
                    assert visible(xy[selected], module_bounds[module]), (
                        "module view clips transport",
                        mode,
                        p,
                        i,
                        module,
                    )
            else:
                g = model["gates"][i]
                illuminated = sorted({a for edge in g["edges"] for a in edge})
                assert active_atoms.isdisjoint(illuminated), ("moving gate atom", mode, p, i)
                pairs = {
                    tuple(sorted((illuminated[a], illuminated[b])))
                    for a, b in cKDTree(xy[illuminated]).query_pairs(2.000001)
                }
                assert pairs == {tuple(sorted(e)) for e in g["edges"]}, (
                    "false physical gate",
                    mode,
                    p,
                    i,
                )
                all_indices = []
                for group in g["algebra"]:
                    for cell in group["active"]:
                        assert cell["exponent"] in group["matrix"][cell["row"]][cell["col"]]
                        for ei in cell["edgeIndices"]:
                            a, b = g["edges"][ei]
                            assert (atoms[b][5] - atoms[a][5]) % p == cell["exponent"]
                        all_indices += cell["edgeIndices"]
                    for cell in group["skipped"]:
                        assert group["matrix"][cell["row"]][cell["col"]] is None
                        for a in cell["checkFibres"]:
                            # Home support, allowing a retained sheet permutation.
                            home = {tuple(v[:2]) for v in atoms if v[2:5] == atoms[a][2:5]}
                            assert tuple(np.round(xy[a], 6)) in home, (
                                "zero left home",
                                mode,
                                p,
                                i,
                                a,
                            )
                assert sorted(all_indices) == list(range(len(g["edges"])))
        assert np.array_equal(xy[:fixed], start[:fixed])
        for pair in range(model["capacity"]):
            moves = sorted((v for v in model["moves"] if v[4] == pair), key=lambda v: v[0])
            assert all(a[0] + a[1] <= b[0] + 1e-5 for a, b in zip(moves, moves[1:], strict=False))
            tasks = sorted(
                (t for t in model["tasks"] if pair in t["pairs"]), key=lambda t: t["start"]
            )
            assert all(
                a["end"] <= b["start"] + 1e-5 for a, b in zip(tasks, tasks[1:], strict=False)
            )
        for i, gate in enumerate(model["gates"]):
            used = {a for edge in gate["edges"] for a in edge}
            for move in model["moves"]:
                if min(gate["end"], move[0] + move[1]) > max(gate["t"], move[0]) + 1e-5:
                    assert used.isdisjoint(move[5]), "Gate atom moves during its pulse"
            for other in model["gates"][i + 1 :]:
                if min(gate["end"], other["end"]) > max(gate["t"], other["t"]) + 1e-5:
                    assert used.isdisjoint(a for edge in other["edges"] for a in edge), (
                        "overlapping gates on one atom"
                    )
        # Sparse carve X checks only move in a nonzero C interaction window.
        for key in model["sparseX"]:
            ids = {i for i, a in enumerate(atoms) if a[2] == 1 and a[4] == key}
            windows = [
                (g["prepStart"], g["returnEnd"])
                for g in model["gates"]
                if any(
                    atoms[a][4] == key and atoms[a][2] == 1
                    for group in g["algebra"]
                    if group["factor"] == "C"
                    for c in group["active"]
                    for ei in c["edgeIndices"]
                    for a, b in [g["edges"][ei]]
                )
            ]
            for mv in model["moves"]:
                if ids.intersection(mv[5]) and model["tasks"][mv[6]].get("factor") == "C":
                    assert any(
                        lo - 1e-5 <= mv[0] and mv[0] + mv[1] <= hi + 1e-5 for lo, hi in windows
                    ), ("zero carve transport", mode, p, key)
        reports.append(
            dict(
                mode=mode,
                lift=p,
                pairs=model["pairsPerCode"],
                frames=len(model["gates"]),
                total_ms=model["total"] / 1000,
            )
        )
    return reports
