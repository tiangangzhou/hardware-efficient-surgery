"""The model families of the animation: algebraic surgery, memory and bare LP routes.

Strings and provenance labels are those of the published page (``provenance.source``
names the original modules: ``surgery_modes`` = ``layout`` + ``rounds`` + ``replay``,
``bare`` = ``bare_replay`` here).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .layout import ATOM_PITCH, CHECK_FIBRES, DATA_FIBRES, MODES, TILE_PITCH, make_case
from .replay import Replay, witnesses
from .rounds import streams
from .schedule import apply_timing

LIFTS = (9, 15, 33)
PAIRS = (1, 2)  # AOD pairs per code


def animation_models() -> tuple[dict, list[dict]]:
    """(``aod-model-data`` payload, bare LP models), in the order the page stores them."""
    models = [
        surgery_model(lift, mode, pairs) for mode in MODES for lift in LIFTS for pairs in PAIRS
    ]
    bare_models = []
    for lift in LIFTS:
        for pairs in PAIRS:
            bare_models.append(bare_model(lift, pairs))
            models.append(memory_model(lift, pairs))
    payload = dict(
        models=models,
        assumptions=dict(
            acceleration=5500,
            gate_us=1,
            spacing_um=12,
            docking_um=2,
            included="Whole-group then overflow (1 pair), complementary subsets together (2 pairs); wrap lanes and blocked-return detours",
            excluded="Readout/reset, trap transfers and within-block collision avoidance",
        ),
    )
    return payload, bare_models


def surgery_model(p, mode, pairs):
    """Fibre, packet or adapter surgery at lift ``p`` with ``pairs`` AOD pairs per code."""
    c = make_case(p, mode)
    c["factorName"] = {"fibre": "A′", "packet": "A′g", "adapter": "A′ad"}[mode]
    model = apply_timing(Replay(c, pairs).finish())
    witnesses(model)
    model["layout"]["localMotion"] = dict(
        version="pair-budget-v1",
        collisionAvoidance=False,
        pairsUsed=min(pairs, 2),
        pairDefinition="Two perpendicular AODs, one x and one y; one 2D trap grid",
    )
    return model


def bare_replay(p, pairs, *, restore_labels=True):
    """Compile a syndrome cycle, optionally retaining the final fibre shifts.

    Graph attachment starts from labelled check homes and needs the default.
    Memory can retain permutations and reassign checks after readout/reset,
    using the same endpoint policy as fibre surgery.
    """
    c = make_case(p, "fibre")
    c["factorName"] = "A′"
    data_streams = {b: streams(c, b)[0] for b in "XZ"}
    c["fibres"] = {k: v for k, v in c["fibres"].items() if v["module"] == 0}
    # Replay initializes one bank per original code before we retain the sole
    # data bank. Auxiliary atoms never enter the replay or pulse calculation.
    r = Replay(c, pairs)
    c["modules"] = [m for m in c["modules"] if m["id"] == 0]
    r.modules = c["modules"]
    r.banks = [b for b in r.banks if b["id"] == 0]
    times = {}
    for basis in "XZ":
        start = r.clock
        data = data_streams[basis]
        kind = 1 if basis == "X" else 2
        for k, s in enumerate(data):
            if k and s["factor"] != data[k - 1]["factor"]:
                checks = {q for group in data[k - 1]["groups"] for q in group}
                r.prepare(
                    {(kind, q): (c["fibres"][kind, q]["home"], 0) for q in checks},
                    basis,
                    "return",
                    "Restore factor sheet order",
                )
                r.gates[-1]["returnEnd"] = r.clock
            r.round(basis, [s], f"Bare {basis} round {k + 1}")
        r.prepare(
            {
                key: (v["home"], 0 if restore_labels else None)
                for key, v in c["fibres"].items()
                if key[0] == kind
            },
            basis,
            "return",
            f"Bare {basis} "
            + ("exact labelled home" if restore_labels else "return to home fibres"),
        )
        r.gates[-1]["returnEnd"] = r.clock
        times[basis] = r.clock - start
        for ids in r.fids.values():
            assert set(map(tuple, r.pos[ids])) == {tuple(r.atoms[i][:2]) for i in ids}
        if restore_labels:
            assert np.allclose(r.pos, np.array([a[:2] for a in r.atoms]))
    times["all"] = r.clock
    return r, times


def bare_model(p, pairs, *, restore_labels=True):
    """Bare 2D transport of one X + Z syndrome cycle of the data code (labelled homes
    restored), the routes the graph gadgets attach to."""
    r, times = bare_replay(p, pairs, restore_labels=restore_labels)
    c = r.c
    atoms = [
        dict(
            id=i,
            kind=("data", "X", "Z")[a[2]],
            index=a[4] * p + a[5],
            x=a[0],
            y=a[1],
            fibre=a[4],
            sheet=a[5],
        )
        for i, a in enumerate(r.atoms)
    ]
    gates = []
    for g in r.gates:
        out = {k: g[k] for k in ("t", "end", "prepStart", "returnEnd", "basis", "algebra")}
        out["pairs"] = [[atoms[a]["index"], atoms[b]["index"]] for a, b in g["edges"]]
        gates.append(out)
    moves = [mv[:6] for mv in r.moves]
    xy = np.array([a[:2] for a in r.atoms], float)
    lo = xy.copy()
    hi = xy.copy()
    for _t, _d, dx, dy, _pair, ids in moves:
        xy[ids] += [dx, dy]
        lo[ids] = np.minimum(lo[ids], xy[ids])
        hi[ids] = np.maximum(hi[ids], xy[ids])
    bounds = [
        float(lo[:, 0].min() - 18),
        float(hi[:, 0].max() + 18),
        float(lo[:, 1].min() - 24),
        float(hi[:, 1].max() + 24),
    ]
    frames = []
    for name, kind, span in [
        ("V×V data", "data", [0, 5, 0, 5]),
        ("X checks", "X", [0, 5, 5, 8]),
        ("Z checks", "Z", [5, 8, 0, 5]),
        ("C×C data", "data", [5, 8, 5, 8]),
    ]:
        x0, x1, y0, y1 = span
        frames.append(
            dict(
                name=name,
                kind=kind,
                bounds=[TILE_PITCH * x0 - 6, TILE_PITCH * x1 - 6, 4 * p * y0 - 6, 4 * p * y1 - 6],
            )
        )
    fibres = [
        dict(kind=("data", "X", "Z")[kind], index=i, home=v["home"], atomIds=r.fids[kind, i])
        for (kind, i), v in c["fibres"].items()
    ]
    return dict(
        lift=p,
        pairs=pairs,
        atoms=atoms,
        moves=moves,
        gates=gates,
        total=r.clock,
        sectorTimesUs=times,
        bounds=bounds,
        blockFrames=frames,
        fibres=fibres,
        Aexponents=[[None if v is None else v[0] for v in row] for row in c["A"]],
        layout=dict(
            fibre_shape=[3, p // 3],
            tile_pitch_um=[TILE_PITCH, 4 * p],
            atom_pitch_um=[ATOM_PITCH, ATOM_PITCH],
            placement="row-major" if p == 9 else "integer-CRT",
        ),
        provenance=dict(
            source="surgery_modes.make_case + streams(data only) + Replay",
            closedCycle=(
                "Every check returns to its exact labelled home, including shift zero."
                if restore_labels
                else "Checks return to home fibre slots; reassign checks only after readout/reset."
            ),
            transport="One/two crossed pairs execute distinct axis stages; blocked whole-tile returns use 6 um passing lanes, with local collision avoidance excluded.",
            excluded="Readout/reset, trap transfer, check/data crossings during macro transport and full-device collision validation.",
        ),
    )


def memory_model(lift: int, pairs: int) -> dict[str, Any]:
    """One X + Z syndrome cycle of the data code alone (checks may end permuted)."""
    replay, times = bare_replay(lift, pairs, restore_labels=False)
    xy = np.array([a[:2] for a in replay.atoms], float)
    low, high = xy.copy(), xy.copy()
    for move in replay.moves:
        ids = move[5]
        xy[ids] += move[2:4]
        low[ids] = np.minimum(low[ids], xy[ids])
        high[ids] = np.maximum(high[ids], xy[ids])
    bounds = [
        float(low[:, 0].min() - 18),
        float(high[:, 0].max() + 18),
        float(low[:, 1].min() - 24),
        float(high[:, 1].max() + 24),
    ]
    replay.modules[0]["transportBounds"] = bounds
    for gate in replay.gates:
        gate.update(bank=0, block="bank0")
        for group in gate["algebra"]:
            group.update(basis=gate["basis"], label="Data code")
    sectors = {}
    for basis in "XZ":
        offset = 0 if basis == "X" else times["X"]
        sectors[basis] = dict(
            taskStarts=[t["start"] - offset if t["basis"] == basis else None for t in replay.tasks],
            gateTimes=[
                [g[k] - offset for k in ("t", "end", "prepStart", "returnEnd")]
                if g["basis"] == basis
                else None
                for g in replay.gates
            ],
            total=times[basis],
        )
    model = dict(
        mode="memory",
        modeLabel="Data-code memory benchmark",
        mission="memory",
        lift=lift,
        pairsPerCode=pairs,
        capacity=pairs,
        allocation="dedicated-per-code",
        atoms=replay.atoms,
        fixed=replay.fixed,
        moves=replay.moves,
        tasks=replay.tasks,
        gates=replay.gates,
        total=times["all"],
        sectorTimesUs=times,
        sectorSchedules=sectors,
        marks=[dict(t=0, label="X sector"), dict(t=times["X"], label="Z sector")],
        modules=replay.modules,
        banks=replay.banks,
        dataCopies=1,
        factorName="A",
        sparseX=[],
        algebraDimensions=dict(mA=3, nA=5, mD=0, nD=0, mC=0, nC=0),
        readoutLabel=f"{DATA_FIBRES * lift} data qubits · {CHECK_FIBRES * lift} X checks + {CHECK_FIBRES * lift} Z checks · one memory cycle",
        layout=dict(
            bounds=bounds,
            fibre_shape=[3, lift // 3],
            tile_pitch_um=[TILE_PITCH, 4 * lift],
            atom_pitch_um=[ATOM_PITCH, ATOM_PITCH],
            placement="row-major" if lift == 9 else "integer-CRT",
            localMotion=dict(
                version="pair-budget-v1",
                collisionAvoidance=False,
                pairsUsed=min(pairs, 2),
                pairDefinition="One x AOD and one y AOD form one 2D grid",
            ),
        ),
        audit=dict(
            gate_frames=len(replay.gates),
            physical_moves=len(replay.moves),
            home_fibre_slots="verified",
            cartesian_pickups="verified",
        ),
        provenance=dict(
            source="bare.replay with retained final fibre permutations, as in fibre surgery",
            closedCycle="X and Z checks return to home fibre slots; reassign checks only after readout/reset. The parallel runtime restores labelled homes after the final round.",
            endpointPolicy="retain-final-fibre-permutations",
            included="Check transport, blocked-return passing lanes and entangling pulses for one X+Z syndrome cycle",
            excluded="Readout/reset and trap-transfer durations; within-block collision avoidance and check/data crossings",
        ),
    )
    witnesses(model)
    return model
