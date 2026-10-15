"""AOD move times and the axis routes of a cyclic sheet shift inside one tile.

One pair means two perpendicular AODs generating one independently controlled x/y
trap grid. One pair shifts a whole selected group and then corrects its overflow; two
pairs move the complementary wrap / non-wrap subsets simultaneously. Local collision
avoidance is excluded. Durations exclude trap transfer and assume constant acceleration
5500 m/s^2 (accelerate, then decelerate).
"""

from __future__ import annotations

import math
from functools import cache

from .layout import ATOM_PITCH

SPACING = float(ATOM_PITCH)
ACCELERATION = 5500.0  # m/s^2


def move_us(distance_um):
    """Duration (us) of a move over ``distance_um``."""
    return 2e6 * math.sqrt(distance_um * 1e-6 / ACCELERATION)


def placement(lift, label):
    """(column, row) of sheet ``label`` in a tile of 3 x l/3 atoms: row-major for l = 9,
    by the Chinese remainder theorem (label mod 3, label mod l/3) otherwise."""
    return (label % 3, label // 3 if lift == 9 else label % (lift // 3))


@cache
def compile_shift(lift, shift, pairs):
    """Return Cartesian subset moves, barriers, and their actual durations.

    Label masks refer to the current canonical *site* labels. A caller with
    a retained shift must translate these masks back to physical atom IDs.
    Four available pairs still use two for this local construction; whole
    jobs can be scheduled independently only when their routes are compatible.
    """
    if lift not in (9, 15, 33) or pairs not in (1, 2, 4):
        raise ValueError("Supported lifts: 9, 15, 33; crossed-pair budgets: 1, 2, 4")
    shift %= lift
    used_pairs = min(pairs, 2)
    positions = {t: list(placement(lift, t)) for t in range(lift)}
    target = {t: placement(lift, (t + shift) % lift) for t in range(lift)}
    stages = []

    def part(labels, axis, distance, pair):
        return dict(
            labels=labels,
            dx_um=distance if axis == 0 else 0.0,
            dy_um=distance if axis == 1 else 0.0,
            duration_us=move_us(abs(distance)),
            pair=pair,
        )

    def emit(name, parts):
        stages.append(
            dict(name=name, parts=parts, duration_us=max(p["duration_us"] for p in parts))
        )
        for p in parts:
            for t in p["labels"]:
                positions[t][0] += p["dx_um"] / SPACING
                positions[t][1] += p["dy_um"] / SPACING

    def cycle(labels, axis, size, residue, name):
        residue %= size
        if not residue or not labels:
            return
        signed = residue if residue <= size / 2 else residue - size
        wrapping = [t for t in labels if not 0 <= positions[t][axis] + signed < size]
        ordinary = [t for t in labels if t not in wrapping]
        if used_pairs == 1:
            emit(name + " · whole group shift", [part(labels, axis, SPACING * signed, 0)])
            correction = -size if signed > 0 else size
            emit(name + " · overflow correction", [part(wrapping, axis, SPACING * correction, 0)])
        else:
            wrap_delta = signed - size if signed > 0 else signed + size
            wrap = part(wrapping, axis, SPACING * wrap_delta, 0)
            straight = part(ordinary, axis, SPACING * signed, 1)
            emit(name + " · wrap + non-wrap together", [wrap, straight])

    cycle(list(range(lift)), 0, 3, shift, "Horizontal cycle")
    if lift == 9:
        # Determine the carry from the labelled target, after the x cycle.
        # A signed shortest x route still implements the same modular carry.
        groups = {}
        for t in range(lift):
            residue = int(target[t][1] - positions[t][1]) % 3
            groups.setdefault(residue, []).append(t)
        for residue, labels in sorted(groups.items()):
            cycle(labels, 1, 3, residue, "Vertical column group +" + str(residue))
    else:
        cycle(list(range(lift)), 1, lift // 3, shift, "Vertical CRT cycle")
    assert all(tuple(positions[t]) == target[t] for t in range(lift))
    return dict(
        stages=stages,
        duration_us=sum(s["duration_us"] for s in stages),
        pairs=used_pairs,
        shift=shift,
        localStages=len(stages),
    )
