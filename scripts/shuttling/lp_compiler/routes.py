"""Whole-tile shuttles of check groups, with passing lanes where tiles would collide.

Wrapping rows/columns and blocked homeward tiles step 6 um off the lattice, pass the
parked checks, then realign. Interfaces (f0, f1) use a physical-y lane before crossing in
physical x and dock 2 um beside the target. Returns run one group at a time per bank.
Local permutations and general whole-device collision avoidance are excluded.
"""

from collections import defaultdict

import numpy as np

RETURN_CLEARANCE = 3.0
PASSING_OFFSET = 6.0


def return_path(replay, job):
    """Check complete segments against other check atoms in the same bank.

    A tile is a grid of atoms, so a half-pitch lane can pass between the
    parked grid's rows/columns. The 3 um clearance is a routing convention,
    not a model of trap size. Test both sides, including departure/arrival.
    """
    ids = [i for key in job["keys"] for i in replay.fids[key]]
    other = [
        i
        for key, members in replay.fids.items()
        if key[0]
        and key not in job["keys"]
        and replay.bank_of[replay.c["fibres"][key]["module"]] == job["bank"]
        for i in members
    ]
    points = replay.pos[ids]
    obstacles = replay.pos[other]
    delta = np.array([job["dx"], job["dy"]])

    def clear(path):
        origin = points.copy()
        for step in path:
            # Bound memory for large Cartesian groups; each segment is checked
            # continuously, independently of the animation's frame rate.
            for offset in range(0, len(origin), 64):
                relative = obstacles[None, :, :] - origin[offset : offset + 64, None, :]
                fraction = np.clip(relative @ step / np.dot(step, step), 0, 1)
                separation = relative - fraction[:, :, None] * step
                if np.any(np.sum(separation**2, axis=2) < RETURN_CLEARANCE**2 - 1e-8):
                    return False
            origin += step
        return True

    if clear([delta]):
        return
    axis = int(np.argmax(np.abs(delta)))
    for sign in (1, -1):
        offset = np.zeros(2)
        offset[1 - axis] = sign * PASSING_OFFSET
        path = [offset, delta, -offset]
        if clear(path):
            job.update(path=[step.tolist() for step in path], returnOffset_um=offset.tolist())
            return
    raise AssertionError(("No clear return lane", replay.c["mode"], replay.p, job["keys"]))


def route_blocks(
    replay, desired, basis, factor, label, start, shift_times, wraps=None, interface=False
):
    """Dispatch the translations that bring each check group to its ``desired`` position:
    homeward groups first (nearest first, one per bank at a time), then the others, then
    the realignment of wrapping groups."""
    r = replay
    wraps = wraps or {}
    buckets = defaultdict(list)
    for key, (target, _) in desired.items():
        dx, dy = [round(target[i] - r.base[key][i], 8) for i in range(2)]
        if not (dx or dy):
            continue
        bank = r.bank_of[r.c["fibres"][key]["module"]]
        home = np.allclose(target, r.c["fibres"][key]["home"])
        buckets[bank, dx, dy, wraps.get(key), home].append(key)
    jobs, home_jobs = [], defaultdict(list)
    for (bank, dx, dy, axis, home), keys in buckets.items():
        for group in r.split_rectangles(keys):
            job = dict(bank=bank, keys=group, dx=dx, dy=dy, kind="translation")
            if interface:
                # f0/f1 travel between modules along physical x. Enter a
                # negative half-grid y lane before crossing either module,
                # then dock at y - 2, opposite concurrent ordinary checks.
                # Retrace the lane on return before realigning home.
                departure = -PASSING_OFFSET + dy if home else -PASSING_OFFSET
                job.update(
                    path=[[0, departure], [dx, 0], [0, dy - departure]],
                    interfaceOffset_um=[0, -PASSING_OFFSET],
                )
            if home:
                home_jobs[bank].append(job)
                continue
            if axis is not None:
                assert (dy if axis == 0 else dx) == 0, "wrap must follow one row or column"
                job["wrapAxis"] = axis
            jobs.append(job)
    # Vacate skipped slots before another sweep row enters them. Park nearer
    # groups first, so later returns see their actual parked positions. One
    # homeward group per bank keeps passing lanes exclusive, while bank clocks
    # and the local cyclic-permutation pair budget remain independent.
    elapsed = shift_times
    for group in home_jobs.values():
        group.sort(key=lambda job: np.hypot(job["dx"], job["dy"]))
    for index in range(max(map(len, home_jobs.values()), default=0)):
        batch = [group[index] for group in home_jobs.values() if index < len(group)]
        for job in batch:
            if not interface:
                return_path(r, job)
        r.clock = start
        elapsed = r.dispatch(batch, basis, factor, label + " · homeward shuttle", offsets=elapsed)
    # Preserve independent bank clocks and the original direct-move scheduler.
    r.clock = start
    elapsed = r.dispatch(jobs, basis, factor, label + " · shuttle", offsets=elapsed)
    # Only the wrapping blocks wait off the common grid while ordinary jobs
    # vacate their destinations. The ordinary paths do not gain extra steps.
    returns = []
    for job in jobs:
        if "wrapAxis" not in job:
            continue
        axis = job["wrapAxis"]
        returns.append(
            dict(
                bank=job["bank"],
                keys=job["keys"],
                dx=-6 if axis == 1 else 0,
                dy=-6 if axis == 0 else 0,
                kind="translation",
                wrapReturnAxis=axis,
            )
        )
    r.clock = start
    r.dispatch(returns, basis, factor, label + " · realign wrapping blocks", offsets=elapsed)
