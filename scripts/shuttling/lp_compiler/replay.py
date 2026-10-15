"""Replay of a surgery (or memory) check schedule as timed atom moves, and its audit."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

from .layout import ATOM_PITCH, CHECK_FIBRES, DATA_FIBRES, TILE_PITCH, docking_offset
from .motion import compile_shift, move_us, placement
from .rounds import streams
from .routes import route_blocks
from .schedule import schedule_jobs


class Replay:
    """Compile a check schedule into timed moves of the check atoms.

    ``atoms`` are [x, y, kind, module, fibre, sheet] (data atoms first, never moved);
    ``moves`` are [start, duration, dx, dy, pair, atom ids, task]; ``tasks`` group the
    moves of one job, ``gates`` are the entangling pulses with their algebra.  Every
    bank (one per code) has ``pairs`` AOD pairs and its own clock.
    """

    def __init__(self, c, pairs):
        self.c = c
        self.p = c["p"]
        self.pairs = pairs
        self.clock = 0.0
        self.atoms = []
        self.ids = {}
        self.fids = {}
        self.sweep_state = {}
        self.base = {}
        self.shift = {}
        self.moves = []
        self.tasks = []
        self.gates = []
        self.marks = []
        self.modules = c["modules"]
        self.bank_of = {m["id"]: m["bank"] for m in self.modules}
        count = 2 * c["copies"]
        self.banks = [
            dict(
                id=b,
                block=f"bank{b}",
                name=next(m["name"] for m in self.modules if m["id"] == b),
                modules=[m["id"] for m in self.modules if m["bank"] == b],
                pairs=list(range(b * pairs, (b + 1) * pairs)),
            )
            for b in range(count)
        ]
        for key, meta in sorted(c["fibres"].items()):
            kind, idx = key
            self.base[key] = list(meta["home"])
            self.shift[key] = 0
            self.fids[key] = []
            for t in range(self.p):
                u, v = placement(self.p, t)
                a = len(self.atoms)
                self.ids[kind, idx, t] = a
                self.fids[key].append(a)
                self.atoms.append(
                    [
                        meta["home"][0] + ATOM_PITCH * u,
                        meta["home"][1] + ATOM_PITCH * v,
                        kind,
                        meta["module"],
                        idx,
                        t,
                    ]
                )
        self.fixed = sum(a[2] == 0 for a in self.atoms)
        self.pos = np.array([a[:2] for a in self.atoms], float)

    def rectangle(self, ids):
        """Whether the atoms ``ids`` form a full grid (one AOD pickup)."""
        xy = np.round(self.pos[ids], 7)
        return len(ids) == len(set(xy[:, 0])) * len(set(xy[:, 1]))

    def emit(self, t, d, dx, dy, pair, ids, task):
        ids = sorted(ids)
        assert self.rectangle(ids), ("Cartesian pickup", self.c["mode"], self.p, task)
        assert all(i >= self.fixed for i in ids)
        self.moves.append([t, d, dx, dy, pair, ids, task])
        self.pos[ids] += [dx, dy]

    def split_rectangles(self, keys):
        """Partition fibres into groups whose atoms each form a full grid."""
        # Test the complete product grid first: adding one fibre at a time can
        # temporarily look non-Cartesian while a new row is still incomplete.
        if self.rectangle([i for key in keys for i in self.fids[key]]):
            return [keys]
        groups = []
        for key in keys:
            for group in groups:
                ids = [i for k in group + [key] for i in self.fids[k]]
                if self.rectangle(ids):
                    group.append(key)
                    break
            else:
                groups.append([key])
        changed = True
        while changed:
            changed = False
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    if self.rectangle([a for key in groups[i] + groups[j] for a in self.fids[key]]):
                        groups[i] += groups.pop(j)
                        changed = True
                        break
                if changed:
                    break
        return groups

    def dispatch(self, jobs, basis, factor, label, offsets=None):
        """Schedule jobs (sheet permutations or translations) per bank and emit their moves."""
        offsets = offsets or {}
        elapsed = dict(offsets)
        for bank in self.banks:
            subset = [j for j in jobs if j["bank"] == bank["id"]]
            if not subset:
                continue
            entries = []
            for i, j in enumerate(subset):
                if j["kind"] == "permutation":
                    rec = compile_shift(self.p, j["delta"], self.pairs)
                    opts = [(rec["pairs"], rec["duration_us"], "axis_subsets")]
                else:
                    duration = sum(
                        move_us(math.hypot(*step)) for step in j.get("path", [[j["dx"], j["dy"]]])
                    )
                    if "wrapAxis" in j:
                        duration += move_us(6)
                    opts = [(1, duration, "translate")]
                entries.append(dict(id=i, deps=[], options=opts))
            plan = schedule_jobs(entries, self.pairs)
            bank_offset = offsets.get(bank["id"], 0.0)
            elapsed[bank["id"]] = bank_offset + plan["duration_us"]
            for task in sorted(plan["tasks"], key=lambda x: (x["start_us"], x["id"])):
                j = subset[task["id"]]
                start = self.clock + bank_offset + task["start_us"]
                end = self.clock + bank_offset + task["end_us"]
                tid = len(self.tasks)
                pairs = [x + bank["id"] * self.pairs for x in task["pairs"]]
                self.tasks.append(
                    dict(
                        start=start,
                        end=end,
                        label=label
                        + (
                            " · whole-block wrap"
                            if "wrapAxis" in j or "wrapReturnAxis" in j
                            else " · interface passing lane"
                            if "interfaceOffset_um" in j
                            else " · return detour"
                            if "returnOffset_um" in j
                            else ""
                        ),
                        pairs=pairs,
                        kind=j["kind"],
                        basis=basis,
                        block=bank["block"],
                        factor=factor,
                        **{
                            key: j[key]
                            for key in ("returnOffset_um", "interfaceOffset_um")
                            if key in j
                        },
                        **(
                            {"wrapAxis": j.get("wrapAxis", j.get("wrapReturnAxis"))}
                            if "wrapAxis" in j or "wrapReturnAxis" in j
                            else {}
                        ),
                    )
                )
                if j["kind"] == "permutation":
                    rec = compile_shift(self.p, j["delta"], self.pairs)
                    stages = rec["stages"]
                    self.tasks[-1].update(
                        delta=j["delta"], localStages=len(stages), method="axis_subsets"
                    )
                    self.tasks[-1]["label"] += " · " + str(len(stages)) + " local stages"
                    self.tasks[-1]["localPhases"] = []
                    stage_offset = 0.0
                    for stage in stages:
                        self.tasks[-1]["localPhases"].append(
                            dict(
                                start=start + stage_offset,
                                end=start + stage_offset + stage["duration_us"],
                                name=stage["name"],
                            )
                        )
                        for part in stage["parts"]:
                            pair = pairs[part["pair"]]
                            ids = [
                                self.ids[*key, (t - j["prior"]) % self.p]
                                for key in j["keys"]
                                for t in part["labels"]
                            ]
                            self.emit(
                                start + stage_offset,
                                part["duration_us"],
                                part["dx_um"],
                                part["dy_um"],
                                pair,
                                ids,
                                tid,
                            )
                        stage_offset += stage["duration_us"]
                    assert abs(stage_offset - (end - start)) < 1e-6
                    for key in j["keys"]:
                        self.shift[key] = j["target"]
                else:
                    ids = [i for k in j["keys"] for i in self.fids[k]]
                    if "path" in j:
                        at = start
                        for step in j["path"]:
                            duration = move_us(math.hypot(*step))
                            self.emit(at, duration, *step, pairs[0], ids, tid)
                            at += duration
                    elif "wrapAxis" in j:
                        offset = [0, 6] if j["wrapAxis"] == 0 else [6, 0]
                        lift = move_us(6)
                        self.emit(start, lift, *offset, pairs[0], ids, tid)
                        self.emit(
                            start + lift, end - start - lift, j["dx"], j["dy"], pairs[0], ids, tid
                        )
                    else:
                        self.emit(start, end - start, j["dx"], j["dy"], pairs[0], ids, tid)
                    for key in j["keys"]:
                        offset = (
                            ([0, 6] if j["wrapAxis"] == 0 else [6, 0])
                            if "wrapAxis" in j
                            else [0, 0]
                        )
                        self.base[key] = [
                            self.base[key][0] + j["dx"] + offset[0],
                            self.base[key][1] + j["dy"] + offset[1],
                        ]
        self.clock += max(elapsed.values(), default=0.0)
        return elapsed

    def prepare(self, desired, basis, factor, label, wraps=None, interface=False):
        """Bring check fibres to ``desired`` (position, sheet shift): shifts, then shuttles."""
        buckets = defaultdict(list)
        for key, (_target, shift) in desired.items():
            if shift is None or shift == self.shift[key]:
                continue
            prior = self.shift[key]
            bank = self.bank_of[self.c["fibres"][key]["module"]]
            buckets[bank, prior, shift].append(key)
        jobs = []
        for (bank, prior, target), keys in buckets.items():
            for group in self.split_rectangles(keys):
                jobs.append(
                    dict(
                        bank=bank,
                        keys=group,
                        prior=prior,
                        target=target,
                        delta=(target - prior) % self.p,
                        kind="permutation",
                    )
                )
        preparation_start = self.clock
        shift_times = self.dispatch(jobs, basis, factor, label + " · internal fibre shift")
        route_blocks(
            self, desired, basis, factor, label, preparation_start, shift_times, wraps, interface
        )

    def annotation(self, s, kind, edges):
        """The algebra record of one sub-round; appends its gate edges to ``edges``."""
        group = dict(
            code=s["code"],
            factor=s["factor"],
            factorLabel={"C": self.c["factorName"], "C_dagger": self.c["factorName"] + "†"}.get(
                s["factor"]
            ),
            term=s["term"],
            region=s["region"],
            dockingOffset_um=docking_offset(s["basis"], s["region"]),
            matrix=s["F"],
            rowLabels=[str(i + 1) for i in range(len(s["F"]))],
            colLabels=[str(j + 1) for j in range(len(s["F"][0]))],
            active=[],
            skipped=[],
        )
        by = defaultdict(list)
        for item in s["active"]:
            indices = []
            for check, data in zip(item["checks"], item["targets"], strict=False):
                for t in range(self.p):
                    indices.append(len(edges))
                    edges.append(
                        [
                            self.ids[kind, check, t],
                            self.ids[0, data, (t + item["exponent"]) % self.p],
                        ]
                    )
            by[item["row"], item["col"], item["exponent"]].extend(indices)
        for (r, j, e), indices in by.items():
            group["active"].append(
                dict(
                    row=r,
                    col=j,
                    matrixRow=r,
                    matrixCol=j,
                    exponent=e,
                    polynomial=s["F"][r][j],
                    edgeIndices=indices,
                )
            )
        for item in s["skipped"]:
            r, j = item["row"], item["col"]
            group["skipped"].append(
                dict(
                    row=r,
                    col=j,
                    matrixRow=r,
                    matrixCol=j,
                    exponent=None,
                    edgeIndices=[],
                    parkedHome=True,
                    checkFibres=[self.ids[kind, i, 0] for i in item["checks"]],
                    dataFibres=[self.ids[0, i, 0] for i in item["targets"]],
                )
            )
        return group

    def round(self, basis, specs, label):
        """Prepare, pulse and clean up one round of parallel sub-rounds ``specs``."""
        kind = 1 if basis == "X" else 2
        start = self.clock
        desired = {}
        post = {}
        for s in specs:
            ox, oy = docking_offset(basis, s["region"])
            for rr in s["groups"]:
                for r in rr:
                    key = kind, r
                    desired[key] = (self.c["fibres"][key]["home"], None)
            for r, xy in s.get("ghosts", {}).items():
                desired[kind, r] = (xy, None)
            for item in s["active"]:
                for r, d in zip(item["checks"], item["targets"], strict=False):
                    key = kind, r
                    xy = self.c["fibres"][0, d]["home"]
                    desired[key] = ([xy[0] + ox, xy[1] + oy], item["exponent"])
                    if item["sparse"]:
                        post[key] = (self.c["fibres"][key]["home"], 0)
        # Separate factors own disjoint checks; preserve factor provenance on
        # each move, including sparse-row immediate returns.
        for s in specs:
            keys = {(kind, r) for rr in s["groups"] for r in rr}
            s["_desired"] = {k: v for k, v in desired.items() if k in keys}
        # Joint preparation keeps independent banks concurrent. A task's factor
        # is recovered from the unique ownership of its check fibres below.
        # Only an actual consecutive sweep-slot wrap receives a detour.
        # Homeward routes are checked for parked-block obstructions separately.
        # Interfaces have separate passing lanes on approach and return.
        wraps = {}
        for s in specs:
            sweeping = {
                r for item in s["active"] if not item["sparse"] for r in item["checks"]
            } | set(s.get("ghosts", {}))
            for rr in s["groups"]:
                for check in rr:
                    key = kind, check
                    previous = self.sweep_state.pop(key, None)
                    if check not in sweeping:
                        continue
                    target = desired[key][0]
                    if (
                        previous
                        and previous["factor"] == s["factor"]
                        and previous["round"] == s["round"] - 1
                        and np.allclose(self.base[key], previous["target"])
                    ):
                        delta = np.array(target) - self.base[key]
                        for axis, pitch in enumerate((TILE_PITCH, 4 * self.p)):
                            if abs(delta[axis]) > pitch + 1e-6 and abs(delta[1 - axis]) < 1e-6:
                                wraps[key] = axis
                    self.sweep_state[key] = dict(
                        factor=s["factor"], round=s["round"], target=list(target)
                    )
        task0 = len(self.tasks)
        interface = all(s["region"] == "interface" for s in specs)
        self.prepare(desired, basis, "mixed", label, wraps=wraps, interface=interface)
        for task in self.tasks[task0:]:
            ti = self.tasks.index(task)
            ids = next(mv[5] for mv in self.moves if mv[6] == ti)
            key = tuple(self.atoms[ids[0]][k] for k in (2, 4))
            task["factor"] = next(s["factor"] for s in specs if key in s["_desired"])
        edges = []
        algebra = [self.annotation(s, kind, edges) for s in specs]
        if edges:
            actual = cKDTree(self.pos).query_pairs(2.000001)
            expected = {tuple(sorted(e)) for e in edges}
            assert actual == expected, (
                self.c["mode"],
                self.p,
                label,
                "false gate",
                list(actual - expected)[:3],
                list(expected - actual)[:3],
            )
            assert len({a for a, b in edges}) == len(edges) and len({b for a, b in edges}) == len(
                edges
            )
            details = [
                dict(
                    name=s["factor"],
                    round=s["round"],
                    region=s["region"],
                    targets=s["targets"],
                    active=[
                        any(a["row"] == i for a in s["active"]) for i in range(len(s["groups"]))
                    ],
                    shifts=[
                        next((a["exponent"] for a in s["active"] if a["row"] == i), 0)
                        for i in range(len(s["groups"]))
                    ],
                )
                for s in specs
            ]
            gate = dict(
                t=self.clock,
                end=self.clock + 1,
                prepStart=start,
                basis=basis,
                label=label,
                edges=edges,
                details=details,
                algebra=algebra,
            )
            self.clock += 1
            self.prepare(
                post,
                basis,
                "C" if basis == "X" else "C_dagger",
                label + " · return sparse checks home",
                interface=interface,
            )
            gate["returnEnd"] = self.clock
            self.gates.append(gate)
        for s in specs:
            s.pop("_desired", None)

    def interface(self, basis):
        """The f0 (X) or f1^dagger (Z) round coupling data code and ancillary code."""
        c = self.c
        groups = []
        targets = []
        for b in range(c["copies"]):
            for k in range(3 if basis == "X" else 5):
                groups.append(
                    [
                        CHECK_FIBRES * b + 5 * k + 3 + b
                        if basis == "X"
                        else c["Z"](5 * b + k, c["q"])
                    ]
                )
                targets.append(
                    c["T"](4 * b + k, c["q"]) if basis == "X" else DATA_FIBRES * b + 5 * k + 3 + b
                )
        count = len(groups)
        F = [[[0] if i == j else None for j in range(count)] for i in range(count)]
        factor = "f0" if basis == "X" else "f1_dagger"
        active = [
            dict(
                checks=rr,
                targets=[targets[i]],
                exponent=0,
                row=i,
                col=i,
                polynomial=[0],
                sparse=True,
            )
            for i, rr in enumerate(groups)
        ]
        s = dict(
            factor=factor,
            base=factor,
            basis=basis,
            code=int(basis == "Z"),
            F=F,
            active=active,
            skipped=[],
            groups=groups,
            region="interface",
            term=factor,
            round=1,
            targets=[i + 1 for i in range(count)],
        )
        self.round(basis, [s], ("f₀" if basis == "X" else "f₁†") + " interface")

    def finish(self):
        """Run the X then Z sector (interface, both streams, return home); the model."""
        c = self.c
        for basis in ("X", "Z"):
            self.marks.append(dict(t=self.clock, label=basis + " sector"))
            self.interface(basis)
            data, anc = streams(c, basis)
            for k in range(max(len(data), len(anc))):
                transition = {}
                kind = 1 if basis == "X" else 2
                for seq in (data, anc):
                    if 0 < k < len(seq) and seq[k]["factor"] != seq[k - 1]["factor"]:
                        for rr in seq[k - 1]["groups"]:
                            for r in rr:
                                transition[kind, r] = (c["fibres"][kind, r]["home"], 0)
                if transition:
                    # Restore row-wise sheet labels before switching to a
                    # column-wise factor, so the next repeated fibre pickup
                    # does not fragment into one job per prior row shift.
                    self.prepare(transition, basis, "return", "Restore factor sheet order")
                    self.gates[-1]["returnEnd"] = self.clock
                specs = [seq[k] for seq in (data, anc) if k < len(seq)]
                # A stream that finishes early parks before the next pulse.
                if k == len(data) or k == len(anc):
                    done = data if k == len(data) else anc
                    rr = {r for s in done for group in s["groups"] for r in group}
                    kind = 1 if basis == "X" else 2
                    self.prepare(
                        {(kind, r): (c["fibres"][kind, r]["home"], None) for r in rr},
                        basis,
                        "return",
                        "Park completed stream",
                    )
                    self.gates[-1]["returnEnd"] = self.clock
                self.round(basis, specs, f"{basis} round {k + 1}")
            kind = 1 if basis == "X" else 2
            self.prepare(
                {k: (v["home"], None) for k, v in c["fibres"].items() if k[0] == kind},
                basis,
                "return",
                basis + " checks return home",
            )
            self.gates[-1]["returnEnd"] = self.clock
        # Camera bounds include temporary virtual slots and interface journeys.
        # Physical module bounds stay unchanged for module-title placement.
        path = np.array([a[:2] for a in self.atoms], float)
        lo = path.copy()
        hi = path.copy()
        for mv in self.moves:
            ids = mv[5]
            path[ids] += mv[2:4]
            lo[ids] = np.minimum(lo[ids], path[ids])
            hi[ids] = np.maximum(hi[ids], path[ids])
        for module in self.modules:
            ids = [i for i, a in enumerate(self.atoms) if a[3] == module["id"]]
            b = module["bounds"]
            module["transportBounds"] = [
                min(b[0], float(lo[ids, 0].min()) - 12),
                max(b[1], float(hi[ids, 0].max()) + 12),
                min(b[2], float(lo[ids, 1].min()) - 12),
                max(b[3], float(hi[ids, 1].max()) + 12),
            ]
        bounds = [
            min(m["transportBounds"][0] for m in self.modules),
            max(m["transportBounds"][1] for m in self.modules),
            min(m["transportBounds"][2] for m in self.modules) - 6,
            max(m["transportBounds"][3] for m in self.modules) + 12,
        ]
        return dict(
            mode=c["mode"],
            modeLabel={
                "fibre": "Fibre surgery",
                "packet": "CRT packet · 1+x",
                "adapter": "Fibre adapter",
            }[c["mode"]],
            lift=self.p,
            pairsPerCode=self.pairs,
            capacity=len(self.banks) * self.pairs,
            allocation="dedicated-per-code",
            logicalSector="Z",
            atoms=self.atoms,
            fixed=self.fixed,
            moves=self.moves,
            tasks=self.tasks,
            gates=self.gates,
            total=self.clock,
            marks=self.marks,
            modules=self.modules,
            banks=self.banks,
            dataCopies=c["copies"],
            factorName=c["factorName"],
            algebraDimensions=dict(mA=3, nA=5, mD=c["d"], nD=c["n"], mC=c["m"], nC=c["r"]),
            sparseX=[c["X"](a, v) for a in c["logical"] for v in range(c["d"])],
            readoutLabel={
                "fibre": f"{self.p} canonical readouts + 2 residuals",
                "packet": f"1 canonical parity + 2 residuals · {3 * self.p + 1} extra ancillary logicals",
                "adapter": f"{self.p} joint canonical products + 4 residuals; α = 1",
            }[c["mode"]],
            layout=dict(
                bounds=bounds,
                fibre_shape=[3, self.p // 3],
                tile_pitch_um=[TILE_PITCH, 4 * self.p],
                atom_pitch_um=[ATOM_PITCH, ATOM_PITCH],
                ancilla_origin_um=TILE_PITCH * c["origin"],
                placement="row-major" if self.p == 9 else "integer-CRT",
                blockPassing=dict(
                    wrapOffset_um=6,
                    returnOffset_um=6,
                    interfaceOffset_um=[0, -6],
                    returnClearance_um=3,
                    scope="wrapping rows/columns, blocked homeward check tiles and f0/f1 interface trips",
                    recipe="perpendicular offset, pass, realign; interfaces cross at y - 6 and dock at y - 2, opposite ordinary checks; reverse on return; homeward groups park sequentially per bank",
                ),
                module_centres=[m["centre"][0] for m in self.modules],
                quadrants=[["data", "Z"], ["X", "data"]],
            ),
            audit=dict(
                gate_frames=len(self.gates),
                physical_moves=len(self.moves),
                labelled_endpoints="verified",
                cartesian_pickups="verified",
            ),
        )


def witnesses(model):
    """Locate explicit real crossings, without claiming an exhaustive path audit."""
    atoms = model["atoms"]
    moves = model["moves"]
    xy = np.array([a[:2] for a in atoms], float)
    fixed = xy[: model["fixed"]]
    starts = []
    centres = []
    for mv in moves:
        starts.append(xy[mv[5][0]].copy())
        fibre = mv[5][0] // model["lift"]
        ids = [i for i in mv[5] if i // model["lift"] == fibre]
        centres.append(xy[ids].mean(axis=0))
        xy[mv[5]] += mv[2:4]
    found = []
    for mv, start in zip(moves, starts, strict=False):
        v = np.array(mv[2:4])
        vv = v @ v
        if vv < 1:
            continue
        us = (fixed - start) @ v / vv
        dist = np.linalg.norm(fixed - (start + us[:, None] * v), axis=1)
        candidates = np.flatnonzero((us > 0.01) & (us < 0.99) & (dist < 0.1))
        if not len(candidates):
            continue
        j = int(candidates[0])
        u = float(us[j])
        phase = math.sqrt(u / 2) if u < 0.5 else 1 - math.sqrt((1 - u) / 2)
        found.append(
            dict(
                t=mv[0] + mv[1] * phase,
                atoms=[mv[5][0], j],
                distance_um=float(dist[j]),
                position=fixed[j].tolist(),
            )
        )
        if len(found) == 3:
            break
    parallel = None
    for left in moves:
        if left[4] >= model["pairsPerCode"]:
            continue
        for r in moves:
            if not (model["pairsPerCode"] <= r[4] < 2 * model["pairsPerCode"]):
                continue
            lo = max(left[0], r[0])
            hi = min(left[0] + left[1], r[0] + r[1])
            if hi - lo > 1:
                parallel = dict(
                    t=(lo + hi) / 2, atoms=[left[5][0], r[5][0]], movingCodes=["data", "ancilla"]
                )
                break
        if parallel:
            break
    model["pathAudit"] = dict(
        method="Explicit straight-segment/fixed-atom intersection witnesses; not exhaustive.",
        witnesses=found,
    )
    model["parallelWitness"] = parallel
    passing = []
    for basis in ("X", "Z"):
        for _i, mv in enumerate(moves):
            task = model["tasks"][mv[6]]
            axis = task.get("wrapAxis")
            if task["basis"] == basis and axis is not None and abs(mv[2 + axis]) > 6:
                passing.append(dict(t=mv[0] + mv[1] / 2, atoms=[mv[5][0]], basis=basis, axis=axis))
                break
    model["blockPassWitnesses"] = passing
