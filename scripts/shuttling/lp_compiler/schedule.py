"""Schedulers: one bank's jobs on its AOD pairs, and the bank clocks of a whole model.

``schedule_jobs`` places the moves of one preparation step on a bank's pairs.
``apply_timing`` then removes cross-bank barriers without changing routes or the order of
noncommuting gates: pulses are addressed per check bank, operations keep their
within-bank interval order (including wrap realignment) and start as soon as their
dependencies and earlier reservations allow. Neither is a globally optimal schedule.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

EPS = 1e-6


def _split_gates(model: dict[str, Any]) -> list[dict[str, Any]]:
    owner = {m["id"]: m["bank"] for m in model["modules"]}
    bank_of = lambda atom: owner[model["atoms"][atom][3]]
    names = {b["id"]: b["name"] for b in model["banks"]}
    gates = []
    for gate in model["gates"]:
        for bank in sorted({bank_of(a) for a, _ in gate["edges"]}):
            indices = [i for i, (a, _) in enumerate(gate["edges"]) if bank_of(a) == bank]
            remap = {old: new for new, old in enumerate(indices)}
            algebra = []
            for group in gate["algebra"]:
                active, skipped = [], []
                for cell in group["active"]:
                    edges = [remap[i] for i in cell["edgeIndices"] if i in remap]
                    if edges:
                        active.append({**cell, "edgeIndices": edges})
                for cell in group["skipped"]:
                    selected = [i for i, a in enumerate(cell["checkFibres"]) if bank_of(a) == bank]
                    if selected:
                        skipped.append(
                            {
                                **cell,
                                "checkFibres": [cell["checkFibres"][i] for i in selected],
                                "dataFibres": [cell["dataFibres"][i] for i in selected],
                            }
                        )
                if active or skipped:
                    algebra.append(
                        {
                            **group,
                            "active": active,
                            "skipped": skipped,
                            "basis": gate["basis"],
                            "label": names[bank],
                        }
                    )
            factors = {g["factor"] for g in algebra}
            gates.append(
                {
                    **gate,
                    "bank": bank,
                    "block": f"bank{bank}",
                    "label": gate["label"] + " · " + names[bank],
                    "edges": [gate["edges"][i] for i in indices],
                    "algebra": algebra,
                    "details": [d for d in gate["details"] if d["name"] in factors],
                }
            )
    return gates


def _bank_order(
    tasks: list[dict[str, Any]], gates: list[dict[str, Any]], basis: str | None = None
) -> dict[str, Any]:
    """Return task starts and pulse windows, retaining source indices."""
    nodes = []
    for i, task in enumerate(tasks):
        if basis is None or task["basis"] == basis:
            nodes.append(
                dict(
                    kind="task",
                    index=i,
                    start=task["start"],
                    end=task["end"],
                    bank=task["block"],
                    basis=task["basis"],
                    atoms=[],
                )
            )
    for i, gate in enumerate(gates):
        if basis is None or gate["basis"] == basis:
            nodes.append(
                dict(
                    kind="gate",
                    index=i,
                    start=gate["t"],
                    end=gate["end"],
                    bank=gate["block"],
                    basis=gate["basis"],
                    atoms={a for e in gate["edges"] for a in e},
                )
            )
    nodes.sort(key=lambda n: (n["start"], n["end"], n["bank"]))
    bank_nodes = defaultdict(list)
    atom_gates = defaultdict(list)
    for node in nodes:
        # Retain the original interval partial order in each bank. In particular,
        # a wrap realignment still waits for that bank's ordinary shuttles.
        deps = [n for n in bank_nodes[node["bank"]] if n["end"] <= node["start"] + EPS]
        previous_gates = {id(g): g for a in node["atoms"] for g in atom_gates[a]}.values()
        # X check->data gates commute with other X gates; likewise data->Z.
        # Preserve opposite-basis order on shared atoms, but allow commuting
        # gates to pass one another when the faster bank is ready first.
        deps.extend(g for g in previous_gates if g["basis"] != node["basis"])
        node["at"] = max((n["until"] for n in deps), default=0.0)
        duration = node["end"] - node["start"]
        while conflicts := [
            g
            for g in previous_gates
            if min(node["at"] + duration, g["until"]) > max(node["at"], g["at"]) + EPS
        ]:
            node["at"] = max(g["until"] for g in conflicts)
        node["until"] = node["at"] + node["end"] - node["start"]
        bank_nodes[node["bank"]].append(node)
        for atom in node["atoms"]:
            atom_gates[atom].append(node)
    task_starts = [None] * len(tasks)
    gate_times = [None] * len(gates)
    for node in nodes:
        i = node["index"]
        if node["kind"] == "task":
            task_starts[i] = node["at"]
            continue
        gate = gates[i]
        window = [
            n
            for n in bank_nodes[node["bank"]]
            if n["kind"] == "task"
            and tasks[n["index"]]["basis"] == gate["basis"]
            and n["start"] >= gate["prepStart"] - EPS
            and n["end"] <= gate["returnEnd"] + EPS
        ]
        gate_times[i] = [
            node["at"],
            node["until"],
            min([node["at"], *(n["at"] for n in window)]),
            max([node["until"], *(n["until"] for n in window)]),
        ]
    return dict(
        taskStarts=task_starts,
        gateTimes=gate_times,
        total=max((n["until"] for n in nodes), default=0.0),
    )


def apply_timing(model: dict[str, Any]) -> dict[str, Any]:
    """Retime the model in place across banks and add the standalone X / Z sector times."""
    original_total = model["total"]
    tasks = model["tasks"]
    gates = _split_gates(model)
    full = _bank_order(tasks, gates)
    sectors = {basis: _bank_order(tasks, gates, basis) for basis in "XZ"}
    for move in model["moves"]:
        move[0] += full["taskStarts"][move[6]] - tasks[move[6]]["start"]
    for i, task in enumerate(tasks):
        delta = full["taskStarts"][i] - task["start"]
        for item in [task, *task.get("localPhases", [])]:
            item["start"] += delta
            item["end"] += delta
    for gate, times in zip(gates, full["gateTimes"], strict=True):
        gate.update(zip(("t", "end", "prepStart", "returnEnd"), times, strict=True))
    order = sorted(range(len(gates)), key=lambda i: (gates[i]["t"], gates[i]["bank"]))
    model["gates"] = [gates[i] for i in order]
    for sector in sectors.values():
        sector["gateTimes"] = [sector["gateTimes"][i] for i in order]
    model["sectorSchedules"] = sectors
    model["total"] = full["total"]
    model["sectorTimesUs"] = {**{b: s["total"] for b, s in sectors.items()}, "all": full["total"]}
    model["marks"] = [
        dict(t=min(t["start"] for t in tasks if t["basis"] == b), label=f"{b} sector") for b in "XZ"
    ]
    model["audit"].update(
        gate_frames=len(gates),
        synchronizedTotalUs=original_total,
        scheduling="Preserve bank routes and noncommuting gate order; reserve shared atoms for pulses",
        gateAddressing="Independent addressed bank pulses",
    )
    return model


def schedule_jobs(tasks, capacity, trials=80):
    """List-schedule one bank's jobs on its ``capacity`` AOD pairs.

    Each task has ``deps`` and ``options`` (pairs, duration, method); the best of
    ``trials`` seeded runs (critical path first, then random priorities) is kept.
    """
    if not tasks:
        return dict(duration_us=0, tasks=[])
    best = None
    for trial in range(trials):
        rng = random.Random(1729 + trial)
        chosen = []
        for task in tasks:
            opts = [v for v in task["options"] if v[0] <= capacity]
            if trial % 4 == 0:
                opt = min(opts, key=lambda z: (z[1], z[0]))
            elif trial % 4 == 1:
                opt = min(opts, key=lambda z: (z[0], z[1]))
            else:
                opt = rng.choice(opts)
            chosen.append(opt)
        successors = [[] for _ in tasks]
        for task in tasks:
            for d in task["deps"]:
                successors[d].append(task["id"])
        critical = [0.0] * len(tasks)
        for i in reversed(range(len(tasks))):
            critical[i] = chosen[i][1] + max((critical[j] for j in successors[i]), default=0)
        priorities = [rng.random() for _ in tasks]
        done = set()
        pending = set(range(len(tasks)))
        running = {}
        records = []
        now = 0.0
        while pending or running:
            finished = [i for i, (end, ids) in running.items() if end <= now + 1e-8]
            for i in finished:
                done.add(i)
                del running[i]
            occupied = {a for end, ids in running.values() for a in ids}
            free = sorted(set(range(capacity)) - occupied)
            while True:
                ready = [
                    i
                    for i in pending
                    if set(tasks[i]["deps"]) <= done and chosen[i][0] <= len(free)
                ]
                if not ready:
                    break
                if trial < 4:
                    i = max(ready, key=lambda k: (critical[k], -k))
                else:
                    i = max(ready, key=lambda k: (priorities[k], critical[k]))
                q, duration, method = chosen[i]
                allocated = free[:q]
                free = free[q:]
                running[i] = (now + duration, allocated)
                pending.remove(i)
                record = {k: v for k, v in tasks[i].items() if k != "options"}
                record.update(start_us=now, end_us=now + duration, pairs=allocated, method=method)
                records.append(record)
            if running:
                now = min(end for end, ids in running.values())
            elif pending:
                raise AssertionError("scheduler deadlock")
        candidate = dict(
            duration_us=now, tasks=sorted(records, key=lambda r: (r["start_us"], r["id"]))
        )
        if best is None or now < best["duration_us"]:
            best = candidate
    # Independent checks of every dependency and every allocated AOD timeline.
    index = {r["id"]: r for r in best["tasks"]}
    for r in best["tasks"]:
        assert all(index[d]["end_us"] <= r["start_us"] + 1e-7 for d in r["deps"])
        assert len(r["pairs"]) <= capacity
    for a in range(capacity):
        rr = sorted((r for r in best["tasks"] if a in r["pairs"]), key=lambda r: r["start_us"])
        assert all(x["end_us"] <= y["start_us"] + 1e-7 for x, y in zip(rr, rr[1:], strict=False))
    return best
