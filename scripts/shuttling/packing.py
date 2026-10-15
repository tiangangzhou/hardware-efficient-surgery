"""Lossless packing of the animation's JSON data blocks.

``aod-model-data`` (``pack_models``, unpacked by the player's ``unpackModels``) and
``hybrid-source`` (``pack_hybrid``, unpacked by ``unpackHybrid``) store exact values:
per-fibre runs of atoms, gate edges and moved atom ids are stored once per fibre, every
container repeated anywhere in the block is stored once in a ``pool`` and referenced as
``{"$modelRef": i}`` or ``{"$hybridRef": i}``, and the JSON is gzipped (level 9, mtime 0)
and base64-encoded.  Each shortcut is validated against the full values; data that does
not fit one stays raw.
"""

from __future__ import annotations

import base64
import collections
import copy
import gzip
import json


def encode(block: dict) -> str:
    raw = json.dumps(block, separators=(",", ":"), ensure_ascii=False).encode()
    return base64.b64encode(gzip.compress(raw, compresslevel=9, mtime=0)).decode("ascii")


def decode(body: str) -> dict:
    return json.loads(gzip.decompress(base64.b64decode(body)))


def resolve(value, pool: list, key: str):
    """Replace every reference ``{key: i}`` by ``pool[i]``, recursively."""
    if isinstance(value, list):
        return [resolve(v, pool, key) for v in value]
    if isinstance(value, dict):
        if key in value:
            return resolve(pool[value[key]], pool, key)
        return {k: resolve(v, pool, key) for k, v in value.items()}
    return value


def intern(payload, key: str, min_length: int, format: str) -> dict:
    """Store every container whose JSON (at least ``min_length`` characters) occurs more
    than once in a pool, referenced as ``{key: index}``."""
    signature = lambda x: json.dumps(x, separators=(",", ":"), ensure_ascii=False)
    counts: collections.Counter[str] = collections.Counter()

    def count(x) -> None:
        if isinstance(x, (list, dict)):
            assert not isinstance(x, dict) or key not in x
            if len(s := signature(x)) >= min_length:
                counts[s] += 1
            for v in x if isinstance(x, list) else x.values():
                count(v)

    count(payload)
    pool: list = []
    lookup: dict[str, int] = {}

    def share(x):
        if not isinstance(x, (list, dict)):
            return x
        s = signature(x)
        shared = counts.get(s, 0) > 1
        if shared and s in lookup:
            return {key: lookup[s]}
        body = [share(v) for v in x] if isinstance(x, list) else {k: share(v) for k, v in x.items()}
        if not shared:
            return body
        lookup[s] = len(pool)
        pool.append(body)
        return {key: lookup[s]}

    root = share(payload)
    return dict(format=format, pool=pool, payload=root)


def canonical_destinations(layer: dict, n: int) -> list[int]:
    """Route of a matching layer: pairs map r -> c, other active slots park at n, n + 1, ...,
    and the remaining slots fill the free positions in order."""
    dest = [-1] * (layer["active"] + n)
    for r, c in layer["pairs"]:
        dest[r] = c
    parking = iter(range(n, len(dest)))
    for i in range(layer["active"]):
        if dest[i] < 0:
            dest[i] = next(parking)
    occupied = set(dest)
    free = iter(i for i in range(len(dest)) if i not in occupied)
    return [next(free) if x < 0 else x for x in dest]


def compact_destinations(layer: dict, n: int) -> dict:
    if layer["destinations"] != canonical_destinations(layer, n):
        return layer
    return dict(layer, destinations={"hybridCanonicalDestinations": True})


def fibre_ids(ids: list[int], period: int):
    """Atom ids as {fibres, sheets} when every fibre contributes the same sheets."""
    if not ids:
        return ids
    groups: list[tuple[int, list[int]]] = []
    for atom_id in ids:
        fibre, sheet = divmod(atom_id, period)
        if not groups or groups[-1][0] != fibre:
            groups.append((fibre, []))
        groups[-1][1].append(sheet)
    sheets = groups[0][1]
    if any(group_sheets != sheets for _, group_sheets in groups):
        return ids
    return {"fibres": [fibre for fibre, _ in groups], "sheets": sheets}


def fibre_edges(edges: list[list[int]], period: int):
    """Gate edges as [check fibre, data fibre, shift] when each fibre pair is a full shift."""
    if len(edges) % period:
        return edges
    fibres = []
    for start in range(0, len(edges), period):
        first = edges[start]
        if len(first) != 2 or first[0] % period:
            return edges
        check, data = first[0] // period, first[1] // period
        shift = first[1] % period
        expected = [
            [check * period + sheet, data * period + (sheet + shift) % period]
            for sheet in range(period)
        ]
        if edges[start : start + period] != expected:
            return edges
        fibres.append([check, data, shift])
    return {"fibres": fibres}


def ranges(indices: list[int]):
    """Runs [start, count] of consecutive indices, when that halves the list."""
    if not indices:
        return indices
    runs: list[list[int]] = []
    for index in indices:
        if runs and index == runs[-1][0] + runs[-1][1]:
            runs[-1][1] += 1
        else:
            runs.append([index, 1])
    return {"ranges": runs} if 2 * len(runs) < len(indices) else indices


def _model_atoms(atoms: list[list], model: dict):
    """Atom rows [x, y, kind, module, fibre, sheet] stored once per fibre."""
    period = model["lift"]
    width, height = model["layout"]["fibre_shape"]
    pitch_x, pitch_y = model["layout"]["atom_pitch_um"]
    crt = model["layout"]["placement"] == "integer-CRT"
    if len(atoms) % period:
        return atoms
    fibres = []
    for start in range(0, len(atoms), period):
        base = atoms[start][:5]
        if len(base) != 5:
            return atoms
        for sheet, atom in enumerate(atoms[start : start + period]):
            row = sheet % height if crt else sheet // width
            expected = [
                base[0] + pitch_x * (sheet % width),
                base[1] + pitch_y * row,
                *base[2:],
                sheet,
            ]
            if atom != expected:
                return atoms
        fibres.append(base)
    return {"fibres": fibres}


def pack_models(payload: dict) -> dict:
    """``aod-model-data``: memory and algebraic surgery models, with each model's move and
    task times stored once in ``clockValues`` and indexed."""
    result = copy.deepcopy(payload)
    for model in result["models"]:
        period = model["lift"]
        model["atoms"] = _model_atoms(model["atoms"], model)
        for move in model["moves"]:
            move[5] = fibre_ids(move[5], period)
        for gate in model["gates"]:
            gate["edges"] = fibre_edges(gate["edges"], period)
            for algebra in gate.get("algebra", []):
                for active in algebra.get("active", []):
                    if "edgeIndices" in active:
                        active["edgeIndices"] = ranges(active["edgeIndices"])
        assert "clockValues" not in model
        values: list[float] = []
        index: dict[float, int] = {}

        def clock(value: float) -> int:
            if value not in index:
                index[value] = len(values)
                values.append(value)
            return index[value]

        for move in model["moves"]:
            move[0] = clock(move[0])
            move[1] = clock(move[1])
        for task in model["tasks"]:
            task["start"] = clock(task["start"])
            task["end"] = clock(task["end"])
        model["clockValues"] = values
    return intern(result, "$modelRef", 32, "surgery-models-v1")


def _bare_atoms(model: dict):
    """Bare-model atom dicts stored once per fibre as [id, kind, index, x, y, fibre]."""
    atoms, p = model["atoms"], model["lift"]
    w, h = model["layout"]["fibre_shape"]
    dx, dy = model["layout"]["atom_pitch_um"]
    crt = model["layout"]["placement"] == "integer-CRT"
    if len(atoms) % p:
        return atoms
    result = []
    for start in range(0, len(atoms), p):
        first = atoms[start]
        for s, a in enumerate(atoms[start : start + p]):
            expected = dict(
                id=first["id"] + s,
                kind=first["kind"],
                index=first["index"] + s,
                x=first["x"] + dx * (s % w),
                y=first["y"] + dy * (s % h if crt else s // w),
                fibre=first["fibre"],
                sheet=s,
            )
            if a != expected:
                return atoms
        result.append(
            [first["id"], first["kind"], first["index"], first["x"], first["y"], first["fibre"]]
        )
    return {"hybridAtomFibres": result}


def pack_hybrid(graph: dict, bare: dict) -> dict:
    """``hybrid-source``: the Graph-WY / Graph-IGND models and the bare LP routes."""
    g, b = copy.deepcopy(graph), copy.deepcopy(bare)
    g["models"] = [
        dict(model, layers=[compact_destinations(layer, model["n"]) for layer in model["layers"]])
        for model in g["models"]
    ]
    for model in b["models"]:
        p = model["lift"]
        model["atoms"] = _bare_atoms(model)
        for move in model["moves"]:
            move[5] = fibre_ids(move[5], p)
        for f in model.get("fibres", []):
            f["atomIds"] = ranges(f["atomIds"])
        for gate in model["gates"]:
            gate["pairs"] = fibre_edges(gate["pairs"], p)
            for algebra in gate.get("algebra", []):
                for active in algebra.get("active", []):
                    if "edgeIndices" in active:
                        active["edgeIndices"] = ranges(active["edgeIndices"])
    return intern(dict(graph=g, bare=b), "$hybridRef", 40, "hybrid-source-v1")
