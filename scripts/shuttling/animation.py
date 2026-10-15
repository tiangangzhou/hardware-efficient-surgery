"""Read (and, for the graph models, rewrite) the data of the animation ``lp-surgery-aod-polynomials.html``.

The page keeps its player in an iframe ``data-srcdoc`` attribute.  The player holds the
JavaScript schedulers and two gzip/base64 JSON blocks: ``aod-model-data`` (memory and
algebraic fibre-surgery models) and ``hybrid-source`` (Graph-WY / Graph-IGND models and
the bare LP routes they attach to).  Repeated objects of a block are stored once in its
``pool`` and referenced as ``{"$modelRef": i}`` or ``{"$hybridRef": i}``; the player's
``unpackModels`` / ``unpackHybrid`` expand them.
"""

from __future__ import annotations

import base64
import collections
import gzip
import html
import json
import re
from pathlib import Path

DOCUMENT = Path(__file__).resolve().parents[2] / "lp-surgery-aod-polynomials.html"


def fragment(document: Path = DOCUMENT) -> str:
    """The unescaped player embedded in the page."""
    match = re.search(r'data-srcdoc="([\s\S]*?)"', document.read_text())
    if not match:
        raise ValueError(f"{document} has no embedded player")
    return html.unescape(match[1])


def block_body(player: str, element: str) -> str:
    """The gzip/base64 text of one block (base64 needs no HTML escaping in the page)."""
    bodies = re.findall(r'<script[^>]*id="' + element + r'"[^>]*>([^<]*)</script>', player)
    if len(bodies) != 1:
        raise ValueError(f"expected one #{element} block, found {len(bodies)}")
    return bodies[0]


def packed(player: str, element: str) -> dict:
    """Decode one gzip/base64 JSON block, still in its packed form."""
    return json.loads(gzip.decompress(base64.b64decode(block_body(player, element))))


def resolve(value, pool: list, key: str):
    """Replace every reference ``{key: i}`` by ``pool[i]``, recursively."""
    if isinstance(value, list):
        return [resolve(v, pool, key) for v in value]
    if isinstance(value, dict):
        if key in value:
            return resolve(pool[value[key]], pool, key)
        return {k: resolve(v, pool, key) for k, v in value.items()}
    return value


def graph_models(player: str) -> list[dict]:
    """The Graph-WY / Graph-IGND models of ``hybrid-source``, expanded as by ``unpackHybrid``."""
    source = packed(player, "hybrid-source")
    models = resolve(source["payload"], source["pool"], "$hybridRef")["graph"]["models"]
    for model in models:
        for layer in model["layers"]:
            if isinstance(layer["destinations"], dict):  # compact "canonical destinations"
                layer["destinations"] = canonical_destinations(layer, model["n"])
    return models


def write_graph_models(models: list[dict], document: Path = DOCUMENT) -> None:
    """Replace the graph models of ``hybrid-source`` and repack the block as the animation's
    builder does (``scripts/aod_animation/pack_hybrid.py`` of algebraic-qec): canonical
    destinations compacted, repeated containers shared through the pool, gzip level 9
    with mtime 0.  Repacking the unchanged block reproduces it byte for byte."""
    text = document.read_text()
    body = block_body(fragment(document), "hybrid-source")
    source = json.loads(gzip.decompress(base64.b64decode(body)))
    payload = resolve(source["payload"], source["pool"], "$hybridRef")
    payload["graph"]["models"] = [
        dict(model, layers=[compact_destinations(layer, model["n"]) for layer in model["layers"]])
        for model in models
    ]
    raw = json.dumps(intern(payload), separators=(",", ":"), ensure_ascii=False).encode()
    new_body = base64.b64encode(gzip.compress(raw, compresslevel=9, mtime=0)).decode("ascii")
    if text.count(body) != 1:
        raise ValueError("the hybrid-source block is not unique in the page")
    document.write_text(text.replace(body, new_body))


def compact_destinations(layer: dict, n: int) -> dict:
    if layer["destinations"] != canonical_destinations(layer, n):
        return layer
    return dict(layer, destinations={"hybridCanonicalDestinations": True})


def intern(payload) -> dict:
    """Share every container whose JSON (at least 40 characters) occurs more than once."""
    signature = lambda x: json.dumps(x, separators=(",", ":"), ensure_ascii=False)
    counts: collections.Counter[str] = collections.Counter()

    def count(x) -> None:
        if isinstance(x, (list, dict)):
            if len(s := signature(x)) >= 40:
                counts[s] += 1
            for v in x if isinstance(x, list) else x.values():
                count(v)

    count(payload)
    pool: list = []
    lookup: dict[str, int] = {}

    def encode(x):
        if not isinstance(x, (list, dict)):
            return x
        s = signature(x)
        shared = counts.get(s, 0) > 1
        if shared and s in lookup:
            return {"$hybridRef": lookup[s]}
        body = [encode(v) for v in x] if isinstance(x, list) else {k: encode(v) for k, v in x.items()}
        if not shared:
            return body
        lookup[s] = len(pool)
        pool.append(body)
        return {"$hybridRef": lookup[s]}

    root = encode(payload)
    return dict(format="hybrid-source-v1", pool=pool, payload=root)


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


def function_source(player: str, name: str) -> str:
    """Source of the JavaScript function ``name`` (its body has no braces in strings)."""
    start = player.index(f"function {name}(")
    depth = 0
    for end in range(player.index("{", start), len(player)):
        depth += {"{": 1, "}": -1}.get(player[end], 0)
        if depth == 0:
            return player[start : end + 1]
    raise ValueError(f"unbalanced braces in {name}")


def compiler_inputs(document: Path = DOCUMENT) -> dict:
    """The schedulers, payload decoders and packed models replayed by ``extract.cjs``."""
    player = fragment(document)
    scripts = re.findall(r"<script(?:\s[^>]*)?>([\s\S]*?)</script>", player)
    runtime = [s for s in scripts if s.lstrip().startswith("/* Xu et al. Algorithm 1")]
    parallel = re.findall(r'<script id="aod-parallel-compiler">([\s\S]*?)</script>', player)
    if len(runtime) != 1 or len(parallel) != 1:
        raise ValueError("animation does not contain the expected schedulers")
    return dict(
        runtime=runtime[0],
        decoders="\n".join(function_source(player, f) for f in ("unpackModels", "unpackHybrid")),
        parallel=parallel[0],
        lp=packed(player, "aod-model-data"),
        hybrid=packed(player, "hybrid-source"),
    )
