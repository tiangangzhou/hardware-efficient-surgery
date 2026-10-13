"""Read the data embedded in the published animation ``lp-surgery-aod-polynomials.html``.

The page keeps its player in an iframe ``data-srcdoc`` attribute.  The player holds the
JavaScript schedulers and two gzip/base64 JSON blocks: ``aod-model-data`` (memory and
algebraic fibre-surgery models) and ``hybrid-source`` (Graph-WY / Graph-IGND models and
the bare LP routes they attach to).  Repeated objects of a block are stored once in its
``pool`` and referenced as ``{"$modelRef": i}`` or ``{"$hybridRef": i}``; the player's
``unpackModels`` / ``unpackHybrid`` expand them.
"""

from __future__ import annotations

import base64
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


def packed(player: str, element: str) -> dict:
    """Decode one gzip/base64 JSON block, still in its packed form."""
    bodies = re.findall(r'<script[^>]*id="' + element + r'"[^>]*>([^<]*)</script>', player)
    if len(bodies) != 1:
        raise ValueError(f"expected one #{element} block, found {len(bodies)}")
    return json.loads(gzip.decompress(base64.b64decode(bodies[0])))


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
