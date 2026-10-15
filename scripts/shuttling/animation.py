"""Read and rewrite the data of the animation ``lp-surgery-aod-polynomials.html``.

The page keeps its player in an iframe ``data-srcdoc`` attribute.  The player holds the
JavaScript schedulers and two gzip/base64 JSON blocks (encoded by ``packing``):
``aod-model-data`` (memory and algebraic surgery models, ``build_lp_models.py``) and
``hybrid-source`` (Graph-WY / Graph-IGND models, ``build_graph_gadgets.py``, and the bare
LP routes they attach to, ``build_lp_models.py``).  The player itself is not generated.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

from packing import canonical_destinations, compact_destinations, decode, encode, intern, resolve

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
    return decode(block_body(player, element))


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
    """Replace the graph models of ``hybrid-source``, keeping its bare LP routes."""
    source = packed(fragment(document), "hybrid-source")
    payload = resolve(source["payload"], source["pool"], "$hybridRef")
    payload["graph"]["models"] = [
        dict(model, layers=[compact_destinations(layer, model["n"]) for layer in model["layers"]])
        for model in models
    ]
    write_blocks({"hybrid-source": encode(intern(payload, "$hybridRef", 40, source["format"]))})


def write_blocks(bodies: dict[str, str], document: Path = DOCUMENT) -> None:
    """Replace the gzip/base64 text of the given blocks."""
    text, player = document.read_text(), fragment(document)
    for element, body in bodies.items():
        old = block_body(player, element)
        if text.count(old) != 1:
            raise ValueError(f"the #{element} block is not unique in the page")
        text = text.replace(old, body)
    document.write_text(text)


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
