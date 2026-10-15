"""Compile the LP-code models of the shuttling animation and compare them with the page.

The animation ``lp-surgery-aod-polynomials.html`` embeds, besides its player, the models
of ``aod-model-data`` (memory and algebraic fibre, packet and adapter surgery at
l = 9, 15, 33 with one or two AOD pairs per code) and the bare LP routes of
``hybrid-source`` (the 2D data transport the Graph-WY / Graph-IGND gadgets of
``build_graph_gadgets.py`` attach to).  This script compiles them with ``lp_compiler``,
checks every model independently (``lp_compiler.checks``: measured matrices, data never
moved, Cartesian pickups, no double-booked atoms or AOD pairs), and compares both
blocks byte for byte with the page; ``--write`` replaces them.

    uv run python scripts/shuttling/build_lp_models.py   # ~10 s
"""

from __future__ import annotations

import argparse

from animation import block_body, fragment, graph_models, write_blocks
from lp_compiler.checks import check_models
from lp_compiler.models import animation_models
from packing import encode, pack_hybrid, pack_models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--write", action="store_true", help="replace the two data blocks")
    args = parser.parse_args()
    payload, bare_models = animation_models()
    check_models(payload)
    player = fragment()
    graph = dict(schemaVersion=2, models=graph_models(player))
    blocks = {
        "aod-model-data": encode(pack_models(payload)),
        "hybrid-source": encode(pack_hybrid(graph, dict(schemaVersion=2, models=bare_models))),
    }
    if args.write:
        write_blocks(blocks)
        player = fragment()
    for element, body in blocks.items():
        assert block_body(player, element) == body, f"#{element} differs from the page"
    print(
        f"{len(payload['models'])} models (checked) and {len(bare_models)} bare LP routes "
        "match the animation"
    )


if __name__ == "__main__":
    main()
