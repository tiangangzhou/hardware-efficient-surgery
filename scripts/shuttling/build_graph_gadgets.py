"""Build the Graph-WY and Graph-IGND gadgets compared in Fig. 3 and Table S2.

Each gadget measures one canonical logical Z operator of the target fibre of the data
code, the operator recorded in the animation (support weights 10, 22, 44 for l = 9, 15,
33).  Graph-WY is the gauging construction of Williamson and Yoder (arXiv:2410.02213),
Graph-IGND the edge-expanded homological measurement of Ide, Gowda, Nadkarni and
Dauphinais (arXiv:2410.02753).  Each merged code is checked to commute, to measure exactly
one logical and to read out the target; its auxiliary graph to have Cheeger constant
h >= 1, computed exactly over all cuts (so the deformed code keeps the distance); and,
with its compiled matching layers, to equal the graph model embedded in the animation.
The gadgets are compared with ``data/shuttling/gadgets/l{l}_{method}.npz``, the input of
``compute_space_overhead.py``, whose SHA-256 the animation records.  ``--write`` replaces
the gadgets that changed and the graph models of the animation.

    uv run python scripts/shuttling/build_graph_gadgets.py   # ~2 min
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
from animation import fragment, graph_models, write_graph_models
from aod import compile_check_round
from qldpc.codes.common import CSSCode
from qldpc.objects import Pauli

from algebraic_qec import gf2_rank
from algebraic_qec.examples.surgery_instances import instance
from qldpc_surgery import build_gadget, cheeger_constant

GADGETS = Path(__file__).resolve().parents[2] / "data" / "shuttling" / "gadgets"

LIFTS = (9, 15, 33)
METHODS = ("gauging", "edge_expansion")
CERTIFICATION = (
    "Cheeger constant h >= 1, exact over all cuts (build_graph_gadgets.py); "
    "fault distance not certified"
)


def sparse_json(matrix: np.ndarray) -> dict:
    return dict(shape=list(matrix.shape), entries=np.argwhere(matrix).tolist())


def matching_layers(hx: np.ndarray, hz: np.ndarray) -> list[dict]:
    """Gate layers of one X+Z check round with their 1D AOD routes."""
    layers = [
        dict(
            basis=layer.basis,
            pairs=[list(p) for p in layer.pairs],
            destinations=list(layer.route.destinations),
            active=layer.route.active_count,
            sourceIndex=j,
        )
        for j, layer in enumerate(compile_check_round(hx, hz).layers)
    ]
    for basis, matrix in (("X", hx), ("Z", hz)):
        pairs = sorted(
            tuple(p) for layer in layers if layer["basis"] == basis for p in layer["pairs"]
        )
        assert pairs == list(zip(*np.nonzero(matrix))), f"{basis} gates are not covered once"
    return layers


def build(ell: int, method: str, support: list[int]) -> tuple[dict[str, np.ndarray], dict]:
    data = instance(ell).data_code()
    hx = data.binary_matrix_x.toarray().astype(np.uint8)
    hz = data.binary_matrix_z.toarray().astype(np.uint8)
    logical = np.zeros(data.n, dtype=np.uint8)
    logical[support] = 1
    assert not (hx @ logical % 2).any(), "not a Z logical"

    gadget = build_gadget(
        CSSCode(hx.astype(int), hz.astype(int)), logical, basis=Pauli.Z, method=method
    )
    gx, gz = gadget.HX_merged.astype(np.uint8), gadget.HZ_merged.astype(np.uint8)
    readout = gadget.effective_readout_vectors[0]
    assert not (gx @ gz.T % 2).any()
    assert gx.shape[1] - gf2_rank(gx) - gf2_rank(gz) == data.k - 1, "measures exactly one logical"
    assert np.array_equal(readout @ gadget.f1.T % 2, logical)
    assert gf2_rank(gadget.incidence) == gadget.incidence.shape[1] - 1

    arrays = dict(
        hx=gx,
        hz=gz,
        f1=gadget.f1,
        f0=gadget.f0,
        incidence=gadget.incidence,
        partial_0=gadget.partial_0,
        targets=logical[None],
        readouts=gadget.effective_readout_vectors,
    )
    model = dict(
        n=int(gx.shape[1]),
        nData=data.n,
        mX=len(gx),
        mZ=len(gz),
        mBare=len(hx),
        support=np.flatnonzero(logical).tolist(),
        readout=np.flatnonzero(readout).tolist(),
        layers=matching_layers(gx, gz),
        blocks={
            "f0": sparse_json(gadget.f0),
            "f1": sparse_json(gadget.f1),
            "incidence": sparse_json(gadget.incidence),
            "partial0": sparse_json(gadget.partial_0),
        },
    )
    return arrays, model


def archived(path: Path, arrays: dict[str, np.ndarray]) -> bool:
    if not path.exists():
        return False
    with np.load(path) as archive:
        return archive.files == list(arrays) and all(
            np.array_equal(archive[k], v) for k, v in arrays.items()
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--write", action="store_true", help="replace the archived gadgets")
    args = parser.parse_args()
    embedded = graph_models(fragment())
    built = {}
    for reference in embedded:
        ell, method = reference["lift"], reference["method"]
        arrays, model = build(ell, method, reference["support"])
        h = cheeger_constant(arrays["incidence"])
        assert h >= 1, f"l={ell} {method}: Cheeger constant {h} < 1"
        built[ell, method] = arrays, model, h
    assert sorted(built) == sorted((ell, method) for ell in LIFTS for method in METHODS)
    if args.write:
        models = []
        for reference in embedded:
            arrays, model, _ = built[reference["lift"], reference["method"]]
            path = GADGETS / f"l{reference['lift']}_{reference['method']}.npz"
            provenance = dict(reference["provenance"], expansionCertification=CERTIFICATION)
            if not archived(path, arrays):
                np.savez_compressed(path, **arrays)
                provenance.update(
                    archiveStem=path.stem,
                    gadgetSha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                )
            models.append({**reference, **model, "provenance": provenance})
        write_graph_models(models)
        embedded = graph_models(fragment())
    for reference in embedded:
        ell, method = reference["lift"], reference["method"]
        arrays, model, h = built[ell, method]
        changed = sorted(key for key, value in model.items() if reference[key] != value)
        assert not changed, f"l={ell} {method}: differs from the animation in {changed}"
        path = GADGETS / f"l{ell}_{method}.npz"
        assert archived(path, arrays), path
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert reference["provenance"]["gadgetSha256"] == digest, f"{path} is not the animation's"
        print(
            f"l={ell:2d} {method:14s} support {len(model['support']):2d}  "
            f"auxiliary qubits {model['n'] - model['nData']:4d}  h = {h:.3f}  "
            "matches the animation"
        )


if __name__ == "__main__":
    main()
