"""Physical qubits of the bottom panel of Fig. 3 (all l canonical logical Z of one fibre).

- memory: the data code alone;
- algebraic fibre surgery: the merged code scheduled in the animation (its code qubits
  are the n of the merged code of the instance: 585, 975, 2442);
- Graph-WY / Graph-IGND: the l cyclic translates of the single-logical gadget of
  ``data/shuttling/gadgets/`` on one shared data code with disjoint auxiliary systems
  (``parallel_gadgets``), checked to commute and to measure exactly the l translates.

Total = code qubits (data + auxiliary) + one syndrome atom per X and Z check row.
Compares the counts with ``data/shuttling/space_overhead.json`` (``--write`` replaces it).

    uv run python scripts/shuttling/compute_space_overhead.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from animation import fragment, packed, resolve
from qldpc.codes.common import CSSCode
from qldpc.objects import Pauli
from scipy import sparse

from algebraic_qec import gf2_rank
from algebraic_qec.examples.surgery_instances import instance
from qldpc_surgery.hmatrix.gadget import GadgetSpec
from qldpc_surgery.hmatrix.gauging import parallel_gadgets

DATA = Path(__file__).resolve().parents[2] / "data" / "shuttling"
REPORT = DATA / "space_overhead.json"
LIFTS = (9, 15, 33)


def translate(matrix: np.ndarray, ell: int, shift: int) -> np.ndarray:
    """Apply x^shift to every cyclic block along the first axis."""
    blocks = matrix.reshape(-1, ell, *matrix.shape[1:])
    return np.roll(blocks, shift, axis=1).reshape(matrix.shape)


def fibre_surgery_counts(lp: dict, ell: int) -> tuple[int, int, int]:
    """(code qubits, X-check atoms, Z-check atoms) of the scheduled fibre surgery."""
    model = next(
        m
        for m in lp["payload"]["models"]
        if m["lift"] == ell and m["mode"] == "fibre" and m["pairsPerCode"] == 1
    )
    atoms = resolve(model["atoms"], lp["pool"], "$modelRef")
    copies, entries = (ell, atoms["fibres"]) if isinstance(atoms, dict) else (1, atoms)
    # Atom kinds: 0 = code qubit, 1 = X check, 2 = Z check.
    code, x, z = (copies * sum(atom[2] == kind for atom in entries) for kind in (0, 1, 2))
    assert code == model["fixed"] == instance(ell).merged_parameters[0]
    return code, x, z


def graph_union(code: CSSCode, hz: np.ndarray, data_k: int, ell: int, method: str) -> dict:
    """Combine the l translates of one gadget and verify the union."""
    with np.load(DATA / "gadgets" / f"l{ell}_{method}.npz") as single:
        gadgets = [
            GadgetSpec.from_maps(
                code,
                translate(single["targets"][0], ell, s),
                basis=Pauli.Z,
                f1=translate(single["f1"], ell, s),
                f0=translate(single["f0"], ell, s),
                incidence=single["incidence"],
                partial_0=single["partial_0"],
                readout_vectors=single["readouts"],
                validate=False,
            )
            for s in range(ell)
        ]
        canonical = np.vstack([translate(single["targets"][0], ell, s) for s in range(ell)])
        single_flux = single["partial_0"].shape[0]
    union = parallel_gadgets(gadgets, validate=False)  # restores flux checks lost to sharing
    ux, uz = union.HX_merged, union.HZ_merged
    n = ux.shape[1]
    assert not np.any((sparse.csr_matrix(ux) @ sparse.csr_matrix(uz).T).data % 2)
    assert np.array_equal(union.effective_batch_x, canonical)
    assert not np.any(union.incidence @ union.effective_readout_vectors.T % 2)
    assert gf2_rank(np.vstack([hz, canonical])) - gf2_rank(hz) == ell  # independent readouts
    assert data_k - (n - gf2_rank(ux) - gf2_rank(uz)) == ell  # measures exactly l logicals
    return dict(
        total_qubits=n,
        checks_x=ux.shape[0],
        checks_z=uz.shape[0],
        restored_flux_checks=union.partial_0.shape[0] - ell * single_flux,
    )


def space_report() -> dict:
    lp = packed(fragment(), "aod-model-data")
    rows = []
    for ell in LIFTS:
        data = instance(ell).data_code()
        hx = data.binary_matrix_x.toarray().astype(np.uint8)
        hz = data.binary_matrix_z.toarray().astype(np.uint8)
        code = CSSCode(hx.astype(int), hz.astype(int))
        total, x, z = fibre_surgery_counts(lp, ell)
        rows.append(
            dict(lift=ell, method="memory", total_qubits=data.n, checks_x=len(hx), checks_z=len(hz))
        )
        rows.append(dict(lift=ell, method="algebraic", total_qubits=total, checks_x=x, checks_z=z))
        for method in ("gauging", "edge_expansion"):
            rows.append(dict(lift=ell, method=method, **graph_union(code, hz, data.k, ell, method)))
    for row in rows:
        row["data_qubits"] = instance(row["lift"]).data_parameters[0]
        row["auxiliary_qubits"] = row["total_qubits"] - row["data_qubits"]
        row["total_physical_qubits"] = row["total_qubits"] + row["checks_x"] + row["checks_z"]
    return dict(
        metric="code qubits (data + auxiliary) plus one syndrome atom per X/Z check row",
        scope="all l canonical logical Z operators of one fibre; graph methods as an l-fold union",
        rows=rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--write", action="store_true", help=f"replace {REPORT.name}")
    args = parser.parse_args()
    report = space_report()
    for row in report["rows"]:
        print(
            f"l={row['lift']:>2} {row['method']:<15} {row['total_physical_qubits']:>5} physical qubits"
        )
    text = json.dumps(report, indent=2) + "\n"
    if args.write:
        REPORT.write_text(text)
    elif REPORT.read_text() != text:
        raise SystemExit(f"differs from {REPORT}")
    else:
        print(f"matches {REPORT.name}")


if __name__ == "__main__":
    main()
