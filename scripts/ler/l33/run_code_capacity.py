"""Code-capacity LER of the bare l = 33 code and the merged surgery code (SM table).

Noise: independent bit flips with probability p on every qubit and one perfect syndrome
measurement.  Basis Z: X errors, decoded with H_Z; a shot fails if the residual error
anticommutes with any logical Z operator.  Basis X: the same with Z errors, H_X and the
logical X operators.  Decoder: ldpc's BpLsdDecoder with the BP-LSD settings of the
circuit simulations (``LSD_KWARGS``) and prior p on every qubit.

Each (code, basis, p) point draws chunks of 2,000 shots from one random generator seeded
with ``--seed`` until it has ``--max-errors`` failures or ``--shots`` shots, and appends
one row to ``--out``.  The published table (``summarize_code_capacity.py``) adds up
single-process cluster shards of this kind, each with its own seed; reruns agree with it
statistically, not bit for bit.

    uv run python scripts/ler/l33/run_code_capacity.py --codes merged --basis Z \\
        --p-list 0.05 --shots 4000 --out code_capacity.csv
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from codes import css_code, data_code, merged_checks
from ldpc import BpLsdDecoder
from qldpc.objects import Pauli
from scipy.sparse import csr_matrix

from qldpc_surgery.decoding import LSD_KWARGS

CHUNK = 2_000


def sample(code, basis: str, p: float, max_shots: int, max_errors: int, rng) -> tuple[int, int]:
    """(shots, logical failures) of one (code, basis, p) point."""
    pauli = Pauli.Z if basis == "Z" else Pauli.X
    h = np.asarray(code.matrix_z if basis == "Z" else code.matrix_x, dtype=np.uint8)
    logicals = np.asarray(code.get_logical_ops(pauli), dtype=np.uint8)
    decoder = BpLsdDecoder(csr_matrix(h), error_channel=[p] * h.shape[1], **LSD_KWARGS)
    shots = failures = 0
    while shots < max_shots and failures < max_errors:
        errors = (rng.random((min(CHUNK, max_shots - shots), h.shape[1])) < p).astype(np.uint8)
        for error, syndrome in zip(errors, errors @ h.T % 2, strict=True):
            residual = error ^ decoder.decode(syndrome) if syndrome.any() else error
            failures += bool((logicals @ residual % 2).any())
        shots += len(errors)
    return shots, failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--codes", default="mem,merged", help="mem (bare) and/or merged")
    parser.add_argument("--basis", default="X,Z")
    parser.add_argument("--p-list", default="0.015,0.02,0.03,0.04,0.05")
    parser.add_argument("--shots", type=int, default=100_000, help="shot cap per point")
    parser.add_argument("--max-errors", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True, help="CSV to append to")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    new_file = not args.out.exists()
    with args.out.open("a", newline="") as stream:
        writer = csv.writer(stream)
        if new_file:
            writer.writerow(["code", "basis", "p", "seed", "shots", "errors", "seconds"])
        for name in args.codes.split(","):
            code = data_code() if name == "mem" else css_code(*merged_checks())
            print(f"{name}: [[{code.num_qudits}, {code.dimension}]]", flush=True)
            for basis in args.basis.split(","):
                for p in map(float, args.p_list.split(",")):
                    start = time.time()
                    shots, fails = sample(code, basis, p, args.shots, args.max_errors, rng)
                    seconds = f"{time.time() - start:.1f}"
                    writer.writerow([name, basis, p, args.seed, shots, fails, seconds])
                    stream.flush()
                    print(f"{name} {basis} p={p}: {fails}/{shots} LER {fails / shots:.2e}")


if __name__ == "__main__":
    main()
