"""QDistRnd distance estimates for the merged codes (SM "Distance estimates for merged codes
from QDistRnd" and its table).

Subcommands:

* ``summarize`` prints the table from the run records in ``data/codes/qdistrnd`` (or
  ``--results DIR``) and checks the shipped records against the published table.  N_P is
  the number of information sets, m_P the number of distinct tracked minimum-weight words
  (at most 100), h_P how often they were found and mu_P = h_P / m_P; d_hat = min(d_X, d_Z)
  is an upper bound on d, and C_XZ = max(0, 1 - exp(-mu_X) - exp(-mu_Z)) is a heuristic
  score, not a calibrated confidence level.
* ``export`` writes the merged HX, HZ of instances I-III as Matrix Market files and checks
  shape, rank and SHA256 against the inputs of the paper's runs.
* ``run`` exports the matrices and runs GAP/QDistRnd (``gap`` with the QDistRnd package on
  the PATH), writing scripts, logs and JSON records to ``--output``.  With debug = 15
  DistRandCSS also reports the tracked minimum-weight words and their multiplicities.
  The paper used QDistRnd 0.9.5, GAP 4.15.1, seed 260920, mindist = 0, debug = 15:

    uv run python scripts/codes/qdistrnd.py summarize               # < 1 s
    uv run python scripts/codes/qdistrnd.py export --output DIR     # ~1 s
    uv run python scripts/codes/qdistrnd.py run --ell 9 --sets 1000                # ~10 s
    uv run python scripts/codes/qdistrnd.py run --ell 15 --sets 1000000 --maxav 5.3  # ~1 min
    uv run python scripts/codes/qdistrnd.py run --ell 33 --sets 10000000 --maxav 5.3 \
        --timeout 604800                                            # X: ~1 day, Z: timeout

``--maxav`` is QDistRnd's stopping rule (stop once the mean multiplicity of the tracked words
exceeds it).  The l = 33 Z run reached the seven-day timeout; a timed-out run is recorded
from the last progress checkpoint in its log.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import scipy.sparse
from common import DATA
from scipy.stats import chi2

from algebraic_qec.bitmask import Rowspace, row_masks
from algebraic_qec.examples.surgery_instances import instance

RECORDS = DATA / "qdistrnd"
LIFTS = (9, 15, 33)
# Published table: (l, P) -> (d_P, N_P, m_P, h_P, mu_P); l -> (d_hat, C_XZ in %).
PAPER_ROWS = {
    (9, "X"): (9, 1000, 9, 82, "9.1111"),
    (9, "Z"): (8, 1000, 100, 784, "7.8400"),
    (15, "X"): (11, 1566, 29, 154, "5.3103"),
    (15, "Z"): (10, 2967, 100, 531, "5.3100"),
    (33, "X"): (21, 986948, 33, 175, "5.3030"),
    (33, "Z"): (20, 6823800, 100, 396, "3.9600"),
}
PAPER_MERGED = {9: (8, "99.95"), 15: (10, "99.01"), 33: (20, "97.60")}
NUMBER = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
CHECKPOINT = re.compile(
    r"Round (\d+) of \d+; lowest wgt = (\d+) count=(\d+)\s+"
    rf"Average number of times vectors with the lowest weight were found = ({NUMBER})\s+"
    r"Number of different vectors = (\d+)"
)


def recorded_inputs(ell: int) -> dict:
    """Input records (file, shape, rank, sha256) of the paper's runs at this lift."""
    x, z = (json.loads((RECORDS / f"l{ell}_{p}.json").read_text())["inputs"] for p in "XZ")
    assert x == z
    return x


def export(ell: int, out: Path) -> None:
    """Write l{ell}_hx.mtx, l{ell}_hz.mtx (sorted coordinate format, one-based) and check them."""
    code = instance(ell).merged_code()
    for name, matrix in (("hx", code.binary_matrix_x), ("hz", code.binary_matrix_z)):
        coo = scipy.sparse.coo_array(matrix)
        order = np.lexsort((coo.col, coo.row))
        lines = [
            "%%MatrixMarket matrix coordinate integer general",
            "%",
            f"{coo.shape[0]} {coo.shape[1]} {coo.nnz}",
        ]
        content = "\n".join(lines + [f"{coo.row[i] + 1} {coo.col[i] + 1} 1" for i in order]) + "\n"
        record = recorded_inputs(ell)[name]
        assert list(matrix.shape) == record["shape"]
        assert Rowspace(row_masks(matrix)).rank == record["rank"]
        assert hashlib.sha256(content.encode()).hexdigest() == record["sha256"], f"l={ell} {name}"
        (out / record["file"]).write_text(content)


def gap_script(ell: int, sector: str, args) -> str:
    """DistRandCSS(GZ, GX, ...) searches X logicals, DistRandCSS(GX, GZ, ...) Z logicals."""
    first, second = ("GZ", "GX") if sector == "X" else ("GX", "GZ")
    options = "field:=GF(2)" + (f",maxav:={args.maxav:g}" if args.maxav is not None else "")
    return f"""if LoadPackage("QDistRnd") <> true then Error("QDistRnd unavailable"); fi;
Print("GAP_VERSION=", GAPInfo.Version, "\\n");
Print("QDIST_VERSION=", PackageInfo("QDistRnd")[1].Version, "\\n");
Reset(GlobalMersenneTwister,{args.seed});
GX:=ReadMTXE("l{ell}_hx.mtx",0)[3];;
GZ:=ReadMTXE("l{ell}_hz.mtx",0)[3];;
Print("INPUT_RANKS=",RankMat(GX),",",RankMat(GZ),"\\n");
t0:=Runtime();;
result:=DistRandCSS({first},{second},{args.sets},0,{args.debug}:{options});;
Print("QDIST_RESULT=",result,"\\n");
Print("RUNTIME_MS=",Runtime()-t0,"\\n");
QUIT;
"""


def parse_log(text: str) -> dict:
    """Record fields from a GAP log: the result or, without one, the last progress checkpoint."""
    text = text.replace("\\\n", "")  # GAP continues long numbers with backslash-newline
    version = lambda tag: (match := re.search(rf"{tag}=(\S+)", text)) and match[1]
    out = dict(gap_version=version("GAP_VERSION"), qdistrnd_version=version("QDIST_VERSION"))
    if result := re.search(r"QDIST_RESULT=(-?\d+)", text):
        distinct = re.findall(r"Number of different vectors = (\d+)", text)
        out.update(
            status="complete",
            information_sets=int(re.search(r"(\d+) rounds of \d+ were made", text)[1]),
            distance_upper_bound=int(result[1]),
            total_best_weight_hits=int(re.search(r"vector found (\d+) times", text)[1]),
            tracked_distinct_vectors=int(distinct[-1]) if distinct else None,
            mean_multiplicity=float(re.findall(rf"<n>=({NUMBER})", text)[-1]),
            pearson_diagnostic=None,
        )
        if statistic := re.search(rf"X\^2_(\d+)=({NUMBER})", text):
            # QDistRnd prints X^2 with (distinct words - 1) degrees of freedom.
            df, value = int(statistic[1]), float(statistic[2])
            p_value = float(chi2.sf(value, df))
            out["tracked_distinct_vectors"] = df + 1
            out["pearson_diagnostic"] = dict(df=df, statistic=value, asymptotic_p_value=p_value)
    elif checkpoints := CHECKPOINT.findall(text):
        rounds, weight, hits, mean, distinct = checkpoints[-1]
        out.update(
            status="checkpoint",
            information_sets=int(rounds),
            distance_upper_bound=int(weight),
            total_best_weight_hits=int(hits),
            tracked_distinct_vectors=int(distinct),
            mean_multiplicity=float(mean),
        )
    if out.get("tracked_distinct_vectors"):
        out["tracked_hits"] = round(out["mean_multiplicity"] * out["tracked_distinct_vectors"])
    return out


def run(args) -> None:
    gap = shutil.which("gap")
    if gap is None:
        raise SystemExit("GAP is not on the PATH; install GAP and its QDistRnd package.")
    out = args.output or Path(tempfile.mkdtemp(prefix="qdistrnd-"))
    out.mkdir(parents=True, exist_ok=True)
    export(args.ell, out)
    n, k, _ = instance(args.ell).merged_parameters
    for sector in (args.pauli,) if args.pauli else ("X", "Z"):
        tag, start, exit_code = f"l{args.ell}_{sector}", time.monotonic(), None
        print(f"Running {tag} in {out}", flush=True)
        (out / f"{tag}.g").write_text(gap_script(args.ell, sector, args))
        with (out / f"{tag}.log").open("w") as log:
            command = [gap, "-q", "--quitonbreak", f"{tag}.g"]
            try:
                process = subprocess.run(
                    command, cwd=out, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout
                )
                exit_code = process.returncode
            except subprocess.TimeoutExpired:
                pass
        parsed = parse_log((out / f"{tag}.log").read_text())
        versions = {key: parsed.pop(key) for key in ("gap_version", "qdistrnd_version")}
        record = dict(ell=args.ell, pauli=sector, n=n, k=k, **versions, seed=args.seed)
        record.update(
            debug=args.debug, mindist=0, maxav=args.maxav, maximum_information_sets=args.sets
        )
        record.update(**parsed, seconds=round(time.monotonic() - start, 1))
        record["inputs"] = recorded_inputs(args.ell)
        if exit_code is None:
            record["status"] = "timeout"
        elif exit_code != 0 or record.get("status") != "complete":
            raise RuntimeError(f"GAP failed for {tag} (exit {exit_code}); see {out / tag}.log")
        (out / f"{tag}.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps({key: v for key, v in record.items() if key != "inputs"}), flush=True)


def summarize(results: Path) -> None:
    records = {}
    for path in sorted(results.glob("l*_[XZ].json")):
        record = json.loads(path.read_text())
        records[record["ell"], record["pauli"]] = record
    published = results.resolve() == RECORDS.resolve()
    header = f"{'l':>3} {'[[n,k]]':>13} {'P':>2} {'d_P':>4} {'N_P':>10} {'m_P':>4} {'h_P':>5}"
    header += f" {'mu_P':>7} {'d_hat':>6} {'C_XZ(%)':>8}  status"
    print(header + "\n" + "-" * len(header))
    for ell in sorted({ell for ell, _ in records}):
        sectors = {p: records[ell, p] for p in "XZ" if (ell, p) in records}
        estimate = min(r["distance_upper_bound"] for r in sectors.values())
        score = max(0.0, 1 - sum(math.exp(-r["mean_multiplicity"]) for r in sectors.values()))
        for index, (pauli, r) in enumerate(sectors.items()):
            mu = r["tracked_hits"] / r["tracked_distinct_vectors"]
            assert abs(mu - r["mean_multiplicity"]) < 1e-9
            first = index == 0
            code = f"[[{r['n']},{r['k']}]]" if first else ""
            row = (r["distance_upper_bound"], r["information_sets"])
            row += (r["tracked_distinct_vectors"], r["tracked_hits"], f"{mu:.4f}")
            merged = f"{estimate:6d} {100 * score:8.2f}" if first and len(sectors) == 2 else ""
            print(
                f"{ell if first else '':>3} {code:>13} {pauli:>2} {row[0]:4d} {row[1]:10d}"
                f" {row[2]:4d} {row[3]:5d} {row[4]:>7} {merged:15}  {r['status']}"
            )
            assert not published or row == PAPER_ROWS[ell, pauli], f"l={ell} {pauli}: {row}"
        if published:
            assert (estimate, f"{100 * score:.2f}") == PAPER_MERGED[ell]
    if published:
        assert len(records) == len(PAPER_ROWS)
        print("All rows agree with the published table.")
        print("The l = 33 Z record is the last checkpoint before the seven-day timeout.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("summarize").add_argument("--results", type=Path, default=RECORDS)
    exporting = commands.add_parser("export")
    exporting.add_argument("--ell", type=int, choices=LIFTS)
    exporting.add_argument("--output", type=Path)
    running = commands.add_parser("run")
    running.add_argument("--ell", type=int, choices=LIFTS, required=True)
    running.add_argument("--pauli", choices=("X", "Z"), help="one sector (default: both)")
    running.add_argument(
        "--sets", type=int, default=1000, help="maximum number of information sets"
    )
    running.add_argument("--maxav", type=float, help="QDistRnd maxav stopping threshold")
    running.add_argument("--seed", type=int, default=260920)
    running.add_argument("--debug", type=int, default=15)
    running.add_argument("--timeout", type=float, help="wall-clock limit per sector in seconds")
    running.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "summarize":
        summarize(args.results)
    elif args.command == "export":
        out = args.output or Path(tempfile.mkdtemp(prefix="qdistrnd-inputs-"))
        out.mkdir(parents=True, exist_ok=True)
        for ell in LIFTS if args.ell is None else (args.ell,):
            export(ell, out)
            print(f"l={ell}: wrote l{ell}_hx.mtx, l{ell}_hz.mtx to {out} (hashes match)")
    else:
        run(args)


if __name__ == "__main__":
    main()
