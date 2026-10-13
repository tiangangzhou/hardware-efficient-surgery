"""Circuit-level LER of idling vs X-type fibre surgery at l = 33 (Fig. 4).

For each physical error rate p and basis, samples the 20-round memory experiment of the
[[1122, 148]] code and the 20-round surgery experiment of the merged [[2442, 113]] code
(``DepolarizingNoiseModel(p)``: gates, preparations and measurements, no idling noise;
EdgeColoring schedule) with sinter and Relay-BP + BP-LSD (``qldpc_surgery.decoding``).
Observables: idling 148 data logicals; surgery 148 data logicals + 33 time-like fibre
readouts (X) or 148 - (33 + 2) surviving data logicals (Z).  Failure masks per observable
are counted, so ``circuit_error_table.py`` can split block and time-like failures.  The
output CSV is sinter's resume file; ``plot_fig4.py --stats`` draws Fig. 4 from it.

    uv run python scripts/ler/l33/run_circuits.py --dry-run
    uv run python scripts/ler/l33/run_circuits.py --p-list 0.003 --shots 20 --out l33.csv

The published counts (``data/ler/l33/counts.json``, 0.45-7 million shots per point) were
sampled on a cluster, the p = 0.002-0.003 points with ``run_batched.py``; a surgery shot
takes roughly 1 s.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import sinter
from codes import data_code, surgery_gadget
from qldpc.circuits import get_memory_experiment
from qldpc.circuits.noise_model import DepolarizingNoiseModel
from qldpc.objects import Pauli

from qldpc_surgery import build_batch_ppm_circuit, check_noiseless, sinter_decoder

DECODER = "relay_bp_lsd"
ROUNDS = 20
SHOTS = {0.0016: 5_000_000, 0.002: 5_000_000, 0.0025: 3_000_000, 0.003: 500_000}  # default


def tasks(experiments: list[str], bases: list[str], p_list: list[float], rounds: int):
    code = data_code() if "idling" in experiments else None
    gadget = surgery_gadget() if "surgery" in experiments else None
    for experiment in experiments:
        for p in p_list:
            for basis in bases:
                pauli, noise = getattr(Pauli, basis), DepolarizingNoiseModel(p)
                if experiment == "surgery":
                    circuit = build_batch_ppm_circuit(
                        gadget,
                        rounds=rounds,
                        experiment_basis=pauli,
                        noise_model=noise,
                        single_sector=True,
                    )
                else:
                    circuit = get_memory_experiment(
                        code, basis=pauli, num_rounds=rounds, noise_model=noise
                    )
                print(
                    f"{experiment} {basis} p={p}: {circuit.num_observables} observables", flush=True
                )
                yield sinter.Task(
                    circuit=circuit,
                    decoder=DECODER,
                    detector_error_model=circuit.detector_error_model(decompose_errors=False),
                    json_metadata=dict(experiment=experiment, p=p, basis=basis, rounds=rounds),
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--experiments", default="surgery,idling")
    parser.add_argument("--basis", default="X,Z")
    parser.add_argument("--p-list", default=",".join(map(str, SHOTS)))
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--shots", type=int, help="shots per task (default: SHOTS[p])")
    parser.add_argument("--max-errors", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, help="sinter CSV (created or resumed)")
    parser.add_argument("--dry-run", action="store_true", help="check noiseless circuits and exit")
    args = parser.parse_args()
    p_list = [float(p) for p in args.p_list.split(",")]
    all_tasks = list(tasks(args.experiments.split(","), args.basis.split(","), p_list, args.rounds))
    if args.dry_run:
        for task in all_tasks:
            check_noiseless(task.circuit)
        print("noiseless circuits flip no detector or observable")
        return
    if args.out is None:
        parser.error("--out is required")
    for p in p_list:
        sinter.collect(
            tasks=[t for t in all_tasks if t.json_metadata["p"] == p],
            decoders=[DECODER],
            custom_decoders={DECODER: sinter_decoder()},
            num_workers=args.workers,
            max_shots=args.shots or SHOTS[p],
            max_errors=args.max_errors,
            count_observable_error_combos=True,  # failure masks per observable
            save_resume_filepath=str(args.out),
            print_progress=True,
        )


if __name__ == "__main__":
    main()
