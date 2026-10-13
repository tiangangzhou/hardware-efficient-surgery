"""Circuit-level LER at l = 9: idling vs fibre, packet and joint surgery (SM figure).

Ten noisy syndrome rounds, separate X and Z experiments, ``DepolarizingNoiseModel(p)``
(gates, preparations and measurements; no idling noise) at five log-spaced p from 0.001
to 0.005, and Relay-BP + BP-LSD decoding (see ``codes.py`` for the four modes).  Only
canonical readouts become time-like observables: the X experiment records 52 / 61 / 55 /
113 observables and the Z experiment 52 / 41 / 47 / 91 (idling / fibre / packet / joint).
Each task stops at 200,000 shots or 100 failures, as in the published parent campaign.
Task metadata and decoder name match the published run, so ``plot_figure.py --stats``
can summarise the output CSV.

    uv run python scripts/ler/l9/run_circuits.py --dry-run
    uv run python scripts/ler/l9/run_circuits.py --p-list 0.005 --shots 20 --out l9.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import sinter
from codes import CANONICAL, MODES, RESIDUAL, code_and_gadget
from qldpc.circuits import get_memory_experiment
from qldpc.circuits.noise_model import DepolarizingNoiseModel
from qldpc.objects import Pauli

from qldpc_surgery import build_batch_ppm_circuit, check_noiseless, sinter_decoder
from qldpc_surgery.decoding import LSD_KWARGS, RELAY_KWARGS

DECODER = "l9_relay_bp_lsd_ms0625_v1"
ROUNDS = 10
# Five log-spaced points 0.001 ... 0.005, written so the floats match the published tasks.
P_GRID = tuple(0.001 * 5 ** (i / 8) for i in range(0, 9, 2))


def tasks(modes: list[str], bases: list[str], p_list: list[float], rounds: int):
    for mode in modes:
        code, gadget = code_and_gadget(mode)
        for p in p_list:
            for basis in bases:
                pauli, noise = getattr(Pauli, basis), DepolarizingNoiseModel(p)
                if gadget is None:
                    circuit = get_memory_experiment(
                        code, basis=pauli, num_rounds=rounds, noise_model=noise
                    )
                else:
                    circuit = build_batch_ppm_circuit(
                        gadget,
                        rounds=rounds,
                        experiment_basis=pauli,
                        noise_model=noise,
                        single_sector=True,
                    )
                time_like = CANONICAL[mode] if basis == "X" else 0
                metadata = dict(
                    campaign="l9_v1",
                    mode=mode,
                    basis=basis,
                    p=p,
                    rounds=rounds,
                    n=code.num_qudits,
                    k=code.dimension,
                    block_observables=circuit.num_observables - time_like,
                    time_like_observables=time_like,
                    canonical_ppm=CANONICAL[mode],
                    residual_ppm=RESIDUAL[mode],
                    joint_shift=0 if mode == "joint" else None,
                    packet_multiplier=[0, 3] if mode == "packet" else None,
                    fibre=[3, 3],
                    surgery_basis="X" if gadget else None,
                    single_sector=bool(gadget),
                    noise_model="DepolarizingNoiseModel",
                    relay_kwargs=RELAY_KWARGS,
                    lsd_kwargs=LSD_KWARGS,
                )
                print(
                    f"{mode} {basis} p={p:.6g}: {circuit.num_observables} observables", flush=True
                )
                yield sinter.Task(
                    circuit=circuit,
                    detector_error_model=circuit.detector_error_model(decompose_errors=False),
                    decoder=DECODER,
                    json_metadata=metadata,
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--basis", default="X,Z")
    parser.add_argument("--p-list", default=",".join(format(p, ".17g") for p in P_GRID))
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--shots", type=int, default=200_000, help="maximum shots per task")
    parser.add_argument("--max-errors", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, help="sinter CSV (created or resumed)")
    parser.add_argument("--dry-run", action="store_true", help="check noiseless circuits and exit")
    args = parser.parse_args()
    p_list = [float(p) for p in args.p_list.split(",")]
    all_tasks = list(tasks(args.modes.split(","), args.basis.split(","), p_list, args.rounds))
    if args.dry_run:
        for task in all_tasks:
            check_noiseless(task.circuit)
        print("noiseless circuits flip no detector or observable")
        return
    if args.out is None:
        parser.error("--out is required")
    sinter.collect(
        tasks=all_tasks,
        decoders=[DECODER],
        custom_decoders={DECODER: sinter_decoder()},
        num_workers=args.workers,
        max_shots=args.shots,
        max_errors=args.max_errors,
        max_batch_size=64,
        count_observable_error_combos=True,
        count_detection_events=True,
        save_resume_filepath=str(args.out),
        print_progress=True,
    )


if __name__ == "__main__":
    main()
