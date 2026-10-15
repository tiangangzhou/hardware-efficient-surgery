"""Batched two-stage campaign at l = 33 on a cluster (the p = 0.002-0.003 points of Fig. 4).

Each task (experiment, basis, p) of ``run_circuits.py`` is sampled in 25,000-shot shards
and decoded in two stages (:mod:`qldpc_surgery.delay_decoding`):

* ``collect``  samples one shard and decodes it with Relay-BP (multithreaded);
* ``finalize`` decodes the shots on which Relay-BP did not converge with BP-LSD;
* ``summary``  merges the finalized shards into a sinter CSV (for ``plot_fig4.py
  --stats``) and a failure-mask file (for ``circuit_error_table.py --masks``).

Shards live in ``OUT/<experiment>-<basis>-<p>/``, with p written as ``repr(float(p))`` (so
``0.003`` and ``0.0030`` name the same directory); other subdirectories of OUT are skipped.
Finished shards are skipped, so every command can be rerun, and a shard's seed depends only
on its task and index.  ``summary`` leaves out tasks without a finalized shard.

    uv run python scripts/ler/l33/run_batched.py collect --task surgery,X,0.003 --shard 1 --out runs
    uv run python scripts/ler/l33/run_batched.py finalize --out runs
    uv run python scripts/ler/l33/run_batched.py summary --out runs --csv l33.csv --masks masks.json

On Slurm, run one array element per shard, e.g. ``sbatch --array=1-20 --cpus-per-task=8``
with ``collect --task ... --shard $SLURM_ARRAY_TASK_ID``, then the same array with
``--cpus-per-task=1`` and ``finalize``. The published points used 18-280 shards per task.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import sinter
from run_circuits import ROUNDS, tasks

from qldpc_surgery.delay_decoding import (
    MASK,
    RELAY_MASK,
    collect_shard,
    finalize_shard,
    merge_shards,
    shard_paths,
)

SHARD_SHOTS = 25_000
BLOCK_OBSERVABLES = 148


def task_name(spec: str) -> str:
    """The directory name ("surgery-X-0.003") of a ``--task`` value ("surgery,X,0.003") or a
    directory name; raises ValueError for anything that is not a task."""
    experiment, basis, p = spec.replace(",", "-").split("-", 2)
    if experiment not in ("surgery", "idling") or basis not in ("X", "Z"):
        raise ValueError(f"not a task: {spec!r}")
    return f"{experiment}-{basis}-{float(p)!r}"


def task_of(name: str) -> sinter.Task:
    """The task of a shard directory or ``--task`` value."""
    experiment, basis, p = task_name(name).split("-", 2)
    return next(tasks([experiment], [basis], [float(p)], ROUNDS))


def directories(out: Path, name: str | None) -> list[Path]:
    """The directory of task ``name``, or every task directory of the campaign."""
    if name:
        return [out / task_name(name)]
    found = []
    for directory in sorted(d for d in out.iterdir() if d.is_dir()):
        try:
            is_task = task_name(directory.name) == directory.name
        except ValueError:
            is_task = False
        if is_task:
            found.append(directory)
        else:
            print(f"skipping {directory}: not a task directory", file=sys.stderr)
    return found


def failures(stats: sinter.TaskStats) -> dict:
    """One entry of the failure-mask file read by ``circuit_error_table.py``."""
    counts, meta = stats.custom_counts, stats.json_metadata
    relay = {k.removeprefix(RELAY_MASK): v for k, v in counts.items() if k.startswith(RELAY_MASK)}
    lsd = {
        k.removeprefix(MASK): v - relay.get(k.removeprefix(MASK), 0)
        for k, v in counts.items()
        if k.startswith(MASK)
    }
    return dict(
        experiment=meta["experiment"].capitalize(),
        basis=meta["basis"],
        p=meta["p"],
        num_shots=stats.shots,
        shards=counts["shards"],
        relay_failures=relay,
        lsd_failures={k: v for k, v in lsd.items() if v},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("collect", "finalize", "summary"))
    parser.add_argument("--out", type=Path, required=True, help="campaign directory")
    parser.add_argument("--task", help="experiment,basis,p (default for finalize/summary: all)")
    parser.add_argument("--shard", type=int, help="shard index (default for finalize: all)")
    parser.add_argument("--shots", type=int, default=SHARD_SHOTS, help="shots per shard")
    parser.add_argument("--csv", type=Path, help="summary: sinter CSV")
    parser.add_argument("--masks", type=Path, help="summary: failure-mask JSON")
    args = parser.parse_args()

    if args.command == "collect":
        if args.task is None or args.shard is None:
            parser.error("collect needs --task and --shard")
        directory = args.out / task_name(args.task)
        print(collect_shard(task_of(args.task), directory, args.shard, args.shots))
    elif args.command == "finalize":
        for directory in directories(args.out, args.task):
            task = task_of(directory.name)
            for shard in shard_paths(directory, "relay"):
                if args.shard in (None, shard):
                    print(finalize_shard(task, directory, shard), flush=True)
    else:
        all_stats = [merge_shards(task_of(d.name), d) for d in directories(args.out, args.task)]
        for stats in all_stats:
            meta, nonconverged = stats.json_metadata, stats.custom_counts["relay_nonconverged"]
            print(
                f"{meta['experiment']} {meta['basis']} p={meta['p']}: {stats.errors}/{stats.shots} "
                f"failures, {stats.custom_counts['shards']} shards, {nonconverged} to BP-LSD"
                + ("" if stats.shots else " (no finalized shard: left out)")
            )
        finished = [stats for stats in all_stats if stats.shots]
        if args.csv:
            lines = [sinter.CSV_HEADER] + [stats.to_csv_line() for stats in finished]
            args.csv.write_text("\n".join(lines) + "\n")
        if args.masks:
            results = [failures(stats) for stats in finished]
            content = dict(block_observables=BLOCK_OBSERVABLES, results=results)
            args.masks.write_text(json.dumps(content, indent=2) + "\n")


if __name__ == "__main__":
    main()
