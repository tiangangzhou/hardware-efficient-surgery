"""Delayed two-stage decoding of sharded sampling campaigns (cluster runs of Fig. 4).

A task is sampled in independent shards, each decoded in two stages that can run as
separate jobs:

1. ``collect_shard`` samples the shard with Stim and decodes every shot with Relay-BP
   (``RELAY_KWARGS``).  It keeps the logical failures of the converged shots and the
   syndromes of the shots on which Relay-BP did not converge.
2. ``finalize_shard`` decodes those syndromes with BP-LSD (``LSD_KWARGS``).

The decoder chain is that of :func:`qldpc_surgery.decoding.sinter_decoder`; only the
scheduling differs.  Each stage writes one ``.npz`` per shard (``relay-NNNN.npz``,
``final-NNNN.npz``; ``shard_paths`` lists them) and skips shards already on disk, so both
stages can be restarted.  The seed of a shard depends only on the task and the shard index.
``merge_shards`` adds the finalized shards of a task into ``sinter.TaskStats``.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import sinter
from qldpc.decoders import DetectorErrorModelArrays

from .decoding import RELAY_KWARGS, DeferredRelayBatch, TieredRelayDecoder

MASK = "obs_mistake_mask="  # sinter's custom-count key: one character per observable
RELAY_MASK = "relay:" + MASK  # failures already made by Relay-BP (T1)


def shard_seed(task: sinter.Task, shard: int) -> int:
    """Stim seed of one shard, derived from the task's strong id and the shard index."""
    digest = hashlib.sha256(f"{task.strong_id()}/{shard}".encode()).digest()
    return int.from_bytes(digest[:8], "little") >> 1


def _arrays(task: sinter.Task):
    arrays = DetectorErrorModelArrays(task.detector_error_model)
    return arrays.detector_flip_matrix, arrays.observable_flip_matrix, arrays.error_probs


def _failures(actual: np.ndarray, corrections: np.ndarray, observables) -> Counter:
    """Mistake masks ('E' = wrong observable) of the shots whose prediction is wrong."""
    wrong = actual ^ (observables @ corrections.T).T.astype(np.uint8) % 2
    return Counter(MASK + "".join("_E"[b] for b in row) for row in wrong if row.any())


def _save(path: Path, **arrays) -> None:
    """Write atomically, through a hidden temporary file that ``shard_paths`` never lists."""
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def shard_paths(directory: Path, stage: str) -> dict[int, Path]:
    """{shard: file} of the ``stage`` ("relay" or "final") files ``<stage>-NNNN.npz`` of a task
    directory, in shard order; other files (e.g. temporary ones) are ignored."""
    paths = {}
    for path in Path(directory).glob(f"{stage}-*.npz"):
        index = path.stem.removeprefix(f"{stage}-")
        if index.isdigit():
            paths[int(index)] = path
    return dict(sorted(paths.items()))


def _load(path: Path, task: sinter.Task) -> dict:
    with np.load(path) as data:
        content = {key: data[key] for key in data.files}
    if str(content["strong_id"]) != task.strong_id():
        raise ValueError(f"{path} belongs to another task")
    return content


def collect_shard(
    task: sinter.Task, directory: Path, shard: int, shots: int, *, batch_size: int = 1000
) -> Path:
    """Stage 1: sample shard ``shard`` and decode it with Relay-BP (``relay-NNNN.npz``)."""
    path = Path(directory) / f"relay-{shard:04d}.npz"
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm, observables, probs = _arrays(task)
    relay = TieredRelayDecoder(pcm, error_channel=probs, fallback=False, **RELAY_KWARGS)
    start = time.monotonic()
    sampler = task.circuit.compile_detector_sampler(seed=shard_seed(task, shard))
    detectors, actual = sampler.sample(shots, separate_observables=True)
    failures, nonconverged = Counter(), []
    for i in range(0, shots, batch_size):
        batch = relay.decode_relay_batch(detectors[i : i + batch_size])
        converged = batch.success
        failures += _failures(
            actual[i : i + batch_size][converged], batch.corrections[converged], observables
        )
        nonconverged.extend(i + batch.failed_indices)
    _save(
        path,
        strong_id=task.strong_id(),
        shots=shots,
        seconds=time.monotonic() - start,
        failures=json.dumps(failures),
        detectors=np.packbits(detectors[nonconverged], axis=1),
        observables=np.packbits(actual[nonconverged], axis=1),
    )
    return path


def finalize_shard(task: sinter.Task, directory: Path, shard: int) -> Path:
    """Stage 2: decode the non-converged shots of a collected shard with BP-LSD."""
    path = Path(directory) / f"final-{shard:04d}.npz"
    if path.exists():
        return path
    relay = _load(Path(directory) / f"relay-{shard:04d}.npz", task)
    pcm, observables, probs = _arrays(task)
    start = time.monotonic()
    syndromes = np.unpackbits(relay["detectors"], axis=1, count=pcm.shape[0])
    actual = np.unpackbits(relay["observables"], axis=1, count=observables.shape[0])
    decoder = TieredRelayDecoder(pcm, error_channel=probs, parallel=False, **RELAY_KWARGS)
    shots = len(syndromes)
    unfinished = DeferredRelayBatch(
        syndromes, np.zeros((shots, pcm.shape[1]), np.uint8), np.zeros(shots, bool)
    )
    corrections = decoder.decode_deferred_batch(unfinished)
    _save(
        path,
        strong_id=task.strong_id(),
        shots=relay["shots"],
        seconds=relay["seconds"] + time.monotonic() - start,
        nonconverged=shots,
        relay_failures=relay["failures"],
        lsd_failures=json.dumps(_failures(actual, corrections, observables)),
    )
    return path


def merge_shards(task: sinter.Task, directory: Path) -> sinter.TaskStats:
    """Sum the finalized shards of a task.

    Custom counts: ``obs_mistake_mask=...`` for every failure (as sinter writes them),
    ``relay:obs_mistake_mask=...`` for the T1 failures among them, ``relay_nonconverged``
    for the shots passed to BP-LSD and ``shards``.
    """
    shots, seconds, counts = 0, 0.0, Counter()
    for path in shard_paths(directory, "final").values():
        shard = _load(path, task)
        relay = Counter(json.loads(str(shard["relay_failures"])))
        lsd = Counter(json.loads(str(shard["lsd_failures"])))
        shots, seconds = shots + int(shard["shots"]), seconds + float(shard["seconds"])
        counts += relay + lsd + Counter({k.replace(MASK, RELAY_MASK): v for k, v in relay.items()})
        counts += Counter(relay_nonconverged=int(shard["nonconverged"]), shards=1)
    return sinter.TaskStats(
        strong_id=task.strong_id(),
        decoder=task.decoder,
        json_metadata=task.json_metadata,
        shots=shots,
        errors=sum(v for k, v in counts.items() if k.startswith(MASK)),
        discards=0,
        seconds=seconds,
        custom_counts=counts,
    )
