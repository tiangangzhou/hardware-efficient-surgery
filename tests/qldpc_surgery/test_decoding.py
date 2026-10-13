"""Relay-BP with the BP-LSD fallback, synchronous and in two stages (``batched``)."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import relay_bp
import sinter
import stim
from qldpc.decoders import DetectorErrorModelArrays

from qldpc_surgery.batched import collect_shard, finalize_shard, merge_shards, shard_seed
from qldpc_surgery.decoding import RELAY_KWARGS, TieredRelayDecoder, sinter_decoder

# One check on two equally likely errors: min-sum BP cannot break the symmetry, so this
# truncated Relay-BP does not converge.
PCM, PRIORS, SYNDROME = np.array([[1, 1]]), np.array([0.1, 0.1]), np.array([1])
WEAK_RELAY = dict(pre_iter=2, num_sets=1, set_max_iter=1)


def test_relay_settings_complete_the_paper_table() -> None:
    """RELAY_KWARGS sets relay-bp options; the SM table's alpha_iteration_scaling_factor
    = 1 is relay-bp's default and is left out, as in the published task metadata."""
    defaults = {
        name: p.default
        for name, p in inspect.signature(relay_bp.RelayDecoderF32).parameters.items()
        if p.default is not inspect.Parameter.empty
    }
    assert set(RELAY_KWARGS) - {"name"} <= set(defaults)
    assert "alpha_iteration_scaling_factor" not in RELAY_KWARGS
    assert defaults["alpha_iteration_scaling_factor"] == 1.0


def test_fallback_decodes_what_relay_bp_cannot() -> None:
    relay_only = TieredRelayDecoder(PCM, error_channel=PRIORS, fallback=False, **WEAK_RELAY)
    deferred = relay_only.decode_relay_batch(np.vstack([SYNDROME, [0]]))
    assert deferred.failed_indices.tolist() == [0]
    assert not np.array_equal(PCM @ deferred.corrections[0] % 2, SYNDROME)
    tiered = TieredRelayDecoder(PCM, error_channel=PRIORS, **WEAK_RELAY)
    corrections = tiered.decode_deferred_batch(deferred)
    assert np.array_equal(PCM @ corrections[0] % 2, SYNDROME) and not corrections[1].any()
    assert np.array_equal(tiered.decode(SYNDROME), corrections[0])


def surface_code_task(p: float = 0.02) -> sinter.Task:
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_x", distance=5, rounds=3, after_clifford_depolarization=p
    )
    return sinter.Task(
        circuit=circuit,
        decoder="relay_bp_lsd",
        detector_error_model=circuit.detector_error_model(),
        json_metadata=dict(p=p),
    )


def test_sinter_decoder_corrects_single_errors() -> None:
    dem = surface_code_task().detector_error_model
    arrays = DetectorErrorModelArrays(dem)
    pcm = np.asarray(arrays.detector_flip_matrix.todense())
    compiled = sinter_decoder().compile_decoder_for_dem(dem)
    syndromes = pcm.T[:20].astype(np.uint8)  # the syndromes of single error mechanisms
    packed = np.packbits(syndromes, axis=1, bitorder="little")
    predicted = compiled.decode_shots_bit_packed(bit_packed_detection_event_data=packed)
    flips = np.asarray(arrays.observable_flip_matrix.todense()).T[:20]
    assert np.array_equal(
        np.unpackbits(predicted, axis=1, bitorder="little")[:, : flips.shape[1]], flips
    )


def test_batched_stages_match_synchronous_decoding(tmp_path) -> None:
    task, shots = surface_code_task(), 500
    for shard in (1, 2):
        collect_shard(task, tmp_path, shard, shots, batch_size=200)
        finalize_shard(task, tmp_path, shard)
    stats = merge_shards(task, tmp_path)

    arrays = DetectorErrorModelArrays(task.detector_error_model)
    decoder = TieredRelayDecoder(task.detector_error_model, **RELAY_KWARGS)
    errors = 0
    for shard in (1, 2):
        sampler = task.circuit.compile_detector_sampler(seed=shard_seed(task, shard))
        detectors, actual = sampler.sample(shots, separate_observables=True)
        corrections = np.concatenate(
            [decoder.decode_batch(detectors[i : i + 200]) for i in range(0, shots, 200)]
        )
        predicted = (arrays.observable_flip_matrix @ corrections.T).T % 2
        errors += int((predicted.astype(bool) ^ actual).any(axis=1).sum())
    counts = stats.custom_counts
    assert (stats.shots, stats.errors, counts["shards"]) == (2 * shots, errors, 2)
    assert 0 < counts["relay:obs_mistake_mask=E"] <= counts["obs_mistake_mask=E"] == errors
    assert counts["relay_nonconverged"] > 0


def test_finished_shards_are_kept_and_tied_to_their_task(tmp_path) -> None:
    task = surface_code_task()
    path = collect_shard(task, tmp_path, 1, 50)
    stamp = path.stat().st_mtime_ns
    assert collect_shard(task, tmp_path, 1, 50).stat().st_mtime_ns == stamp
    with pytest.raises(ValueError, match="another task"):
        finalize_shard(surface_code_task(0.01), tmp_path, 1)
