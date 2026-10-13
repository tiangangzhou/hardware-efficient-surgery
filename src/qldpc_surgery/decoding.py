"""Decoder of the circuit-level simulations: Relay-BP with a BP-LSD fallback.

T1 is one ``relay_bp.RelayDecoderF32`` call: up to ``pre_iter`` iterations of plain
min-sum BP, then relay legs if BP has not converged (Mueller et al., "Improved belief
propagation is sufficient for real-time decoding of quantum memory",
arXiv:2506.01779). T2 decodes the original syndrome of every shot on which T1 does
not converge with a fresh BP-LSD decoder; it inherits no Relay-BP messages (tiers
as in Zhao et al. arXiv:2604.16209 App. B.2). Both use the error probabilities of
the undecomposed detector error model as priors. The settings are those of the
SM table "Shared parameters for the circuit simulations of logical error rates".
"""

from __future__ import annotations

import dataclasses
import functools

import numpy as np
import scipy.sparse
import stim
from qldpc import decoders
from qldpc.decoders import DetectorErrorModelArrays

# T2 (BP-LSD; also the code-capacity decoder). omp_thread_count keeps ldpc's default 1.
LSD_KWARGS: dict[str, object] = {
    "bp_method": "minimum_sum",
    "schedule": "serial",
    "max_iter": 100,
    "ms_scaling_factor": 0.625,
    "lsd_method": "lsd_e",
    "lsd_order": 5,
}

# T1 (Relay-BP). gamma_dist_interval matters: relay-bp's default (-0.24, 0.66) admits
# negative memory strengths, which stall BP on these codes. alpha_iteration_scaling_factor
# keeps relay-bp's default 1.0 (left out, as in the published task metadata).
RELAY_KWARGS: dict[str, object] = {
    "name": "RelayDecoderF32",
    "pre_iter": 120,
    "num_sets": 30,
    "set_max_iter": 60,
    "gamma0": 0.1,
    "alpha": 0.85,
    "gamma_dist_interval": (0.1, 0.8),
    "stop_nconv": 1,
    "seed": 0,
}


@dataclasses.dataclass(frozen=True)
class DeferredRelayBatch:
    """T1 output awaiting T2: the syndromes, Relay-BP's corrections (error vectors over
    the error mechanisms) and whether Relay-BP converged on each shot."""

    syndromes: np.ndarray
    corrections: np.ndarray
    success: np.ndarray

    @property
    def failed_indices(self) -> np.ndarray:
        return np.flatnonzero(~self.success)


class TieredRelayDecoder:
    """T1 Relay-BP and T2 BP-LSD (module docstring) on a detector error model, or on a
    check matrix with ``error_channel``.

    ``decode_relay_batch`` runs T1 and ``decode_deferred_batch`` finishes its output with
    T2, so the tiers can run as separate jobs (``batched.py``); ``decode_batch`` does
    both. ``fallback=False`` disables T2, ``lsd_kwargs`` overrides entries of
    ``LSD_KWARGS``, and ``relay_kwargs`` go to Relay-BP (relay-bp's defaults otherwise).
    ``parallel`` multithreads Relay-BP over the shots of a batch.
    """

    def __init__(
        self,
        pcm_or_dem: np.ndarray | scipy.sparse.spmatrix | stim.DetectorErrorModel,
        *,
        error_channel: np.ndarray | None = None,
        fallback: bool = True,
        lsd_kwargs: dict[str, object] | None = None,
        name: str = "RelayDecoderF32",
        parallel: bool = True,
        **relay_kwargs: object,
    ) -> None:
        if isinstance(pcm_or_dem, stim.DetectorErrorModel):
            arrays = DetectorErrorModelArrays(pcm_or_dem)
            pcm = arrays.detector_flip_matrix
            if error_channel is None:
                error_channel = arrays.error_probs
        else:
            pcm = pcm_or_dem
        # Priors in (0, 1/2) keep the log-likelihood ratios finite and positive.
        priors = np.clip(np.asarray(error_channel, float), 1e-12, 0.5 - 1e-12)
        self.num_errors, self.parallel = pcm.shape[1], parallel
        self.relay = decoders.RelayBPDecoder(pcm, priors, name=name, **relay_kwargs)
        self.lsd = None
        if fallback:
            settings = LSD_KWARGS | (lsd_kwargs or {})
            self.lsd = decoders.get_decoder_BP_LSD(pcm, error_channel=priors, **settings)

    def decode_relay_batch(self, syndromes: np.ndarray) -> DeferredRelayBatch:
        syndromes = np.atleast_2d(np.asarray(syndromes, np.uint8))
        if not len(syndromes):
            return DeferredRelayBatch(
                syndromes, np.zeros((0, self.num_errors), np.uint8), np.zeros(0, bool)
            )
        results = self.relay.decode_detailed_batch(
            syndromes, parallel=self.parallel, progress_bar=False
        )
        corrections = np.array([r.decoding for r in results], np.uint8)
        return DeferredRelayBatch(syndromes, corrections, np.array([r.success for r in results]))

    def decode_deferred_batch(self, batch: DeferredRelayBatch) -> np.ndarray:
        corrections = np.array(batch.corrections, np.uint8)
        if self.lsd is not None:
            for i in batch.failed_indices:
                corrections[i] = self.lsd.decode(batch.syndromes[i])
        return corrections

    def decode_batch(self, syndromes: np.ndarray) -> np.ndarray:
        return self.decode_deferred_batch(self.decode_relay_batch(syndromes))

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        return self.decode_batch(np.atleast_2d(syndrome))[0]


def sinter_decoder() -> decoders.SinterDecoder:
    """The paper's decoder (``RELAY_KWARGS`` then ``LSD_KWARGS``) for sinter, on the
    undecomposed detector error model; sinter parallelises over processes instead of
    threads."""
    constructor = functools.partial(TieredRelayDecoder, parallel=False, **RELAY_KWARGS)
    return decoders.SinterDecoder(decompose_errors=False, decoder=constructor)
