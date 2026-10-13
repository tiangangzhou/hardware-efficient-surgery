"""Surgery circuits with one gadget: a single PPM, or a batch of PPMs.

``build_single_ppm_circuit`` measures one logical (Cain et al. arXiv:2603.28627
App. D); its time-like observable is the product of all new checks.
``build_batch_ppm_circuit`` measures the t logicals ``gadget.effective_batch_x`` with
one shared auxiliary code (SM, "Parallel measurements with one auxiliary code"):
time-like observable j is the product of the new checks selected by
``gadget.effective_readout_vectors[j]``. This is the circuit of the paper's
logical-error-rate simulations.

Both emit, in the match basis (``experiment_basis`` equal to ``gadget.basis``), k block
observables followed by the time-like ones; in the opposite basis, the block
logicals that commute with the measured operators (k - t when these are independent
modulo stabilisers). The batch builder further keeps only logicals that can be
frame-corrected through the shared auxiliary code.
"""

from __future__ import annotations

import numpy as np
import stim
from qldpc.circuits.noise_model import NoiseModel
from qldpc.codes.common import CSSCode
from qldpc.objects import PauliXZ

from ..hmatrix.gadget import GadgetSpec
from .engine import commuting_logicals, deformable_logicals, surgery_circuit


def _ppm_circuit(
    gadget: GadgetSpec,
    *,
    batch: bool,
    rounds: int,
    experiment_basis: PauliXZ | None,
    noise_model: NoiseModel | None,
    data_init: str | None,
    destructive_measure_data: bool,
    single_sector: bool,
) -> stim.Circuit:
    if experiment_basis is None:
        experiment_basis = gadget.basis
    merged = CSSCode(
        gadget.HX_merged.astype(np.int_), gadget.HZ_merged.astype(np.int_), is_subsystem_code=False
    )
    code, n = gadget.code, gadget.code.num_qudits
    logical_ops = np.asarray(code.get_logical_ops(experiment_basis), np.uint8)
    match = experiment_basis is gadget.basis
    if batch:
        measured = np.zeros((1, n), np.uint8) if match else gadget.effective_batch_x
        block = deformable_logicals(logical_ops, measured, merged, experiment_basis, n)
        readouts = [np.flatnonzero(r) for r in gadget.effective_readout_vectors]
    else:
        block = commuting_logicals(logical_ops, np.zeros(n, np.uint8) if match else gadget.x)
        readouts = [np.arange(gadget.f1.shape[1])]
    return surgery_circuit(
        merged,
        basis=gadget.basis,
        experiment_basis=experiment_basis,
        n_data=n,
        n_gadget_qubits=len(gadget.incidence),
        n_original_checks=(len(code.matrix_x), len(code.matrix_z)),
        n_adapter_checks=0,
        block_logicals=block,
        readouts=readouts,
        rounds=rounds,
        data_init=data_init,
        destructive_measure_data=destructive_measure_data,
        single_sector=single_sector,
        noise_model=noise_model,
    )


def build_single_ppm_circuit(
    gadget: GadgetSpec,
    *,
    rounds: int,
    experiment_basis: PauliXZ | None = None,
    noise_model: NoiseModel | None = None,
    data_init: str | None = None,
    destructive_measure_data: bool = True,
    single_sector: bool = False,
) -> stim.Circuit:
    """Measure the logical ``gadget.x`` (k + 1 observables in the match basis, k - 1
    in the opposite one). See ``engine.surgery_circuit`` for the other arguments;
    ``experiment_basis`` defaults to ``gadget.basis``."""
    return _ppm_circuit(
        gadget,
        batch=False,
        rounds=rounds,
        experiment_basis=experiment_basis,
        noise_model=noise_model,
        data_init=data_init,
        destructive_measure_data=destructive_measure_data,
        single_sector=single_sector,
    )


def build_batch_ppm_circuit(
    gadget: GadgetSpec,
    *,
    rounds: int,
    experiment_basis: PauliXZ | None = None,
    noise_model: NoiseModel | None = None,
    data_init: str | None = None,
    destructive_measure_data: bool = True,
    single_sector: bool = False,
) -> stim.Circuit:
    """Measure the t logicals ``gadget.effective_batch_x`` (k + t observables in the
    match basis). See ``engine.surgery_circuit`` for the other arguments;
    ``experiment_basis`` defaults to ``gadget.basis``."""
    return _ppm_circuit(
        gadget,
        batch=True,
        rounds=rounds,
        experiment_basis=experiment_basis,
        noise_model=noise_model,
        data_init=data_init,
        destructive_measure_data=destructive_measure_data,
        single_sector=single_sector,
    )
