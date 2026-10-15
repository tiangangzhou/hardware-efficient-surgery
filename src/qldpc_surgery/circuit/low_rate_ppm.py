"""Low-rate surgery: a circuit measuring one logical with one gadget.

``build_single_ppm_circuit`` measures the logical ``gadget.x`` (Cain et al.
arXiv:2603.28627 App. D); its time-like observable is the product of all new checks. In
the match basis (``experiment_basis`` equal to ``gadget.basis``) it emits k block
observables followed by the time-like one; in the opposite basis, the k - 1 block
logicals that commute with the measured operator. High-rate surgery, several logicals
through one shared auxiliary code, is ``high_rate_ppm``.
"""

from __future__ import annotations

import numpy as np
import stim
from qldpc.circuits.noise_model import NoiseModel
from qldpc.codes.common import CSSCode
from qldpc.objects import PauliXZ

from ..hmatrix.gadget import GadgetSpec
from .engine import commuting_logicals, require_single_readout, surgery_circuit


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
    in the opposite one). The new checks must read out ``gadget.x`` alone (a connected
    graph gadget); see ``engine.surgery_circuit`` for the other arguments;
    ``experiment_basis`` defaults to ``gadget.basis``."""
    require_single_readout(gadget)
    if experiment_basis is None:
        experiment_basis = gadget.basis
    merged = CSSCode(
        gadget.HX_merged.astype(np.int_), gadget.HZ_merged.astype(np.int_), is_subsystem_code=False
    )
    code, n = gadget.code, gadget.code.num_qudits
    logical_ops = np.asarray(code.get_logical_ops(experiment_basis), np.uint8)
    match = experiment_basis is gadget.basis
    measured = np.zeros(n, np.uint8) if match else gadget.x
    return surgery_circuit(
        merged,
        basis=gadget.basis,
        experiment_basis=experiment_basis,
        n_data=n,
        n_gadget_qubits=len(gadget.incidence),
        n_original_checks=(len(code.matrix_x), len(code.matrix_z)),
        n_adapter_checks=0,
        block_logicals=commuting_logicals(logical_ops, measured),
        readouts=[np.arange(gadget.f1.shape[1])],
        rounds=rounds,
        data_init=data_init,
        destructive_measure_data=destructive_measure_data,
        single_sector=single_sector,
        noise_model=noise_model,
    )
