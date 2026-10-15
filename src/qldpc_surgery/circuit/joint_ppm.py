"""Surgery circuit measuring X_l X_r (or Z_l Z_r) through a bridge (``hmatrix.joint``).

The two gadgets act on two codes (data qubits of the left code first) or on the
same code. Observables as in ``low_rate_ppm``: in the match basis the k_l + k_r (or k)
block logicals and one time-like observable, the product of all new vertex checks
of both gadgets; in the opposite basis the block logicals commuting with the
measured product.
"""

from __future__ import annotations

import numpy as np
import stim
from qldpc.circuits.noise_model import NoiseModel
from qldpc.codes.common import CSSCode
from qldpc.objects import PauliXZ
from scipy.linalg import block_diag

from ..hmatrix.gadget import GadgetSpec
from ..hmatrix.joint import JointSpec, joint_merged_code
from .engine import commuting_logicals, require_single_readout, surgery_circuit


def build_joint_ppm_circuit(
    g_l: GadgetSpec,
    g_r: GadgetSpec,
    bridge: JointSpec,
    *,
    rounds: int,
    experiment_basis: PauliXZ | None = None,
    noise_model: NoiseModel | None = None,
    data_init: str | tuple[str, str] | None = None,
    destructive_measure_data: bool = True,
    single_sector: bool = False,
) -> tuple[stim.Circuit, CSSCode]:
    """(circuit, merged code) of the joint measurement.

    ``data_init`` is a string over all data qubits or, for two codes, a pair of
    strings, one per code (each may be a single broadcast character). Each gadget must
    read out its ``x`` alone (a connected graph). See ``engine.surgery_circuit`` for the
    other arguments; ``experiment_basis`` defaults to ``bridge.basis``.
    """
    for g in (g_l, g_r):
        require_single_readout(g)
    if experiment_basis is None:
        experiment_basis = bridge.basis
    merged = joint_merged_code(g_l, g_r, bridge)
    codes = (g_l.code,) if g_l.code is g_r.code else (g_l.code, g_r.code)
    if isinstance(data_init, tuple):
        data_init = "".join(
            s * c.num_qudits if len(s) == 1 else s for s, c in zip(data_init, codes, strict=True)
        )
    logical_ops = block_diag(
        *(np.asarray(c.get_logical_ops(experiment_basis), np.uint8) for c in codes)
    )
    measured = np.concatenate([g_l.x, g_r.x]) if len(codes) == 2 else g_l.x ^ g_r.x
    match = experiment_basis is bridge.basis
    n_data = sum(c.num_qudits for c in codes)
    block = commuting_logicals(logical_ops, np.zeros(n_data, np.uint8) if match else measured)
    return surgery_circuit(
        merged,
        basis=bridge.basis,
        experiment_basis=experiment_basis,
        n_data=n_data,
        n_gadget_qubits=len(bridge.g_l_aug.incidence) + len(bridge.g_r_aug.incidence),
        n_original_checks=(
            sum(len(c.matrix_x) for c in codes),
            sum(len(c.matrix_z) for c in codes),
        ),
        n_adapter_checks=bridge.width - 1,
        block_logicals=block,
        readouts=[np.arange(g_l.f1.shape[1] + g_r.f1.shape[1])],
        rounds=rounds,
        data_init=data_init,
        destructive_measure_data=destructive_measure_data,
        single_sector=single_sector,
        noise_model=noise_model,
    ), merged
