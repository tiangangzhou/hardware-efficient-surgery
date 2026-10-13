"""Surgery circuits: deterministic detectors, observable sets and their values."""

from __future__ import annotations

import numpy as np
import pytest
from qldpc import codes
from qldpc.circuits import DepolarizingNoiseModel
from qldpc.objects import Pauli
from scipy.linalg import block_diag

from qldpc_surgery import (
    build_batch_ppm_circuit,
    build_bridge,
    build_gadget,
    build_joint_ppm_circuit,
    build_single_ppm_circuit,
    check_noiseless,
)
from qldpc_surgery.hmatrix.gadget import GadgetSpec
from tests.qldpc_surgery.fixtures import (
    bb36,
    logical,
    logical_state,
    observable_values,
    undetectable_logical_errors,
)

NOISE = DepolarizingNoiseModel(1e-3)
OTHER = {Pauli.X: Pauli.Z, Pauli.Z: Pauli.X}


def gadget(code, basis, i=0, method="edge_expansion"):
    return build_gadget(code, logical(code, basis, i), basis=basis, method=method)


def batch_gadget(code, basis) -> GadgetSpec:
    """Two single gadgets side by side, measuring two logicals at once."""
    g0, g1 = gadget(code, basis, 0), gadget(code, basis, 1)
    return GadgetSpec.from_maps(
        code,
        g0.x,
        basis=basis,
        f1=np.hstack([g0.f1, g1.f1]),
        incidence=block_diag(g0.incidence, g1.incidence),
        f0=np.hstack([g0.f0, g1.f0]),
        partial_0=block_diag(g0.partial_0, g1.partial_0),
        batch_x=np.stack([g0.x, g1.x]),
        readout_vectors=block_diag(np.ones(g0.f1.shape[1]), np.ones(g1.f1.shape[1])),
    )


@pytest.mark.parametrize("method", ["edge_expansion", "gauging"])
@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
def test_single_ppm_observables(basis, method) -> None:
    code = bb36()
    g = gadget(code, basis, method=method)
    for experiment, expected in [(basis, code.dimension + 1), (OTHER[basis], code.dimension - 1)]:
        for single_sector in (False, True):
            circuit = build_single_ppm_circuit(
                g,
                rounds=3,
                experiment_basis=experiment,
                single_sector=single_sector,
                noise_model=NOISE,
            )
            check_noiseless(circuit)
            assert circuit.num_observables == expected
            assert undetectable_logical_errors(circuit) == 0


@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
def test_batch_ppm_observables(basis) -> None:
    code = bb36()
    single = gadget(code, basis)
    assert build_batch_ppm_circuit(single, rounds=3) == build_single_ppm_circuit(single, rounds=3)
    g = batch_gadget(code, basis)
    for experiment, expected in [(basis, code.dimension + 2), (OTHER[basis], code.dimension - 2)]:
        circuit = build_batch_ppm_circuit(
            g, rounds=3, experiment_basis=experiment, single_sector=True, noise_model=NOISE
        )
        check_noiseless(circuit)
        assert circuit.num_observables == expected
        assert undetectable_logical_errors(circuit) == 0


@pytest.mark.parametrize("rounds", [1, 2, 3])
@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
def test_block_and_time_like_observables_read_the_measured_eigenvalue(basis, rounds) -> None:
    """Logical qubit 0 prepared in an eigenstate of the measured operator: the first
    block observable and the time-like one both read its eigenvalue."""
    code = bb36()
    g = gadget(code, basis)
    for state, value in zip("+-" if basis is Pauli.X else "01", (0, 1)):
        circuit = build_single_ppm_circuit(g, rounds=rounds, data_init=logical_state(code, state))
        values = observable_values(circuit)
        assert (values[:, 0] == value).all() and (values[:, code.dimension] == value).all()
        assert not values[:, 1 : code.dimension].any()


def test_detectors_and_coordinates() -> None:
    code = codes.SteaneCode()
    g = gadget(code, Pauli.X)
    circuit = build_single_ppm_circuit(g, rounds=1)
    lanes = {tuple(c) for c in circuit.get_final_qubit_coordinates().values()}
    assert {y for _, y in lanes} == {0, 1, 2, 3, 4, 5}  # data, aux, old/new X and Z checks
    # First-round detectors: the X checks of the |+> data, and the flux checks of the |0>
    # edges; the new vertex checks are random.
    first = {(int(c[1]), int(c[2])) for c in circuit.get_detector_coordinates().values()}
    assert {lane for lane, t in first if t == 0} == {2, 5}
    for single_sector, kept in [(False, {2, 3, 4, 5}), (True, {2, 3})]:
        circuit = build_single_ppm_circuit(g, rounds=3, single_sector=single_sector)
        assert {int(c[1]) for c in circuit.get_detector_coordinates().values()} == kept
    detached = build_single_ppm_circuit(g, rounds=2, destructive_measure_data=False)
    assert detached.num_observables == 0
    check_noiseless(detached)


@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
def test_joint_ppm_observables_and_values(basis) -> None:
    c_l, c_r = codes.SteaneCode(), codes.SteaneCode()
    g_l, g_r = gadget(c_l, basis), gadget(c_r, basis)
    bridge = build_bridge(g_l, g_r)
    for experiment, expected in [(basis, 3), (OTHER[basis], 1)]:
        for single_sector in (False, True):
            circuit, merged = build_joint_ppm_circuit(
                g_l,
                g_r,
                bridge,
                rounds=3,
                experiment_basis=experiment,
                single_sector=single_sector,
                noise_model=NOISE,
            )
            check_noiseless(circuit)
            assert circuit.num_observables == expected and merged.dimension == 1
            assert undetectable_logical_errors(circuit) == 0
    # The time-like observable reads the product of the two prepared eigenvalues.
    states = "+-" if basis is Pauli.X else "01"
    for s_l in states:
        for s_r in states:
            data_init = (logical_state(c_l, s_l), logical_state(c_r, s_r))
            circuit, _ = build_joint_ppm_circuit(g_l, g_r, bridge, rounds=2, data_init=data_init)
            values = observable_values(circuit)
            assert (values[:, 2] == states.index(s_l) ^ states.index(s_r)).all()


def test_intra_code_joint_ppm() -> None:
    code = bb36()
    g_l, g_r = gadget(code, Pauli.Z, 0), gadget(code, Pauli.Z, 1)
    circuit, merged = build_joint_ppm_circuit(
        g_l, g_r, build_bridge(g_l, g_r), rounds=3, noise_model=NOISE
    )
    check_noiseless(circuit)
    assert merged.dimension == code.dimension - 1 and circuit.num_observables == code.dimension + 1
    assert undetectable_logical_errors(circuit) == 0
