"""Stim circuit of a surgery experiment on a merged CSS code.

Protocol (SM, "Simulation methods and decoder settings: Circuit construction"; Cain
et al. arXiv:2603.28627 App. D). The first ``n_data`` qubits of the merged code are
data qubits, the others auxiliary (gadget edges, then adapter qubits); the new
checks measure ``basis``-type logicals.

1. Data qubits are prepared in ``experiment_basis`` (or per ``data_init``), auxiliary
   qubits in the other basis than ``basis`` (|0> for an X-type measurement).
2. All merged checks are measured for ``rounds`` rounds (qldpc ``EdgeColoring``).
3. The auxiliary qubits are measured in the same basis as they were prepared
   (detaching them), then the data qubits in ``experiment_basis``.

Detectors. After a product-state preparation a check has a deterministic outcome iff
all its qubits are prepared in the check's own Pauli basis; the same checks are the
ones the final single-qubit readouts reconstruct. These checks get first-round and
final detectors; every later round compares consecutive outcomes.

Observables. (i) A basis of the ``experiment_basis`` logicals that survive the
surgery, read out from the final measurements. Each data representative w is
completed on the auxiliary qubits to a merged-code logical w + c (a frame
correction); c exists iff w commutes with every logical that the merged code
measures. (ii) If ``experiment_basis`` equals ``basis``, one time-like observable per
readout: the product of the first-round outcomes of a set of new checks, which
equals a measured logical.
"""

from __future__ import annotations

import numpy as np
import stim
from qldpc.circuits.bookkeeping import MeasurementRecord, QubitIDs
from qldpc.circuits.memory.syndrome_measurement import EdgeColoring
from qldpc.circuits.noise_model import NoiseModel
from qldpc.codes.common import CSSCode
from qldpc.objects import Pauli, PauliXZ

from ..hmatrix.gadget import GF2, GadgetSpec, gf2_rank

# Coordinate lanes (y) of the qubits; x is the position within the lane.
DATA, AUX, X_CHECK, NEW_X_CHECK, Z_CHECK, NEW_Z_CHECK, ADAPTER, ADAPTER_CHECK = range(8)


def commuting_logicals(logical_ops: np.ndarray, functionals: np.ndarray) -> np.ndarray:
    """Basis of the span of ``logical_ops`` (rows) annihilated by every functional.

    For a single functional a with a_i = a . logical_i, the rows with a_i = 0 are kept
    and every other row is added to the first row p with a_p = 1 (p itself dropped).
    Several functionals are handled by a null space.
    """
    logical_ops = np.asarray(logical_ops, np.uint8)
    functionals = np.asarray(functionals, np.uint8)
    if functionals.ndim == 1 or len(functionals) <= 1:
        a = logical_ops @ functionals.reshape(-1) % 2
        anticommuting = np.flatnonzero(a)
        if not len(anticommuting):
            return logical_ops.copy()
        p = anticommuting[0]
        rows = [*logical_ops[a == 0], *(logical_ops[i] ^ logical_ops[p] for i in anticommuting[1:])]
        return np.array(rows, np.uint8).reshape(-1, logical_ops.shape[1])
    combinations = np.asarray(GF2(logical_ops @ functionals.T % 2).T.null_space())
    return np.asarray(GF2(combinations) @ GF2(logical_ops), np.uint8).reshape(
        -1, logical_ops.shape[1]
    )


def deformable_logicals(
    logical_ops: np.ndarray,
    functionals: np.ndarray,
    merged_code: CSSCode,
    experiment_basis: PauliXZ,
    n_data: int,
) -> np.ndarray:
    """``commuting_logicals`` restricted to the data logicals w that have a frame
    correction on the auxiliary qubits.

    w is completable iff its syndrome under the opposite-type merged checks M = [M_d | M_a]
    lies in the column space of M_a, i.e. l M_d w = 0 for all l with l M_a = 0: iff w
    commutes with all of f1(ker d1). A shared auxiliary code may measure logicals
    beyond ``batch_x`` (residual directions), which this removes as well.
    """
    M = _opposite_checks(merged_code, experiment_basis)
    left_null = np.asarray(GF2(M[:, n_data:].T).null_space())
    if not len(left_null):
        return commuting_logicals(logical_ops, functionals)
    constraints = left_null @ M[:, :n_data] % 2
    return commuting_logicals(logical_ops, np.vstack([np.atleast_2d(functionals), constraints]))


def _opposite_checks(merged_code: CSSCode, experiment_basis: PauliXZ) -> np.ndarray:
    matrix = merged_code.matrix_z if experiment_basis is Pauli.X else merged_code.matrix_x
    return np.asarray(matrix, np.uint8)


def _gf2_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The solution of A c = b over GF(2) with all free variables zero; raises if b is not in
    the image (a block logical without frame correction)."""
    reduced = np.asarray(GF2(np.hstack([A, b.reshape(-1, 1)])).row_reduce())
    c = np.zeros(A.shape[1], np.uint8)
    for row in reduced:
        if (pivots := np.flatnonzero(row[:-1])).size:
            c[pivots[0]] = row[-1]
        elif row[-1]:
            raise ValueError("a block logical has no frame correction on the auxiliary qubits")
    return c


def require_single_readout(gadget: GadgetSpec) -> None:
    """Raise unless the product of all new checks of ``gadget`` is its only readout and equals
    the logical on ``gadget.x``: ker d1 = {0, 1} (a connected graph) and f1 1 = x."""
    ones = np.ones(gadget.f1.shape[1], np.uint8)
    incidence = gadget.incidence.astype(np.int_)
    if (
        gf2_rank(incidence) != len(ones) - 1
        or (incidence @ ones % 2).any()
        or not np.array_equal(gadget.f1.astype(np.int_) @ ones % 2, gadget.x)
    ):
        raise ValueError(
            "the new checks must read out gadget.x alone (ker d1 = {0, 1}); "
            "use build_batch_ppm_circuit for gadgets with readout vectors"
        )


def _frame_corrected_support(
    merged_code: CSSCode, experiment_basis: PauliXZ, w: np.ndarray, n_data: int
) -> list[int]:
    """Qubits of the merged-code logical w + c, with c on the auxiliary qubits."""
    M = _opposite_checks(merged_code, experiment_basis)
    if not len(M):
        return np.flatnonzero(w).tolist()
    c = _gf2_solve(M[:, n_data:], M[:, :n_data] @ w % 2)
    return [*np.flatnonzero(w).tolist(), *(n_data + np.flatnonzero(c)).tolist()]


def _check_lanes(
    qubit_ids: QubitIDs, basis: PauliXZ, n_original: tuple[int, int], n_adapter_checks: int
) -> dict[int, tuple[int, int]]:
    """check id -> (lane, x): original checks, new checks, and adapter cycle checks
    (the last ``n_adapter_checks`` checks of the type complementary to ``basis``)."""
    lanes = {}
    for checks, m, original, new, comp in [
        (qubit_ids.checks_x, n_original[0], X_CHECK, NEW_X_CHECK, basis is Pauli.Z),
        (qubit_ids.checks_z, n_original[1], Z_CHECK, NEW_Z_CHECK, basis is Pauli.X),
    ]:
        end = len(checks) - (n_adapter_checks if comp else 0)
        for i, cid in enumerate(checks):
            lanes[cid] = (
                (original, i) if i < m else (new, i - m) if i < end else (ADAPTER_CHECK, i - end)
            )
    return lanes


def _deterministic_checks(
    merged_code: CSSCode,
    qubit_ids: QubitIDs,
    *,
    basis: PauliXZ,
    experiment_basis: PauliXZ,
    n_data: int,
) -> set[int]:
    """Nonzero checks whose qubits are all prepared (and read out) in their own basis."""
    x_basis = np.full(merged_code.num_qudits, basis is Pauli.Z)  # auxiliary qubits
    x_basis[:n_data] = experiment_basis is Pauli.X
    reliable = set()
    for H, checks, own in [
        (merged_code.matrix_x, qubit_ids.checks_x, x_basis),
        (merged_code.matrix_z, qubit_ids.checks_z, ~x_basis),
    ]:
        H = np.asarray(H, np.uint8)
        ok = H.any(axis=1) & ~(H[:, ~own].any(axis=1))
        reliable.update(np.asarray(checks)[ok].tolist())
    return reliable


def _initialization(
    qubit_ids: QubitIDs,
    n_data: int,
    basis: PauliXZ,
    experiment_basis: PauliXZ,
    data_init: str | None,
) -> stim.Circuit:
    """Data in "0", "1", "+" or "-" per character of ``data_init`` (one character is
    broadcast; default: the +1 eigenstate of ``experiment_basis``), auxiliary qubits in
    the +1 eigenstate of the basis other than ``basis``."""
    states = data_init or ("+" if experiment_basis is Pauli.X else "0")
    states = states * n_data if len(states) == 1 else states
    if len(states) != n_data or not set(states) <= set("01+-"):
        raise ValueError(f"data_init needs one of '0', '1', '+', '-' for each of {n_data} qubits")
    circuit = stim.Circuit()
    for gate, chars in [("R", "01"), ("RX", "+-"), ("X", "1"), ("Z", "-")]:
        if targets := [q for q, s in zip(qubit_ids.data, states) if s in chars]:
            circuit.append(gate, targets)
    if auxiliary := qubit_ids.data[n_data:]:
        circuit.append("R" if basis is Pauli.X else "RX", auxiliary)
    return circuit


def surgery_circuit(
    merged_code: CSSCode,
    *,
    basis: PauliXZ,
    experiment_basis: PauliXZ,
    n_data: int,
    n_gadget_qubits: int,
    n_original_checks: tuple[int, int],
    n_adapter_checks: int,
    block_logicals: np.ndarray,
    readouts: list[np.ndarray],
    rounds: int,
    data_init: str | None,
    destructive_measure_data: bool,
    single_sector: bool,
    noise_model: NoiseModel | None,
) -> stim.Circuit:
    """The experiment of the module docstring.

    ``n_gadget_qubits`` auxiliary qubits follow the data (the rest are adapter qubits),
    ``n_original_checks`` = (X, Z) data checks come first in each check type, and
    ``readouts[j]`` indexes the new ``basis``-type checks whose first-round product is
    the j-th time-like observable. ``data_init`` should only use eigenstates of
    ``experiment_basis``, as the detectors assume. ``single_sector`` keeps only the
    detectors of ``experiment_basis``-type checks (SM: "Detectors are added only in the
    stabilizer basis corresponding to the experiment"): the observables are flipped
    only by errors of the other Pauli type, which only these checks detect. Without
    ``destructive_measure_data`` the data stay encoded and there are no final
    detectors and no observables.
    """
    qubit_ids = QubitIDs.from_code(merged_code)
    data = qubit_ids.data[:n_data]
    lanes = _check_lanes(qubit_ids, basis, n_original_checks, n_adapter_checks)
    reliable = _deterministic_checks(
        merged_code, qubit_ids, basis=basis, experiment_basis=experiment_basis, n_data=n_data
    )
    kept = set(qubit_ids.checks_x if experiment_basis is Pauli.X else qubit_ids.checks_z)
    detected = [cid for cid in qubit_ids.check if not single_sector or cid in kept]

    circuit = stim.Circuit()
    for i, q in enumerate(qubit_ids.data):
        if i < n_data:
            circuit.append("QUBIT_COORDS", q, (i, DATA))
        elif i < n_data + n_gadget_qubits:
            circuit.append("QUBIT_COORDS", q, (i - n_data, AUX))
        else:
            circuit.append("QUBIT_COORDS", q, (i - n_data - n_gadget_qubits, ADAPTER))
    for cid, (lane, x) in sorted(lanes.items(), key=lambda item: item[1]):
        circuit.append("QUBIT_COORDS", cid, (x, lane))
    circuit += _initialization(qubit_ids, n_data, basis, experiment_basis, data_init)

    one_round, round_record = EdgeColoring().get_circuit(merged_code, qubit_ids)
    record = MeasurementRecord()
    circuit += one_round
    record.append(round_record)
    for cid in detected:
        if cid in reliable:
            circuit.append("DETECTOR", [record.get_target_rec(cid)], (*lanes[cid][::-1], 0))
    if rounds > 1:
        repeat = one_round.copy()
        record.append(round_record)
        repeat.append("SHIFT_COORDS", [], (0, 0, 1))
        for cid in detected:
            targets = [record.get_target_rec(cid, -1), record.get_target_rec(cid, -2)]
            repeat.append("DETECTOR", targets, (*lanes[cid][::-1], 0))
        circuit.append(stim.CircuitRepeatBlock(rounds - 1, repeat))
        record.append(round_record, repeat=rounds - 2)

    auxiliary = qubit_ids.data[n_data:]
    circuit.append("M" if basis is Pauli.X else "MX", auxiliary)
    record.append({q: i for i, q in enumerate(auxiliary)})
    if destructive_measure_data:
        circuit.append("SHIFT_COORDS", [], (0, 0, 1))
        circuit.append("MX" if experiment_basis is Pauli.X else "M", data)
        record.append({q: i for i, q in enumerate(data)})
        final = reliable & kept if single_sector else reliable
        sectors = [
            (merged_code.matrix_x, qubit_ids.checks_x),
            (merged_code.matrix_z, qubit_ids.checks_z),
        ]
        for H, checks in sectors if basis is Pauli.X else sectors[::-1]:
            for row, cid in zip(np.asarray(H, np.uint8), checks):
                if cid in final:
                    targets = [
                        record.get_target_rec(qubit_ids.data[q]) for q in np.flatnonzero(row)
                    ]
                    targets.append(record.get_target_rec(cid, -1))
                    circuit.append("DETECTOR", targets, (*lanes[cid][::-1], 0))

        observables = [
            [record.get_target_rec(qubit_ids.data[q]) for q in support]
            for support in (
                _frame_corrected_support(merged_code, experiment_basis, w, n_data)
                for w in block_logicals
            )
        ]
        if experiment_basis is basis:
            if basis is Pauli.X:
                new_checks = np.asarray(qubit_ids.checks_x)[n_original_checks[0] :]
            else:
                new_checks = np.asarray(qubit_ids.checks_z)[n_original_checks[1] :]
            observables += [
                [record.get_target_rec(cid, 0) for cid in new_checks[readout]]
                for readout in readouts
            ]
        for index, targets in enumerate(observables):
            circuit.append("OBSERVABLE_INCLUDE", targets, index)

    return circuit if noise_model is None else noise_model.noisy_circuit(circuit)


def check_noiseless(circuit: stim.Circuit, shots: int = 16) -> None:
    """Raise if any detector or observable flips in the noiseless circuit."""
    detectors, observables = (
        circuit.without_noise()
        .compile_detector_sampler(seed=1)
        .sample(shots, separate_observables=True)
    )
    if detectors.any() or observables.any():
        raise AssertionError("noiseless circuit flips a detector or an observable")
