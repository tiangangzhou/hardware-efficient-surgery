"""Codes and helpers shared by the surgery tests."""

from __future__ import annotations

import numpy as np
import stim
import sympy
from qldpc import codes
from qldpc.codes.common import CSSCode
from qldpc.objects import Pauli, PauliXZ

# Generalised bicycle codes [[2l, k]] listed by Webster, Smith, Cohen arXiv:2511.15989
# App. A: H_X = [A | B], H_Z = [B^T | A^T], with A, B sums of powers of the l x l shift.
GB_CODES = {
    62: (31, (0, 6, 15), (0, 5, 7)),
    126: (63, (0, 4, 37), (0, 29, 49)),
    254: (127, (0, 32, 100), (0, 28, 49)),
    510: (255, (0, 39, 55), (0, 70, 127)),
}


def gb_code(n: int) -> CSSCode:
    ell, a, b = GB_CODES[n]
    shift = np.roll(np.eye(ell, dtype=int), -1, axis=0)
    A = sum(np.linalg.matrix_power(shift, e) for e in a) % 2
    B = sum(np.linalg.matrix_power(shift, e) for e in b) % 2
    return CSSCode(np.hstack([A, B]), np.hstack([B.T, A.T]), is_subsystem_code=False)


def bb36() -> CSSCode:
    """The [[36, 8]] bivariate bicycle code."""
    x, y = sympy.symbols("x y")
    return codes.BBCode({x: 3, y: 6}, x**3 + y + y**2, y**3 + x + x**2)


def logical(code: CSSCode, basis: PauliXZ, i: int = 0) -> np.ndarray:
    return np.asarray(code.get_logical_ops(basis)[i], np.uint8)


def logical_state(code: CSSCode, state: str, i: int = 0) -> str:
    """``data_init`` for logical qubit i in |state> and the others in |0> (for "0", "1")
    or |+> (for "+", "-"): X_i or Z_i applied to |0...0> or |+...+>."""
    if state in "0+":
        return state
    flip = logical(code, Pauli.X if state == "1" else Pauli.Z, i)
    return "".join(state if f else {"1": "0", "-": "+"}[state] for f in flip)


def observable_values(circuit: stim.Circuit, shots: int = 16) -> np.ndarray:
    """Measured values (not flips) of the observables, which all follow the last
    measurement, per shot."""
    samples = circuit.compile_sampler(seed=0).sample(shots).astype(np.uint8)
    values = np.zeros((shots, circuit.num_observables), np.uint8)
    for op in circuit.flattened():
        if op.name == "OBSERVABLE_INCLUDE":
            index = int(op.gate_args_copy()[0])
            for target in op.targets_copy():
                values[:, index] ^= samples[:, circuit.num_measurements + target.value]
    return values


def undetectable_logical_errors(circuit: stim.Circuit) -> int:
    """Number of error mechanisms that flip an observable but no detector."""
    return sum(
        1
        for e in circuit.detector_error_model().flattened()
        if e.type == "error"
        and any(t.is_logical_observable_id() for t in e.targets_copy())
        and not any(t.is_relative_detector_id() for t in e.targets_copy())
    )
