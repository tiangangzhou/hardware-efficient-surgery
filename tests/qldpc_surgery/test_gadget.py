"""The mapping cone of the Graph-IGND and Graph-WY gadgets (``hmatrix/gadget.py``)."""

from __future__ import annotations

import numpy as np
import pytest
from qldpc import codes
from qldpc.codes.common import CSSCode
from qldpc.objects import Pauli

from qldpc_surgery import build_gadget, cheeger_constant
from qldpc_surgery.hmatrix.gadget import GadgetSpec, basis_matrices, gf2_rank
from tests.qldpc_surgery.fixtures import bb36, gb_code, logical


def merged_code(g: GadgetSpec) -> CSSCode:
    return CSSCode(g.HX_merged.astype(int), g.HZ_merged.astype(int), is_subsystem_code=False)


@pytest.mark.parametrize("method", ["edge_expansion", "gauging"])
@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
@pytest.mark.parametrize("make_code", [codes.SteaneCode, bb36, lambda: gb_code(62)])
def test_gadget_is_a_mapping_cone_measuring_one_logical(make_code, basis, method) -> None:
    code = make_code()
    x = logical(code, basis)
    g = build_gadget(code, x, basis=basis, method=method, validate=True)
    H_meas, H_comp = basis_matrices(code, basis)
    d1, d0 = g.incidence.astype(int), g.partial_0.astype(int)
    assert not ((H_comp @ g.f1 + g.f0 @ d1) % 2).any()  # chain map H f1 = f0 d1
    assert not (d0 @ d1 % 2).any()  # d0 d1 = 0
    assert gf2_rank(d1) == d1.shape[1] - 1  # connected graph: ker d1 = {0, 1}
    # The new checks multiply to the measured logical, so it is the only one measured
    # and the merged code keeps the other k - 1 logical qubits.
    product = g.HX_merged if basis is Pauli.X else g.HZ_merged
    product = product[len(H_meas) :].sum(axis=0) % 2
    assert np.array_equal(product, np.r_[x, np.zeros(len(d1), np.uint8)])
    assert merged_code(g).dimension == code.dimension - 1
    if method == "edge_expansion":
        assert cheeger_constant(g.incidence) >= 1


@pytest.mark.parametrize("method", ["edge_expansion", "gauging"])
def test_z_measurement_is_the_x_measurement_of_the_dual_code(method) -> None:
    code = bb36()
    dual = CSSCode(code.matrix_z, code.matrix_x, is_subsystem_code=False)
    z = logical(code, Pauli.Z)
    g = build_gadget(code, z, basis=Pauli.Z, method=method)
    h = build_gadget(dual, z, basis=Pauli.X, method=method)
    for field in ("f1", "incidence", "f0", "partial_0"):
        assert np.array_equal(getattr(g, field), getattr(h, field))
    assert np.array_equal(g.HX_merged, h.HZ_merged) and np.array_equal(g.HZ_merged, h.HX_merged)


def test_from_maps_rejects_invalid_maps() -> None:
    code = codes.SteaneCode()
    x = logical(code, Pauli.X)
    g = build_gadget(code, x, basis=Pauli.X)
    maps = dict(basis=Pauli.X, f1=g.f1, incidence=g.incidence, f0=g.f0, partial_0=g.partial_0)
    GadgetSpec.from_maps(code, x, **maps)
    f0 = g.f0.copy()
    f0[0, 0] ^= 1  # breaks H_Z f1 = f0 d1
    with pytest.raises(ValueError, match="commute"):
        GadgetSpec.from_maps(code, x, **(maps | {"f0": f0}))
    with pytest.raises(ValueError, match="ker d1"):
        readout = np.eye(1, g.f1.shape[1], dtype=np.uint8)
        GadgetSpec.from_maps(code, x, **maps, batch_x=x[None], readout_vectors=readout)


def test_build_gadget_rejects_a_non_logical_and_an_unknown_method() -> None:
    code = codes.SteaneCode()
    with pytest.raises(ValueError, match="commute"):
        build_gadget(code, np.eye(7, dtype=np.uint8)[0], basis=Pauli.X)
    with pytest.raises(ValueError, match="unknown construction"):
        build_gadget(code, logical(code, Pauli.X), basis=Pauli.X, method="lattice")
