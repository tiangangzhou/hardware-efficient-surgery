"""Instances I-III: chain-map identities and the code parameters quoted in the paper."""

import pytest

from algebraic_qec import PolynomialMatrix
from algebraic_qec.examples.surgery_instances import SURGERY_INSTANCES, instance


@pytest.mark.parametrize("case", SURGERY_INSTANCES, ids=lambda c: f"l{c.ell}")
def test_attachment_is_a_chain_map(case):
    maps, hx = case.surgery_maps(), case.data_code().matrix_x
    d1, d0, f1, f0 = maps["d1"], maps["d0"], maps["f1"], maps["f0"]
    assert hx @ f1 == f0 @ d1
    assert d0 @ d1 == PolynomialMatrix.zero(case.ell, d0.shape[0], d1.shape[1])


@pytest.mark.parametrize("case", SURGERY_INSTANCES, ids=lambda c: f"l{c.ell}")
def test_code_parameters(case):
    data, merged = case.data_code(), case.merged_code()
    assert (data.n, data.k) == case.data_parameters[:2]
    assert (merged.n, merged.k) == case.merged_parameters[:2]
    assert merged.k == data.k - (case.ell + 2)  # l canonical and two residual readouts


def test_lookup_by_lift():
    assert instance(15).name == "II"
