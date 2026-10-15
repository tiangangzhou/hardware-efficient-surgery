"""LP(A, A^dagger) codes of instances I-III and Cain et al. (arXiv:2603.28627), checked
against qldpc."""

import numpy as np
import pytest

from algebraic_qec.examples.surgery_instances import lifted_product_code
from tests.algebraic_qec.fixtures import LP_CODES


@pytest.mark.parametrize("ell, seed, k", LP_CODES.values(), ids=LP_CODES)
def test_parameters_and_check_weights(ell, seed, k):
    code = lifted_product_code(ell, seed)
    rows, cols = len(seed), len(seed[0])
    assert (code.n, code.k) == (ell * (rows**2 + cols**2), k)
    assert code.k >= (cols - rows) ** 2 * ell
    for binary in (code.binary_matrix_x, code.binary_matrix_z):
        assert set(np.asarray(binary.sum(axis=1)).ravel()) == {rows + cols}


@pytest.mark.parametrize("ell, seed, k", LP_CODES.values(), ids=LP_CODES)
def test_lifts_match_qldpc(ell, seed, k):
    """qldpc lifts x^e to our P(-e) = P(e)^T, so its protograph uses the negated exponents."""
    from qldpc import abstract, codes

    group = abstract.CyclicGroup(ell)
    g = group.generators[0]
    protograph = [[abstract.Element(group, g ** (-e)) for e in row] for row in seed]
    theirs = codes.LPCode(abstract.Protograph(protograph))
    ours = lifted_product_code(ell, seed)
    assert np.array_equal(ours.binary_matrix_x.toarray(), np.asarray(theirs.matrix_x) % 2)
    assert np.array_equal(ours.binary_matrix_z.toarray(), np.asarray(theirs.matrix_z) % 2)
