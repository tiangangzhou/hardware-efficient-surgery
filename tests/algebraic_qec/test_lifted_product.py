"""LP(A, A^dagger) codes, checked against Cain et al. (arXiv:2603.28627) and qldpc."""

import numpy as np

from algebraic_qec.examples.lifted_product import CAIN_LP_CODES

# name -> (l, n, k): n = l (n_A^2 + m_A^2); k from the binary ranks.
PARAMETERS = {
    "lp^{3,5}_{20}": (33, 1122, 148),
    "lp^{3,7}_{16}": (45, 2610, 744),
    "lp^{3,7}_{20}": (75, 4350, 1224),
    "lp^{3,7}_{24}": (91, 5278, 1480),
}


def test_parameters_and_check_weights():
    for spec in CAIN_LP_CODES:
        code = spec.build()
        assert (spec.period, code.n, code.k) == PARAMETERS[spec.name]
        rows, cols = len(spec.seed), len(spec.seed[0])
        assert code.k >= (cols - rows) ** 2 * spec.period
        for binary in (code.binary_matrix_x, code.binary_matrix_z):
            assert set(np.asarray(binary.sum(axis=1)).ravel()) == {rows + cols}


def test_lifts_match_qldpc():
    """qldpc lifts x^e to our P(-e) = P(e)^T, so its protograph uses the negated exponents."""
    from qldpc import abstract, codes

    for spec in CAIN_LP_CODES:
        group = abstract.CyclicGroup(spec.period)
        g = group.generators[0]
        seed = [[abstract.Element(group, g ** (-e)) for e in row] for row in spec.seed]
        theirs = codes.LPCode(abstract.Protograph(seed))
        ours = spec.build()
        assert np.array_equal(ours.binary_matrix_x.toarray(), np.asarray(theirs.matrix_x) % 2)
        assert np.array_equal(ours.binary_matrix_z.toarray(), np.asarray(theirs.matrix_z) % 2)
