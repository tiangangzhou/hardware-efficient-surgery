"""LP(A, A^dagger) codes shared by the algebraic tests."""

from __future__ import annotations

from algebraic_qec.examples.surgery_instances import SURGERY_INSTANCES

# The 3 x 7 codes lp^{3,7}_d of Cain et al., arXiv:2603.28627, Appendix A, as
# name -> (l, seed, k); their 3 x 5 code lp^{3,5}_{20} is the data code of instance III.
CAIN_3X7_CODES = {
    "lp^{3,7}_{16}": (
        45,
        ((29, 21, 31, 15, 37, 25, 27), (13, 25, 19, 26, 11, 18, 29), (31, 2, 27, 32, 41, 41, 18)),
        744,
    ),
    "lp^{3,7}_{20}": (
        75,
        ((0, 71, 73, 68, 33, 50, 47), (38, 39, 60, 26, 18, 1, 23), (73, 6, 5, 42, 20, 22, 73)),
        1224,
    ),
    "lp^{3,7}_{24}": (
        91,
        ((57, 75, 42, 80, 7, 67, 27), (57, 73, 34, 12, 27, 50, 87), (21, 53, 70, 18, 1, 3, 18)),
        1480,
    ),
}

# Every LP code under test, name -> (l, seed, k): instances I-III, then the 3 x 7 codes.
LP_CODES = {
    f"l{c.ell}": (c.ell, c.seed, c.data_parameters[1]) for c in SURGERY_INSTANCES
} | CAIN_3X7_CODES
