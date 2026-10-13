"""Lifted-product codes over R_l = F2[x]/(x^l + 1) and their algebraic surgery.

Modules, in dependency order:

* ``polynomial``, ``polymatrix``: R_l, matrices over R_l and their binary lifts;
* ``gf2poly``, ``ring_kernel``: the CRT decomposition of R_l and kernels over R_l;
* ``bitmask``: GF(2) linear algebra on bit-packed vectors; ``css``: CSS codes over R_l;
* ``canonical``: aligned information sets and canonical kernel generators u_i;
* ``tube``: tube factors D and selected factors A' of the auxiliary code LP(A', D);
* ``distance`` (+ ``exhaustive.cpp``): exhaustive minimum-weight logical search;
* ``examples.lifted_product``, ``examples.surgery_instances``: the paper's codes.
"""

__all__ = (
    "CPMCSSCode",
    "CyclicPolynomial",
    "PolynomialMatrix",
    "RingDecomposition",
    "gf2_rank",
    "kernel",
)

from .bitmask import gf2_rank
from .css import CPMCSSCode
from .polymatrix import PolynomialMatrix
from .polynomial import CyclicPolynomial
from .ring_kernel import RingDecomposition, kernel
