"""Polynomials over F2 stored as Python ints: bit i is the coefficient of x^i.

These are the representatives used for the CRT decomposition of R_l = F2[x]/(x^l + 1)
(SM "Extended LP construction and logical bases"): ``factor_modulus`` splits x^l + 1 into
its irreducible factors g_c, and the other functions do arithmetic modulo x^l + 1 and in the
component fields E_c = F2[x]/(g_c).
"""

from __future__ import annotations


def deg(p: int) -> int:
    """Degree of p, with deg(0) = -1."""
    return p.bit_length() - 1


def mul(a: int, b: int) -> int:
    """a * b in F2[x]."""
    product = 0
    while b:
        if b & 1:
            product ^= a
        a <<= 1
        b >>= 1
    return product


def divmod_poly(a: int, b: int) -> tuple[int, int]:
    """Quotient and remainder of a divided by b in F2[x]."""
    if not b:
        raise ZeroDivisionError("polynomial division by zero")
    quotient = 0
    while deg(a) >= deg(b):
        shift = deg(a) - deg(b)
        quotient |= 1 << shift
        a ^= b << shift
    return quotient, a


def mulmod(a: int, b: int, m: int) -> int:
    """a * b mod m."""
    return divmod_poly(mul(a, b), m)[1]


def inverse(a: int, m: int) -> int:
    """a^-1 mod m by the extended Euclidean algorithm (s_i a = r_i mod m throughout)."""
    r0, r1, s0, s1 = m, divmod_poly(a, m)[1], 0, 1
    while r1:
        q, r = divmod_poly(r0, r1)
        r0, r1, s0, s1 = r1, r, s1, s0 ^ mul(q, s1)
    if r0 != 1:
        raise ZeroDivisionError(f"{a:#x} is not invertible modulo {m:#x}")
    return divmod_poly(s0, m)[1]


def factor_modulus(period: int) -> list[int]:
    """The irreducible factors of x^period + 1, period odd, in increasing order as ints.

    For odd l, x^l + 1 is coprime to its derivative x^(l-1), hence squarefree, and R_l is a
    product of fields.  Trial division by g = 1 + x, ... in increasing order: a g that divides
    the remaining part r has no smaller factor left in r, so g is irreducible; once
    2 deg(g) > deg(r), r has no factor of degree <= deg(r) / 2 and is irreducible (or 1).
    The cost grows like 2^(degree of the second-largest factor), fine for the lifts here.
    """
    if period % 2 == 0:
        raise ValueError(f"x^{period} + 1 is not squarefree: the period must be odd")
    rest, factors, g = (1 << period) | 1, [], 0b11
    while deg(rest) >= 2 * deg(g):
        quotient, remainder = divmod_poly(rest, g)
        if not remainder:
            factors.append(g)
            rest = quotient
        g += 2  # x does not divide x^l + 1: only odd candidates
    return factors + [rest] if rest != 1 else factors
