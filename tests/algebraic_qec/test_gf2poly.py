"""F2[x] arithmetic and the factorization of x^l + 1."""

import pytest

from algebraic_qec.gf2poly import divmod_poly, factor_modulus, inverse, mul, mulmod

# SM Eqs. (s-lp9-crt-factors), (s-lp15-factors) and Table "Selected CRT unions for l = 33".
SM_FACTORS = {
    9: [[0, 1], [0, 1, 2], [0, 3, 6]],
    15: [[0, 1], [0, 1, 2], [0, 1, 4], [0, 3, 4], [0, 1, 2, 3, 4]],
    33: [[0, 1], [0, 1, 2], [0, 3, 5, 7, 10], [0, 1, 5, 9, 10], list(range(11))],
}


def test_arithmetic():
    assert mul(0b11, 0b11) == 0b101  # (1 + x)^2 = 1 + x^2
    q, r = divmod_poly(0b1101011, 0b1011)
    assert mul(q, 0b1011) ^ r == 0b1101011 and r < 0b1000
    assert mulmod(0b10, inverse(0b10, 0b1011), 0b1011) == 1
    with pytest.raises(ZeroDivisionError):
        inverse(0b11, 0b1111)  # 1 + x divides 1 + x + x^2 + x^3


@pytest.mark.parametrize("ell", SM_FACTORS)
def test_factors_match_the_sm(ell):
    assert [[e for e in range(g.bit_length()) if g >> e & 1] for g in factor_modulus(ell)] == (
        SM_FACTORS[ell]
    )


@pytest.mark.parametrize("ell", range(1, 46, 2))
def test_factors_are_the_irreducible_factorization(ell):
    factors = factor_modulus(ell)
    product = 1
    for g in factors:
        product = mul(product, g)
    assert product == (1 << ell) | 1 and factors == sorted(factors)
    # x^l + 1 has one irreducible factor of degree |C| per 2-cyclotomic coset C of Z_l, so a
    # factorization with as many nonconstant factors of these degrees is the irreducible one.
    cosets = {frozenset(a * 2**k % ell for k in range(ell)) for a in range(ell)}
    assert sorted(g.bit_length() - 1 for g in factors) == sorted(map(len, cosets))


def test_even_period_is_not_squarefree():
    with pytest.raises(ValueError, match="squarefree"):
        factor_modulus(4)
