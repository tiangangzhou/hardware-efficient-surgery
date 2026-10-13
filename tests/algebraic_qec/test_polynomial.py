"""The ring R_l = F2[x]/(x^l + 1)."""

from algebraic_qec import CyclicPolynomial as P


def test_exponents_reduce_mod_period_and_cancel_in_pairs():
    assert P(5, [0, 6, 8, -1]).exponents == {0, 1, 3, 4}
    assert P(5, [1, 2, 2, 3, 3, 3]).exponents == {1, 3}


def test_bits_round_trip_reduces_mod_x_l_plus_1():
    assert P.from_bits(5, 0b1100000).bits == 0b11  # x^5 + x^6 = 1 + x
    assert P.from_bits(7, P(7, [0, 2, 6]).bits) == P(7, [0, 2, 6])


def test_ring_operations():
    p, q = P(7, [0, 1]), P(7, [1, 3])
    assert p + q == P(7, [0, 3]) and p + p == P.zero(7)
    assert p * p == P(7, [0, 2])  # (1 + x)^2 = 1 + x^2 over F2
    assert P(3, [2]) * P(3, [2]) == P(3, [1])  # x^4 = x
    assert p * P.one(7) == p and not p * P.zero(7)


def test_adjoint_is_an_involutive_ring_automorphism():
    p, q = P(8, [1, 2]), P(8, [0, 3])
    assert P(5, [0, 1, 3]).adjoint() == P(5, [0, 4, 2])
    assert p.adjoint().adjoint() == p
    assert (p * q).adjoint() == p.adjoint() * q.adjoint()


def test_repr():
    assert repr(P(5, [2, 0])) == "x^0 + x^2 (mod x^5 + 1)"
