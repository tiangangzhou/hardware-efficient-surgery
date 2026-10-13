"""Elements of the ring R_l = F2[x]/(x^l + 1) = F2[Z_l] (SM "Extended LP construction").

A polynomial is stored as the set of exponents (mod l) of its nonzero monomials.

Lift convention (SM "LP complex and binary checks"): x^s becomes the l x l circulant P(s)
with P(s)[a, a + s] = 1, indices mod l.  Then P(s) P(t) = P(s + t), so lifting preserves
products, and P(s)^T = P(-s), so the adjoint p(x) -> p(x^-1) lifts to the transpose.
The ``qldpc`` package lifts x^s to P(-s) = P(s)^T instead.
"""

from __future__ import annotations

from collections.abc import Iterable


class CyclicPolynomial:
    """An element of F2[x]/(x^period + 1)."""

    __slots__ = ("exponents", "period")

    def __init__(self, period: int, exponents: Iterable[int] = ()) -> None:
        """sum_e x^e with exponents reduced mod period; repeated terms cancel in pairs."""
        terms: set[int] = set()
        for e in exponents:
            terms ^= {e % period}
        self.period = period
        self.exponents = frozenset(terms)

    @classmethod
    def zero(cls, period: int) -> CyclicPolynomial:
        return cls(period)

    @classmethod
    def one(cls, period: int) -> CyclicPolynomial:
        return cls(period, [0])

    @classmethod
    def monomial(cls, period: int, exponent: int) -> CyclicPolynomial:
        return cls(period, [exponent])

    @classmethod
    def from_bits(cls, period: int, bits: int) -> CyclicPolynomial:
        """The residue mod x^period + 1 of an F2[x] polynomial stored as an int (bit e: x^e)."""
        return cls(period, (e for e in range(bits.bit_length()) if bits >> e & 1))

    @property
    def bits(self) -> int:
        """This element as an F2[x] polynomial of degree < period, stored as an int."""
        return sum(1 << e for e in self.exponents)

    @property
    def weight(self) -> int:
        """Number of monomials."""
        return len(self.exponents)

    def _check_period(self, other: CyclicPolynomial) -> None:
        if self.period != other.period:
            raise ValueError(f"period mismatch: {self.period} != {other.period}")

    def __add__(self, other: CyclicPolynomial) -> CyclicPolynomial:
        self._check_period(other)
        return CyclicPolynomial(self.period, self.exponents ^ other.exponents)

    def __mul__(self, other: CyclicPolynomial) -> CyclicPolynomial:
        self._check_period(other)
        return CyclicPolynomial(
            self.period, (a + b for a in self.exponents for b in other.exponents)
        )

    def adjoint(self) -> CyclicPolynomial:
        """p(x) -> p(x^-1); lifts to the transpose."""
        return CyclicPolynomial(self.period, (-e for e in self.exponents))

    def __bool__(self) -> bool:
        return bool(self.exponents)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CyclicPolynomial):
            return NotImplemented
        return self.period == other.period and self.exponents == other.exponents

    def __hash__(self) -> int:
        return hash((self.period, self.exponents))

    def __repr__(self) -> str:
        terms = " + ".join(f"x^{e}" for e in sorted(self.exponents)) or "0"
        return f"{terms} (mod x^{self.period} + 1)"
