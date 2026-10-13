"""CRT decomposition of R_l = F2[x]/(x^l + 1), l odd, and kernels of matrices over R_l.

SM "Extended LP construction and logical bases": x^l + 1 = prod_c g_c with distinct
irreducible g_c, so R_l ~= prod_c E_c with fields E_c = F2[x]/(g_c) of order 2^(deg g_c).
The CRT idempotents

    eps_c = M_c (M_c^-1 mod g_c) mod (x^l + 1),   M_c = (x^l + 1) / g_c,

satisfy eps_c mod g_d = delta_cd and sum_c eps_c = 1.  ``project`` is the map
R_l -> E_c, p -> p mod g_c, and ``lift`` its section E_c -> R_l, v -> eps_c v.

A matrix A over R_l acts componentwise, so (SM "Auxiliary LP code construction")

    ker A = sum_c eps_c ker(A_c),   A_c = A mod g_c.

``kernel`` returns eps_c v for every vector v of an RREF basis of every ker(A_c).  These
generate ker A as an R_l-module, and one from component c spans deg g_c binary dimensions
of the kernel of the lift.
"""

from __future__ import annotations

from algebraic_qec.gf2poly import deg, divmod_poly, factor_modulus, inverse, mulmod
from algebraic_qec.polymatrix import PolynomialMatrix
from algebraic_qec.polynomial import CyclicPolynomial


class RingDecomposition:
    """The factors g_c of x^l + 1 (increasing as ints) and the idempotents eps_c of R_l.

    Polynomials g_c, eps_c and component-field elements are ints (see gf2poly).
    """

    def __init__(self, period: int) -> None:
        self.period = period
        self.modulus = (1 << period) | 1
        self.factors = tuple(factor_modulus(period))
        cofactors = [divmod_poly(self.modulus, g)[0] for g in self.factors]
        self.idempotents = tuple(
            mulmod(m, inverse(m, g), self.modulus) for m, g in zip(cofactors, self.factors)
        )

    @property
    def component_degrees(self) -> list[int]:
        """deg g_c for every component c."""
        return [deg(g) for g in self.factors]

    def project(self, poly: CyclicPolynomial, c: int) -> int:
        """p mod g_c, an element of E_c."""
        return divmod_poly(poly.bits, self.factors[c])[1]

    def lift(self, value: int, c: int) -> CyclicPolynomial:
        """eps_c v: the element of R_l equal to v in E_c and zero in the other components."""
        return CyclicPolynomial.from_bits(
            self.period, mulmod(value, self.idempotents[c], self.modulus)
        )

    def inverse(self, poly: CyclicPolynomial) -> CyclicPolynomial:
        """p^-1 = sum_c eps_c (p mod g_c)^-1 for a unit p (nonzero in every component)."""
        result = CyclicPolynomial.zero(self.period)
        for c, g in enumerate(self.factors):
            result += self.lift(inverse(self.project(poly, c), g), c)
        return result

    def component_kernel(self, matrix: PolynomialMatrix, c: int) -> dict[int, list[int]]:
        """RREF kernel basis of A_c over E_c, as {free column j: vector}.

        Columns are scanned left to right; the vector for free column j is 1 at j, 0 at the
        other free columns, and solves A_c v = 0 on the pivot columns.
        """
        g, (nrows, ncols) = self.factors[c], matrix.shape
        rows = [[0] * ncols for _ in range(nrows)]
        for (i, j), poly in matrix.nonzero_entries():
            rows[i][j] = self.project(poly, c)
        pivots: list[int] = []
        for col in range(ncols):
            rank = len(pivots)
            r = next((i for i in range(rank, nrows) if rows[i][col]), None)
            if r is None:
                continue
            scale = inverse(rows[r][col], g)
            pivot = [mulmod(v, scale, g) for v in rows[r]]
            rows[r], rows[rank] = rows[rank], pivot
            for i, row in enumerate(rows):
                if i != rank and row[col]:
                    rows[i] = [v ^ mulmod(row[col], w, g) for v, w in zip(row, pivot)]
            pivots.append(col)
        basis = {}
        for free in (j for j in range(ncols) if j not in pivots):
            vector = [0] * ncols
            vector[free] = 1
            for rank, col in enumerate(pivots):
                vector[col] = rows[rank][free]  # -a = a in characteristic two
            basis[free] = vector
        return basis


def kernel(
    matrix: PolynomialMatrix, decomposition: RingDecomposition | None = None
) -> list[PolynomialMatrix]:
    """R_l-module generators eps_c v of ker(matrix), as column vectors, component by component."""
    dec = decomposition or RingDecomposition(matrix.period)
    generators = []
    for c in range(len(dec.factors)):
        for vector in dec.component_kernel(matrix, c).values():
            entries = {(j, 0): dec.lift(v, c) for j, v in enumerate(vector) if v}
            generators.append(PolynomialMatrix(matrix.period, (len(vector), 1), entries))
    return generators
