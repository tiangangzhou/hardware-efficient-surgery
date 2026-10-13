"""The paper's instances I-III (lifts 9, 15, 33) of algebraic fibre surgery.

SM "Three code examples".  Each instance has a data code LP(A, A^dagger) over R_l with a
3 x 5 monomial seed A, a tube factor D (m x n with n = m + 1, Eqs. (s-lp9-tube),
(s-lp15-tube), (s-lp33-seed-tube)) and a coupled coordinate q with ker D = R_l t, t_q = 1.
Fibre surgery on (i, j) = (3, 3) uses H_L = e_4^T, A' = [A; e_4^T] and, with s = 1 selection
row (SM Eqs. (tube) and (em-tube-wiring)),

    d1 = (A' (x) I_n ; I_5 (x) D),   d0 = (I_4 (x) D | A' (x) I_m),
    f1 = (I_5 (x) W ; 0),            f0 = (P_A (x) W | 0),   W = e_j e_q^T,  P_A = (I_3 | 0).

Then H_X f1 = f0 d1, d0 d1 = 0, and the merged code of Eq. (sm-css-surgery-checks) measures
the Z classes [(z (x) e_j, 0)], z in ker A'.

=========  ====  ====================  ====================
Instance   l     data code             merged code
=========  ====  ====================  ====================
I          9     [[306, 52, 8]]        [[585, 41, 8]]
II         15    [[510, 76, 10]]       [[975, 59, 10]]
III        33    [[1122, 148, <= 20]]  [[2442, 113, <= 20]]
=========  ====  ====================  ====================

Exponent tables: entry e is the monomial x^e and None is 0.
"""

from __future__ import annotations

from dataclasses import dataclass

from algebraic_qec.css import CPMCSSCode
from algebraic_qec.examples.lifted_product import lifted_product_code
from algebraic_qec.polymatrix import PolynomialMatrix
from algebraic_qec.polynomial import CyclicPolynomial

Exponents = tuple[tuple[int | None, ...], ...]


@dataclass(frozen=True)
class SurgeryInstance:
    """Seed A, tube D, coupled coordinate q and the code parameters quoted in the paper."""

    name: str
    ell: int
    seed: Exponents
    tube: Exponents
    couple_column: int
    data_parameters: tuple[int, int, int]
    merged_parameters: tuple[int, int, int]
    fibre: tuple[int, int] = (3, 3)  # (i, j): kernel direction u_i, cokernel coordinate e_j
    selection_coordinate: int = 4  # H_L = e_4^T removes the other aligned direction u_4

    @property
    def factor_a(self) -> PolynomialMatrix:
        return PolynomialMatrix.from_monomials(self.ell, self.seed)

    @property
    def factor_d(self) -> PolynomialMatrix:
        return PolynomialMatrix.from_monomials(self.ell, self.tube)

    def data_code(self) -> CPMCSSCode:
        return lifted_product_code(self.ell, self.seed)

    def selected_factor(self) -> PolynomialMatrix:
        """A' = [A; e_k^T] with k = selection_coordinate."""
        a = self.factor_a
        one = CyclicPolynomial.one(self.ell)
        return a.vstack(PolynomialMatrix(self.ell, (1, 5), {(0, self.selection_coordinate): one}))

    def surgery_maps(self) -> dict[str, PolynomialMatrix]:
        """The auxiliary boundaries d1, d0 and the attachment maps f1, f0 over R_l."""
        ell, a, a_sel, d = self.ell, self.factor_a, self.selected_factor(), self.factor_d
        (m, n), (rows, cols), q, j = d.shape, a.shape, self.couple_column, self.fibre[1]
        eye = lambda size: PolynomialMatrix.identity(ell, size)
        one = CyclicPolynomial.one(ell)
        d1 = a_sel.kron(eye(n)).vstack(eye(cols).kron(d))
        d0 = eye(rows + 1).kron(d).hstack(a_sel.kron(eye(m)))
        # Data qubit (k, j) of A_1 (x) B_0 and X check (k, j) of A_0 (x) B_0 (B_0 = R^cols)
        # meet auxiliary coordinate (k, q) of R^cols (x) R^n, resp. R^(rows+1) (x) R^n.
        num_data = cols * cols + rows * rows
        f1 = {(k * cols + j, k * n + q): one for k in range(cols)}
        f0 = {(k * cols + j, k * n + q): one for k in range(rows)}
        return dict(
            d1=d1,
            d0=d0,
            f1=PolynomialMatrix(ell, (num_data, cols * n), f1),
            f0=PolynomialMatrix(ell, (rows * cols, d1.shape[0]), f0),
        )

    def merged_code(self) -> CPMCSSCode:
        """H_X~ = (H_X f0; 0 d0), H_Z~ = (H_Z 0; f1^dagger d1^dagger), Eq. (sm-css-surgery-checks)."""
        data, maps = self.data_code(), self.surgery_maps()
        d1, d0, f1, f0 = maps["d1"], maps["d0"], maps["f1"], maps["f0"]
        zero = lambda rows, cols: PolynomialMatrix.zero(self.ell, rows, cols)
        hx = data.matrix_x.hstack(f0).vstack(zero(d0.shape[0], data.matrix_x.shape[1]).hstack(d0))
        hz = data.matrix_z.hstack(zero(data.matrix_z.shape[0], d1.shape[0]))
        return CPMCSSCode(hx, hz.vstack(f1.adjoint().hstack(d1.adjoint())))


SURGERY_INSTANCES: tuple[SurgeryInstance, ...] = (
    SurgeryInstance(
        name="I",
        ell=9,
        seed=((0, 0, 0, 0, 0), (0, 5, 2, 6, 1), (0, 8, 4, 5, 2)),
        tube=((0, 0, 0, None), (None, 2, 6, 0), (8, None, 5, 0)),
        couple_column=3,
        data_parameters=(306, 52, 8),
        merged_parameters=(585, 41, 8),
    ),
    SurgeryInstance(
        name="II",
        ell=15,
        seed=((0, 0, 0, 0, 0), (0, 8, 13, 7, 6), (0, 14, 6, 4, 13)),
        tube=((0, 0, 0, None), (None, 9, 10, 0), (11, None, 4, 0)),
        couple_column=0,
        data_parameters=(510, 76, 10),
        merged_parameters=(975, 59, 10),
    ),
    SurgeryInstance(
        name="III",
        ell=33,
        seed=((0, 0, 0, 0, 0), (0, 14, 19, 11, 26), (0, 13, 2, 15, 21)),
        tube=(
            (27, 19, 13, 0, None),
            (None, 22, 8, 13, 20),
            (8, None, 29, 22, 20),
            (19, 29, None, 12, 20),
        ),
        couple_column=2,
        data_parameters=(1122, 148, 20),  # d <= 20: an upper bound
        merged_parameters=(2442, 113, 20),  # d <= 20: an upper bound (QDistRnd)
    ),
)


def instance(ell: int) -> SurgeryInstance:
    """The instance with lift ell."""
    return next(case for case in SURGERY_INSTANCES if case.ell == ell)
