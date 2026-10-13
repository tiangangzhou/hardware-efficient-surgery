"""Worked examples with l = 9 and l = 15 (SM "Three code examples", instances I and II).

For the [[306,52,8]] (l = 9) and [[510,76,10]] (l = 15) data codes this recomputes:

* data code: both binary check ranks (127, 217); the CRT factors g_c of x^l + 1 and their
  idempotents; the component ranks of A; the aligned information set J = {3, 4}; the
  packet vectors u_i^(c) and the glued generators u_3, u_4; the matrices T_l, T_l^{-1};
* fibre selection A' = [A; e_4^T]: ker A' = R u_3 + K_res of binary dimension l + 2;
* l = 15 packet selection H_{L,S}: singletons give 1, 2, 4, 4, 4 canonical readouts, only
  S = {0} keeps the residuals (removed by eps_0 e_1^T, eps_0 e_2^T), and S = {2, 3} gives
  g_S = 1+x+x^3+x^4+x^5+x^7+x^8 with eight readouts;
* tube: the deleted-q minor Delta (x+x^5+x^7, resp. x^4+x^9+x^10), the displayed t, and
  ker(lift D) as a classical [36, 9, 8], resp. [60, 15, 11] code (15 words of weight 11);
* adapter between fibres (3,3) and (4,4) of two blocks with offset 1: factor 9 x 10,
  boundaries 66 x 40 and 27 x 66, attachments 68 x 40 and 30 x 66; [[1206, 91]] with check
  ranks (501, 614), resp. [[2010, 133]] with (843, 1034); l(9*4 + 10*3) auxiliary qubits;
  readout of dimension l + 4 (l products Z_{33;a} Z_{44;a-1} and four residuals) that meets
  each canonical register trivially and each full data register in two dimensions; a
  common CRT selection S keeps sum_{c in S} deg g_c products.

The merged codes and their exact distances are checked by ``verify_instances.py``.

    uv run python scripts/codes/worked_examples.py   # ~5 s
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from common import column, e_S, g_S, kernel_space, readout_images, residuals

from algebraic_qec import CPMCSSCode, CyclicPolynomial, PolynomialMatrix, RingDecomposition
from algebraic_qec.bitmask import (
    Rowspace,
    logical_quotient_basis,
    nullspace_basis,
    orbit_masks,
    row_masks,
    set_bits,
    span_words,
)
from algebraic_qec.canonical import aligned_information_set, systematic_kernel_generators
from algebraic_qec.examples.surgery_instances import instance
from algebraic_qec.tube import component_ranks, maximal_minors, tube_kernel_generator


def b(*exponents: int) -> int:
    """A component-field element as a bitmask (bit e = coefficient of x^e)."""
    return sum(1 << e for e in exponents)


# Displayed SM values.  Polynomials are exponent lists, packets[i][c] = u_i^(c) mod g_c, and
# delta_t = Delta t with Delta the deleted-q minor of D.
SM = {
    9: dict(
        data_rank=127,
        factors=[[0, 1], [0, 1, 2], [0, 3, 6]],
        idempotents=[list(range(9)), [1, 2, 4, 5, 7, 8], [3, 6]],
        component_ranks=[1, 3, 3],
        packets={
            3: [[1, 0, 0, 1, 0], [1, b(1), b(1), 1, 0],
                [b(0, 3, 5), b(2, 5), b(2, 3), 1, 0]],
            4: [[1, 0, 0, 0, 1], [b(0, 1), b(0, 1), 1, 0, 1],
                [b(0, 2, 3, 4, 5), b(3, 5), b(2, 4), 0, 1]],
        },
        u3=[[2, 6, 8], [0, 3, 6, 8], [2, 3], [0], []],
        u4=[[0, 1, 3, 7, 8], [1, 2, 3, 4, 7, 8], [2, 4], [], [0]],
        T=["111111111", "011011011", "101101101", "000100100", "000010010",
           "000001001", "100000100", "010000010", "001000001"],
        T_inverse=["110100000", "101010000", "111001000", "110000100", "101000010",
                   "111000001", "110100100", "101010010", "111001001"],
        delta_t=[[2, 5, 6], [5, 6, 8], [2, 8], [1, 5, 7]],  # Delta = x+x^5+x^7
        tube_distance=8,
        adapter=(1206, 91, (501, 614), (2,)),
    ),
    15: dict(
        data_rank=217,
        factors=[[0, 1], [0, 1, 2], [0, 1, 4], [0, 3, 4], [0, 1, 2, 3, 4]],
        idempotents=[list(range(15)), [1, 2, 4, 5, 7, 8, 10, 11, 13, 14],
                     [1, 2, 3, 4, 6, 8, 9, 12], [3, 6, 7, 9, 11, 12, 13, 14],
                     [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14]],
        component_ranks=[1, 3, 3, 3, 3],
        packets={
            3: [[1, 0, 0, 1, 0], [b(0, 1), b(1), 0, 1, 0], [b(0, 3), 0, b(3), 1, 0],
                [b(1, 2), b(0, 1, 2), 0, 1, 0], [b(0, 2), b(0, 1, 2), b(0, 1), 1, 0]],
            4: [[1, 0, 0, 0, 1], [b(1), b(1), 1, 0, 1], [b(2, 3), b(0, 2), b(3), 0, 1],
                [b(0, 1, 2), b(0, 1, 2, 3), b(0, 3), 0, 1], [b(3), 1, b(3), 0, 1]],
        },
        u3=[[2, 3, 4, 7, 9, 11, 14], [0, 1, 2, 3, 10, 11, 12, 14], [1, 4, 7, 9, 10, 12], [0], []],
        u4=[[1, 2, 4, 7, 8, 11, 14], [3, 5, 7, 10, 11, 14], [0, 1, 2, 3, 4, 5, 8, 10], [], [0]],
        T=["111111111111111", "011011011011011", "101101101101101", "011110101100100",
           "001111010110010", "000111101011001", "100011110101100", "000100110101111",
           "100010011010111", "110001001101011", "111000100110101", "011110111101111",
           "101111011110111", "110111101111011", "111011110111101"],
        T_inverse=["110100010001000", "101010001000100", "111001000100010", "110000100010001",
                   "101110010011111", "111011011011000", "110001111110100", "101110111100010",
                   "111101001110001", "110010110101111", "101111001011000", "111011110110100",
                   "110111111000010", "101101101100001", "111100100111111"],
        delta_t=[[4, 9, 10], [4, 10, 11], [9, 11], [5, 6, 13]],  # Delta = x^4+x^9+x^10
        tube_distance=11,
        adapter=(2010, 133, (843, 1034), (2, 3)),
    ),
}  # fmt: skip


def check_data_code(case, dec: RingDecomposition, sm: dict) -> list[PolynomialMatrix]:
    """Data code, CRT decomposition, canonical generators, basis change and ker A'."""
    ell, a, data = case.ell, case.factor_a, case.data_code()
    ranks = [Rowspace(row_masks(h)).rank for h in (data.binary_matrix_x, data.binary_matrix_z)]
    assert ranks == [sm["data_rank"]] * 2 and data.k == data.n - 2 * sm["data_rank"]
    assert [set_bits(f) for f in dec.factors] == sm["factors"]
    assert [set_bits(e) for e in dec.idempotents] == sm["idempotents"]
    assert component_ranks(a, dec) == sm["component_ranks"]
    assert aligned_information_set(a, 2, dec) == (3, 4)
    generators = systematic_kernel_generators(a, (3, 4), dec)
    for i, u in zip((3, 4), generators):
        assert [sorted(u[j, 0].exponents) for j in range(5)] == sm[f"u{i}"]
        for c in range(len(dec.factors)):
            assert [dec.project(u[j, 0], c) for j in range(5)] == sm["packets"][i][c]

    # Eq. (s-basis-change-rule): T_l[(c, b), a] = [x^a](eps_c x^b), T_l^-1[a, (c, b)] =
    # [x^b](x^a mod g_c), rows (c, b) ordered by c, then b.
    x = lambda a: CyclicPolynomial.monomial(ell, a)
    labels = [(c, b) for c, deg in enumerate(dec.component_degrees) for b in range(deg)]
    forward = [[int(a in (dec.lift(1, c) * x(b)).exponents) for a in range(ell)] for c, b in labels]
    inverse = [[dec.project(x(a), c) >> b & 1 for c, b in labels] for a in range(ell)]
    assert ["".join(map(str, row)) for row in forward] == sm["T"]
    assert ["".join(map(str, row)) for row in inverse] == sm["T_inverse"]
    assert ((np.array(forward) @ np.array(inverse)) % 2 == np.eye(ell)).all()

    # Fibre selection A' = [A; e_4^T]: ker A' = R u_3 + K_res.
    selected = case.selected_factor()
    kernel = kernel_space(selected)
    assert kernel.rank == ell + 2
    for vector in [generators[0], *residuals(ell)]:
        assert selected @ vector == PolynomialMatrix.zero(ell, 4, 1)
        assert all(kernel.contains(word) for word in orbit_masks(vector))
    return generators


def check_packet_selection(case, dec: RingDecomposition) -> None:
    """ker(A; H_{L,S}) at l = 15 for the singletons, S = {0} with extra rows, S = {2, 3}."""
    ell = case.ell
    one, eps0 = CyclicPolynomial.one(ell), CyclicPolynomial(ell, range(ell))

    def dimension(g: CyclicPolynomial, extra: tuple[int, ...] = ()) -> int:
        """H_{L,S} = (e_4^T; g e_0^T; ...; g e_3^T) plus rows eps_0 e_r^T for r in extra."""
        rows = {(0, 4): one} | {(1 + r, r): g for r in range(4)}
        rows |= {(5 + k, r): eps0 for k, r in enumerate(extra)}
        selector = PolynomialMatrix(ell, (5 + len(extra), 5), rows)
        return kernel_space(case.factor_a.vstack(selector)).rank

    degrees = dec.component_degrees
    assert [dimension(g_S(dec, [c])) for c in range(5)] == [degrees[0] + 2, *degrees[1:]]
    assert degrees == [1, 2, 4, 4, 4] and dimension(g_S(dec, [0]), extra=(1, 2)) == 1
    assert g_S(dec, [2, 3]) == CyclicPolynomial(ell, [0, 1, 3, 4, 5, 7, 8])
    assert dimension(g_S(dec, [2, 3])) == 8


def check_tube(case, dec: RingDecomposition, sm: dict) -> None:
    """Delta, the displayed t, and the classical code ker(lift D)."""
    ell, d, poly = case.ell, case.factor_d, lambda e: CyclicPolynomial(case.ell, e)
    t = tube_kernel_generator(d, case.couple_column, dec)
    delta = maximal_minors(d)[case.couple_column, 0]
    assert [t[j, 0] * delta for j in range(4)] == [poly(e) for e in sm["delta_t"]]
    kernel = nullspace_basis(row_masks(d.lift()), 4 * ell)
    weights = Counter(word.bit_count() for word in span_words(kernel))
    assert sum(weights.values()) == 2**ell - 1 and min(weights) == sm["tube_distance"]
    if ell == 9:
        assert delta == poly([1, 5, 7]) and dec.inverse(delta) == poly([1, 2, 3, 5, 6])
    else:  # t = (1, h, 1 + h, x^10 + (x^9 + x^10) h) with h = Delta^{-1} (x^4 + x^10 + x^11)
        h = dec.inverse(delta) * poly([4, 10, 11])
        assert delta == poly([4, 9, 10]) and h == poly([0, 3, 5, 7, 8, 9, 10, 11, 12])
        one = CyclicPolynomial.one(ell)
        assert [t[j, 0] for j in range(4)] == [one, h, one + h, poly([10]) + poly([9, 10]) * h]
        assert weights[11] == 15


def check_adapter(
    case, dec: RingDecomposition, u3: PolynomialMatrix, u4: PolynomialMatrix, sm: dict
) -> None:
    """Equal-lift adapter between fibre (3,3) of block 1 and fibre (4,4) of block 2."""
    ell, q, a, data = case.ell, case.couple_column, case.factor_a, case.data_code()
    one, x = CyclicPolynomial.one(ell), CyclicPolynomial.monomial(ell, 1)
    eye = lambda size: PolynomialMatrix.identity(ell, size)
    zero = lambda rows, cols: PolynomialMatrix.zero(ell, rows, cols)
    unit = lambda r: PolynomialMatrix(ell, (1, 5), {(0, r): one})
    link = PolynomialMatrix(ell, (1, 10), {(0, 3): one, (0, 9): x})  # eta_1 + x eta_2 = 0
    factor = a.vstack(unit(4)).block_diag(a.vstack(unit(3))).vstack(link)
    d1 = factor.kron(eye(4)).vstack(eye(10).kron(case.factor_d))
    d0 = eye(9).kron(case.factor_d).hstack(factor.kron(eye(3)))
    # Block k attaches at cokernel coordinate j_k = 3 + k; data coordinate (r, j) of block k.
    at = lambda k, r, j: 34 * k + 5 * r + j
    f1 = {(at(k, r, 3 + k), 4 * (5 * k + r) + q): one for k in (0, 1) for r in range(5)}
    f0 = {(15 * k + 5 * r + 3 + k, 4 * (4 * k + r) + q): one for k in (0, 1) for r in range(3)}
    f1, f0 = PolynomialMatrix(ell, (68, 40), f1), PolynomialMatrix(ell, (30, 66), f0)
    data_x = data.matrix_x.block_diag(data.matrix_x)
    data_z = data.matrix_z.block_diag(data.matrix_z)
    shapes = [m.shape for m in (factor, d1, d0, f1, f0)]
    assert shapes == [(9, 10), (66, 40), (27, 66), (68, 40), (30, 66)]
    assert data_x @ f1 == f0 @ d1 and d0 @ d1 == zero(27, 40)
    code = CPMCSSCode(
        data_x.hstack(f0).vstack(zero(27, 68).hstack(d0)),
        data_z.hstack(zero(30, 66)).vstack(f1.adjoint().hstack(d1.adjoint())),
    )
    hx, hz = code.binary_matrix_x, code.binary_matrix_z
    assert not ((hx @ hz.T).toarray() % 2).any()
    n, k, ranks, packet = sm["adapter"]
    assert (code.n, code.k) == (n, k) and k == 2 * data.k - (ell + 4)
    assert (Rowspace(row_masks(hx)).rank, Rowspace(row_masks(hz)).rank) == ranks
    assert code.n - 2 * data.n == ell * (9 * 4 + 10 * 3)

    def on_data(vector: PolynomialMatrix, k: int, j: int, scale=one) -> list[int]:
        """Orbit (bitmasks) of scale * vector of block k placed on data fibre j."""
        entries = {(at(k, r, j), 0): scale * vector[r, 0] for r in range(5)}
        return orbit_masks(PolynomialMatrix(ell, (68, 1), entries))

    # Intended readout: R (u_3, x^{-1} u_4) and eps_0 (e_0 + e_r), r = 1, 2, in each block.
    x_inv = CyclicPolynomial.monomial(ell, -1)
    product = column([u3[r, 0] for r in range(5)] + [x_inv * u4[r, 0] for r in range(5)])
    assert factor @ product == zero(9, 1)
    joint = lambda s: [a ^ b for a, b in zip(on_data(u3, 0, 3, s), on_data(u4, 1, 4, s * x_inv))]
    intended = joint(one) + [on_data(v, k, 3 + k)[0] for k in (0, 1) for v in residuals(ell)]
    images = readout_images(d1.lift(), f1.lift())
    image = Rowspace(images)
    assert image.rank == len(images) == Rowspace(intended).rank == ell + 4, "no relations"
    assert all(image.contains(v) for v in intended)

    stabilizers = row_masks(data_z.lift())
    span = lambda *parts: Rowspace(stabilizers + [v for p in parts for v in p]).rank
    meet = lambda rows: span(images) + span(rows) - span(images, rows) - span()  # mod stabilizers
    assert span(images) - span() == ell + 4
    # Canonical register of block k: fibres (i, j) in {3, 4}^2; full register: all data logicals.
    registers = [[w for u in (u3, u4) for j in (3, 4) for w in on_data(u, k, j)] for k in (0, 1)]
    logicals = logical_quotient_basis(
        row_masks(data.binary_matrix_x), row_masks(data.binary_matrix_z), data.n
    )
    assert [meet(rows) for rows in registers] == [0, 0]
    assert [meet(logicals), meet([v << data.n for v in logicals])] == [2, 2]
    # A common CRT selection S keeps sum_{c in S} deg g_c products and no residual.
    assert Rowspace(joint(e_S(dec, packet))).rank == sum(dec.component_degrees[c] for c in packet)
    assert e_S(dec, packet) * CyclicPolynomial(ell, range(ell)) == CyclicPolynomial.zero(ell)


def main() -> None:
    for ell, sm in SM.items():
        case, dec = instance(ell), RingDecomposition(ell)
        u3, u4 = check_data_code(case, dec, sm)
        if ell == 15:
            check_packet_selection(case, dec)
        check_tube(case, dec, sm)
        check_adapter(case, dec, u3, u4, sm)
        n, k, ranks, _ = sm["adapter"]
        tube = f"[{4 * ell},{ell},{sm['tube_distance']}]"
        print(
            f"l={ell}: dim ker A' = {ell + 2}, tube code {tube}, adapter [[{n},{k}]] ranks {ranks}"
        )
    print("All worked-example checks passed.")


if __name__ == "__main__":
    main()
