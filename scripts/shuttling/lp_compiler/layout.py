"""Tile layout of the LP data code(s) and the ancillary code of one surgery mode.

Every fibre (a row of R_l: one data qubit or check per sheet) is a tile of l atoms.  A
data code LP(A, A^dagger), A the 3 x 5 seed, has 5 x 5 V x V data tiles (columns 0-4,
rows 0-4), 3 x 3 C x C data tiles (columns 5-7, rows 5-7), 3 x 5 X check tiles (columns
0-4, rows 5-7) and 5 x 3 Z check tiles (columns 5-7, rows 0-4).  The ancillary code of
the factor C (fibre: A' = [A; e_4]; packet: CRT packet rows 1 + x; adapter: two copies
joined by one row) and the tube D starts at column ``origin``: its m x n data T and
m x d X checks share the rows of the data code's X checks, its r x d data U and r x n Z
checks those of the V x V data (columns ordered so that the coupled column q is first).
"""

from __future__ import annotations

from algebraic_qec.examples.surgery_instances import instance

DATA_FIBRES = 34  # data tiles of one LP code: 5 x 5 (V x V) and 3 x 3 (C x C)
CHECK_FIBRES = 15  # X (3 x 5) or Z (5 x 3) check tiles of one LP code
TILE_PITCH = 36  # um between tile columns; tile rows are 4 l um apart
ATOM_PITCH = 12  # um between the atoms of one tile
MODES = ("fibre", "packet", "adapter")


def terms(matrix):
    """Exponent matrix -> polynomial term lists (None for a zero entry)."""
    return [[None if x is None else [x] for x in row] for row in matrix]


def dagger(matrix, p):
    """Conjugate transpose over R_l of a matrix of term lists."""
    return [
        [
            None if matrix[i][j] is None else sorted((-s) % p for s in matrix[i][j])
            for i in range(len(matrix))
        ]
        for j in range(len(matrix[0]))
    ]


def make_case(p, mode, origin=10):
    """Factors, fibre homes and modules of one surgery mode at lift ``p``.

    Fibre keys are (kind, index), kind 0 = data, 1 = X check, 2 = Z check; ``T``, ``U``,
    ``X``, ``Z`` index the ancillary data and checks after the ``ND`` data and ``NC``
    check fibres of the data code(s).
    """
    source = instance(p)
    A, D = terms(source.seed), terms(source.tube)
    d, n = len(D), len(D[0])
    q = source.couple_column
    H = [[None] * 4 + [[0]]]
    C = A + H
    copies = 1
    logical = {3}
    if mode == "packet":
        H = H + [[[0, 1] if i == j else None for j in range(5)] for i in range(4)]
        C = A + H
        logical = set(range(3, 8))
    if mode == "adapter":
        C = [row + [None] * 5 for row in A + H] + [
            [None] * 5 + row for row in A + [[None, None, None, [0], None]]
        ]
        C += [[None, None, None, [0], None, None, None, None, None, [1]]]
        copies = 2
        logical = {3, 7, 8}
    m, r = len(C), len(C[0])
    ND = DATA_FIBRES * copies
    NC = CHECK_FIBRES * copies
    T = lambda a, j: ND + a * n + j
    U = lambda i, v: ND + m * n + i * d + v
    X = lambda a, v: NC + a * d + v
    Z = lambda i, j: NC + i * n + j
    rows = lambda a: (0, a) if mode != "adapter" or a < 4 else ((1, a - 4) if a < 8 else (2, 0))
    ys = (0, 12)  # first tile row of each data code
    fibres = {}
    modules = []

    def add(kind, idx, module, x, y):
        fibres[kind, idx] = dict(module=module, home=[TILE_PITCH * x, 4 * p * y])

    for b in range(copies):
        y0 = ys[b]
        code = 2 * b
        modules.append(
            dict(
                id=code,
                name="Data code" if copies == 1 else f"Data {b + 1}",
                kind="data",
                bank=code,
                bounds=[-18, 310, 4 * p * y0 - 24, 4 * p * (y0 + 8)],
                centre=[138, 4 * p * y0 - 12],
            )
        )
        for i in range(5):
            for j in range(5):
                add(0, DATA_FIBRES * b + 5 * i + j, code, j, y0 + i)
        for a in range(3):
            for z in range(3):
                add(0, DATA_FIBRES * b + 25 + 3 * a + z, code, 5 + z, y0 + 5 + a)
            for j in range(5):
                add(1, CHECK_FIBRES * b + 5 * a + j, code, j, y0 + 5 + a)
        for i in range(5):
            for z in range(3):
                add(2, CHECK_FIBRES * b + 3 * i + z, code, 5 + z, y0 + i)
        bottom = m if copies == 1 else 4
        modules.append(
            dict(
                id=code + 1,
                name="Ancillary code" if copies == 1 else f"Ancilla {b + 1}",
                kind="ancilla",
                bank=code + 1,
                bounds=[
                    TILE_PITCH * origin - 18,
                    TILE_PITCH * (origin + d + n) + 10,
                    4 * p * y0 - 24,
                    4 * p * (y0 + 5 + bottom),
                ],
                centre=[TILE_PITCH * (origin + (d + n) / 2) - 6, 4 * p * y0 - 12],
            )
        )
    if copies == 2:
        modules.append(
            dict(
                id=4,
                name="Adapter strip",
                kind="adapter",
                bank=1,
                bounds=[
                    TILE_PITCH * origin - 18,
                    TILE_PITCH * (origin + d + n) + 10,
                    4 * p * 10 - 15,
                    4 * p * 11,
                ],
                centre=[TILE_PITCH * (origin + (d + n) / 2) - 6, 4 * p * 10 - 12],
            )
        )
    for a in range(m):
        b, local = rows(a)
        owner = 2 * b + 1 if b < 2 else 4
        y = ys[b] + 5 + local if b < 2 else 10
        for j in range(n):
            add(0, T(a, j), owner, origin + d + (j - q) % n, y)
        for v in range(d):
            add(1, X(a, v), owner, origin + v, y)
    for i in range(r):
        b, local = divmod(i, 5)
        owner = 2 * b + 1
        y = ys[b] + local
        for v in range(d):
            add(0, U(i, v), owner, origin + v, y)
        for j in range(n):
            add(2, Z(i, j), owner, origin + d + (j - q) % n, y)
    return dict(
        p=p,
        mode=mode,
        A=A,
        D=D,
        C=C,
        H=H,
        d=d,
        n=n,
        m=m,
        r=r,
        q=q,
        copies=copies,
        ND=ND,
        NC=NC,
        T=T,
        U=U,
        X=X,
        Z=Z,
        fibres=fibres,
        modules=modules,
        logical=logical,
        origin=origin,
    )


def docking_offset(basis, region):
    """Place checks 2 um perpendicular to their quadrant's sweep direction.

    In the default transposed view this puts X checks below Data 1 and to
    the right of Data 2. Z checks use the complementary directions. Both
    interfaces cross physical x and dock on the negative-y side, opposite
    ordinary checks that may already be touching the same data qubits.
    """
    if region == "interface":
        return [0, -2]
    if (basis == "X") == (region == "LPData1"):
        return [2, 0]
    return [0, 2]
