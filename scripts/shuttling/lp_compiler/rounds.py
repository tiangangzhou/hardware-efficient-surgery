"""Transport rounds of the factors of the check matrices.

A factor F (A, its conjugate transpose, C or D) acts on one quadrant of the layout: round
k pairs check group i with data group i + k (cyclically; i - k for F^dagger), and each
monomial x^e of F[i][i + k] is one sub-round in which the check fibres sit on the data
fibres with sheet shift e.  Rectangular factors keep cycling through empty (virtual)
columns, whose checks park on "ghost" slots beside the quadrant.
"""

from __future__ import annotations

from .layout import CHECK_FIBRES, DATA_FIBRES, TILE_PITCH, dagger, docking_offset


def matrix_rounds(
    case, basis, factor, groups, target, transpose=False, rounds=None, region="LPData1"
):
    """Sub-rounds of ``factor`` (``groups[i]``: check fibres of row i, ``target(j)``: data
    fibres of column j), each with its active cells and skipped zero cells."""
    p = case["p"]
    F = {"A": case["A"], "C": case["C"], "D": case["D"]}[factor]
    base = factor
    if transpose:
        F = dagger(F, p)
        factor += "_dagger"
    n = max(len(F), len(F[0]))
    count = n if rounds is None else rounds
    code = int(base != "A")
    term = {
        ("X", "A", False): "data-x-1",
        ("X", "A", True): "data-x-2",
        ("Z", "A", False): "data-z-1",
        ("Z", "A", True): "data-z-2",
        ("X", "C", False): "anc-x-1",
        ("X", "D", False): "anc-x-2",
        ("Z", "D", True): "anc-z-1",
        ("Z", "C", True): "anc-z-2",
    }[basis, base, transpose]
    out = []
    for k in range(count):
        cols = [(i - k if transpose else i + k) % n for i in range(len(F))]
        width = max(
            (len(F[i][j]) for i, j in enumerate(cols) if j < len(F[i]) and F[i][j]), default=1
        )
        for ti in range(width):
            active = []
            skipped = []
            ghosts = {}
            for i, j in enumerate(cols):
                if j >= len(F[i]):
                    # Empty virtual targets in rectangular dense factors retain
                    # the cyclic shuttle. They are not zero entries of H_L.
                    if transpose:
                        kind = 1 if basis == "X" else 2
                        for check in groups[i]:
                            home = case["fibres"][kind, check]["home"]
                            x, y = home
                            if base == "A":
                                b, local = divmod(check, CHECK_FIBRES)
                                if basis == "X":
                                    x, y = TILE_PITCH * (5 + j), 4 * p * (12 * b + 5 + local // 5)
                                else:
                                    x, y = TILE_PITCH * (5 + local % 3), 4 * p * (12 * b + 5 + j)
                            elif base == "D":
                                x = TILE_PITCH * (case["origin"] + j)
                            elif base == "C":
                                y = 4 * p * (11 if case["mode"] == "adapter" else 5 + j)
                            ox, oy = docking_offset(basis, region)
                            ghosts[check] = [x + ox, y + oy]
                    continue
                targets = target(j)
                poly = F[i][j]
                if poly is None:
                    skipped.append(dict(row=i, col=j, checks=groups[i], targets=targets))
                elif ti < len(poly):
                    sparse = base == "C" and (j if transpose else i) in case["logical"]
                    active.append(
                        dict(
                            checks=groups[i],
                            targets=targets,
                            exponent=poly[ti],
                            row=i,
                            col=j,
                            polynomial=poly,
                            sparse=sparse,
                        )
                    )
            out.append(
                dict(
                    factor=factor,
                    base=base,
                    basis=basis,
                    code=code,
                    F=F,
                    active=active,
                    skipped=skipped,
                    ghosts=ghosts,
                    groups=groups,
                    region=region,
                    term=term,
                    round=k + 1,
                    monomial=ti + 1,
                    targets=[j + 1 for j in cols],
                )
            )
    return out


def streams(c, basis):
    """(data-code rounds, ancillary-code rounds) of one basis; the two streams run in
    parallel, one pulse per round index."""
    B, d, n, m, r = c["copies"], c["d"], c["n"], c["m"], c["r"]
    T, U, X, Z = (c[k] for k in ("T", "U", "X", "Z"))
    nd, nc = DATA_FIBRES, CHECK_FIBRES
    if basis == "X":
        dat = matrix_rounds(
            c,
            basis,
            "A",
            [[nc * b + 5 * a + j for b in range(B) for j in range(5)] for a in range(3)],
            lambda i: [nd * b + 5 * i + j for b in range(B) for j in range(5)],
        )
        dat += matrix_rounds(
            c,
            basis,
            "A",
            [[nc * b + 5 * a + j for b in range(B) for a in range(3)] for j in range(5)],
            lambda z: [nd * b + 25 + 3 * a + z for b in range(B) for a in range(3)],
            True,
            region="LPData2",
        )
        anc = matrix_rounds(
            c,
            basis,
            "C",
            [[X(a, v) for v in range(d)] for a in range(m)],
            lambda i: [U(i, v) for v in range(d)],
        )
        anc += matrix_rounds(
            c,
            basis,
            "D",
            [[X(a, v) for a in range(m)] for v in range(d)],
            lambda j: [T(a, j) for a in range(m)],
            rounds=d,
            region="LPData2",
        )
    else:
        dat = matrix_rounds(
            c,
            basis,
            "A",
            [[nc * b + 3 * i + z for b in range(B) for i in range(5)] for z in range(3)],
            lambda j: [nd * b + 5 * i + j for b in range(B) for i in range(5)],
        )
        dat += matrix_rounds(
            c,
            basis,
            "A",
            [[nc * b + 3 * i + z for b in range(B) for z in range(3)] for i in range(5)],
            lambda a: [nd * b + 25 + 3 * a + z for b in range(B) for z in range(3)],
            True,
            region="LPData2",
        )
        anc = matrix_rounds(
            c,
            basis,
            "D",
            [[Z(i, j) for i in range(r)] for j in range(n)],
            lambda v: [U(i, v) for i in range(r)],
            True,
            rounds=d,
        )
        anc += matrix_rounds(
            c,
            basis,
            "C",
            [[Z(i, j) for j in range(n)] for i in range(r)],
            lambda a: [T(a, j) for j in range(n)],
            True,
            region="LPData2",
        )
    return dat, anc
