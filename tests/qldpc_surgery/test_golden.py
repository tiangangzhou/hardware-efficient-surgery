"""SHA-256 hashes of deterministic constructions (Graph-IGND gadgets, joint merged codes).

The hashes were generated before the code was simplified; do not regenerate them
unless a change of the constructions is intended.
"""

from __future__ import annotations

import hashlib

import numpy as np
from qldpc import codes
from qldpc.objects import Pauli

from qldpc_surgery import build_bridge, build_gadget
from qldpc_surgery.hmatrix.joint import joint_merged_code
from tests.qldpc_surgery.fixtures import bb36, gb_code, logical

GADGETS = {
    "Steane|X|plain": {
        "HX_merged": "4bdb995400f54d1a3b31b2c2ce00161724ec255a93f2e865ad50f0add49a9468",
        "HZ_merged": "59dbd7c3d60edd7f180a1ed3c2c885b0fab671bed39fbd2b191efde95d84d6d9",
        "Q_prime": "c3cfab2cb19ff893e979d8c2e443bdc496416fc8e9ad578c62f5c5db9b77a89a",
        "incidence": "a1a1cc2299c58b0cd0bfc937862f9c0d8299ac6f8e5c0527bada7239c4b7caad",
        "partial_0": "2f1535a4022a4d7483b3b6b1953de8cf6ebb2c1e93a5198a7c145ea687902537",
        "support": "9798549418cb6f2fc21948af401a9e76b0d76be4decd1832af85ea57ce6c2082",
    },
    "Steane|Z|plain": {
        "HX_merged": "aa0aea4b5ce27ceff33fb92a0db056bdc9fd099b9cf14b5ddd5aef213a0d9f13",
        "HZ_merged": "a1a3657619e8e17d4702f35d881c1cbd3a72de936f2f54e3cf232814bc2caf08",
        "Q_prime": "0beef7e903fce1e25b443e987bf86c1466cd3a48b991fa656afa62cf7931e4c7",
        "incidence": "614d552da5a5c27c0d355bf05ee824dd82c4bcf2418231cd36a51747d475fdcd",
        "partial_0": "1f64673779413bef9a029dc55b8d5d14cb3039370b90148a649a5baa4584a2b2",
        "support": "21f98a1faba3ccf33fb30d737299bc45f52cd099d43bfcbf5e4c2da7eeb76eac",
    },
    "Webster0|X|plain": {
        "HX_merged": "a382a22c7fc9c4113b5b2a9394fff7f36bc4c4e5ad26c5bd5be1e239c6409b66",
        "HZ_merged": "6cca9af4e28ea6ea949138289c34eb944d61a636cd8b2e82a863bd9a8955de3c",
        "Q_prime": "2ebb3180fbadced19db474c80b30a0dcaf4ff709af7eddc812957e04229595f1",
        "incidence": "f0ef586b0165812ff64cca8d5963aadd04030bff24b13f982922a0fdf6446de7",
        "partial_0": "b7c5a349555e811b870270dce582d4b1af762819c68237e65ff8266972d52f14",
        "support": "8f0cb21653e4611b4ec38d5cc558eb8f3a1b8c0658ba3622db6327ff76cc8c90",
    },
    "Webster0|Z|plain": {
        "HX_merged": "0cbb6428b2d536b5921f5371264a110a0285aacf3496d78e8746e2117f23d903",
        "HZ_merged": "5008e919a9d9150eeff39bf4871c4f24fc8773b3f2c037433ebb4d8d99c903dd",
        "Q_prime": "96d5c58b36038b45fe7d9bf0ef605c2f1656d11b51168d13dfc45ec721a366fb",
        "incidence": "884a966ae9c985a5d27518aecdc8f457982c932216e42749e92174bcc44699cc",
        "partial_0": "4f3c1faeea8516123148a1c65536818d86350d2e46ca250cd66b54cf6978c75e",
        "support": "2275ddb8918d641f134d11a8d06284ce4605bb9348b86f256459fc87aa3f42c9",
    },
}

JOINTS = {
    "inter Steane X": {
        "matrix_x": "52483653350257be9e8a2c502c372c12837d624d0d0e684bea6da92cfdcbe604",
        "matrix_z": "5962040d195066388740c9f4fb70851762f5f5b44d21b4b812bfd66f8fbf42be",
    },
    "inter Steane Z": {
        "matrix_x": "e2f5b46b375d8323a8ed872f489b9679bcd0d7ea6ebe1a17b3a443c19eab7712",
        "matrix_z": "4387ea4d6966ca8666732807436545c13f69ab496f0431d3a168c6ed76ab4770",
    },
    "inter Surface3xSteane X": {
        "matrix_x": "22f411647406920fcaba9301b71acd35413d4f6f01de5514cfd3fd68895e0c2f",
        "matrix_z": "47c449d93e1524346fa39f5fe1e69869541dd944bd6b165a2d067bc71cff584a",
    },
    "intra BB36 Z": {
        "matrix_x": "9952e1a9d032965806f7647a10e1675379c0793113d2950c04e444193d203be1",
        "matrix_z": "9e080edee296e3d7cfe42a84991989e1a43ef87741f8dd7260efe7585098a8a7",
    },
    "intra Steane X": {
        "matrix_x": "a59a11b70b5efcc704e62861ba12a5b15df17cebf17d8f7711d75d475ee6325f",
        "matrix_z": "2d1398a22945a06d89459ad5b4ad91db85d745c675b6db100a26c5b40050d087",
    },
    "intra Steane Z": {
        "matrix_x": "a118a85a2214b34f266ce40784edda414f6dd8a582691b59423972c104e200ed",
        "matrix_z": "36b4198d331a01be71edaf15d2ab66e51d96f24e337649fdfa36f881ec798bf1",
    },
}


def digest(value) -> str:
    """Shape-aware SHA-256 of an integer array."""
    a = np.ascontiguousarray(np.asarray(value, dtype=np.int64))
    return hashlib.sha256(a.tobytes() + repr(a.shape).encode()).hexdigest()


def test_edge_expansion_gadgets_match_golden() -> None:
    for name, code in [("Steane", codes.SteaneCode()), ("Webster0", gb_code(62))]:
        for basis in (Pauli.X, Pauli.Z):
            x = logical(code, basis)
            g = build_gadget(code, x, basis=basis)
            n, n_aux = code.num_qudits, len(g.incidence)
            fields = {
                "support": np.flatnonzero(x),
                "incidence": g.incidence,
                "partial_0": g.partial_0,
                "HX_merged": g.HX_merged,
                "HZ_merged": g.HZ_merged,
                "Q_prime": np.arange(n, n + n_aux),
            }
            expected = GADGETS[f"{name}|{basis.name}|plain"]
            assert {k: digest(v) for k, v in fields.items()} == expected, (name, basis)


def test_joint_merged_codes_match_golden() -> None:
    steane = codes.SteaneCode()
    bb = bb36()
    cases = {
        "intra Steane X": (steane, steane, Pauli.X, 0, 0),
        "intra Steane Z": (steane, steane, Pauli.Z, 0, 0),
        "intra BB36 Z": (bb, bb, Pauli.Z, 0, 1),
        "inter Steane X": (codes.SteaneCode(), codes.SteaneCode(), Pauli.X, 0, 0),
        "inter Steane Z": (codes.SteaneCode(), codes.SteaneCode(), Pauli.Z, 0, 0),
        "inter Surface3xSteane X": (codes.SurfaceCode(3), codes.SteaneCode(), Pauli.X, 0, 0),
    }
    for name, (c_l, c_r, basis, i, j) in cases.items():
        g_l = build_gadget(c_l, logical(c_l, basis, i), basis=basis)
        g_r = build_gadget(c_r, logical(c_r, basis, j), basis=basis)
        merged = joint_merged_code(g_l, g_r, build_bridge(g_l, g_r))
        actual = {"matrix_x": digest(merged.matrix_x), "matrix_z": digest(merged.matrix_z)}
        assert actual == JOINTS[name], name
