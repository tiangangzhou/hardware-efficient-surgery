"""CSS code surgery on top of the ``qldpc`` package.

- ``hmatrix.gadget``: ``GadgetSpec``, the mapping cone of a gadget, and ``build_gadget``;
- ``hmatrix.edge_expansion``: Graph-IGND, the edge-expanded homological measurement of
  Ide, Gowda, Nadkarni, Dauphinais (arXiv:2410.02753);
- ``hmatrix.gauging``: Graph-WY, the gauging measurement of Williamson and Yoder
  (arXiv:2410.02213), and ``parallel_gadgets``;
- ``hmatrix.joint``: the bridge (repetition-code adapter) of Swaroop, Jochym-O'Connor,
  Yoder (arXiv:2410.03628) for joint measurements;
- ``circuit``: Stim circuits for single, batch and joint measurements;
- ``decoding`` and ``batched``: Relay-BP with a BP-LSD fallback, synchronous or in two
  stages;
- ``aod``: gate layers and check routes of the graph gadgets for the shuttling model.
"""

from __future__ import annotations

from .circuit import (
    build_batch_ppm_circuit,
    build_joint_ppm_circuit,
    build_single_ppm_circuit,
    check_noiseless,
)
from .decoding import DeferredRelayBatch, TieredRelayDecoder, sinter_decoder
from .hmatrix.edge_expansion import cheeger_constant
from .hmatrix.gadget import GadgetSpec, build_gadget
from .hmatrix.gauging import parallel_gadgets
from .hmatrix.joint import JointSpec, build_bridge

__all__ = [
    "DeferredRelayBatch",
    "GadgetSpec",
    "JointSpec",
    "TieredRelayDecoder",
    "build_batch_ppm_circuit",
    "build_bridge",
    "build_gadget",
    "build_joint_ppm_circuit",
    "build_single_ppm_circuit",
    "check_noiseless",
    "cheeger_constant",
    "parallel_gadgets",
    "sinter_decoder",
]
