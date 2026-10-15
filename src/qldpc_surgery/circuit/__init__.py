"""Stim circuits of surgery experiments: low-rate surgery of one logical
(``low_rate_ppm``), high-rate surgery of a batch of logicals with one gadget
(``high_rate_ppm``), or a product of two logicals through a bridge (``joint_ppm``);
``engine`` has the protocol."""

from __future__ import annotations

from .engine import check_noiseless
from .high_rate_ppm import build_batch_ppm_circuit
from .joint_ppm import build_joint_ppm_circuit
from .low_rate_ppm import build_single_ppm_circuit

__all__ = [
    "build_batch_ppm_circuit",
    "build_joint_ppm_circuit",
    "build_single_ppm_circuit",
    "check_noiseless",
]
