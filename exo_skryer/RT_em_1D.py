"""
RT_em_1D.py
===========
"""

from __future__ import annotations

from typing import Dict, Mapping, Tuple

import jax.numpy as jnp

__all__ = ["compute_emission_spectrum_1d"]


def compute_emission_spectrum_1d(
    state: Dict[str, jnp.ndarray],
    params: Dict[str, jnp.ndarray],
    opacity_components: Mapping[str, jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    raise RuntimeError(
        "compute_emission_spectrum_1d is deprecated. "
        "Use compute_emission_spectrum_1d_ck or compute_emission_spectrum_1d_lbl instead."
    )
