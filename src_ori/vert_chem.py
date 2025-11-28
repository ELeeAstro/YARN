"""
vert_chem.py
============

Overview:
    TODO: Describe the purpose and responsibilities of this module.

Sections to complete:
    - Usage
    - Key Functions
    - Notes
"""

from __future__ import annotations

from typing import Dict

import jax.numpy as jnp

solar_h2 = 0.5
solar_he = 0.085114
solar_h2_he = solar_h2 + solar_he


def constant_vmr(
    p_lay: jnp.ndarray,
    T_lay: jnp.ndarray,
    params: Dict[str, jnp.ndarray],
    nlay: int,
) -> Dict[str, jnp.ndarray]:
    del p_lay, T_lay  # unused but kept for consistent signature

    vmr: Dict[str, jnp.ndarray] = {}
    for k, v in params.items():
        if k.startswith("log_10_f_"):
            sp = k[len("log_10_f_") :]
            vmr[sp] = 10.0 ** jnp.asarray(v)
        elif k.startswith("f_"):
            sp = k[len("f_") :]
            vmr[sp] = jnp.asarray(v)

    trace_values = list(vmr.values())
    if trace_values:
        total_trace_vmr = jnp.sum(jnp.stack(trace_values))
    else:
        total_trace_vmr = jnp.asarray(0.0)
    background_vmr = 1.0 - total_trace_vmr

    vmr["H2"] = background_vmr * solar_h2 / solar_h2_he
    vmr["He"] = background_vmr * solar_he / solar_h2_he

    vmr_lay = {species: jnp.full((nlay,), value) for species, value in vmr.items()}
    return vmr_lay


def chemical_equilibrium(
    p_lay: jnp.ndarray,
    T_lay: jnp.ndarray,
    params: Dict[str, jnp.ndarray],
    nlay: int,
) -> Dict[str, jnp.ndarray]:
    del p_lay, T_lay, params, nlay
    raise NotImplementedError("chemical_equilibrium is not implemented yet.")
