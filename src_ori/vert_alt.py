"""
vert_alt.py
===========

Overview:
    TODO: Describe the purpose and responsibilities of this module.

Sections to complete:
    - Usage
    - Key Functions
    - Notes
"""

from __future__ import annotations

from typing import Dict

import jax
import jax.numpy as jnp

from data_constants import amu, kb, R_jup


def hypsometric(p_lev: jnp.ndarray, T_lay: jnp.ndarray, mu_lay: jnp.ndarray, params: Dict[str, jnp.ndarray]):
    log_g = jnp.asarray(params["log_g"])
    g = 10.0 ** log_g
    scale_heights = (kb * T_lay) / (mu_lay * amu * g)
    log_ratio = jnp.log(p_lev[:-1] / p_lev[1:])
    delta_z = scale_heights * log_ratio
    cumulative = jnp.concatenate([jnp.zeros((1,)), jnp.cumsum(delta_z)])
    return cumulative


def hypsometric_variable_g(p_lev: jnp.ndarray, T_lay: jnp.ndarray, mu_lay: jnp.ndarray, params: Dict[str, jnp.ndarray]):
    log_g = jnp.asarray(params["log_g"])
    base_radius = jnp.asarray(params["R_p"]) * R_jup
    g0 = 10.0 ** log_g
    nlev = p_lev.shape[0]

    def body(i, z):
        g = g0 * (base_radius / (base_radius + z[i])) ** 2
        scale_height = (kb * T_lay[i]) / (mu_lay[i] * amu * g)
        dz = scale_height * jnp.log(p_lev[i] / p_lev[i + 1])
        return z.at[i + 1].set(z[i] + dz)

    z0 = jnp.zeros_like(p_lev)
    z = jax.lax.fori_loop(0, nlev - 1, body, z0)
    return z
