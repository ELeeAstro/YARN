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


def hypsometric(p_lev, T_lay, mu_lay, params):
    log_g = jnp.asarray(params["log_g"])
    g = 10.0 ** log_g 
    H = (kb * T_lay) / (mu_lay * amu * g)
    dlnp = jnp.log(p_lev[:-1] / p_lev[1:])
    dz = H * dlnp
    z0 = jnp.zeros_like(p_lev[:1])

    z_lev = jnp.concatenate([z0, jnp.cumsum(dz)])
    z_lay = (z_lev[:-1] + z_lev[1:]) / 2.0

    return z_lev, z_lay, dz


def g_at_z(R0, z, g_ref):
    return g_ref * (R0 / (R0 + z)) ** 2

def hypsometric_variable_g(p_lev, T_lay, mu_lay, params):


    nlev = p_lev.shape[0]

    log_g = jnp.asarray(params["log_g"])
    R0 = jnp.asarray(params["R_p"]) * R_jup

    g_ref = 10.0 ** log_g

    dlnp = jnp.log(p_lev[:-1] / p_lev[1:])

    # Sweep 1: constant g
    H0 = (kb * T_lay) / (mu_lay * amu * g_ref)
    dz0 = H0 * dlnp
    z1 = jnp.concatenate([jnp.zeros_like(p_lev[:1]), jnp.cumsum(dz0)])

    # Mid-layer altitudes from sweep 1
    z_mid_1 = 0.5 * (z1[:-1] + z1[1:])

    # Sweep 2: use g at mid-layer (from sweep 1), then integrate upward
    g_mid = g_at_z(R0, z_mid_1, g_ref)
    H_mid = (kb * T_lay) / (mu_lay * amu * g_mid)
    dz = H_mid * dlnp
    z_lev = jnp.concatenate([jnp.zeros_like(p_lev[:1]), jnp.cumsum(dz)])
    z_lay = (z_lev[:-1] + z_lev[1:]) / 2.0

    return z_lev, z_lay, dz

def hypsometric_variable_g_step_predictor(p_lev, T_lay, mu_lay, params):

    nlev = p_lev.shape[0]

    log_g = jnp.asarray(params["log_g"])
    R0 = jnp.asarray(params["R_p"]) * R_jup
    g_ref = 10.0 ** log_g

    dlnp = jnp.log(p_lev[:-1] / p_lev[1:])

    z0 = jnp.zeros_like(p_lev)
    dz0 = jnp.zeros_like(p_lev[:-1])

    def body(i, carry):
        z, dz = carry

        # predictor using g at level i
        g_i = g_at_z(R0, z[i], g_ref)
        H_i = (kb * T_lay[i]) / (mu_lay[i] * amu * g_i)
        dz_pred = H_i * dlnp[i]

        # corrector using g at mid-layer altitude (predicted)
        z_mid = z[i] + 0.5 * dz_pred
        g_mid = g_at_z(R0, z_mid, g_ref)
        H_mid = (kb * T_lay[i]) / (mu_lay[i] * amu * g_mid)
        dz_i = H_mid * dlnp[i]

        z = z.at[i + 1].set(z[i] + dz_i)
        dz = dz.at[i].set(dz_i)
        return (z, dz)

    z_lev, dz = jax.lax.fori_loop(0, nlev - 1, body, (z0, dz0))
    z_lay = 0.5 * (z_lev[:-1] + z_lev[1:])
    return z_lev, z_lay, dz
