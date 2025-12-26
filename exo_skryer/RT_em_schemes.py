"""
RT_em_schemes.py
================
"""

from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp
from jax import lax


_MU_NODES = (0.0454586727, 0.2322334416, 0.5740198775, 0.9030775973)
_MU_WEIGHTS = (0.0092068785, 0.1285704278, 0.4323381850, 0.4298845087)
nstreams = len(_MU_NODES) * 2
_DT_THRESHOLD = 1.0e-4
_DT_SAFE = 1.0e-12

__all__ = ["solve_alpha_eaa", "get_emission_solver"]


def solve_alpha_eaa(
    be_levels: jnp.ndarray,
    dtau_layers: jnp.ndarray,
    ssa: jnp.ndarray,
    g_phase: jnp.ndarray,
    be_internal: jnp.ndarray,
    return_layer_contrib: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    nlev, nwl = be_levels.shape
    nlay = nlev - 1

    be_levels = be_levels.astype(jnp.float64)[::-1]
    dtau_layers = dtau_layers.astype(jnp.float64)[::-1]
    ssa = ssa.astype(jnp.float64)[::-1]
    g_phase = g_phase.astype(jnp.float64)[::-1]

    al = be_levels[1:] - be_levels[:-1]
    lw_up_sum = jnp.zeros((nlev, nwl))
    lw_down_sum = jnp.zeros((nlev, nwl))

    mask = g_phase >= 1.0e-4
    fc = jnp.where(mask, g_phase**nstreams, 0.0)
    pmom2 = jnp.where(mask, g_phase**(nstreams + 1), 0.0)
    ratio = jnp.maximum((fc**2) / jnp.maximum(pmom2**2, 1.0e-30), 1.0e-30)
    sigma_sq = jnp.where(mask, ((nstreams + 1) ** 2 - nstreams**2) / jnp.log(ratio), 1.0)
    c = jnp.exp((nstreams**2) / (2.0 * sigma_sq))
    fc_scaled = c * fc

    w_in = jnp.clip(ssa, 0.0, 0.99)
    denom = jnp.maximum(1.0 - fc_scaled * w_in, 1.0e-12)
    w0 = jnp.where(mask, w_in * ((1.0 - fc_scaled) / denom), w_in)
    dtau = jnp.where(mask, (1.0 - w_in * fc_scaled) * dtau_layers, dtau_layers)
    hg = g_phase
    eps = jnp.sqrt((1.0 - w0) * (1.0 - hg * w0))
    dtau_a = eps * dtau

    tau_interface = jnp.concatenate([jnp.zeros((1, nwl), dtype=dtau_a.dtype),
                                     jnp.cumsum(dtau_a, axis=0)], axis=0)
    tau_top_layer = tau_interface[:-1]

    if return_layer_contrib:
        layer_contrib_sum = jnp.zeros((nlay, nwl), dtype=be_levels.dtype)

    for mu, weight in zip(_MU_NODES, _MU_WEIGHTS):
        T_trans = jnp.exp(-dtau_a / mu)
        mu_over_dtau = mu / jnp.maximum(dtau_a, _DT_SAFE)

        def down_body(k, lw):
            linear = (
                lw[k] * T_trans[k]
                + be_levels[k + 1]
                - al[k] * mu_over_dtau[k]
                - (be_levels[k] - al[k] * mu_over_dtau[k]) * T_trans[k]
            )
            iso = (
                lw[k] * T_trans[k]
                + 0.5 * (be_levels[k] + be_levels[k + 1]) * (1.0 - T_trans[k])
            )
            next_val = jnp.where(dtau_a[k] > _DT_THRESHOLD, linear, iso)
            return lw.at[k + 1].set(next_val)

        lw_init = jnp.zeros((nlev, nwl), dtype=be_levels.dtype)
        lw_down = lax.fori_loop(0, nlay, down_body, lw_init)

        lw_up_init = jnp.zeros((nlev, nwl), dtype=be_levels.dtype).at[-1].set(
            lw_down[-1] + be_internal
        )

        if return_layer_contrib:
            T_toa = jnp.exp(-tau_top_layer / mu)

            def up_body(idx, carry):
                lw, layer_acc = carry
                k = nlay - 1 - idx

                linear = (
                    lw[k + 1] * T_trans[k]
                    + be_levels[k]
                    + al[k] * mu_over_dtau[k]
                    - (be_levels[k + 1] + al[k] * mu_over_dtau[k]) * T_trans[k]
                )
                iso = (
                    lw[k + 1] * T_trans[k]
                    + 0.5 * (be_levels[k] + be_levels[k + 1]) * (1.0 - T_trans[k])
                )
                I_top = jnp.where(dtau_a[k] > _DT_THRESHOLD, linear, iso)

                source = I_top - lw[k + 1] * T_trans[k]
                layer_acc = layer_acc.at[k].add(weight * source * T_toa[k])
                lw = lw.at[k].set(I_top)
                return (lw, layer_acc)

            lw_up, layer_contrib_sum = lax.fori_loop(
                0, nlay, up_body, (lw_up_init, layer_contrib_sum)
            )
        else:
            def up_body(idx, lw):
                k = nlay - 1 - idx
                linear = (
                    lw[k + 1] * T_trans[k]
                    + be_levels[k]
                    + al[k] * mu_over_dtau[k]
                    - (be_levels[k + 1] + al[k] * mu_over_dtau[k]) * T_trans[k]
                )
                iso = (
                    lw[k + 1] * T_trans[k]
                    + 0.5 * (be_levels[k] + be_levels[k + 1]) * (1.0 - T_trans[k])
                )
                I_top = jnp.where(dtau_a[k] > _DT_THRESHOLD, linear, iso)
                return lw.at[k].set(I_top)

            lw_up = lax.fori_loop(0, nlay, up_body, lw_up_init)

        lw_down_sum = lw_down_sum + lw_down * weight
        lw_up_sum = lw_up_sum + lw_up * weight

    lw_up_flux = jnp.pi * lw_up_sum
    lw_down_flux = jnp.pi * lw_down_sum

    if return_layer_contrib:
        layer_contrib_flux = jnp.pi * layer_contrib_sum[::-1]
    else:
        layer_contrib_flux = jnp.zeros((nlay, nwl), dtype=lw_up_flux.dtype)

    return lw_up_flux, lw_down_flux, layer_contrib_flux


def get_emission_solver(name: str):
    name = str(name).lower().strip()
    if name in ("eaa", "alpha_eaa"):
        return solve_alpha_eaa
    raise NotImplementedError(f"Unknown emission scheme '{name}'")
