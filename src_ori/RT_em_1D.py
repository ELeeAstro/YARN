"""
RT_em_1D.py
===========

Overview:
    Scaffolding for a 1D emission-spectrum radiative-transfer solver.

Sections to complete:
    - Usage
    - Key Functions
    - Notes
"""

from __future__ import annotations

from typing import Dict, Mapping, Callable

import jax.numpy as jnp


def _sum_opacity_components(
    state: Dict[str, jnp.ndarray],
    opacity_components: Mapping[str, jnp.ndarray],
) -> jnp.ndarray:
    """Return the summed opacity grid for all provided components."""
    if opacity_components:
        first = next(iter(opacity_components.values()))
        k_tot = jnp.zeros_like(first)
        for component in opacity_components.values():
            k_tot = k_tot + component
        return k_tot
    nlay = int(state["nlay"])
    nwl = int(state["nwl"])
    return jnp.zeros((nlay, nwl))


def two_stream_emission(
    state: Dict[str, jnp.ndarray],
    k_tot: jnp.ndarray,
    source_fn: Callable[[Dict[str, jnp.ndarray], jnp.ndarray], jnp.ndarray] | None = None,
) -> jnp.ndarray:
    """
    Placeholder two-stream emission calculation.

    Parameters
    ----------
    state : Dict[str, jnp.ndarray]
        Atmospheric state dictionary (contains T_lay, p_lay, etc.).
    k_tot : jnp.ndarray
        Total opacity grid (nlay, nlambda).
    source_fn : Callable, optional
        Function returning a source term per layer and wavelength.
    """
    k_floor = 1.0e-99
    k_eff = jnp.maximum(k_tot, k_floor)
    dz = jnp.asarray(state["dz"])[:, None]
    tau = jnp.cumsum(k_eff * dz, axis=0)

    if source_fn is None:
        # Default source: Planck-like placeholder using T_lay
        T_lay = jnp.asarray(state["T_lay"])[:, None]
        source = jnp.power(T_lay, 4)
    else:
        source = source_fn(state, k_tot)

    intensity = jnp.trapz(source * jnp.exp(-tau), axis=0)
    return intensity


def compute_emission_spectrum_1d(
    state: Dict[str, jnp.ndarray],
    params: Dict[str, jnp.ndarray],
    opacity_components: Mapping[str, jnp.ndarray],
    solver: Callable[[Dict[str, jnp.ndarray], jnp.ndarray], jnp.ndarray] | None = None,
) -> jnp.ndarray:
    """
    Compute the wavelength-dependent emission spectrum.

    Parameters
    ----------
    state : Dict[str, jnp.ndarray]
        Atmospheric state dictionary.
    params : Dict[str, jnp.ndarray]
        Retrieval parameters (unused placeholder for now).
    opacity_components : Mapping[str, jnp.ndarray]
        Mapping of opacity component names to arrays shaped (nlay, nlambda).
    solver : Callable, optional
        Function handling the RT solution; defaults to `two_stream_emission`.
    """
    k_tot = _sum_opacity_components(state, opacity_components)

    if solver is None:
        solver = two_stream_emission

    return solver(state, k_tot)
