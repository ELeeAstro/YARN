"""
vert_mu.py
==========

Overview:
    TODO: Describe the purpose and responsibilities of this module.

Sections to complete:
    - Usage
    - Key Functions
    - Notes
"""

from __future__ import annotations

from typing import Dict, Tuple, Any

import jax.numpy as jnp

import build_opacities as XS

from data_constants import CHEM_SPECIES_DATA


solar_h2 = 0.5
solar_he = 0.085114
solar_h2_he = solar_h2 + solar_he

_SPECIES_MASS = {entry["symbol"]: float(entry["molecular_weight"]) for entry in CHEM_SPECIES_DATA}


def _broadcast_ratio(value, template):
    arr = jnp.asarray(value)
    if arr.shape == template.shape:
        return arr
    if arr.ndim == 0:
        return jnp.full_like(template, arr)
    return jnp.broadcast_to(arr, template.shape)


def compute_mean_molecular_weight(
    layer_template: jnp.ndarray,
    params: Dict[str, jnp.ndarray],
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray], bool]:
    """
    Placeholder for a chemical-equilibrium solver.

    Parameters
    ----------
    layer_template : jnp.ndarray
        1D array defining the atmospheric layers for broadcasting mixing ratios.
    params : Dict[str, jnp.ndarray]
        Retrieval parameters describing bulk composition and other controls.

    Returns
    -------
    Tuple[jnp.ndarray, Dict[str, jnp.ndarray], bool]
        Mean molecular weight profile, dictionary of per-species mixing ratios,
        and a flag indicating whether this calculation succeeded dynamically.
    """

    layer_template = jnp.asarray(layer_template)
    if layer_template.ndim != 1:
        raise ValueError("layer_template must be a 1D array of atmospheric layers.")

    species_names = XS.line_species_names()
    mixing_components: Dict[str, jnp.ndarray] = {}

    total_trace = jnp.zeros_like(layer_template)
    for name in species_names:
        key = f"f_{name}"
        if key not in params:
            continue
        mix_ratio = _broadcast_ratio(params[key], layer_template)
        total_trace = total_trace + mix_ratio
        mixing_components[name] = mix_ratio

    background = jnp.clip(1.0 - total_trace, a_min=0.0)
    mix_h2 = background * solar_h2 / solar_h2_he
    mix_he = background * solar_he / solar_h2_he

    mixing_components["H2"] = mix_h2
    mixing_components["He"] = mix_he

    unique_species = sorted(mixing_components.keys())
    mass_vector = jnp.array([_SPECIES_MASS[name] for name in unique_species])
    mix_matrix = jnp.stack([mixing_components[name] for name in unique_species], axis=0)
    mu_profile = jnp.einsum("i...,i->...", mix_matrix, mass_vector)

    return mu_profile, mixing_components, True
