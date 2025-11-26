"""
opacity_ray.py
==============

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

import registry_ray as XR


def zero_ray_opacity(state: Dict[str, jnp.ndarray], params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    layer_count = int(state["nlay"])
    wavelength_count = int(state["nwl"])
    return jnp.zeros((layer_count, wavelength_count))


def compute_ray_opacity(state: Dict[str, jnp.ndarray], params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    if not XR.has_ray_data():
        return zero_ray_opacity(state, params)
    wavelengths = state["wl"]
    number_density = state["nd"]
    density = state["rho"]
    layer_count = number_density.shape[0]
    master_wavelength = XR.ray_master_wavelength()
    if master_wavelength.shape != wavelengths.shape:
        raise ValueError("Rayleigh wavelength grid must match forward-model grid.")
    sigma_values = XR.ray_sigma_table()
    species_names = XR.ray_species_names()
    mixing_cache = state.get("mixing_ratios", {})
    mixing_ratio_list = []
    for name in species_names:
        if name in mixing_cache:
            value = mixing_cache[name]
        else:
            key = f"f_{name}"
            if key not in params:
                raise KeyError(f"Missing Rayleigh mixing ratio parameter '{key}'")
            value = jnp.asarray(params[key])
        if value.ndim == 0:
            mixing_ratio_list.append(jnp.full((layer_count,), value))
        elif value.ndim == 1:
            if value.shape[0] != layer_count:
                raise ValueError(f"Rayleigh mixing ratio '{name}' length {value.shape[0]} != {layer_count}.")
            mixing_ratio_list.append(value)
        else:
            raise ValueError(f"Rayleigh mixing ratio '{name}' has unsupported shape {value.shape}.")
    mixing_ratios = jnp.stack(mixing_ratio_list, axis=0)
    absorption = mixing_ratios[:, :, None] * number_density[None, :, None] * sigma_values[:, None, :]
    return jnp.sum(absorption, axis=0) / density[:, None]
