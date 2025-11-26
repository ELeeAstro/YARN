"""
opacity_line.py
===============

Overview:
    TODO: Describe the purpose and responsibilities of this module.

Sections to complete:
    - Usage
    - Key Functions
    - Notes
"""

from typing import Dict

import jax
import jax.numpy as jnp

import build_opacities as XS
from data_constants import amu

_SIGMA_CACHE: jnp.ndarray | None = None


def _load_sigma_cube() -> jnp.ndarray:
    global _SIGMA_CACHE
    if _SIGMA_CACHE is None:
        _SIGMA_CACHE = XS.line_sigma_cube()
    return _SIGMA_CACHE


def _interp_weights(grid: jnp.ndarray, targets: jnp.ndarray):
    idx = jnp.searchsorted(grid, targets, side="right") - 1
    idx = jnp.clip(idx, 0, grid.size - 2)
    lower = jnp.take(grid, idx)
    upper = jnp.take(grid, idx + 1)
    weight = jnp.where(upper > lower, (targets - lower) / (upper - lower), 0.0)
    return idx, jnp.clip(weight, 0.0, 1.0)


def _interpolate_sigma(layer_pressures_bar: jnp.ndarray, layer_temperatures: jnp.ndarray) -> jnp.ndarray:
    sigma_cube = _load_sigma_cube()
    pressure_grid = XS.line_pressure_grid()
    temperature_grids = XS.line_temperature_grids()
    pressure_idx, pressure_weight = _interp_weights(pressure_grid, layer_pressures_bar)
    interpolate_temperatures = jax.vmap(lambda grid: _interp_weights(grid, layer_temperatures), in_axes=0)
    temperature_idx, temperature_weight = interpolate_temperatures(temperature_grids)

    def _interpolate_species(species_sigma: jnp.ndarray, temp_idx: jnp.ndarray, temp_weight: jnp.ndarray) -> jnp.ndarray:
        s00 = species_sigma[pressure_idx, temp_idx, :]
        s01 = species_sigma[pressure_idx, temp_idx + 1, :]
        s10 = species_sigma[pressure_idx + 1, temp_idx, :]
        s11 = species_sigma[pressure_idx + 1, temp_idx + 1, :]
        lower_interp = (1.0 - temp_weight)[:, None] * s00 + temp_weight[:, None] * s01
        upper_interp = (1.0 - temp_weight)[:, None] * s10 + temp_weight[:, None] * s11
        return (1.0 - pressure_weight)[:, None] * lower_interp + pressure_weight[:, None] * upper_interp

    sigma_log = jax.vmap(_interpolate_species, in_axes=(0, 0, 0))(sigma_cube, temperature_idx, temperature_weight)
    return 10.0 ** sigma_log


def zero_line_opacity(state: Dict[str, jnp.ndarray], params: Dict[str, jnp.ndarray]):
    layer_pressures = state["p_lay"]
    wavelengths = state["wl"]
    layer_count = jnp.size(layer_pressures)
    wavelength_count = jnp.size(wavelengths)
    return jnp.zeros((layer_count, wavelength_count))


def compute_line_opacity(state: Dict[str, jnp.ndarray], params: Dict[str, jnp.ndarray]):
    layer_pressures = state["p_lay"]
    layer_temperatures = state["T_lay"]
    layer_mu = state["mu_lay"]
    species_names = XS.line_species_names()
    mixing_ratios = jnp.stack([jnp.asarray(params[f"f_{name}"]) for name in species_names])
    sigma_values = _interpolate_sigma(layer_pressures / 1e6, layer_temperatures)
    normalization = mixing_ratios[:, None] / (layer_mu[None, :] * amu)
    return jnp.sum(sigma_values * normalization[:, :, None], axis=0)
