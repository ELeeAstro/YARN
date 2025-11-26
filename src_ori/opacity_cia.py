"""
opacity_cia.py
==============

Overview:
    TODO: Describe the purpose and responsibilities of this module.

Sections to complete:
    - Usage
    - Key Functions
    - Notes
"""

from typing import Dict, Tuple

import jax.numpy as jnp
from jax import vmap

import build_opacities as XS


def zero_cia_opacity(state: Dict[str, jnp.ndarray], params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    layer_pressures = state["p_lay"]
    wavelengths = state["wl"]
    layer_count = jnp.size(layer_pressures)
    wavelength_count = jnp.size(wavelengths)
    return jnp.zeros((layer_count, wavelength_count))


_CIA_SIGMA_CACHE: jnp.ndarray | None = None


def _load_cia_sigma() -> jnp.ndarray:
    global _CIA_SIGMA_CACHE
    if _CIA_SIGMA_CACHE is None:
        _CIA_SIGMA_CACHE = XS.cia_sigma_cube()
    return _CIA_SIGMA_CACHE


def _compute_linear_weights(grid: jnp.ndarray, targets: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    idx = jnp.searchsorted(grid, targets, side="right") - 1
    idx = jnp.clip(idx, 0, grid.size - 2)
    lower = jnp.take(grid, idx)
    upper = jnp.take(grid, idx + 1)
    weight = jnp.where(upper > lower, (targets - lower) / (upper - lower), 0.0)
    return idx, jnp.clip(weight, 0.0, 1.0)


def _interpolate_sigma(layer_temperatures: jnp.ndarray) -> jnp.ndarray:
    sigma_cube = _load_cia_sigma()
    temperature_grids = XS.cia_temperature_grids()
    idx, weight = vmap(_compute_linear_weights, in_axes=(0, None))(temperature_grids, layer_temperatures)
    idx_expanded = idx[:, :, None]
    sigma_lower = jnp.take_along_axis(sigma_cube, idx_expanded, axis=1)
    sigma_upper = jnp.take_along_axis(sigma_cube, idx_expanded + 1, axis=1)
    weight_expanded = weight[:, :, None]
    sigma_interp = (1.0 - weight_expanded) * sigma_lower + weight_expanded * sigma_upper
    min_temperature = temperature_grids[:, 0]
    mask = layer_temperatures[None, :] < min_temperature[:, None]
    tiny = jnp.array(-199, dtype=sigma_interp.dtype)
    return 10.0**jnp.where(mask[:, :, None], tiny, sigma_interp)


def _is_hminus(name: str) -> bool:
    s = str(name).strip().lower().replace(" ", "")
    return s in {"h-", "h−", "hminus", "hm", "hminusion"}

def _compute_pair_weight(
    name: str,
    params: Dict[str, jnp.ndarray],
    layer_count: int,
    mixing_ratios: Dict[str, jnp.ndarray],
) -> jnp.ndarray:
    # --- special case: H- bound-free continuum uses only f_Hminus ---
    if _is_hminus(name):
        # accept a few common spellings for convenience
        for key in ("f_Hminus", "f_hminus", "f_H-", "f_h-", "f_Hm", "f_hm"):
            if key in params:
                w = jnp.asarray(params[key])
                return jnp.broadcast_to(w, (layer_count,))
        raise KeyError(
            "Missing H- abundance parameter. Provide one of: "
            "params['f_Hminus'] (preferred), or params['f_H-'], params['f_Hm']."
        )

    # --- normal CIA: requires 'A-B' and uses product of mixing ratios ---
    parts = name.split("-")
    if len(parts) != 2 or (parts[0] == "") or (parts[1] == ""):
        raise ValueError(f"CIA species name '{name}' must be of form 'A-B' (or 'H-' special case).")
    species_a, species_b = parts

    def _resolve_ratio(species: str) -> jnp.ndarray:
        if species in mixing_ratios:
            return jnp.asarray(mixing_ratios[species])
        key = f"f_{species}"
        if key in params:
            return jnp.asarray(params[key])
        raise KeyError(f"Missing CIA mixing parameter for '{key}'")

    ratio_a = jnp.broadcast_to(_resolve_ratio(species_a), (layer_count,))
    ratio_b = jnp.broadcast_to(_resolve_ratio(species_b), (layer_count,))
    return ratio_a * ratio_b



def compute_cia_opacity(state: Dict[str, jnp.ndarray], params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    layer_count = int(state["nlay"])
    wavelengths = state["wl"]
    layer_temperatures = state["T_lay"]
    number_density = state["nd"]   # (nlay,)
    density = state["rho"]         # (nlay,)

    master_wavelength = XS.cia_master_wavelength()
    if master_wavelength.shape != wavelengths.shape:
        raise ValueError("CIA wavelength grid must match the forward-model master grid.")

    sigma_values = _interpolate_sigma(layer_temperatures)  # (nspecies, nlay, nwl) presumably
    species_names = XS.cia_species_names()
    mixing_ratios = state.get("mixing_ratios", {})

    pair_weights = jnp.stack(
        [_compute_pair_weight(name, params, layer_count, mixing_ratios) for name in species_names],
        axis=0,
    )  # (nspecies, nlay)

    density = jnp.where(density == 0.0, jnp.inf, density)

    # - CIA pairs:   nd^2 / rho
    # - H-:          nd / rho
    is_hm = jnp.asarray([_is_hminus(n) for n in species_names])[:, None]  # (nspecies, 1) bool

    norm_cia = (number_density ** 2 / density)[None, :]   # (1, nlay)
    norm_hm  = (number_density / density)[None, :]        # (1, nlay)

    normalization = pair_weights * jnp.where(is_hm, norm_hm, norm_cia)  # (nspecies, nlay)

    return jnp.sum(normalization[:, :, None] * sigma_values, axis=0)  # (nlay, nwl)
