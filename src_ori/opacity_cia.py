"""
opacity_cia.py
==============

Overview:
    Collision-induced absorption (CIA) opacity contribution for the forward
    model.

    This module interpolates CIA cross-sections from `build_opacities` to the
    layer temperature structure and combines them with per-layer number density
    and species mixing ratios to produce a mass opacity (cm^2 g^-1) on the
    forward-model wavelength grid.

Sections to complete:
    - Usage
    - Key Functions
    - Notes

Usage:
    CIA opacity uses the same wavelength grid as the forward model. During a
    forward-model call, pass the shared `state` dictionary to
    `compute_cia_opacity(state, params)` to obtain the CIA contribution.

Key Functions:
    - `compute_cia_opacity`: Computes CIA opacity in cm^2 g^-1.
    - `_interpolate_sigma`: Interpolates (log10) cross-sections on log10(T).
    - `_compute_pair_weight`: Computes per-layer CIA pair weights from VMRs.
    - `zero_cia_opacity`: Convenience helper returning an all-zero array.

Notes:
    CIA species names are expected in `A-B` form (e.g., `H2-He`) where the weight
    is `VMR[A] * VMR[B]` and normalization uses `nd^2 / rho`. The H- continuum is
    handled in `opacity_special.py`.
"""

from typing import Dict

import jax.numpy as jnp
from jax import vmap

import build_opacities as XS


def zero_cia_opacity(state: Dict[str, jnp.ndarray], params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Return a zero CIA opacity array.

    Parameters
    ----------
    state : dict[str, jax.numpy.ndarray]
        State dictionary containing `p_lay` and `wl` to determine output shape.
    params : dict[str, jax.numpy.ndarray]
        Unused; kept for API compatibility.

    Returns
    -------
    jax.numpy.ndarray
        Array of zeros with shape `(nlay, nwl)`.
    """
    layer_pressures = state["p_lay"]
    wavelengths = state["wl"]
    layer_count = jnp.size(layer_pressures)
    wavelength_count = jnp.size(wavelengths)
    return jnp.zeros((layer_count, wavelength_count))


def _interpolate_sigma(layer_temperatures: jnp.ndarray) -> jnp.ndarray:
    """Interpolate CIA cross-sections to layer temperatures.

    Performs linear interpolation in log10(T) using per-species temperature
    grids. Cross-sections are stored in log10 space in the registry and
    converted back to linear space after interpolation.

    Parameters
    ----------
    layer_temperatures : jax.numpy.ndarray, shape `(nlay,)`
        Layer temperatures in K.

    Returns
    -------
    jax.numpy.ndarray
        Cross-sections in linear space with shape `(nspecies, nlay, nwl)`.
    """
    sigma_cube = XS.cia_sigma_cube()
    temperature_grids = XS.cia_temperature_grids()

    # Convert to log10 space for interpolation
    log_t_layers = jnp.log10(layer_temperatures)

    def _interp_one_species(sigma_2d, temp_grid):
        """Interpolate log10 cross-sections for a single CIA species.

        Parameters
        ----------
        sigma_2d : jax.numpy.ndarray
            Log10 cross-sections with shape `(nT, nwl)`.
        temp_grid : jax.numpy.ndarray
            Temperature grid (K) with shape `(nT,)`.

        Returns
        -------
        jax.numpy.ndarray
            Log10 cross-sections interpolated to layers with shape `(nlay, nwl)`.
        """
        # sigma_2d: (n_temp, n_wavelength)
        # temp_grid: (n_temp,)

        # Convert temperature grid to log space
        log_t_grid = jnp.log10(temp_grid)

        # Find temperature bracket indices and weights in log space
        t_idx = jnp.searchsorted(log_t_grid, log_t_layers) - 1
        t_idx = jnp.clip(t_idx, 0, log_t_grid.shape[0] - 2)
        t_weight = (log_t_layers - log_t_grid[t_idx]) / (log_t_grid[t_idx + 1] - log_t_grid[t_idx])
        t_weight = jnp.clip(t_weight, 0.0, 1.0)

        # Get lower and upper temperature brackets
        # Indexing: sigma_2d[temp, wavelength]
        s_t0 = sigma_2d[t_idx, :]          # shape: (n_layers, n_wavelength)
        s_t1 = sigma_2d[t_idx + 1, :]

        # Linear interpolation in temperature
        s_interp = (1.0 - t_weight)[:, None] * s_t0 + t_weight[:, None] * s_t1

        # Set cross sections to very small value below minimum temperature
        min_temp = temp_grid[0]
        below_min = layer_temperatures < min_temp
        tiny = jnp.array(-199.0, dtype=s_interp.dtype)
        s_interp = jnp.where(below_min[:, None], tiny, s_interp)

        return s_interp

    # Vectorize over all CIA species
    sigma_log = vmap(_interp_one_species)(sigma_cube, temperature_grids)
    return 10.0 ** sigma_log


def _compute_pair_weight(
    name: str,
    layer_count: int,
    layer_vmr: Dict[str, jnp.ndarray],
) -> jnp.ndarray:
    """Compute a per-layer CIA pair weight from the VMR mapping.

    Parameters
    ----------
    name : str
        CIA species name in `A-B` form (e.g., `H2-He`).
    layer_count : int
        Number of atmospheric layers (`nlay`).
    layer_vmr : dict[str, jax.numpy.ndarray]
        Mapping from species name to VMR values (scalar or `(nlay,)`).

    Returns
    -------
    jax.numpy.ndarray
        Per-layer weight array with shape `(nlay,)`.

    Raises
    ------
    ValueError
        If `name` is not in `A-B` format.
    KeyError
        If required species keys are missing from `layer_vmr`.
    """
    name_clean = name.strip()

    # Normal CIA pair: "H2-He" -> product of H2 and He VMRs
    parts = name_clean.split("-")
    if len(parts) != 2:
        raise ValueError(f"CIA species '{name}' must be in 'A-B' format")

    species_a, species_b = parts[0], parts[1]
    ratio_a = jnp.asarray(layer_vmr[species_a])
    ratio_b = jnp.asarray(layer_vmr[species_b])
    ratio_a = jnp.broadcast_to(ratio_a, (layer_count,))
    ratio_b = jnp.broadcast_to(ratio_b, (layer_count,))
    return ratio_a * ratio_b



def compute_cia_opacity(state: Dict[str, jnp.ndarray], params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Compute CIA mass opacity.

    Parameters
    ----------
    state : dict[str, jax.numpy.ndarray]
        State dictionary with at least:

        - `nlay` : int-like
            Number of layers.
        - `wl` : jax.numpy.ndarray, shape `(nwl,)`
            Forward-model wavelength grid.
        - `T_lay` : jax.numpy.ndarray, shape `(nlay,)`
            Layer temperatures in K.
        - `nd_lay` : jax.numpy.ndarray, shape `(nlay,)`
            Layer number density.
        - `rho_lay` : jax.numpy.ndarray, shape `(nlay,)`
            Layer mass density.
        - `vmr_lay` : Mapping[str, Any]
            Mapping from species name to VMR values (scalar or `(nlay,)`).
    params : dict[str, jax.numpy.ndarray]
        Unused; kept for API compatibility.

    Returns
    -------
    jax.numpy.ndarray
        CIA opacity in cm^2 g^-1 with shape `(nlay, nwl)`.

    Raises
    ------
    ValueError
        If the CIA wavelength grid does not match `state["wl"]`.
    KeyError
        If a required species name is missing from `state["vmr_lay"]`.
    """
    layer_count = int(state["nlay"])
    wavelengths = state["wl"]
    layer_temperatures = state["T_lay"]
    number_density = state["nd_lay"]   # (nlay,)
    density = state["rho_lay"]         # (nlay,)
    layer_vmr = state["vmr_lay"]

    master_wavelength = XS.cia_master_wavelength()
    if master_wavelength.shape != wavelengths.shape:
        raise ValueError("CIA wavelength grid must match the forward-model master grid.")

    sigma_values = _interpolate_sigma(layer_temperatures)  # (nspecies, nlay, nwl)
    species_names = XS.cia_species_names()
    keep_indices = [i for i, name in enumerate(species_names) if name.strip() != "H-"]
    if not keep_indices:
        return zero_cia_opacity(state, params)

    # Compute pair weights for each CIA species (string ops happen once at trace time)
    pair_weights = jnp.stack(
        [_compute_pair_weight(species_names[i], layer_count, layer_vmr) for i in keep_indices],
        axis=0,
    )  # (nspecies_keep, nlay)

    normalization = pair_weights * (number_density**2 / density)[None, :]  # (nspecies_keep, nlay)
    sigma_keep = sigma_values[keep_indices, :, :]  # (nspecies_keep, nlay, nwl)
    return jnp.sum(normalization[:, :, None] * sigma_keep, axis=0)  # (nlay, nwl)
