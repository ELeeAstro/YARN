"""
Mean Molecular Weight Module
=============================

This module provides functions for computing and managing mean molecular weight (mu)
profiles in atmospheric models. The mean molecular weight is essential for converting
between pressure, altitude, and for computing atmospheric scale heights.

Functions
---------
constant_mu : Generate a constant mean molecular weight profile
compute_mu : Compute mean molecular weight from volume mixing ratios

Notes
-----
Mean molecular weight (mu) is measured in atomic mass units (amu) or g/mol.
For Earth's atmosphere, mu ≈ 28.97 g/mol.
For gas giant atmospheres, mu can vary significantly with composition
(e.g., H2/He-dominated atmospheres have mu ≈ 2.3 g/mol).
"""

from __future__ import annotations

from typing import Dict

import jax.numpy as jnp

from data_constants import CHEM_SPECIES_DATA


_SPECIES_MASS = {entry["symbol"]: float(entry["molecular_weight"]) for entry in CHEM_SPECIES_DATA}


def constant_mu(params: Dict[str, jnp.ndarray], nlay: int) -> jnp.ndarray:
    """
    Generate a constant mean molecular weight profile.

    Creates a vertically uniform mean molecular weight profile where all
    atmospheric layers have the same value. This is appropriate for well-mixed
    atmospheres or as a simple approximation.

    Parameters
    ----------
    params : Dict[str, jnp.ndarray]
        Dictionary containing:
        - 'mu' : Mean molecular weight [amu or g/mol].
    nlay : int
        Number of atmospheric layers.

    Returns
    -------
    jnp.ndarray
        Mean molecular weight profile (nlay,) [amu or g/mol].

    Raises
    ------
    ValueError
        If 'mu' parameter is not provided in params dictionary.

    Examples
    --------
    >>> params = {'mu': 2.3}  # H2/He-dominated atmosphere
    >>> mu_profile = constant_mu(params, nlay=50)
    """
    if "mu" not in params:
        raise ValueError("vert_mu='constant' requires a 'mu' parameter.")
    mu_const = jnp.asarray(params["mu"])
    return jnp.full((nlay,), mu_const)


def compute_mu(vmr_lay: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """
    Compute mean molecular weight from volume mixing ratios.

    Calculates the layer-by-layer mean molecular weight based on the chemical
    composition (volume mixing ratios) of atmospheric species. The computation
    uses molecular weights from the CHEM_SPECIES_DATA registry.

    Parameters
    ----------
    vmr_lay : Dict[str, jnp.ndarray]
        Dictionary mapping species symbols to their volume mixing ratio profiles.
        Each entry should be an array of shape (nlay,).
        Species symbols must match those in CHEM_SPECIES_DATA.

    Returns
    -------
    jnp.ndarray
        Mean molecular weight profile (nlay,) [amu or g/mol].

    Raises
    ------
    ValueError
        If no valid species (matching CHEM_SPECIES_DATA) are provided.

    Notes
    -----
    The mean molecular weight is computed as:
        mu = sum(VMR_i * M_i)
    where VMR_i is the volume mixing ratio and M_i is the molecular weight
    of species i.

    Only species present in both vmr_lay and _SPECIES_MASS (derived from
    CHEM_SPECIES_DATA) are included in the calculation.

    Examples
    --------
    >>> vmr_lay = {
    ...     'H2': jnp.full(50, 0.85),
    ...     'He': jnp.full(50, 0.15)
    ... }
    >>> mu_profile = compute_mu(vmr_lay)
    """
    species_list = sorted(species for species in vmr_lay.keys() if species in _SPECIES_MASS)
    if not species_list:
        raise ValueError("No valid species provided to compute mean molecular weight.")

    vmr_arrays = [jnp.asarray(vmr_lay[sp]) for sp in species_list]
    masses = jnp.array([_SPECIES_MASS[sp] for sp in species_list])
    vmr_stack = jnp.stack(vmr_arrays, axis=0)
    mu_profile = jnp.sum(vmr_stack * masses[:, None], axis=0)
    return mu_profile
