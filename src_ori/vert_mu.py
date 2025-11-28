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

from typing import Dict

import jax.numpy as jnp

from data_constants import CHEM_SPECIES_DATA


_SPECIES_MASS = {entry["symbol"]: float(entry["molecular_weight"]) for entry in CHEM_SPECIES_DATA}


def constant_mu(params: Dict[str, jnp.ndarray], nlay: int) -> jnp.ndarray:
    if "mu" not in params:
        raise ValueError("vert_mu='constant' requires a 'mu' parameter.")
    mu_const = jnp.asarray(params["mu"])
    return jnp.full((nlay,), mu_const)


def compute_mu(vmr_lay: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    species_list = sorted(species for species in vmr_lay.keys() if species in _SPECIES_MASS)
    if not species_list:
        raise ValueError("No valid species provided to compute mean molecular weight.")

    vmr_arrays = [jnp.asarray(vmr_lay[sp]) for sp in species_list]
    masses = jnp.array([_SPECIES_MASS[sp] for sp in species_list])
    vmr_stack = jnp.stack(vmr_arrays, axis=0)
    mu_profile = jnp.sum(vmr_stack * masses[:, None], axis=0)
    return mu_profile
