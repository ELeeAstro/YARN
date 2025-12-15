"""
Atmospheric Chemistry Profile Module
=====================================

This module provides functions for computing vertical chemical abundance profiles
(volume mixing ratios) in planetary atmospheres. Various chemistry models are available,
from simple constant abundances to full thermochemical equilibrium calculations.

Functions
---------
constant_vmr : Generate constant volume mixing ratio profiles from parameters
chemical_equilibrium : Placeholder for general chemical equilibrium (not yet implemented)
CE_rate_jax : Compute chemical equilibrium profiles using RateJAX solver
quench_approx : Compute quenched chemical abundance profiles (in development)

Notes
-----
Species Naming Convention:
    Species names in config parameters (after stripping f_ or log_10_f_ prefix)
    must match opacity table species names exactly (including capitalization).
    For example: 'H2O', 'CH4', 'CO2', 'NH3', etc.

Parameter Prefixes:
    - 'f_' : Linear volume mixing ratio (e.g., 'f_H2O' = 1e-4)
    - 'log_10_f_' : Log10 of volume mixing ratio (e.g., 'log_10_f_H2O' = -4.0)

Background Atmosphere:
    H2 and He abundances are automatically computed to fill the remaining atmosphere
    after accounting for trace species, using solar H2/He ratio.

Solar Reference Values:
    Based on Asplund et al. (2021):
    - O/H = 10^(8.69-12.0) = 4.9e-4
    - C/H = 10^(8.46-12.0) = 2.88e-4
    - N/H = 10^(7.83-12.0) = 6.76e-5
    - He/H2 = 10^(10.914-12.0) = 0.082
"""

from __future__ import annotations

from typing import Dict

import jax.numpy as jnp

from rate_jax import RateJAX, get_gibbs_cache

solar_h2 = 0.5
solar_he = 10.0**(10.914-12.0)
solar_h2_he = solar_h2 + solar_he

# Solar reference abundances (relative to H) - Asplund et al. (2021)
solar_O = 10.0**(8.69-12.0)
solar_C = 10.0**(8.46-12.0)
solar_N = 10.0**(7.83-12.0)


def constant_vmr(
    p_lay: jnp.ndarray,
    T_lay: jnp.ndarray,
    params: Dict[str, jnp.ndarray],
    nlay: int,
) -> Dict[str, jnp.ndarray]:
    """
    Generate constant volume mixing ratio profiles from parameters.

    Creates vertically uniform chemical abundance profiles where each species
    has the same VMR throughout all atmospheric layers. H2 and He are automatically
    added to fill the remaining atmosphere using the solar H2/He ratio.

    Parameters
    ----------
    p_lay : jnp.ndarray
        Layer pressures (nlay,) [bar]. Not used but kept for API consistency.
    T_lay : jnp.ndarray
        Layer temperatures (nlay,) [K]. Not used but kept for API consistency.
    params : Dict[str, jnp.ndarray]
        Dictionary of chemical abundance parameters. Keys should be:
        - 'f_{species}' : Linear VMR (e.g., 'f_H2O': 1e-4)
        - 'log_10_f_{species}' : Log10 VMR (e.g., 'log_10_f_CH4': -3.5)
        Species names must match opacity table names exactly.
    nlay : int
        Number of atmospheric layers.

    Returns
    -------
    Dict[str, jnp.ndarray]
        Dictionary mapping species names to VMR profiles.
        Each profile is an array of shape (nlay,).
        Always includes 'H2' and 'He' as background gases.

    Notes
    -----
    The background H2/He mixture is computed as:
        total_trace = sum of all specified trace species VMRs
        background = 1.0 - total_trace
        H2 = background * (solar_H2 / (solar_H2 + solar_He))
        He = background * (solar_He / (solar_H2 + solar_He))

    Species names extracted from parameter keys (after stripping prefixes)
    must match opacity table species names exactly, including capitalization.

    Examples
    --------
    >>> params = {
    ...     'log_10_f_H2O': -4.0,  # 1e-4
    ...     'log_10_f_CH4': -3.5,  # ~3.16e-4
    ...     'f_CO': 1e-5
    ... }
    >>> vmr_profiles = constant_vmr(p_lay, T_lay, params, nlay=50)
    >>> # Returns dict with keys: 'H2O', 'CH4', 'CO', 'H2', 'He'
    """
    del p_lay, T_lay  # unused but kept for consistent signature

    vmr: Dict[str, jnp.ndarray] = {}
    for k, v in params.items():
        if k.startswith("log_10_f_"):
            species = k[len("log_10_f_"):]
            vmr[species] = 10.0 ** jnp.asarray(v)

    trace_values = list(vmr.values())
    if trace_values:
        total_trace_vmr = jnp.sum(jnp.stack(trace_values))
    else:
        total_trace_vmr = jnp.asarray(0.0)
    background_vmr = 1.0 - total_trace_vmr

    vmr["H2"] = background_vmr * solar_h2 / solar_h2_he
    vmr["He"] = background_vmr * solar_he / solar_h2_he

    vmr_lay = {species: jnp.full((nlay,), value) for species, value in vmr.items()}
    return vmr_lay


def build_constant_vmr_kernel(species_order: tuple[str, ...]):
    param_keys = tuple(f"log_10_f_{s}" for s in species_order)

    def _constant_vmr_fixed(p_lay, T_lay, params, nlay):
        del p_lay, T_lay

        values = [10.0 ** jnp.asarray(params[k]) for k in param_keys]
        trace = jnp.stack(values, axis=0) if values else jnp.zeros((0,), dtype=jnp.float32)
        background = 1.0 - jnp.sum(trace) if values else jnp.asarray(1.0)

        vmr = {s: jnp.full((nlay,), trace[i]) for i, s in enumerate(species_order)}
        vmr["H2"] = jnp.full((nlay,), background * (solar_h2 / solar_h2_he))
        vmr["He"] = jnp.full((nlay,), background * (solar_he / solar_h2_he))
        return vmr

    return _constant_vmr_fixed


def chemical_equilibrium(
    p_lay: jnp.ndarray,
    T_lay: jnp.ndarray,
    params: Dict[str, jnp.ndarray],
    nlay: int,
) -> Dict[str, jnp.ndarray]:
    """
    Placeholder for general chemical equilibrium calculation.

    This function is reserved for future implementation of chemical equilibrium
    calculations. Use CE_rate_jax for current thermochemical equilibrium needs.

    Parameters
    ----------
    p_lay : jnp.ndarray
        Layer pressures (nlay,) [bar].
    T_lay : jnp.ndarray
        Layer temperatures (nlay,) [K].
    params : Dict[str, jnp.ndarray]
        Dictionary of chemical parameters (implementation dependent).
    nlay : int
        Number of atmospheric layers.

    Returns
    -------
    Dict[str, jnp.ndarray]
        Dictionary mapping species names to VMR profiles (not implemented).

    Raises
    ------
    NotImplementedError
        This function is not yet implemented.

    See Also
    --------
    CE_rate_jax : Current implementation for chemical equilibrium using RateJAX.
    """
    del p_lay, T_lay, params, nlay
    raise NotImplementedError("chemical_equilibrium is not implemented yet.")


def CE_rate_jax(
    p_lay: jnp.ndarray,
    T_lay: jnp.ndarray,
    params: Dict[str, jnp.ndarray],
    nlay: int,
) -> Dict[str, jnp.ndarray]:
    """
    Chemical equilibrium profiles using RateJAX solver with cached Gibbs data.

    Computes thermochemical equilibrium VMR profiles for 12 species
    (H2O, CH4, CO, CO2, NH3, C2H2, C2H4, HCN, N2, H2, H, He) using
    Gibbs free energy minimization.

    Parameters
    ----------
    p_lay : array, shape (nlay,)
        Layer pressures [bar]
    T_lay : array, shape (nlay,)
        Layer temperatures [K]
    params : dict
        Chemical abundance parameters:
        - 'M/H': Metallicity [dex], log10 scale factor relative to solar (default: 0.0)
        - 'C/O': Carbon to oxygen ratio (default: 0.5, solar)
    nlay : int
        Number of atmospheric layers (unused but kept for API compatibility)

    Returns
    -------
    vmr_lay : dict[str, array]
        Dictionary mapping species names to VMR profiles, shape (nlay,)
        Species: H2O, CH4, CO, CO2, NH3, C2H2, C2H4, HCN, N2, H2, H, He

    Raises
    ------
    RuntimeError
        If Gibbs cache has not been initialized with load_gibbs_cache()

    Notes
    -----
    - Requires Gibbs cache to be pre-loaded via load_gibbs_cache()
    - Uses Lodders & Fegley (2002) polynomial fits for chemistry regimes
    - Automatically selects HCO or HCNO chemistry based on conditions
    - Includes H2 dissociation equilibrium at high temperatures
    - GPU-compatible when JIT-compiled
    - Helium fraction is fixed at solar value (fHe = 0.085114)

    Abundance Scaling:
        Solar reference values (relative to H2):
            O_solar = 5.0e-4
            C_solar = 2.5e-4
            N_solar = 1.0e-4
            Solar C/O = 0.5

        Conversion from M/H and C/O:
            O = O_solar × 10^[M/H]
            C = (C/O) × O
            N = N_solar × 10^[M/H]

    Examples
    --------
    >>> # First, load Gibbs cache (once during initialization):
    >>> from rate_jax import load_gibbs_cache
    >>> load_gibbs_cache("JANAF_data/")
    >>>
    >>> # Then use in forward model:
    >>> # Solar metallicity, solar C/O
    >>> params = {'M/H': 0.0, 'C/O': 0.5}
    >>> vmr = CE_rate_jax(p_lay, T_lay, params, nlay)
    >>>
    >>> # Super-solar metallicity, high C/O (carbon-rich)
    >>> params = {'M/H': 0.5, 'C/O': 0.9}
    >>> vmr = CE_rate_jax(p_lay, T_lay, params, nlay)
    """
    del nlay  # Unused but kept for API compatibility with other vert_chem functions

    # Get cached Gibbs tables (will raise RuntimeError if not loaded)
    gibbs = get_gibbs_cache()

    # Extract metallicity and C/O ratio from params (keep as JAX arrays for JIT compatibility)
    metallicity = params.get('M/H', 0.0)  # [dex]
    CO_ratio = params.get('C/O', solar_C/solar_O)  # dimensionless

    # Convert M/H and C/O to elemental abundances
    # Scale oxygen and nitrogen by metallicity
    O = solar_O * (10.0 ** metallicity)
    N = solar_N * (10.0 ** metallicity)

    # Carbon set by C/O ratio
    C = CO_ratio * O

    # Create RateJAX solver
    rate = RateJAX(gibbs=gibbs, C=C, N=N, O=O, fHe=solar_he)

    # Solve chemical equilibrium profile
    vmr_lay = rate.solve_profile(T_lay, p_lay/1e6)

    return vmr_lay


def quench_approx(
    p_lay: jnp.ndarray,
    T_lay: jnp.ndarray,
    params: Dict[str, jnp.ndarray],
    nlay: int,
) -> Dict[str, jnp.ndarray]:
    """
    Compute quenched chemical abundance profiles (in development).

    Calculates chemical abundances with quenching effects, where certain species
    become "frozen in" at their high-temperature equilibrium values as gas parcels
    are transported upward. This accounts for finite chemical reaction timescales.

    Parameters
    ----------
    p_lay : jnp.ndarray
        Layer pressures (nlay,) [bar].
    T_lay : jnp.ndarray
        Layer temperatures (nlay,) [K].
    params : Dict[str, jnp.ndarray]
        Dictionary of chemical parameters:
        - 'M/H' : Metallicity [dex], log10 scale factor relative to solar.
        - 'C/O' : Carbon to oxygen ratio.
        Additional quenching parameters to be determined.
    nlay : int
        Number of atmospheric layers.

    Returns
    -------
    Dict[str, jnp.ndarray]
        Dictionary mapping species names to quenched VMR profiles (nlay,).
        Species: H2O, CH4, CO, CO2, NH3, C2H2, C2H4, HCN, N2, H2, H, He.

    Raises
    ------
    RuntimeError
        If Gibbs cache has not been initialized with load_gibbs_cache().

    Notes
    -----
    WARNING: This function is currently in development and incomplete.
    The quenching calculation is not yet implemented - currently only computes
    chemical equilibrium profiles without applying quenching.

    Quenching typically affects species like:
    - CO/CH4 ratio (quenched at T ~ 1000-1500 K)
    - NH3 (quenched at similar temperatures)
    - HCN and other nitrogen-bearing species

    The quench level depends on:
    - Chemical reaction timescales
    - Vertical mixing timescales (Kzz profile)
    - Temperature-pressure profile

    See Also
    --------
    CE_rate_jax : Chemical equilibrium without quenching (currently functional).
    """

    # Get cached Gibbs tables (will raise RuntimeError if not loaded)
    gibbs = get_gibbs_cache()

    # Extract metallicity and C/O ratio from params (keep as JAX arrays for JIT compatibility)
    metallicity = params.get('M/H', 0.0)  # [dex]
    CO_ratio = params.get('C/O', solar_C/solar_O)  # dimensionless

    # Convert M/H and C/O to elemental abundances
    # Scale oxygen and nitrogen by metallicity
    O = solar_O * (10.0 ** metallicity)
    N = solar_N * (10.0 ** metallicity)

    # Carbon set by C/O ratio
    C = CO_ratio * O

    # Create RateJAX solver
    rate = RateJAX(gibbs=gibbs, C=C, N=N, O=O, fHe=solar_he)

    # Solve chemical equilibrium profile
    vmr_lay = rate.solve_profile(T_lay, p_lay/1e6)

    # Now solve the quench level for each quenched species - first calculate the chemical timescale of each species
    # For vectorisation reasons, just do the full profile


    Species = [
            "H2O", "CH4", "CO", "CO2", "NH3",
            "C2H2", "C2H4", "HCN", "N2",
            "H2", "H", "He",
    ]

    return vmr_lay
