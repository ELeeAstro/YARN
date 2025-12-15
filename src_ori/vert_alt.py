"""
Altitude Profile Module
=======================

This module provides functions for computing vertical altitude profiles from
pressure, temperature, and mean molecular weight using the hypsometric equation
(barometric formula). Various implementations account for different levels of
physical complexity.

Functions
---------
hypsometric : Basic hypsometric integration with constant gravity
g_at_z : Compute gravity as a function of altitude
hypsometric_variable_g : Hypsometric integration with altitude-dependent gravity
hypsometric_variable_g_pref : Hypsometric integration anchored at reference pressure

Notes
-----
The Hypsometric Equation:
    The fundamental relationship between pressure and altitude in a hydrostatic
    atmosphere is given by:
        dz/dp = -H/p = -(kB * T) / (mu * amu * g * p)
    where H is the pressure scale height.

Integration Methods:
    - hypsometric: Assumes constant g throughout the atmosphere (simplest)
    - hypsometric_variable_g: Accounts for g(z) = g_ref * (R0/(R0+z))^2
    - hypsometric_variable_g_pref: Same as variable_g but anchors at arbitrary
      reference pressure level (useful for retrievals)

Physical Constants:
    All physical constants (kb, amu, R_jup, bar) are imported from data_constants.

Units:
    - Altitude: typically in cm or m (depends on R_jup units)
    - Pressure: bar
    - Temperature: K
    - Gravity: cm/s^2
    - Mean molecular weight: amu (g/mol)
"""

from __future__ import annotations

from typing import Dict

import jax
import jax.numpy as jnp

from data_constants import amu, kb, R_jup, bar


def hypsometric(p_lev, T_lay, mu_lay, params):
    """
    Compute altitude profile using the hypsometric equation with constant gravity.

    This is the simplest altitude integration method, assuming gravity is constant
    throughout the atmosphere. Suitable for thin atmospheres or when high accuracy
    is not required.

    Parameters
    ----------
    p_lev : jnp.ndarray
        Pressure at levels (nlev,) [bar]. Ordered from high to low pressure
        (bottom to top of atmosphere).
    T_lay : jnp.ndarray
        Layer temperatures (nlev-1,) [K].
    mu_lay : jnp.ndarray
        Mean molecular weight at layers (nlev-1,) [amu or g/mol].
    params : Dict[str, jnp.ndarray]
        Dictionary containing:
        - 'log_10_g' : Log10 of surface gravity [cm/s^2].

    Returns
    -------
    z_lev : jnp.ndarray
        Altitude at levels (nlev,) [cm]. Zero at bottom level.
    z_lay : jnp.ndarray
        Altitude at layer midpoints (nlev-1,) [cm].
    dz : jnp.ndarray
        Layer thickness (nlev-1,) [cm].

    Notes
    -----
    The pressure scale height is computed as:
        H = (kB * T) / (mu * amu * g)

    Altitude differences are then:
        dz = H * ln(p_lower / p_upper)

    The integration starts at z=0 at the bottom level and progresses upward.

    Examples
    --------
    >>> params = {'log_10_g': 3.5}  # log10(g) = 3.5, so g ≈ 3162 cm/s^2
    >>> z_lev, z_lay, dz = hypsometric(p_lev, T_lay, mu_lay, params)
    """
    g_ref = 10.0**jnp.asarray(params["log_10_g"])
    H = (kb * T_lay) / (mu_lay * amu * g_ref)
    dlnp = jnp.log(p_lev[:-1] / p_lev[1:])
    dz = H * dlnp
    z0 = jnp.zeros_like(p_lev[:1])

    z_lev = jnp.concatenate([z0, jnp.cumsum(dz)])
    z_lay = (z_lev[:-1] + z_lev[1:]) / 2.0

    return z_lev, z_lay, dz


def g_at_z(R0, z, g_ref):
    """
    Compute gravity at a given altitude above the planet's reference level.

    Gravity decreases with altitude according to the inverse square law.
    This function is used in altitude integration schemes that account for
    variable gravity.

    Parameters
    ----------
    R0 : float
        Reference radius of the planet [cm]. Typically the radius at a
        reference pressure level (e.g., 1 bar or 10 bar).
    z : float or jnp.ndarray
        Altitude above the reference level [cm].
    g_ref : float
        Reference surface gravity at R0 [cm/s^2].

    Returns
    -------
    float or jnp.ndarray
        Gravity at altitude z [cm/s^2].

    Notes
    -----
    The gravity variation follows:
        g(z) = g_ref * (R0 / (R0 + z))^2

    This accounts for the fact that gravity decreases with distance from
    the planet's center according to Newton's law of gravitation.

    For planets with thick atmospheres (e.g., gas giants), this correction
    can be significant over the observable atmosphere.

    Examples
    --------
    >>> R0 = 7.0e9  # 70,000 km radius
    >>> g_ref = 2500.0  # cm/s^2
    >>> z = 1.0e8  # 1000 km altitude
    >>> g = g_at_z(R0, z, g_ref)
    """
    return g_ref * (R0 / (R0 + z)) ** 2

def hypsometric_variable_g(p_lev, T_lay, mu_lay, params):
    """
    Compute altitude profile with altitude-dependent gravity using predictor-corrector.

    This method improves upon the basic hypsometric integration by accounting for
    the variation of gravity with altitude: g(z) = g_ref * (R0/(R0+z))^2.
    Uses a predictor-corrector scheme for better accuracy.

    Parameters
    ----------
    p_lev : jnp.ndarray
        Pressure at levels (nlev,) [bar]. Ordered from high to low pressure
        (bottom to top of atmosphere).
    T_lay : jnp.ndarray
        Layer temperatures (nlev-1,) [K].
    mu_lay : jnp.ndarray
        Mean molecular weight at layers (nlev-1,) [amu or g/mol].
    params : Dict[str, jnp.ndarray]
        Dictionary containing:
        - 'log_10_g' : Log10 of reference surface gravity [cm/s^2].
        - 'R_p' : Planetary radius at reference level [R_jup units].

    Returns
    -------
    z_lev : jnp.ndarray
        Altitude at levels (nlev,) [cm]. Zero at bottom level.
    z_lay : jnp.ndarray
        Altitude at layer midpoints (nlev-1,) [cm].
    dz : jnp.ndarray
        Layer thickness (nlev-1,) [cm].

    Notes
    -----
    Algorithm:
        For each layer:
        1. Predictor: Estimate dz using gravity at current level
        2. Corrector: Refine dz using gravity at predicted mid-layer altitude

    This predictor-corrector approach provides better accuracy than using
    gravity at just the lower level, especially for thick atmospheric layers.

    The integration starts at z=0 at the bottom level (highest pressure) and
    progresses upward through decreasing pressure.

    Important for gas giant planets where atmospheric scale heights can be
    significant compared to the planetary radius.

    Examples
    --------
    >>> params = {
    ...     'log_10_g': 3.5,  # g = 3162 cm/s^2
    ...     'R_p': 1.0  # 1 Jupiter radius
    ... }
    >>> z_lev, z_lay, dz = hypsometric_variable_g(p_lev, T_lay, mu_lay, params)
    """

    g_ref = 10.0**jnp.asarray(params["log_10_g"])
    R0 = jnp.asarray(params["R_p"]) * R_jup

    dlnp = jnp.log(p_lev[:-1] / p_lev[1:])

    def step(z_current, inputs):
        T_i, mu_i, dlnp_i = inputs

        # Predictor using g at current level
        g_i = g_at_z(R0, z_current, g_ref)
        H_i = (kb * T_i) / (mu_i * amu * g_i)
        dz_pred = H_i * dlnp_i

        # Corrector using g at predicted mid-layer altitude
        z_mid = z_current + 0.5 * dz_pred
        g_mid = g_at_z(R0, z_mid, g_ref)
        H_mid = (kb * T_i) / (mu_i * amu * g_mid)
        dz_i = H_mid * dlnp_i

        return z_current + dz_i, dz_i

    z0 = jnp.zeros((), dtype=p_lev.dtype)
    _, dz = jax.lax.scan(step, z0, (T_lay, mu_lay, dlnp))
    z_lev = jnp.concatenate([jnp.zeros((1,), dtype=dz.dtype), jnp.cumsum(dz)])
    z_lay = 0.5 * (z_lev[:-1] + z_lev[1:])
    return z_lev, z_lay, dz


def hypsometric_variable_g_pref(p_lev, T_lay, mu_lay, params):
    """
    Compute altitude profile with variable gravity anchored at reference pressure.

    This method performs hypsometric integration with altitude-dependent gravity,
    but anchors the zero-altitude point at an arbitrary reference pressure level
    rather than at the bottom of the atmosphere. This is particularly useful for
    atmospheric retrievals where the planetary radius is defined at a specific
    pressure (e.g., 1 bar or 10 bar).

    Parameters
    ----------
    p_lev : jnp.ndarray
        Pressure at levels (nlev,) [bar]. Ordered from high to low pressure
        (bottom to top of atmosphere).
    T_lay : jnp.ndarray
        Layer temperatures (nlev-1,) [K].
    mu_lay : jnp.ndarray
        Mean molecular weight at layers (nlev-1,) [amu or g/mol].
    params : Dict[str, jnp.ndarray]
        Dictionary containing:
        - 'log_10_g' : Log10 of reference gravity at p_ref [cm/s^2].
        - 'R_p' : Planetary radius at p_ref level [R_jup units].
        - 'log_10_p_ref' : Log10 of reference pressure level [bar].

    Returns
    -------
    z_lev : jnp.ndarray
        Altitude at levels (nlev,) [cm]. Zero at the reference pressure level.
    z_lay : jnp.ndarray
        Altitude at layer midpoints (nlev-1,) [cm].
    dz : jnp.ndarray
        Layer thickness (nlev-1,) [cm]. Positive values indicate upward layers.

    Notes
    -----
    Algorithm:
        1. Find the layer containing p_ref
        2. Integrate partial steps from p_ref to bracketing levels
        3. Integrate upward (toward lower pressures) from upper bracket
        4. Integrate downward (toward higher pressures) from lower bracket

    The reference pressure p_ref is automatically clipped to lie within the
    pressure grid bounds. If p_ref falls outside the grid, it uses the
    nearest boundary.

    This function uses a predictor-corrector scheme for each integration step,
    similar to hypsometric_variable_g, but with bidirectional integration from
    the reference point.

    Useful for retrievals where:
        - Radius is constrained at a specific pressure (e.g., transit radius at 10 mbar)
        - Gravity is known at a particular pressure level
        - You want altitude = 0 at a meaningful reference level

    Examples
    --------
    >>> params = {
    ...     'log_10_g': 3.5,      # g = 3162 cm/s^2 at p_ref
    ...     'R_p': 1.0,           # R = 1 R_jup at p_ref
    ...     'log_10_p_ref': 0.0   # p_ref = 1 bar
    ... }
    >>> z_lev, z_lay, dz = hypsometric_variable_g_pref(p_lev, T_lay, mu_lay, params)
    >>> # z_lev will be zero at the level closest to 1 bar
    """
    g_ref = 10.0**jnp.asarray(params["log_10_g"])
    R0 = jnp.asarray(params["R_p"]) * R_jup
    p_ref = 10.0**jnp.asarray(params["log_10_p_ref"]) * bar

    nlev = p_lev.shape[0]
    dlnp = jnp.log(p_lev[:-1] / p_lev[1:])

    # Ensure p_ref lies within the grid bounds
    p_ref = jnp.clip(p_ref, p_lev[-1], p_lev[0])

    # Locate the layer whose bounds encompass p_ref
    mask = p_lev >= p_ref
    ref_layer = jnp.sum(mask.astype(jnp.int32)) - 1
    ref_layer = jnp.clip(ref_layer, 0, nlev - 2)

    z_lev = jnp.zeros_like(p_lev)

    def integrate_segment(layer_idx, z_start, delta_ln, direction):
        T = T_lay[layer_idx]
        mu = mu_lay[layer_idx]
        g_i = g_at_z(R0, z_start, g_ref)
        H_i = (kb * T) / (mu * amu * g_i)
        direction = jnp.asarray(direction, dtype=z_start.dtype)
        dz_pred = direction * H_i * delta_ln
        z_mid = z_start + 0.5 * dz_pred
        g_mid = g_at_z(R0, z_mid, g_ref)
        H_mid = (kb * T) / (mu * amu * g_mid)
        dz_val = direction * H_mid * delta_ln
        return z_start + dz_val

    # Partial step from p_ref to the bracketing levels (if needed)
    delta_down = jnp.maximum(jnp.log(p_lev[ref_layer] / p_ref), 0.0)
    z_lower = integrate_segment(ref_layer, 0.0, delta_down, -1.0)
    z_lev = z_lev.at[ref_layer].set(z_lower)

    has_upper = ref_layer + 1 < nlev
    delta_up = jnp.maximum(jnp.log(p_ref / p_lev[ref_layer + 1]), 0.0)

    def set_upper(z_vals):
        z_upper = integrate_segment(ref_layer, 0.0, delta_up, 1.0)
        return z_vals.at[ref_layer + 1].set(z_upper)

    z_lev = jax.lax.cond(has_upper, set_upper, lambda z_vals: z_vals, z_lev)

    # Integrate toward lower pressures (upward in altitude)
    def body_up(i, z_vals):
        use_layer = jnp.logical_and(i >= (ref_layer + 1), i < nlev - 1)

        def update(z_arr):
            z_start = z_arr[i]
            delta = dlnp[i]
            z_next = integrate_segment(i, z_start, delta, 1.0)
            return z_arr.at[i + 1].set(z_next)

        return jax.lax.cond(use_layer, update, lambda z_arr: z_arr, z_vals)

    z_lev = jax.lax.fori_loop(0, nlev - 1, body_up, z_lev)

    # Integrate toward higher pressures (downward in altitude)
    def body_down(i, z_vals):
        idx = ref_layer - 1 - i
        use_layer = idx >= 0

        def update(z_arr):
            z_start = z_arr[idx + 1]
            delta = dlnp[idx]
            z_next = integrate_segment(idx, z_start, delta, -1.0)
            return z_arr.at[idx].set(z_next)

        return jax.lax.cond(use_layer, update, lambda z_arr: z_arr, z_vals)

    z_lev = jax.lax.fori_loop(0, nlev - 1, body_down, z_lev)

    dz = z_lev[1:] - z_lev[:-1]
    z_lay = 0.5 * (z_lev[:-1] + z_lev[1:])
    return z_lev, z_lay, dz
