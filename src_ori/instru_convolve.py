"""
instru_convolve.py
==================

Overview:
    Instrument-response convolution utilities for mapping a high-resolution
    forward-model spectrum onto observed wavelength bins.

    The bin response functions (boxcar edges or bandpass throughputs) are
    prepared once and stored in the `registry_bandpass` module as padded JAX
    arrays. This module performs the final weighted trapezoidal integration to
    produce the binned spectrum.

Sections to complete:
    - Usage
    - Key Functions
    - Notes

Usage:
    Call `apply_response_functions(wl, spectrum)` after generating a high-res
    model spectrum on the master wavelength grid. The bin definitions and
    weights are read from `registry_bandpass`.

Key Functions:
    - `apply_response_functions`: Public entry point used by the forward model.
    - `_convolve_spectrum_core`: JIT-compiled core that performs the batched
      weighted trapezoidal integrations for all bins.

Notes:
    The `wl` argument is currently unused because bin response functions are
    pre-built against the master wavelength grid; it is kept for API
    consistency and potential future validation.
"""

from __future__ import annotations

import jax
from jax import jit
import jax.numpy as jnp

# Adjust the import path/module name to wherever you placed instru_bandpass.py
from registry_bandpass import (
    bandpass_num_bins,
    bandpass_wavelengths_padded,
    bandpass_weights_padded,
    bandpass_indices_padded,
    bandpass_norms,
)

@jit
def _convolve_spectrum_core(
    spec: jnp.ndarray,
    wl_pad: jnp.ndarray,
    w_pad: jnp.ndarray,
    idx_pad: jnp.ndarray,
    norms: jnp.ndarray,
) -> jnp.ndarray:
    """Convolve a single spectrum into observation bins (JIT core).

    Parameters
    ----------
    spec : jax.numpy.ndarray, shape `(nwl_hi,)`
        High-resolution spectrum evaluated on the master wavelength grid.
    wl_pad : jax.numpy.ndarray, shape `(nbin, max_len)`
        Per-bin wavelength samples (padded).
    w_pad : jax.numpy.ndarray, shape `(nbin, max_len)`
        Per-bin response weights/throughputs (padded).
    idx_pad : jax.numpy.ndarray, shape `(nbin, max_len)`
        Per-bin indices into `spec` for the corresponding wavelength samples.
    norms : jax.numpy.ndarray, shape `(nbin,)`
        Per-bin normalization factors (e.g., integral of throughput).

    Returns
    -------
    jax.numpy.ndarray
        Binned spectrum with shape `(nbin,)`.

    Notes
    -----
    This function is pure and JIT-friendly: all inputs are explicit JAX arrays.
    """
    spec_pad = jnp.take(spec, idx_pad, axis=0)  # (n_bins, max_len)
    numerator = jnp.trapezoid(spec_pad * w_pad, x=wl_pad, axis=1)
    return numerator / jnp.maximum(norms, 1e-99)


def apply_response_functions(
    wl: jnp.ndarray,          # currently unused, kept for API consistency
    spectrum: jnp.ndarray,
) -> jnp.ndarray:
    """Apply per-bin response functions to a high-resolution spectrum.

    Parameters
    ----------
    wl : jax.numpy.ndarray
        High-resolution wavelength grid. Currently unused; kept for API
        consistency with older code paths.
    spectrum : jax.numpy.ndarray, shape `(nwl_hi,)`
        High-resolution model spectrum on the master wavelength grid.

    Returns
    -------
    jax.numpy.ndarray
        Binned spectrum with shape `(nbin,)`. If no bins are prepared, returns
        an empty array with dtype matching `spectrum`.
    """

    n_bins = bandpass_num_bins()
    if n_bins == 0:
        # No bins prepared; return empty array of the right dtype
        return jnp.zeros((0,), dtype=spectrum.dtype)

    # Fetch the JAX arrays once, outside the jitted core
    wl_pad = bandpass_wavelengths_padded()   # (n_bins, max_len)
    w_pad = bandpass_weights_padded()        # (n_bins, max_len)
    idx_pad = bandpass_indices_padded()      # (n_bins, max_len)
    norms = bandpass_norms()                 # (n_bins,)

    return _convolve_spectrum_core(
        spec=spectrum,
        wl_pad=wl_pad,
        w_pad=w_pad,
        idx_pad=idx_pad,
        norms=norms,
    )
