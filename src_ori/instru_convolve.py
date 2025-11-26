from __future__ import annotations

import jax
import jax.numpy as jnp

# Adjust the import path/module name to wherever you placed instru_bandpass.py
from registry_bandpass import (
    bandpass_num_bins,
    bandpass_wavelengths_padded,
    bandpass_weights_padded,
    bandpass_indices_padded,
    bandpass_norms,
)


def apply_response_functions(
    wl: jnp.ndarray,          # currently unused, kept for API consistency
    spectrum: jnp.ndarray,
) -> jnp.ndarray:
    """
    Convolve hi-res spectrum with per-bin response functions, using the
    pre-built bandpass registry and padded JAX arrays.
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

    def _convolve_one_spectrum(spec: jnp.ndarray) -> jnp.ndarray:
        """
        Jitted core that assumes bandpass_* arrays are fixed and already built.
        """

        def convolve_bin(carry, i):
            # i is a JAX scalar; indexing along axis 0 is JAX-friendly
            idx_row = idx_pad[i]          # (max_len,)
            wl_row = wl_pad[i]            # (max_len,)
            w_row = w_pad[i]              # (max_len,)

            spec_slice = spec[idx_row]    # (max_len,)
            numerator = jnp.trapezoid(spec_slice * w_row, x=wl_row)
            value = numerator / jnp.maximum(norms[i], 1e-99)
            return carry, value

        _, binned = jax.lax.scan(convolve_bin, None, jnp.arange(n_bins))
        return binned

    return _convolve_one_spectrum(spectrum)
