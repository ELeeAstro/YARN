"""
read_obs.py
===========

Overview:
   Reads in the observation data for the retrieval 

    - Usage
    - Key Functions
    - Notes
"""

import numpy as np

def _load_columns(path: str) -> np.ndarray:
    '''
      Input: path to observational data file
      Output: raw data from the data file
    '''

    raw = np.genfromtxt(path, dtype=str, comments="#", autostrip=True)
    if raw.ndim == 1:
        raw = raw[None, :]

    return raw


def read_obs_data(path):
    '''
      Input: path to observational data file
      Output: Dictionary containing observational data
    '''

    print(f"[info] Reading observational data from: {path}")

    raw = _load_columns(path)
    if raw.shape[1] < 4:
        raise ValueError(f"[error] Observational file '{path}' must have at least four columns (wl,dwl,y,dy).")

    floats = raw[:, :4].astype(float)
    wl = floats[:, 0] # Central wavelengths
    dwl = floats[:, 1] # Wavelength +/- band widths (half width)
    y = floats[:, 2] # Observation y data (usually R_p^2/R_s^2 or F_p/F_s)
    dy = floats[:, 3] # Observation +/- error bars
    if raw.shape[1] >= 5:
        response_mode = raw[:, 4] # Response function for the wavelength band
        print('[info] Using custom wavelength convolution functions for each band')
    else:
        response_mode = np.full(wl.shape, "boxcar", dtype="<U16") # use boxcar as default
        print('[info] All bands have been defaulted to boxcar convolution')

    # Output dictionary values
    obs_dict = {"wl": wl, "dwl": dwl, "y": y, "dy": dy, "response_mode": response_mode}

    return obs_dict
