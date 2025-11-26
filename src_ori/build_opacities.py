"""
build_opacities.py
==================

Overview:
    Reads the individual opacity tables and produces the global arrays used by the retrieval model

Sections to complete:
    - Usage
    - Key Functions
    - Notes
"""

from pathlib import Path
from typing import Optional
import numpy as np

from registry_line import (
    LineRegistryEntry,
    has_line_data,
    line_master_wavelength,
    line_pick_arrays,
    line_pressure_grid,
    line_sigma_cube,
    line_species_names,
    line_temperature_grids,
    line_temperature_grid,
    load_line_registry,
    reset_registry as reset_line_registry,
)
from registry_cia import (
    CiaRegistryEntry,
    cia_master_wavelength,
    cia_pick_arrays,
    cia_sigma_cube,
    cia_species_names,
    cia_temperature_grid,
    cia_temperature_grids,
    has_cia_data,
    load_cia_registry,
    reset_registry as reset_cia_registry,
)
from registry_ray import (
    RayRegistryEntry,
    has_ray_data,
    ray_master_wavelength,
    ray_sigma_table,
    ray_species_names,
    load_ray_registry,
    ray_pick_arrays,
    reset_registry as reset_ray_registry,
)

# Global wavelength array registries
_MASTER_WL: Optional[np.ndarray] = None
_MASTER_WL_CUT: Optional[np.ndarray] = None


def _load_wavelength_file(path: Path) -> np.ndarray:
    resolved = path.expanduser().resolve()
    if resolved.suffix == ".npy":
        return np.asarray(np.load(resolved), dtype=float)

    with resolved.open("r", encoding="utf-8") as handle:
        first = handle.readline().strip()
    try:
        n_expected = int(first.split()[0])
    except (ValueError, IndexError):
        n_expected = None

    arr = np.loadtxt(resolved, comments="#", skiprows=1)
    if arr.ndim == 1:
        lam = np.asarray([arr[-1]], dtype=float)
    else:
        lam = np.asarray(arr[:, 1], dtype=float)

    if n_expected is not None and lam.shape[0] != n_expected:
        print(f"[warn] wavelength file {resolved} header says N={n_expected} but read {lam.shape[0]} rows.")

    return lam


def read_master_wl(cfg, obs, exp_dir: Optional[Path] = None) -> np.ndarray:
    global _MASTER_WL
    if _MASTER_WL is not None:
        return _MASTER_WL

    opac_cfg = getattr(cfg, "opac", None)
    lam_master = getattr(opac_cfg, "wl_master", None) if opac_cfg is not None else None
    master: Optional[np.ndarray] = None

    if lam_master is not None:
        if isinstance(lam_master, str):
            path = Path(lam_master)
            if not path.is_absolute():
                if exp_dir is None:
                    raise ValueError(
                        "cfg.opac.wl_master is a relative path but exp_dir was not provided."
                        " Provide exp_dir or use an absolute path."
                    )
                path = (exp_dir / path).resolve()
            print(f"[master_wl] loading wl_master from file: {path}")
            master = _load_wavelength_file(path)
        else:
            print("[master_wl] using wl_master array from YAML")
            master = np.asarray(lam_master, dtype=float)

    if master is None:
        if "wl" not in obs:
            raise KeyError("Could not determine master wavelength grid: cfg.opac.wl_master not set and obs['wl'] missing.")
        print("[master_wl] using obs['wl'] as master grid")
        master = np.asarray(obs["wl"], dtype=float)

    if master.ndim != 1:
        raise ValueError(f"Master wavelength grid must be 1D, got shape {master.shape}.")
    if not np.all(np.isfinite(master)):
        raise ValueError("Master wavelength grid contains non-finite values.")
    if not np.all(np.diff(master) > 0):
        raise ValueError("Master wavelength grid must be strictly increasing.")

    _MASTER_WL = master
    return _MASTER_WL


def init_cut_master_wl(obs, lam_master: Optional[np.ndarray] = None, full_grid=False) -> np.ndarray:
    global _MASTER_WL_CUT
    if _MASTER_WL_CUT is not None:
        return _MASTER_WL_CUT

    if lam_master is None:
        if _MASTER_WL is None:
            raise RuntimeError("Full master wavelength grid not initialised; call read_master_wl() first.")
        lam_master = _MASTER_WL

    lam_array = np.asarray(lam_master, dtype=float)
    wl_obs = np.asarray(obs["wl"], dtype=float)
    dwl_obs = np.asarray(obs["dwl"], dtype=float)
    left_edges = wl_obs - dwl_obs
    right_edges = wl_obs + dwl_obs

    if full_grid:
        mask = np.ones_like(lam_array, dtype=bool)
        print(f"[info] doing full master wavelength grid calculation")
    else:
        mask = np.any(
            (lam_array[None, :] >= left_edges[:, None]) & (lam_array[None, :] <= right_edges[:, None]),
            axis=0,
        )
    if not np.any(mask):
        raise ValueError("No master wavelengths lie within any observation bins.")

    _MASTER_WL_CUT = lam_array[mask]
    return _MASTER_WL_CUT


def master_wavelength() -> np.ndarray:
    if _MASTER_WL is None:
        raise RuntimeError("Master wavelength grid not initialised; call read_master_wl() first.")
    return _MASTER_WL


def master_wavelength_cut() -> np.ndarray:
    if _MASTER_WL_CUT is None:
        raise RuntimeError("Cut master wavelength grid not initialised; call build_opacities() or init_cut_master_wl().")
    return _MASTER_WL_CUT


def build_opacities(cfg, obs, exp_dir: Optional[Path] = None):

    # Reset all the global registries
    reset_line_registry()
    reset_cia_registry()
    reset_ray_registry()

    # Read the master wavelength grid and calculate the observationally cut grid
    lam_master_full = read_master_wl(cfg, obs, exp_dir=exp_dir)
    lam_master_cut = init_cut_master_wl(obs, lam_master_full, full_grid=cfg.opac.full_grid)

    # Read in or calculate the line, CIA and Rayleigh opacity
    opac_cfg = getattr(cfg, "opac", None)
    if opac_cfg is not None and getattr(opac_cfg, "line", None) not in (None, "None"):
        load_line_registry(cfg, obs, lam_master=lam_master_cut)
    if opac_cfg is not None and getattr(opac_cfg, "cia", None) not in (None, "None"):
        load_cia_registry(cfg, obs, lam_master=lam_master_cut)
    if opac_cfg is not None and getattr(opac_cfg, "ray", None) not in (None, "None"):
        load_ray_registry(cfg, obs, lam_master=lam_master_cut)


__all__ = [
    "build_opacities",
    "read_master_wl",
    "master_wavelength",
    "master_wavelength_cut",
    "LineRegistryEntry",
    "CiaRegistryEntry",
    "RayRegistryEntry",
    "has_line_data",
    "has_cia_data",
    "has_ray_data",
    "line_master_wavelength",
    "line_pressure_grid",
    "line_temperature_grid",
    "line_temperature_grids",
    "line_sigma_cube",
    "line_pick_arrays",
    "line_species_names",
    "load_line_registry",
    "load_cia_registry",
    "load_ray_registry",
    "cia_master_wavelength",
    "cia_temperature_grid",
    "cia_temperature_grids",
    "cia_sigma_cube",
    "cia_pick_arrays",
    "cia_species_names",
    "ray_master_wavelength",
    "ray_sigma_table",
    "ray_species_names",
    "ray_pick_arrays",
]
