#!/usr/bin/env python3
"""
plot_model_band.py — plot median model and 95% credible band using the same
forward model as the retrieval, built via build_forward_model(cfg, obs).

Expected layout (experiment directory):
  - config.yaml           (your YAML)
  - posterior.nc          (ArviZ InferenceData with posterior group)
  - observed_data.csv     (optional; lam_um,dlam_um,depth,depth_sigma)

Source layout:
  - ../../src_ori/build_model.py with:
        def build_forward_model(cfg, obs, return_highres=False) -> predict_fn
  - ../../src_ori/build_opacities.py with:
        def build_opacities(cfg, obs, exp_dir)
        def master_wavelength_cut()
  - ../../src_ori/instru_bandpass.py with:
        def load_bandpass_registry(obs, full_grid, cut_grid)

Outputs (in experiment directory):
  - model_band.png / model_band.pdf
  - model_band_quantiles.npz (depth_p02_5, depth_p50, depth_p97_5, lam, dlam)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import arviz as az
from types import SimpleNamespace

from jax import config as jax_config
jax_config.update("jax_enable_x64", True)


# ---------------- YAML / cfg helpers ----------------


def _to_ns(x):
    if isinstance(x, list):
        return [_to_ns(v) for v in x]
    if isinstance(x, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in x.items()})
    return x


def _read_cfg(path: Path):
    with path.open("r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    return _to_ns(y)


def _is_fixed_param(p) -> bool:
    dist = str(getattr(p, "dist", "")).lower()
    return dist == "delta" or bool(getattr(p, "fixed", False))


def _fixed_value_param(p):
    val = getattr(p, "value", None)
    if val is not None:
        return float(val)
    init = getattr(p, "init", None)
    if init is not None:
        return float(init)
    return None


# ---------------- observed data loading ----------------


def _load_observed(exp_dir: Path, cfg):
    """
    Return lam, dlam, y, dy, response_mode.

    Priority:
      1) observed_data.csv in exp_dir (from save_observed_data_csv)
      2) cfg.obs.path (raw data file; flexible column handling)
    """
    csv_path = exp_dir / "observed_data.csv"
    if csv_path.exists():
        arr = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=str)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape[1] < 2:
            raise ValueError(
                f"{csv_path} must have at least lam_um,dlam_um columns."
            )
        lam = arr[:, 0].astype(float)
        dlam = arr[:, 1].astype(float)
        y = arr[:, 2].astype(float) if arr.shape[1] >= 3 else None
        dy = arr[:, 3].astype(float) if arr.shape[1] >= 4 else None
        resp = arr[:, 4] if arr.shape[1] >= 5 else np.full_like(lam, "boxcar", dtype=object)
        return lam, dlam, y, dy, resp

    # Fall back to raw obs file from YAML
    obs_cfg = getattr(cfg, "obs", None)
    if obs_cfg is None or not getattr(obs_cfg, "path", None):
        raise FileNotFoundError(
            "No observed_data.csv in experiment directory and cfg.obs.path missing. "
            "Cannot load observed data."
        )

    data_path = Path(obs_cfg.path)
    if not data_path.is_absolute():
        data_path = (exp_dir / data_path).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find data file: {data_path}")

    arr = np.loadtxt(data_path)
    if arr.ndim != 2 or arr.shape[1] < 1:
        raise ValueError(
            f"Unsupported data format in {data_path}. "
            "Need columns lam[, dlam, y, dy]."
        )

    lam = arr[:, 0]
    if arr.shape[1] >= 4:
        dlam = arr[:, 1]
        y = arr[:, 2]
        dy = arr[:, 3]
        resp = np.full_like(lam, "boxcar", dtype=object)
    elif arr.shape[1] == 3:
        dlam = np.zeros_like(lam)
        y = arr[:, 1]
        dy = arr[:, 2]
        resp = np.full_like(lam, "boxcar", dtype=object)
    elif arr.shape[1] == 2:
        dlam = np.zeros_like(lam)
        y = arr[:, 1]
        dy = None
        resp = np.full_like(lam, "boxcar", dtype=object)
    else:
        dlam = np.zeros_like(lam)
        y = None
        dy = None
        resp = np.full_like(lam, "boxcar", dtype=object)

    return lam, dlam, y, dy, resp


# ---------------- chain handling (ArviZ posterior.nc) ----------------


def _flatten_param(a: np.ndarray) -> np.ndarray:
    """
    Flatten chains/draws to 1D (N,) vector.

    - 2D: (chain, draw) -> flatten
    - 1D: already flattened
    - >=3D: treat as vector/tensor param; take first component politely.
    """
    arr = np.asarray(a)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        # (chain, draw)
        return arr.reshape(-1)
    if arr.ndim >= 3:
        # (chain, draw, *tail)
        tail = arr.shape[2:]
        if np.prod(tail) == 1:
            return arr.reshape(-1)
        # Collapse tail and take first component
        return arr.reshape(arr.shape[0], arr.shape[1], -1)[..., 0].reshape(-1)
    raise ValueError(f"Unsupported param shape {arr.shape} for flattening")


def _build_param_draws_from_idata(
    posterior_ds,
    params_cfg: List[SimpleNamespace],
) -> Dict[str, np.ndarray]:
    """
    Build dict name->(N_total,) samples for each param in cfg.params, filling
    fixed/delta params from their value/init if missing in posterior.

    posterior_ds: idata.posterior (xarray.Dataset)
    """
    out: Dict[str, np.ndarray] = {}

    # infer total draws from dims
    if "chain" not in posterior_ds.dims or "draw" not in posterior_ds.dims:
        raise ValueError("posterior.nc must have dims ('chain', 'draw').")
    n_chain = int(posterior_ds.sizes["chain"])
    n_draw = int(posterior_ds.sizes["draw"])
    N_total = n_chain * n_draw

    for p in params_cfg:
        name = getattr(p, "name", None)
        if not name:
            continue

        if name in posterior_ds.data_vars:
            arr = posterior_ds[name].values  # (chain, draw[, ...])
            out[name] = _flatten_param(arr)
        elif _is_fixed_param(p):
            val = _fixed_value_param(p)
            if val is None:
                raise ValueError(f"Fixed/delta param '{name}' needs value/init in YAML.")
            out[name] = np.full((N_total,), float(val), dtype=float)
        else:
            raise KeyError(f"Free parameter '{name}' not found in posterior.nc.")

    lengths = {k: v.shape[0] for k, v in out.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Inconsistent parameter lengths: {lengths}")
    return out, N_total


def _bump_opacity_paths_one_level(cfg):
    """
    Prefix '../' to all *relative* cfg.opac.*.path entries.

    This leaves absolute paths unchanged.
    """
    opac = getattr(cfg, "opac", None)
    if opac is None:
        return

    # We go one level up relative to where the YAML paths are interpreted
    prefix = Path("..")

    for attr in dir(opac):
        if attr.startswith("_"):
            continue
        val = getattr(opac, attr)

        # Handle lists, e.g. cfg.opac.line, cfg.opac.cia, etc.
        if isinstance(val, list):
            for spec in val:
                if hasattr(spec, "path"):
                    p = getattr(spec, "path")
                    if not p:
                        continue
                    p_str = str(p)
                    # Only touch relative paths
                    if not Path(p_str).is_absolute():
                        new_path = str(prefix / p_str)
                        setattr(spec, "path", new_path)

        # Handle single objects with .path (in case you ever have those)
        elif hasattr(val, "path"):
            p = getattr(val, "path")
            if not p:
                continue
            p_str = str(p)
            if not Path(p_str).is_absolute():
                new_path = str(prefix / p_str)
                setattr(val, "path", str(prefix / p_str))


# ---------------- plotting / main logic ----------------


def plot_model_band(
    config_path: str,
    outname: str = "model_band",
    max_samples: int | None = 2000,
    random_seed: int = 123,
    show_data: bool = True,
    show_plot: bool = True,
):
    cfg_path = Path(config_path).resolve()
    exp_dir = cfg_path.parent

    # Add ../../src_ori to sys.path so we can import build_forward_model etc.
    src_root = (exp_dir / "../../src_ori").resolve()
    if src_root.is_dir() and str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    try:
        from build_model import build_forward_model
        from build_opacities import build_opacities, master_wavelength_cut
        from instru_bandpass import load_bandpass_registry
    except ImportError as e:
        raise ImportError(
            f"Could not import modeling helpers from {src_root}."
        ) from e

    # Load cfg as SimpleNamespace
    cfg = _read_cfg(cfg_path)

    _bump_opacity_paths_one_level(cfg)

    # Load observed data
    lam, dlam, y_obs, dy_obs, response_mode = _load_observed(exp_dir, cfg)
    lam_arr = np.asarray(lam, dtype=float)
    dlam_arr = np.asarray(dlam, dtype=float)

    # Build obs dict for forward model & bandpass
    obs = {
        "wl": lam_arr,
        "dwl": dlam_arr,
        "y": np.asarray(y_obs, dtype=float) if y_obs is not None else None,
        "dy": np.asarray(dy_obs, dtype=float) if dy_obs is not None else None,
        "response_mode": np.asarray(response_mode, dtype=object) if response_mode is not None else None,
    }

    # Ensure global opacity tables are populated (forward model relies on them)
    build_opacities(cfg, obs, exp_dir)

    # Build bandpass / response metadata on the same hi-res grid used by the forward model
    hi_wl = np.asarray(master_wavelength_cut(), dtype=float)
    # full_grid is currently unused, so we can safely pass hi_wl for both full_grid and cut_grid
    load_bandpass_registry(obs, hi_wl, hi_wl)

    # Build the *same* forward model used in the retrieval; we want hi-res + binned
    predict_fn = build_forward_model(cfg, obs, return_highres=True)

    # Load posterior from ArviZ NetCDF
    posterior_path = exp_dir / "posterior.nc"
    if not posterior_path.exists():
        raise FileNotFoundError(f"Missing {posterior_path}. Run the retrieval and save InferenceData first.")
    idata = az.from_netcdf(posterior_path)
    posterior_ds = idata.posterior

    # Build param draws (name -> (N_total,))
    params_cfg = getattr(cfg, "params", [])
    if not params_cfg:
        raise ValueError("cfg.params is empty or missing; cannot reconstruct parameters.")
    param_draws, N_total = _build_param_draws_from_idata(posterior_ds, params_cfg)

    # Sub-sample draws for model evaluation
    rng = np.random.default_rng(random_seed)
    if max_samples is not None and max_samples > 0 and max_samples < N_total:
        idx = np.sort(rng.choice(N_total, size=max_samples, replace=False))
    else:
        idx = np.arange(N_total)
    M = idx.size

    # Evaluate model for each selected draw
    depth_samples = np.empty((M, lam_arr.size), dtype=float)
    hires_samples = np.empty((M, hi_wl.size), dtype=float)

    for k, ii in enumerate(idx):
        pars: Dict[str, float] = {}
        for p in params_cfg:
            name = getattr(p, "name", None)
            if not name:
                continue
            pars[name] = float(param_draws[name][ii])
        result = predict_fn(pars)
        hires_samples[k, :] = np.asarray(result["hires"], dtype=float)
        depth_samples[k, :] = np.asarray(result["binned"], dtype=float)

    # Pointwise quantiles (binned + hi-res)
    q02_5 = np.quantile(depth_samples, 0.025, axis=0)
    q50   = np.quantile(depth_samples, 0.50,  axis=0)
    q97_5 = np.quantile(depth_samples, 0.975, axis=0)
    hq02_5 = np.quantile(hires_samples, 0.025, axis=0)
    hq50   = np.quantile(hires_samples, 0.50,  axis=0)
    hq97_5 = np.quantile(hires_samples, 0.975, axis=0)

    # Save quantiles
    np.savez_compressed(
        exp_dir / f"{outname}_quantiles.npz",
        lam=lam_arr,
        dlam=dlam_arr,
        depth_p02_5=q02_5,
        depth_p50=q50,
        depth_p97_5=q97_5,
        draw_idx=idx,
        lam_hires=hi_wl,
        depth_hi_p02_5=hq02_5,
        depth_hi_p50=hq50,
        depth_hi_p97_5=hq97_5,
    )

    # Plot
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("colorblind", 4)
    fig, ax = plt.subplots(figsize=(8, 4.5))

    # hi-res median (optional overlay)
    ax.plot(hi_wl, hq50, lw=1.0, alpha=0.6, label="Median (hi-res)", color=palette[0])
    # credible band (binned)
    ax.fill_between(lam_arr, q02_5, q97_5, alpha=0.3, label="95% credible band", color=palette[1])
    # median binned model
    ax.plot(lam_arr, q50, lw=2, label="Median model", color=palette[2])

    # observations (if available)
    if show_data and y_obs is not None:
        if dy_obs is not None:
            ax.errorbar(
                lam_arr,
                y_obs,
                xerr=dlam_arr,
                yerr=dy_obs,
                fmt="o",
                ms=3,
                lw=1,
                alpha=0.9,
                label="Observed",
                color=palette[3],
                ecolor=palette[3],
                capsize=2,
            )
        else:
            ax.errorbar(
                lam_arr,
                y_obs,
                xerr=dlam_arr,
                fmt="o",
                ms=3,
                alpha=0.9,
                label="Observed",
                color=palette[3],
                ecolor=palette[3],
                capsize=2,
            )

    ax.set_xlabel("Wavelength [µm]")
    ax.set_ylabel("Transit depth")
    ax.set_xscale("log")
    ax.legend()
    fig.tight_layout()

    png = exp_dir / f"{outname}.png"
    pdf = exp_dir / f"{outname}.pdf"
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    print(f"[model_band] saved:\n  {png}\n  {pdf}")

    if show_plot:
        plt.show()

    return fig, ax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config", required=True,
        help="Path to YAML config used for the run (in the experiment directory).",
    )
    ap.add_argument(
        "--outname", type=str, default="model_band",
        help="Output filename stem (default: model_band)",
    )
    ap.add_argument(
        "--max-samples", type=int, default=2000,
        help="Evaluate model on at most this many posterior draws "
             "(<=0 means use all draws).",
    )
    ap.add_argument(
        "--seed", type=int, default=123,
        help="Random seed for subsampling posterior draws.",
    )
    ap.add_argument(
        "--no-data", action="store_true",
        help="Do not overlay observed data, even if available.",
    )
    ap.add_argument(
        "--no-show", action="store_true",
        help="Do not display the plot window (useful on headless systems).",
    )
    args = ap.parse_args()

    max_samples = None if args.max_samples <= 0 else args.max_samples

    plot_model_band(
        args.config,
        outname=args.outname,
        max_samples=max_samples,
        random_seed=args.seed,
        show_data=not args.no_data,
        show_plot=not args.no_show,
    )


if __name__ == "__main__":
    main()
