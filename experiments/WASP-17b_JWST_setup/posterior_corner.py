#!/usr/bin/env python3
"""
posterior_corner.py
====================

Overview:
    Generate a corner-style plot (pairwise marginalized posterior) from an
    ArviZ InferenceData NetCDF file produced by the retrieval pipeline.

Sections to complete:
    - Usage
    - Key Functions
    - Notes
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set

import arviz as az
import corner
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml


def _infer_scalar_params(data_vars: Iterable) -> List[str]:
    """
    Return parameter names that are scalar (chain/draw only) to keep the
    corner plot manageable. Higher-dimensional parameters can be selected
    manually via --params if desired.
    """
    scalar_names: List[str] = []
    for name, var in data_vars.items():
        dims = tuple(var.dims)
        if dims == ("chain", "draw"):
            scalar_names.append(name)
        else:
            print(f"[posterior_corner] Skipping non-scalar variable '{name}' with dims {dims}")
    return scalar_names


def _resolve_var_names(posterior_ds, requested: Sequence[str] | None) -> List[str]:
    if requested:
        missing = [v for v in requested if v not in posterior_ds.data_vars]
        if missing:
            raise KeyError(f"Variables not found in posterior: {missing}")
        return list(requested)
    names = _infer_scalar_params(posterior_ds.data_vars)
    if not names:
        raise ValueError(
            "No scalar parameters detected automatically. "
            "Provide --params explicitly (e.g., --params R_p log_g)."
        )
    return names


def _flatten_param(arr: np.ndarray) -> np.ndarray:
    """Flatten (chain, draw[, ...]) arrays to a 1D vector."""
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        return arr.reshape(-1)
    if arr.ndim >= 3:
        tail = int(np.prod(arr.shape[2:]))
        if tail == 1:
            return arr.reshape(-1)
        # Collapse extra dims, take first component for sanity
        reshaped = arr.reshape(arr.shape[0], arr.shape[1], tail)
        return reshaped[..., 0].reshape(-1)
    raise ValueError(f"Unsupported parameter shape {arr.shape}")


def _build_sample_matrix(
    posterior_ds,
    var_names: List[str],
    log_params: Set[str],
) -> np.ndarray:
    samples = []
    for name in var_names:
        arr = posterior_ds[name].values
        vec = _flatten_param(np.asarray(arr, dtype=float))
        if name in log_params:
            if np.any(vec <= 0):
                raise ValueError(
                    f"Parameter '{name}' has non-positive samples; cannot take log10."
                )
            vec = np.log10(vec)
        samples.append(vec)
    stacked = np.vstack(samples).T
    if stacked.shape[0] == 0:
        raise ValueError("No posterior samples available to plot.")
    return stacked


def _load_config_data(config_path: Path | None) -> Dict[str, Any]:
    if config_path is None or not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _infer_log_params_from_config(
    cfg: Dict[str, Any],
    candidate_names: Sequence[str],
) -> Set[str]:
    params_cfg = cfg.get("params", [])
    candidate_set = set(candidate_names)
    log_params: Set[str] = set()

    for entry in params_cfg:
        name = entry.get("name")
        if name not in candidate_set:
            continue
        transform = str(entry.get("transform", "")).lower()
        dist = str(entry.get("dist", "")).lower()
        is_log = transform == "log" or dist.startswith("log_") or dist == "lognormal"
        if is_log:
            log_params.add(name)

    return log_params


def _extract_plot_order(
    cfg: Dict[str, Any],
    candidate_names: Sequence[str],
) -> Dict[str, float]:
    params_cfg = cfg.get("params", [])
    candidate_set = set(candidate_names)
    orders: Dict[str, float] = {}
    for entry in params_cfg:
        name = entry.get("name")
        if name not in candidate_set:
            continue
        if "plot_order" not in entry:
            continue
        try:
            orders[name] = float(entry["plot_order"])
        except (TypeError, ValueError):
            continue
    return orders


def _apply_plot_order(var_names: List[str], order_map: Dict[str, float]) -> List[str]:
    ordered = []
    for idx, name in enumerate(var_names):
        order_value = order_map.get(name)
        if order_value is not None and order_value <= 0:
            continue
        sort_bucket = 0 if order_value is not None else 1
        sort_key = (sort_bucket, order_value if order_value is not None else idx, idx)
        ordered.append((sort_key, name))
    ordered.sort(key=lambda item: item[0])
    return [name for _, name in ordered]


def _load_corner_config(path: Path | None) -> Dict[str, Dict[str, Any]]:
    if path is None:
        return {}
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError("corner_config must be a mapping of parameter names.")
    out: Dict[str, Dict[str, Any]] = {}
    for key, val in raw.items():
        if isinstance(val, dict):
            out[key] = val
        elif isinstance(val, str):
            out[key] = {"label": val}
        else:
            raise ValueError("corner_config entries must be dicts or strings.")
    return out


def plot_corner(
    posterior_path: Path,
    params: Sequence[str] | None = None,
    outname: str = "posterior_corner",
    quantiles: Sequence[float] = (0.16, 0.5, 0.84),
    config_path: Path | None = None,
    extra_log_params: Sequence[str] | None = None,
    label_map_path: Path | None = None,
) -> Path:
    """
    Load posterior.nc, select variables, and save a classic corner plot with
    histograms, scatter points, and density contours.
    """
    if not posterior_path.exists():
        raise FileNotFoundError(f"Could not find posterior file: {posterior_path}")

    idata = az.from_netcdf(posterior_path)
    posterior_ds = idata.posterior

    # --- choose variables ---
    var_names = _resolve_var_names(posterior_ds, params)

    if config_path is None:
        default_cfg = posterior_path.parent / "retrieval_config.yaml"
        config_path = default_cfg if default_cfg.exists() else None
    config_data = _load_config_data(config_path)

    if label_map_path is None:
        default_label_yaml = posterior_path.parent / "corner_config.yaml"
        label_map_path = default_label_yaml if default_label_yaml.exists() else None
    corner_cfg = _load_corner_config(label_map_path)

    # plot_order from retrieval config + from corner_config.yaml
    order_map = _extract_plot_order(config_data, var_names)
    label_order_overrides = {
        name: cfg.get("order")
        for name, cfg in corner_cfg.items()
        if isinstance(cfg, dict) and "order" in cfg
    }
    order_map.update({k: v for k, v in label_order_overrides.items() if v is not None})
    var_names = _apply_plot_order(var_names, order_map)
    if not var_names:
        raise ValueError("No parameters remain after applying plot_order filtering.")

    # --- log10 handling ---
    log_params = _infer_log_params_from_config(config_data, var_names)
    if extra_log_params:
        log_params.update(extra_log_params)
    if log_params:
        print(f"[posterior_corner] Log-scaling parameters: {', '.join(sorted(log_params))}")

    # --- sample matrix & labels ---
    samples = _build_sample_matrix(posterior_ds, var_names, log_params)

    labels = []
    for name in var_names:
        default_label = f"log10 {name}" if name in log_params else name
        labels.append(corner_cfg.get(name, {}).get("label", default_label))
    print(f"[posterior_corner] Plotting variables: {', '.join(labels)}")

    # --- style & corner call ---
    sns.set_theme(style="ticks", palette="colorblind")
    contour_color = sns.color_palette("colorblind", 1)[0]
    scatter_color = "#808080"

    # Make contours visually strong
    contour_kwargs = {
        "colors": [contour_color],
        "linewidths": 1.6,
        "linestyles": "solid",
    }

    # Slightly lighter scatter so contours stand out
    data_kwargs = {
        "alpha": 0.25,
        "ms": 2.0,
        "mew": 0.0,
        "color": scatter_color,
    }

    fig = corner.corner(
        samples,
        labels=labels,
        quantiles=quantiles,
        show_titles=True,
        hist_bin_factor=1.2,
        label_kwargs={"fontsize": 12, "labelpad": 8},
        title_kwargs={"fontsize": 12},
        plot_contours=True,
        plot_density=True,   # <- explicit
        fill_contours=False,
        contour_kwargs=contour_kwargs,
        plot_datapoints=True,
        data_kwargs=data_kwargs,
        max_n_ticks=4,
        smooth=0.75,
        color=contour_color,
        labelpad=0.04,
    )

    if fig is None:
        raise RuntimeError("corner.corner returned None; no plot generated.")

    # Tighten layout a bit
    fig.subplots_adjust(
        left=0.13, right=0.98, bottom=0.14, top=0.98, wspace=0.07, hspace=0.07
    )

    # Example of per-parameter axis tweaks (optional)
    try:
        axes = np.array(fig.axes).reshape(len(var_names), len(var_names))
        if "R_p" in var_names:
            idx = var_names.index("R_p")
            diag_ax = axes[idx, idx]
            diag_ax.set_yticklabels([])
            if idx == len(var_names) - 1:
                formatter = plt.FuncFormatter(lambda x, _: f"{x:.2f}")
                diag_ax.xaxis.set_major_formatter(formatter)
            else:
                diag_ax.set_xticklabels([])
    except ValueError:
        # Not a square grid; just ignore
        pass

    out_png = posterior_path.parent / f"{outname}.png"
    out_pdf = posterior_path.parent / f"{outname}.pdf"
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    print(f"[posterior_corner] saved:\n  {out_png}\n  {out_pdf}")
    plt.show()
    return out_png


def main():
    ap = argparse.ArgumentParser(description="ArviZ corner plot helper for posterior.nc.")
    ap.add_argument(
        "--posterior",
        type=str,
        default="posterior.nc",
        help="Path to the ArviZ NetCDF file (default: posterior.nc in current directory).",
    )
    ap.add_argument(
        "--params",
        nargs="+",
        help="Specific parameter names to include. Defaults to scalar parameters only.",
    )
    ap.add_argument(
        "--config",
        type=str,
        help="Retrieval YAML config path used to infer log-scaled parameters (defaults to retrieval_config.yaml next to posterior).",
    )
    ap.add_argument(
        "--log-params",
        nargs="+",
        help="Additional parameter names to plot in log10 space (applied after YAML inference).",
    )
    ap.add_argument(
        "--outname",
        type=str,
        default="posterior_corner",
        help="Filename stem for outputs (default: posterior_corner).",
    )
    ap.add_argument(
        "--label-map",
        type=str,
        help="Path to YAML/JSON file mapping param names to custom axis labels.",
    )
    args = ap.parse_args()

    posterior_path = Path(args.posterior).resolve()
    config_path = Path(args.config).resolve() if args.config else None
    label_map_path = Path(args.label_map).resolve() if args.label_map else None
    plot_corner(
        posterior_path,
        params=args.params,
        outname=args.outname,
        config_path=config_path,
        extra_log_params=args.log_params,
        label_map_path=label_map_path,
    )


if __name__ == "__main__":
    main()
