"""
sampler_ultranest_NS.py
=======================
"""

from __future__ import annotations
from typing import Dict, Any, Tuple, List
from pathlib import Path

import numpy as np
import jax.numpy as jnp

from .build_prepared import Prepared

try:
    import ultranest
    from ultranest import ReactiveNestedSampler
    ULTRANEST_AVAILABLE = True
except ImportError:
    ULTRANEST_AVAILABLE = False
    ultranest = None
    ReactiveNestedSampler = None

__all__ = [
    "build_prior_transform_ultranest",
    "build_loglikelihood_ultranest",
    "run_nested_ultranest",
]


def build_prior_transform_ultranest(cfg) -> Tuple[callable, List[str]]:
    """
    Build UltraNest prior transform from cfg.params.

    Parameters
    ----------
    cfg : object
        Configuration object with cfg.params list

    Returns
    -------
    prior_transform : callable
        Function with signature (u) -> theta returning transformed array
    param_names : List[str]
        Ordered list of non-delta parameter names
    """
    from scipy.stats import norm, lognorm

    params_cfg = [p for p in cfg.params if str(getattr(p, "dist", "")).lower() != "delta"]
    param_names = [p.name for p in params_cfg]

    def prior_transform(u: np.ndarray) -> np.ndarray:
        theta = np.empty_like(u)
        eps = 1e-10
        u = np.clip(u, eps, 1.0 - eps)

        for i, p in enumerate(params_cfg):
            dist_name = str(getattr(p, "dist", "")).lower()

            if dist_name == "uniform":
                low = float(getattr(p, "low"))
                high = float(getattr(p, "high"))
                transform = str(getattr(p, "transform", "identity")).lower()
                if transform == "logit":
                    z = np.log(u[i] / (1.0 - u[i]))
                    t = 1.0 / (1.0 + np.exp(-z))
                    theta[i] = low + (high - low) * t
                else:
                    theta[i] = low + u[i] * (high - low)
            elif dist_name in ("gaussian", "normal"):
                mu = float(getattr(p, "mu"))
                sigma = float(getattr(p, "sigma"))
                theta[i] = norm.ppf(u[i], loc=mu, scale=sigma)
            elif dist_name == "lognormal":
                mu = float(getattr(p, "mu"))
                sigma = float(getattr(p, "sigma"))
                theta[i] = lognorm.ppf(u[i], s=sigma, scale=np.exp(mu))
            else:
                raise ValueError(f"Unsupported distribution '{dist_name}' for parameter '{p.name}'")

        return theta

    return prior_transform, param_names


def build_loglikelihood_ultranest(cfg, prep: Prepared, param_names: List[str]) -> callable:
    """
    Build UltraNest log-likelihood function using NumPy for the statistics.
    """
    y_obs = jnp.asarray(prep.y)
    dy_obs_p = jnp.asarray(prep.dy_p)
    dy_obs_m = jnp.asarray(prep.dy_m)

    def loglikelihood(theta: np.ndarray) -> float:
        def invalid_ll() -> float:
            return -1e100

        try:
            theta_dict = {param_names[i]: float(theta[i]) for i in range(len(theta))}
            mu = prep.fm(theta_dict)
            if not bool(jnp.all(jnp.isfinite(mu))):
                return invalid_ll()
            r = y_obs - mu

            c = theta_dict.get("c", -99.0)
            sig_jit = 10.0**c
            sig_jit2 = sig_jit * sig_jit

            sigp_eff = jnp.sqrt(dy_obs_p**2 + sig_jit2)
            sigm_eff = jnp.sqrt(dy_obs_m**2 + sig_jit2)
            sig_eff = jnp.where(r >= 0.0, sigp_eff, sigm_eff)

            norm = jnp.clip(sigm_eff + sigp_eff, 1e-300, jnp.inf)
            sig_eff = jnp.clip(sig_eff, 1e-300, jnp.inf)

            logC = 0.5 * jnp.log(2.0 / jnp.pi) - jnp.log(norm)
            logL = jnp.sum(logC - 0.5 * (r / sig_eff) ** 2)

            result = float(logL)
            if not np.isfinite(result):
                return invalid_ll()
            return result
        except Exception as e:
            print(f"[UltraNest] Likelihood evaluation error: {e}")
            return invalid_ll()

    return loglikelihood


def run_nested_ultranest(
    cfg,
    prep: Prepared,
    exp_dir: Path,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Run UltraNest nested sampling.
    """
    if not ULTRANEST_AVAILABLE:
        raise ImportError(
            "UltraNest is not installed. Install with:\n"
            "  pip install ultranest"
        )

    un_cfg = getattr(cfg.sampling, "ultranest", None)
    if un_cfg is None:
        raise ValueError("Missing cfg.sampling.ultranest configuration.")

    n_live = int(getattr(un_cfg, "num_live_points", 500))
    dlogz = float(getattr(un_cfg, "dlogz", 0.5))
    max_iters = int(getattr(un_cfg, "max_iters", 0))
    min_num_live_points = int(getattr(un_cfg, "min_num_live_points", n_live))
    verbose = bool(getattr(un_cfg, "verbose", True))
    show_status = bool(getattr(un_cfg, "show_status", True))

    exp_dir.mkdir(parents=True, exist_ok=True)

    prior_fn, param_names = build_prior_transform_ultranest(cfg)
    loglike_fn = build_loglikelihood_ultranest(cfg, prep, param_names)

    sampler = ReactiveNestedSampler(param_names, loglike_fn, prior_fn)

    print(f"[UltraNest] Running nested sampling...")
    print(f"[UltraNest] Free parameters: {len(param_names)}")
    print(f"[UltraNest] Parameter names: {param_names}")
    print(f"[UltraNest] Live points: {n_live}")
    print(f"[UltraNest] dlogz: {dlogz}")

    run_kwargs = dict(
        min_num_live_points=min_num_live_points,
        dlogz=dlogz,
        show_status=show_status,
        viz_callback=None,
    )
    if max_iters > 0:
        run_kwargs["max_iters"] = max_iters

    results = sampler.run(**run_kwargs)
    sampler.print_results()

    logz = float(results.get("logz", np.nan))
    logzerr = float(results.get("logzerr", np.nan))

    evidence_info: Dict[str, Any] = {
        "logZ": logz,
        "logZ_err": logzerr,
        "sampler": "ultranest",
        "n_live": n_live,
    }

    if "samples" in results:
        samples = np.asarray(results["samples"])
    elif "weighted_samples" in results and "points" in results["weighted_samples"]:
        samples = np.asarray(results["weighted_samples"]["points"])
    else:
        raise RuntimeError("UltraNest results did not include posterior samples.")

    n_samples = samples.shape[0]
    print(f"[UltraNest] Posterior samples: {n_samples}")

    samples_dict: Dict[str, np.ndarray] = {}
    for i, name in enumerate(param_names):
        samples_dict[name] = samples[:, i]

    for param in cfg.params:
        name = param.name
        if name not in samples_dict:
            dist_name = str(getattr(param, "dist", "")).lower()
            if dist_name == "delta":
                val = getattr(param, "value", getattr(param, "init", None))
                if val is not None:
                    samples_dict[name] = np.full(
                        (n_samples,),
                        float(val),
                        dtype=np.float64,
                    )

    return samples_dict, evidence_info
