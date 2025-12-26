"""
sampler_dynesty_NS.py
=====================
"""

from __future__ import annotations
from typing import Dict, Any, Tuple, List
from pathlib import Path
import pickle

import numpy as np
import jax.numpy as jnp

from .build_prepared import Prepared

# Try to import Dynesty
try:
    import dynesty
    from dynesty import NestedSampler, DynamicNestedSampler
    from dynesty import utils as dyutils
    DYNESTY_AVAILABLE = True
except ImportError:
    DYNESTY_AVAILABLE = False
    dynesty = None
    NestedSampler = None
    DynamicNestedSampler = None

__all__ = [
    "build_prior_transform_dynesty",
    "build_loglikelihood_dynesty",
    "run_nested_dynesty"
]


def build_prior_transform_dynesty(cfg) -> Tuple[callable, List[str]]:
    """
    Build Dynesty prior transform from cfg.params.

    Parameters
    ----------
    cfg : object
        Configuration object with cfg.params list

    Returns
    -------
    prior_transform : callable
        Function with signature (u) -> theta that returns transformed array
    param_names : List[str]
        Ordered list of non-delta parameter names
    """
    from scipy.stats import norm, lognorm

    # Extract non-delta parameters
    params_cfg = [p for p in cfg.params if str(getattr(p, "dist", "")).lower() != "delta"]
    param_names = [p.name for p in params_cfg]

    def prior_transform(u: np.ndarray) -> np.ndarray:
        """
        Dynesty prior transform: maps unit cube to physical parameters.

        Returns new array (cleaner than PyMultiNest's in-place modification).

        Parameters
        ----------
        u : np.ndarray
            Unit cube values [0, 1]^n

        Returns
        -------
        theta : np.ndarray
            Physical parameter values
        """
        theta = np.empty_like(u)

        # Clip to avoid edge cases in inverse CDF
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


def build_loglikelihood_dynesty(cfg, prep: Prepared, param_names: List[str]) -> callable:
    """
    Build Dynesty log-likelihood function.

    Implements the same split-normal likelihood as JAXNS and BlackJAX NS,
    but uses NumPy operations (except for the JAX forward model).

    Parameters
    ----------
    cfg : object
        Configuration object
    prep : Prepared
        Prepared model bundle with forward model and observed data
    param_names : List[str]
        Ordered list of parameter names

    Returns
    -------
    loglikelihood : callable
        Function with signature (theta) -> logL returning log-likelihood
    """
    # Observed data - keep in JAX for single-scalar output conversion
    y_obs = jnp.asarray(prep.y)
    dy_obs_p = jnp.asarray(prep.dy_p)
    dy_obs_m = jnp.asarray(prep.dy_m)

    def loglikelihood(theta: np.ndarray) -> float:
        """
        Dynesty log-likelihood: split-normal likelihood.

        Parameters
        ----------
        theta : np.ndarray
            Parameter values in constrained space

        Returns
        -------
        logL : float
            Log-likelihood value
        """
        def invalid_ll() -> float:
            return -1e100

        try:
            # Build parameter dictionary
            theta_dict = {param_names[i]: float(theta[i]) for i in range(len(theta))}

            # Call forward model (JAX) and convert to NumPy
            mu = prep.fm(theta_dict)  # (N,)
            if not bool(jnp.all(jnp.isfinite(mu))):
                return invalid_ll()
            r = y_obs - mu  # residuals

            # Split-normal likelihood using NumPy operations
            c = theta_dict.get("c", -99.0)  # log10(sigma_jit)
            sig_jit = 10.0**c
            sig_jit2 = sig_jit * sig_jit

            # Inflate BOTH sides in quadrature
            sigp_eff = jnp.sqrt(dy_obs_p**2 + sig_jit2)
            sigm_eff = jnp.sqrt(dy_obs_m**2 + sig_jit2)

            # Choose side for exponent
            sig_eff = jnp.where(r >= 0.0, sigp_eff, sigm_eff)

            # Normalisation must use the SAME effective scales
            norm = jnp.clip(sigm_eff + sigp_eff, 1e-300, jnp.inf)
            sig_eff = jnp.clip(sig_eff, 1e-300, jnp.inf)

            logC = 0.5 * jnp.log(2.0 / jnp.pi) - jnp.log(norm)
            logL = jnp.sum(logC - 0.5 * (r / sig_eff) ** 2)

            # Convert to Python float
            result = float(logL)

            # Handle non-finite values
            if not np.isfinite(result):
                return invalid_ll()

            return result

        except Exception as e:
            print(f"[Dynesty] Likelihood evaluation error: {e}")
            return invalid_ll()

    return loglikelihood


def run_nested_dynesty(
    cfg,
    prep: Prepared,
    exp_dir: Path,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Run Dynesty nested sampling.

    Parameters
    ----------
    cfg : object
        Configuration with cfg.sampling.dynesty settings
    prep : Prepared
        Prepared model bundle
    exp_dir : Path
        Output directory

    Returns
    -------
    samples_dict : Dict[str, np.ndarray]
        Posterior samples for each parameter
    evidence_info : Dict[str, Any]
        Evidence and diagnostic information
    """
    if not DYNESTY_AVAILABLE:
        raise ImportError(
            "Dynesty is not installed. Install with:\n"
            "  pip install dynesty"
        )

    dy_cfg = cfg.sampling.dynesty

    # Extract configuration with defaults
    nlive = int(getattr(dy_cfg, "nlive", 500))
    bound = str(getattr(dy_cfg, "bound", "multi"))
    sample = str(getattr(dy_cfg, "sample", "auto"))
    dlogz = float(getattr(dy_cfg, "dlogz", 0.5))
    maxiter = getattr(dy_cfg, "maxiter", None)
    maxcall = getattr(dy_cfg, "maxcall", None)
    bootstrap = int(getattr(dy_cfg, "bootstrap", 0))
    enlarge = getattr(dy_cfg, "enlarge", None)
    update_interval = getattr(dy_cfg, "update_interval", None)
    dynamic = bool(getattr(dy_cfg, "dynamic", False))
    print_progress = bool(getattr(dy_cfg, "print_progress", True))
    seed = int(getattr(dy_cfg, "seed", 42))

    # Setup output directory
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Build prior and likelihood from cfg (not prep!)
    prior_fn, param_names = build_prior_transform_dynesty(cfg)
    loglike_fn = build_loglikelihood_dynesty(cfg, prep, param_names)

    ndim = len(param_names)

    print(f"[Dynesty] Running nested sampling...")
    print(f"[Dynesty] Free parameters: {ndim}")
    print(f"[Dynesty] Parameter names: {param_names}")
    print(f"[Dynesty] Live points: {nlive}")
    print(f"[Dynesty] Bound: {bound}, Sample: {sample}")
    print(f"[Dynesty] dlogZ stopping criterion: {dlogz}")

    # Choose sampler type
    if dynamic:
        print(f"[Dynesty] Using dynamic nested sampling")
        sampler = DynamicNestedSampler(
            loglike_fn,
            prior_fn,
            ndim,
            nlive=nlive,
            bound=bound,
            sample=sample,
            bootstrap=bootstrap,
            enlarge=enlarge,
            update_interval=update_interval,
            rstate=np.random.default_rng(seed),
        )
        # Run dynamic nested sampling
        sampler.run_nested(dlogz_init=dlogz, print_progress=print_progress)
    else:
        print(f"[Dynesty] Using static nested sampling")
        sampler = NestedSampler(
            loglike_fn,
            prior_fn,
            ndim,
            nlive=nlive,
            bound=bound,
            sample=sample,
            bootstrap=bootstrap,
            enlarge=enlarge,
            update_interval=update_interval,
            rstate=np.random.default_rng(seed),
        )
        # Run static nested sampling
        sampler.run_nested(
            dlogz=dlogz,
            maxiter=maxiter,
            maxcall=maxcall,
            print_progress=print_progress,
        )

    print(f"[Dynesty] Sampling complete. Extracting results...")

    # Get results
    results = sampler.results

    # Extract evidence information
    evidence_info: Dict[str, Any] = {
        "logZ": float(results.logz[-1]),
        "logZ_err": float(results.logzerr[-1]),
        "ESS": int(results.ess),
        "n_like": int(results.ncall),
        "H": float(results.h[-1]),
        "sampler": "dynesty",
        "n_live": nlive,
    }

    print(f"[Dynesty] Evidence: {evidence_info['logZ']:.2f} ± {evidence_info['logZ_err']:.2f}")
    print(f"[Dynesty] ESS: {evidence_info['ESS']}")
    print(f"[Dynesty] Likelihood evaluations: {evidence_info['n_like']}")

    # Save full results object
    results_path = exp_dir / "dynesty_results.pkl"
    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    evidence_info["results_file"] = str(results_path)

    print(f"[Dynesty] Saved full results to {results_path}")

    # Extract equal-weighted posterior samples
    weights = np.exp(results.logwt - results.logz[-1])
    samples = dyutils.resample_equal(results.samples, weights)
    n_samples = samples.shape[0]

    print(f"[Dynesty] Posterior samples: {n_samples}")

    # Build samples_dict (same as JAXNS/BlackJAX)
    samples_dict: Dict[str, np.ndarray] = {}

    # Add free parameters
    for i, name in enumerate(param_names):
        samples_dict[name] = samples[:, i]

    # Add fixed/delta parameters (same as JAXNS/BlackJAX)
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
