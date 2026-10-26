"""Survival confidence bands around the exponential of predicted negative CHF.

Requires a variance-enabled survival prediction. Two approaches:

- "naive": marginal SDs = sqrt(diag(Cov)); MC band with full covariance.
- "smoothed": tensor-product penalized B-spline smoothing of the
  covariance surface, rank-k PSD truncation, positive-spectrum residual
  correction, and eigenvalue-ratio weights. The C++ smoother uses actual
  failure times and is an alternative to R's index-coordinate mgcv GAM;
  the two methods do not have identical smoothing scales.
"""
from __future__ import annotations

from numbers import Real
import warnings

import numpy as np

from ._core import mc_band, rlt_tp_smooth


def _nearest_psd(M):
    vals, vecs = np.linalg.eigh(M)
    vals = np.clip(vals, 0.0, None)
    Mp = (vecs * vals) @ vecs.T
    return (Mp + Mp.T) / 2.0


def _rank_k_psd(Sigma_psd, k_rank):
    p = Sigma_psd.shape[0]
    k = min(max(k_rank, 0), p)
    if k == 0:
        return np.zeros_like(Sigma_psd)
    vals, vecs = np.linalg.eigh(Sigma_psd)
    vals = np.clip(vals, 0.0, None)
    Uk = vecs[:, -k:]                      # eigh ascending -> top-k last
    lamk = vals[-k:]
    SigK = (Uk * lamk) @ Uk.T
    return (SigK + SigK.T) / 2.0


def _pos_spectrum_psd(M):
    vals, vecs = np.linalg.eigh(M)
    pos = vals > 0
    if not pos.any():
        return np.zeros_like(M)
    Mp = (vecs[:, pos] * vals[pos]) @ vecs[:, pos].T
    return (Mp + Mp.T) / 2.0


def _integer_parameter(value, name, minimum=0):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or value < minimum or int(value) != value):
        qualifier = "positive" if minimum == 1 else "nonnegative"
        raise ValueError(f"{name} must be a finite {qualifier} integer")
    return int(value)


def get_surv_band(model, X, i=0, alpha=0.05, approach="naive", nsim=5000,
                  k_rank=10, k_mode="fixed", k_prop=0.99, seed=0,
                  band_grid_size=0):
    """Subject-specific simultaneous bands around ``exp(-CHF)``.

    The nominal simultaneous target is across the returned evaluation
    times for each subject, not jointly across subjects or between omitted
    times. Linear-combination forests return a mean product-limit survival
    curve that can differ from this reference curve; these bands do not
    directly estimate the variation of that mean product-limit estimator.
    Both approaches simulate from the raw CHF covariance. Smoothing changes
    only the marginal standardizing scales, using the Python B-spline
    alternative rather than R's GAM smoother.

    Parameters
    ----------
    model : fitted RLT_surv with var_mode != "none"
    X : test covariates
    i : subject index (0 = all); 1-based like R when positive
    alpha : level or array of levels
    approach : {"naive", "smoothed"}
    nsim : MC simulations for the critical value
    k_rank / k_mode / k_prop : rank truncation controls ("smoothed")
    seed : RNG seed for the MC band (deterministic given seed)
    band_grid_size : optional reduced grid (see predict_var)

    Returns
    -------
    dict with ``lower``, ``upper`` (n, T) for scalar alpha or
    (n, n_alpha, T) for multiple levels, ``timepoints``, per-subject
    ``diagnostics``, and final marginal scales ``marsd`` (n, T).
    """
    if approach not in ("naive", "smoothed"):
        raise ValueError(f"approach {approach!r} not available")
    if k_mode not in ("fixed", "proportion"):
        raise ValueError("k_mode must be 'fixed' or 'proportion'")
    i = _integer_parameter(i, "i")
    nsim = _integer_parameter(nsim, "nsim", minimum=1)
    seed = _integer_parameter(seed, "seed")
    alphas = np.atleast_1d(np.asarray(alpha, dtype=np.float64))
    if (alphas.ndim != 1 or alphas.size == 0 or not np.all(np.isfinite(alphas))
            or np.any(alphas < 0) or np.any(alphas > 0.5)):
        raise ValueError("alpha must contain finite levels in [0, 0.5]")
    if k_mode == "proportion":
        if (isinstance(k_prop, (bool, np.bool_)) or not isinstance(k_prop, Real)
                or not np.isfinite(k_prop) or not 0 < k_prop <= 1):
            raise ValueError("k_prop must be in (0, 1]")
    elif approach == "smoothed":
        k_rank = _integer_parameter(k_rank, "k_rank")

    out = model._predict_raw(X, var_est=True, band_grid_size=band_grid_size)
    tp_grid, _ = model._grid_and_mapping(band_grid_size)
    tp_grid = np.asarray(tp_grid, dtype=np.float64)
    if "Cov" not in out:
        raise ValueError("prediction covariance is unavailable; fit with var_mode")
    chf = np.asarray(out["CHF"], dtype=np.float64)
    cov = np.asarray(out["Cov"], dtype=np.float64)
    if (chf.ndim != 2 or cov.shape != (chf.shape[0], chf.shape[1], chf.shape[1])
            or tp_grid.shape != (chf.shape[1],) or chf.shape[1] == 0):
        raise ValueError("CHF, covariance, and time grid dimensions must agree")
    if not np.all(np.isfinite(chf)) or not np.all(np.isfinite(cov)):
        raise ValueError("CHF and covariance must contain finite values")
    N, nt = chf.shape

    if i == 0:
        allid = np.arange(N)
    else:
        if i > N:
            raise ValueError(f"observation {i} does not exist")
        allid = [i - 1]  # 1-based

    if approach == "smoothed" and nt < 3:
        warnings.warn(
            "Smoothing requires at least three time points; using approach = 'naive'.",
            UserWarning, stacklevel=2)
        approach = "naive"

    eps = np.finfo(float).eps
    lowers, uppers, diags, scales = [], [], [], []

    for idx in allid:
        Sigma_raw = np.asarray(cov[idx], dtype=np.float64)
        raw_sd = np.sqrt(np.maximum(np.diag(Sigma_raw), eps))

        if approach == "naive":
            marsd = raw_sd.copy()
            bandk = np.atleast_1d(
                np.asarray(mc_band(marsd, Sigma_raw, alphas, nsim, seed),
                            dtype=np.float64))
            diag_info = {"timepoint": tp_grid, "raw_sd": raw_sd,
                         "final_sd": marsd}
            ve_info = {"var_explained": np.nan if np.trace(Sigma_raw) == 0 else 1.0,
                       "k_rank": None,
                       "total_var": float(np.trace(Sigma_raw)),
                       "lowrank_var": float(np.trace(Sigma_raw))}
            eig_info = None
        elif approach == "smoothed":
            if k_mode == "proportion":
                vals = np.clip(
                    np.linalg.eigvalsh(_nearest_psd(Sigma_raw))[::-1], 0, None)
                total = vals.sum()
                if total <= eps:
                    k_used = 0
                    cum = np.zeros_like(vals)
                    explained = np.nan if total == 0 else 0.0
                else:
                    cum = np.cumsum(vals) / total
                    k_used = int(np.searchsorted(cum, k_prop) + 1)
                    k_used = min(k_used, vals.shape[0])
                    explained = float(cum[k_used - 1])
                eig_info = {"vals": vals, "cum_ratio": cum,
                            "k_used": k_used, "k_prop": float(k_prop),
                            "var_explained": explained}
            else:
                k_used = min(k_rank, nt)
                eig_info = None

            # C++ tensor-product penalized B-spline smoother (GCV lambda)
            Sigma_s = np.asarray(rlt_tp_smooth(
                Sigma_raw, tp_grid.astype(np.float64).ravel(),
                max(k_used, 4), -1.0))
            Sigma_psd = _nearest_psd(Sigma_s)
            Sigma_smoothK = _rank_k_psd(Sigma_psd, k_used) \
                if k_used > 0 else np.zeros_like(Sigma_psd)
            smoothK_sd = np.sqrt(np.maximum(np.diag(Sigma_smoothK), eps))

            R = (Sigma_raw - Sigma_smoothK)
            R = (R + R.T) / 2.0
            R_pos = _pos_spectrum_psd(R)
            sdB = np.sqrt(max(float(np.diag(R_pos).max()), 0.0))

            tr_smoothK = float(np.clip(np.diag(Sigma_smoothK), 0, None).sum())
            tr_R_pos = float(np.clip(np.diag(R_pos), 0, None).sum())
            tr_total = tr_smoothK + tr_R_pos
            w_k = tr_smoothK / tr_total if tr_total > eps else 1.0
            w_r = 1.0 - w_k if tr_total > eps else 0.0

            marsd = np.sqrt(np.maximum(
                w_k * smoothK_sd ** 2 + w_r * sdB ** 2, eps))
            bandk = np.atleast_1d(
                np.asarray(mc_band(marsd, Sigma_raw, alphas, nsim, seed),
                            dtype=np.float64))
            diag_info = {"timepoint": tp_grid, "raw_sd": raw_sd,
                         "smoothK_sd": smoothK_sd, "final_sd": marsd,
                         "resid_sd": np.full(tp_grid.shape, sdB)}
            ve_info = {
                "var_explained":
                    float(np.trace(Sigma_smoothK) / np.trace(Sigma_raw))
                    if np.trace(Sigma_raw) != 0 else np.nan,
                "k_rank": int(k_used),
                "total_var": float(np.trace(Sigma_raw)),
                "lowrank_var": tr_smoothK,
                "resid_var": float(np.trace(Sigma_raw) - tr_smoothK),
            }

        k_crit = bandk[0] if bandk.shape[0] == 1 else bandk
        lowers.append(np.exp(-chf[idx] - k_crit))
        uppers.append(np.exp(-chf[idx] + k_crit))
        diags.append({"diag_sd": diag_info, "var_explained": ve_info,
                      "eig": eig_info})
        scales.append(marsd)

    return {
        "lower": np.asarray(lowers),
        "upper": np.asarray(uppers),
        "timepoints": tp_grid,
        "marsd": np.asarray(scales),
        "diagnostics": diags,
    }
