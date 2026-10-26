"""Internal: shared parameter handling for rlt estimators."""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np

from ._core import CoreParams

# public sklearn-ish names -> CoreParams attribute names
_PARAM_MAP = {
    "n_estimators": "ntrees",
    "min_samples_leaf": "nmin",
    "mtry": "mtry",
    "nsplit": "nsplit",
    "resample_replace": "replacement",
    "resample_prob": "resample_prob",
    "importance": "importance",
    "resample_track": "obs_track",
    "var_mode": "var_mode",
    "linear_comb": "linear_comb",
    "linear_comb_method": "linear_comb_method",
    "split_rule": "split_rule",
    "alpha": "alpha",
    "embed_ntrees": "embed_ntrees",
    "embed_mtry": "embed_mtry",
    "embed_nmin": "embed_nmin",
    "embed_nsplit": "embed_nsplit",
    "embed_resample_replace": "embed_replacement",
    "embed_resample_prob": "embed_resample_prob",
    "embed_mute": "embed_mute",
    "embed_protect": "embed_protect",
    "embed_threshold": "embed_threshold",
    "reinforcement": "reinforcement",
}

# importance: "none"/"permute"/"distribute" -> 0/1/2
_IMPORTANCE_CODES = {"none": 0, "permute": 1, "distribute": 2}

# var_mode: "none"/"matched"/"ij"/"jack" -> 0/1/2/3
_VAR_MODE_CODES = {"none": 0, "matched": 1, "ij": 2, "jack": 3}


def _encode(value, table, name):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        value = int(value)
    if isinstance(value, str):
        key = value.lower()
        if key not in table:
            raise ValueError(
                f"{name}={value!r} not recognized; choose from {sorted(table)}"
            )
        return table[key]
    number = _number(value, name)
    if number != int(number) or int(number) not in table.values():
        raise ValueError(f"{name} must be one of {sorted(table)} or {sorted(table.values())}")
    return int(number)


def _number(value, name):
    if isinstance(value, Integral):
        return int(value)
    if not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite numeric scalar")
    return float(value)


def _integer(value, name, minimum=0, maximum=None):
    # Primary R parameters are converted to integers before range checks.
    number = int(_number(value, name))
    if number < minimum or (maximum is not None and number > maximum):
        bound = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
        raise ValueError(f"{name} must be {bound}")
    return number


def check_grid_size(value, name):
    """Grid sizes are exact nonnegative integer scalars, as in R."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite nonnegative integer scalar")
    number = _number(value, name)
    if number < 0 or number != int(number):
        raise ValueError(f"{name} must be a finite nonnegative integer scalar")
    return int(number)


def _flag(value, name):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return bool(_number(value, name))


def _clamp(value, name, minimum, maximum=None):
    number = max(minimum, _number(value, name))
    return min(maximum, number) if maximum is not None else number


def build_core_params(estimator, n: int, p: int, seed: int) -> CoreParams:
    """Translate an estimator's public hyper-parameters to CoreParams."""
    cp = CoreParams()
    cp.n = int(n)
    cp.p = int(p)

    imp = _encode(estimator.importance, _IMPORTANCE_CODES, "importance")
    if imp is None:
        imp = 0
    vm = _encode(estimator.var_mode, _VAR_MODE_CODES, "var_mode")
    if vm is None:
        vm = 0

    replacement = _flag(estimator.resample_replace, "resample_replace")
    resample_prob = (1.0 if replacement else 0.8) if estimator.resample_prob is None else _number(estimator.resample_prob, "resample_prob")
    if not 0 < resample_prob <= 1:
        raise ValueError("resample_prob must be in (0, 1]")
    linear_comb = min(p, _integer(estimator.linear_comb, "linear_comb", 1))
    protect = (int(np.ceil(np.log(n))) if estimator.embed_protect is None
               else int(_clamp(estimator.embed_protect, "embed_protect", 0, p)))
    reinforcement = _flag(estimator.reinforcement, "reinforcement")
    if (reinforcement or linear_comb > 1) and protect < 1:
        raise ValueError("embed_protect must be positive when reinforcement or linear combinations are enabled")
    values = {
        "ntrees": _integer(estimator.n_estimators, "n_estimators", 1),
        "nmin": _integer(estimator.min_samples_leaf, "min_samples_leaf", 1),
        "mtry": _integer(estimator.mtry if estimator.mtry is not None else max(1, p // 2), "mtry", 1, p),
        "nsplit": _integer(estimator.nsplit, "nsplit", 0),
        "replacement": replacement,
        "resample_prob": resample_prob,
        "use_obs_w": False,  # set by fit() when sample_weight is given
        "use_var_prob": False,  # set by fit() when var_prob is given
        "importance": imp,
        "reinforcement": reinforcement,
        "obs_track": _flag(estimator.resample_track, "resample_track") or vm > 0,
        "var_mode": vm,
        "linear_comb": linear_comb,
        "alpha": _clamp(estimator.alpha, "alpha", 0, 0.5),
        "split_rule": 1,
        "linear_comb_method": 1,
        "embed_ntrees": int(_clamp(estimator.embed_ntrees, "embed_ntrees", 1)),
        "embed_mtry": _clamp(estimator.embed_mtry, "embed_mtry", 0, p),
        "embed_nmin": int(_clamp(estimator.embed_nmin, "embed_nmin", 1)),
        "embed_nsplit": int(_clamp(estimator.embed_nsplit, "embed_nsplit", 0)),
        "embed_replacement": _flag(estimator.embed_resample_replace, "embed_resample_replace"),
        "embed_resample_prob": _clamp(estimator.embed_resample_prob, "embed_resample_prob", 0, 0.99),
        "embed_mute": _clamp(estimator.embed_mute, "embed_mute", 0, p),
        "embed_protect": protect,
        "embed_threshold": _clamp(estimator.embed_threshold, "embed_threshold", 0, 1),
        "ncores": 0 if estimator.n_jobs is None or _number(estimator.n_jobs, "n_jobs") <= 0 else _integer(estimator.n_jobs, "n_jobs", 1),
        "verbose": _integer(estimator.verbose, "verbose", 0),
        "seed": _integer(seed, "random_state", 0),
    }
    for k, v in values.items():
        setattr(cp, k, v)
    return cp


def _normalize_weights(value, length, name):
    weights = np.asarray(value, dtype=np.float64).ravel()
    if weights.size != length:
        raise ValueError(f"{name} must have length {length}, got {weights.size}")
    if not np.all(np.isfinite(weights)):
        raise ValueError(f"{name} must contain finite weights")
    if np.any(weights < 0):
        raise ValueError(f"{name} cannot be negative")
    if not weights.size or not np.any(weights > 0):
        raise ValueError(f"{name} must contain at least one positive weight")
    weights = weights / weights.max()
    return weights / weights.sum()


def check_var_prob(var_prob, p):
    return _normalize_weights(var_prob, p, "var_prob")


def check_obs_weight(sample_weight, n):
    return _normalize_weights(sample_weight, n, "sample_weight")
