"""Shared R/native repairs, with checks that the intended branches run."""

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from rlt import _core


def _params(n=64, p=5, *, comb=False, ncores=1):
    cp = _core.CoreParams()
    cp.n, cp.p, cp.ntrees = n, p, 2
    cp.mtry, cp.nmin, cp.nsplit = p, n - 1, 0
    cp.linear_comb = 2 if comb else 1
    cp.linear_comb_method = 1
    cp.embed_nmin, cp.embed_protect = 100, 1
    cp.replacement, cp.resample_prob = False, 1.
    cp.ncores, cp.seed = ncores, 7
    return cp


def _fit(model, comb, X, y, cp, *, weights=None, var_prob=None, tracking=None):
    n, p = X.shape
    cp.use_obs_w, cp.use_var_prob = weights is not None, var_prob is not None
    weights = np.empty(0) if weights is None else np.asarray(weights, dtype=float)
    var_prob = np.empty(0) if var_prob is None else np.asarray(var_prob, dtype=float)
    tracking = (np.empty((0, 0), dtype=np.int64) if tracking is None
                else np.asarray(tracking, dtype=np.int64))
    ncat = np.ones(p, dtype=np.int64)
    suffix = "CombForestFit" if comb else "ForestFit"
    if model == "regression":
        return getattr(_core, "RegUni" + suffix)(
            X, np.asarray(y, dtype=float), ncat, weights, var_prob, tracking, cp)
    if model == "classification":
        return getattr(_core, "ClaUni" + suffix)(
            X, np.asarray(y, dtype=np.int64), ncat, 2, weights, var_prob, tracking, cp)
    time, event = y
    return getattr(_core, "SurvUni" + suffix)(
        X, np.asarray(time, dtype=np.int64), np.asarray(event, dtype=np.int64),
        ncat, weights, var_prob, tracking, cp)


def _check_sampler_boundaries(ncores):
    rng = np.random.default_rng(183)
    X = rng.normal(size=(64, 5))
    responses = {"regression": X[:, 0] + X[:, 1],
                 "classification": (X[:, 0] > 0).astype(np.int64),
                 "survival": (np.arange(1, 65), np.ones(64, dtype=np.int64))}
    # Regression/classification LC marginal screening uses uniform sampling.
    # Exercise each real outer weighted-sampler caller.
    for model, comb in [("regression", False), ("classification", False),
                        ("survival", False), ("survival", True)]:
        cp = _params(comb=comb, ncores=ncores)
        cp.mtry = 4
        with pytest.raises((RuntimeError, ValueError), match="fewer positive weights"):
            _fit(model, comb, X, responses[model], cp, var_prob=[1, 0, 0, 0, 0])
        # Positive support exists, but floating subtraction exhausts the mass.
        cp.mtry = 2
        with pytest.raises((RuntimeError, ValueError), match="remaining weight"):
            _fit(model, comb, X, responses[model], cp,
                 var_prob=[1, 1e-300, 1e-300, 0, 0])
        cp.mtry = 4
        supported = _fit(model, comb, X, responses[model], cp,
                         var_prob=[.1, .2, .3, .4, 0])
        assert all(np.isfinite(v).all() for v in supported["FittedForest"]["NodeWeight"])
        # R explicitly bypasses probabilities when all variables are requested.
        cp.mtry = 5
        full = _fit(model, comb, X, responses[model], cp,
                    var_prob=[1, 0, 0, 0, 0])
        plain = _fit(model, comb, X, responses[model], cp)
        for key in ("SplitVar", "SplitValue", "LeftNode", "RightNode"):
            for actual, expected in zip(full["FittedForest"][key],
                                        plain["FittedForest"][key]):
                np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("ncores", [1, 2])
def test_partial_weighted_sampling_terminates_and_full_pool_bypass_is_preserved(ncores):
    # The old implementation hangs, and an uncaught OpenMP throw can abort.
    path = str(Path(__file__).resolve())
    repo = str(Path(__file__).resolve().parents[1])
    code = (f"import runpy,sys; sys.path.insert(0,{repo!r}); "
            f"ns=runpy.run_path({path!r}); ns['_check_sampler_boundaries']({ncores})")
    child = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, timeout=30)
    assert child.returncode == 0, child.stdout + child.stderr


@pytest.mark.parametrize("ncores", [1, 2])
def test_weighted_SIR_skips_empty_slices_and_finds_a_real_direction(ncores):
    rng = np.random.default_rng(5)
    X = rng.normal(size=(64, 4))
    y = np.arange(64, dtype=float)
    X[:, 0] = y / 64 + .1 * X[:, 0]
    weights = np.r_[np.zeros(16), np.ones(48)] / 48
    cp = _params(p=4, comb=True, ncores=ncores)
    cp.linear_comb_method = 4
    forest = _fit("regression", True, X, y, cp, weights=weights)["FittedForest"]
    for variables, load, means in zip(forest["SplitVar"], forest["SplitLoad"],
                                       forest["NodeAve"]):
        assert np.count_nonzero(np.abs(load[0]) > 1e-12) == 2
        assert np.all(variables[0] >= 0)
        assert np.isfinite(load).all() and np.isfinite(means).all()
        # Marginal-screen naive fallback has equal absolute loadings of one.
        assert not np.allclose(np.abs(load[0]), 1., rtol=0, atol=1e-8)
        np.testing.assert_allclose(np.linalg.norm(load[0]), 1., atol=1e-12)


def test_one_positive_weight_SIR_slice_uses_the_existing_naive_fallback():
    X = np.random.default_rng(5).normal(size=(64, 4))
    y = np.arange(64, dtype=float)
    X[:, 0] = y / 64 + .1 * X[:, 0]
    weights = np.r_[np.zeros(56), np.ones(8)] / 8
    cp = _params(p=4, comb=True)
    cp.linear_comb_method = 4
    forest = _fit("regression", True, X, y, cp, weights=weights)["FittedForest"]
    for variables, load in zip(forest["SplitVar"], forest["SplitLoad"]):
        # Marginal screening supplies unit VI. Preserve the established
        # unweighted-correlation naive fallback rather than an arbitrary
        # eigenbasis for the zero between-slice matrix.
        expected = np.sign([np.corrcoef(X[:, j], y)[0, 1] for j in variables[0]])
        np.testing.assert_array_equal(load[0], expected)
        assert np.count_nonzero(load[0]) == 2


def _survival_data():
    rng = np.random.default_rng(45682)
    X = rng.normal(size=(64, 5))
    X[:, 3] += .35 * X[:, 0]
    signal = 1.5 * X[:, 0] - .7 * X[:, 1] + .25 * X[:, 2] * X[:, 3]
    # Consume the regression/classification draws in the live audit fixture.
    rng.normal(scale=.3, size=64)
    rng.normal(scale=.4, size=64)
    time = np.maximum(1, np.round(rng.exponential(np.exp(-.4 * signal)) * 4)).astype(np.int64)
    event = (np.arange(64) % 5 != 0).astype(np.int64)
    return X, time, event


def _sup_score(projected, time, event, cut):
    """Independent finite-population Renyi score on a proposed partition."""
    failures = np.unique(time[event == 1])
    total_risk = np.array([np.sum(time >= t) for t in failures], dtype=float)
    total_fail = np.array([np.sum((time == t) & (event == 1)) for t in failures])
    left = projected <= cut
    left_risk = np.array([np.sum(left & (time >= t)) for t in failures], dtype=float)
    left_fail = np.array([np.sum(left & (time == t) & (event == 1)) for t in failures])
    right_risk = total_risk - left_risk
    w = left_risk * right_risk / total_risk
    valid = (left_risk > 0) & (right_risk > 0)
    increment = np.zeros(len(failures))
    increment[valid] = (left_fail[valid] / left_risk[valid]
                        - (total_fail[valid] - left_fail[valid]) / right_risk[valid])
    variance = np.zeros(len(failures))
    enough = total_risk > 1
    variance[enough] = ((total_risk[enough] - total_fail[enough])
                        / (total_risk[enough] - 1) * total_fail[enough] / total_risk[enough])
    denominator = np.sum(variance * w)
    return np.max(np.cumsum(w * increment)**2) / denominator if denominator > 0 else -1.


@pytest.mark.parametrize("nsplit", [0, 3])
@pytest.mark.parametrize("ncores", [1, 2])
def test_unweighted_LC_suplogrank_has_real_loadings_and_finite_cutpoint_scores(nsplit, ncores):
    X, time, event = _survival_data()
    cp = _params(comb=True, ncores=ncores)
    cp.split_rule, cp.linear_comb_method, cp.nsplit, cp.alpha = 2, 2, nsplit, .1
    forest = _fit("survival", True, X, (time, event), cp)["FittedForest"]
    for variables, load, cut, hazards in zip(forest["SplitVar"], forest["SplitLoad"],
                                             forest["SplitValue"], forest["NodeHaz"]):
        assert np.count_nonzero(np.abs(load[0]) > 1e-12) == 2
        projected = X[:, variables[0]] @ load[0]
        chosen_score = _sup_score(projected, time, event, cut[0])
        assert np.isfinite(chosen_score) and chosen_score > 0
        assert all(np.isfinite(h).all() for h in hazards)
        if nsplit == 0:
            ranked = np.sort(projected)
            # Match the documented child-fraction restriction.
            nmin = max(1, int(len(time) * cp.alpha))
            cuts = (ranked[nmin - 1:len(time) - nmin]
                    + ranked[nmin:len(time) - nmin + 1]) / 2
            scores = [_sup_score(projected, time, event, c) for c in cuts]
            np.testing.assert_allclose(chosen_score, max(scores), rtol=1e-12, atol=1e-12)


def test_suplogrank_zero_variance_cannot_win_a_split():
    X, _, _ = _survival_data()
    cp = _params(comb=True)
    cp.split_rule = 2
    forest = _fit("survival", True, X, (np.ones(64), np.ones(64)), cp)["FittedForest"]
    assert all(v[0, 0] == -1 for v in forest["SplitVar"])
    assert all(np.isfinite(h).all() for tree in forest["NodeHaz"] for h in tree)


@pytest.mark.parametrize("model", ["regression", "classification", "survival"])
@pytest.mark.parametrize("comb", [False, True])
def test_native_OOB_counts_do_not_require_retaining_tracking(model, comb):
    X = np.random.default_rng(36).normal(size=(32, 5))
    y = {"regression": np.arange(32, dtype=float),
         "classification": np.arange(32) % 2,
         "survival": (np.arange(1, 33), np.ones(32))}[model]
    cp = _params(n=32, comb=comb)
    cp.ntrees, cp.nmin, cp.replacement, cp.resample_prob = 3, 100, True, .8
    tracking = np.ones((32, 3), dtype=np.int64)
    tracking[1], tracking[2], tracking[3] = -1, 0, [-1, 0, 1]
    out = _fit(model, comb, X, y, cp, tracking=tracking)
    assert "ObsTrack" not in out
    np.testing.assert_array_equal(out["OOBCount"], np.sum(tracking == 0, axis=1))
    assert out["OOBCount"].shape == (32,)
    cp.obs_track = True
    kept = _fit(model, comb, X, y, cp, tracking=tracking)
    np.testing.assert_array_equal(kept["OOBCount"], np.sum(kept["ObsTrack"] == 0, axis=1))
    # Nominal full sampling disables OOB output even when a preset contains 0.
    cp.replacement, cp.resample_prob = False, 1.
    disabled = _fit(model, comb, X, y, cp, tracking=tracking)
    assert "OOBCount" not in disabled and "Prediction" not in disabled
