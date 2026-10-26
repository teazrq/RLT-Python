"""Numerical and process-safety regressions from the corrected R core."""

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from rlt import _core


def _roots(comb, ntrees=4):
    """Create saved root-only trees without depending on fitting or RNG."""
    splitvar = [np.array([[-1]] if comb else [-1], dtype=np.int64)
                for _ in range(ntrees)]
    values = [np.array([0.0]) for _ in range(ntrees)]
    nodes = [np.array([0], dtype=np.int64) for _ in range(ntrees)]
    weights = [np.array([1.0]) for _ in range(ntrees)]
    if comb:
        return [splitvar, [np.array([[1.0]]) for _ in range(ntrees)],
                values, nodes, nodes, weights]
    return [splitvar, values, nodes, nodes, weights]


def _ij_reference(prediction, counts):
    counts = np.maximum(counts, 0).astype(float)
    centered_prediction = prediction - prediction.mean(axis=1, keepdims=True)
    centered_counts = counts - counts.mean(axis=1, keepdims=True)
    B = prediction.shape[1]
    # Sum squared empirical covariances with each training observation.
    raw = np.sum((centered_prediction @ centered_counts.T / B)**2, axis=1)
    correction = (np.var(counts, axis=1).sum()
                  * np.var(prediction, axis=1) / B)
    return np.maximum(raw - correction, 0)


@pytest.mark.parametrize("comb", [False, True])
@pytest.mark.parametrize("ncores", [1, 2])
def test_ij_uses_fractional_count_moments(comb, ncores):
    counts = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]],
                      dtype=np.int64)
    prediction = np.arange(4.0)[None, :]
    X = np.zeros((1, 1))
    ncat = np.array([1], dtype=np.int64)
    expected = _ij_reference(prediction, counts)
    np.testing.assert_allclose(expected, [0.265625], rtol=0, atol=1e-15)

    reg = _core.RegUniCombForestPred if comb else _core.RegUniForestPred
    out = reg(*_roots(comb), [np.array([v]) for v in prediction[0]],
              X, ncat, counts, 2, True, ncores, 0)
    np.testing.assert_allclose(out["Prediction"], [1.5], rtol=0, atol=1e-15)
    np.testing.assert_allclose(out["Variance"], expected, rtol=0, atol=1e-15)

    cla = _core.ClaUniCombForestPred if comb else _core.ClaUniForestPred
    probabilities = [np.array([[v / 4, 1 - v / 4]]) for v in prediction[0]]
    out = cla(*_roots(comb), probabilities, X, ncat, counts, 2, True, ncores, 0)
    expected_cla = np.repeat(_ij_reference(prediction / 4, counts)[:, None], 2, axis=1)
    np.testing.assert_allclose(out["Prob"], [[.375, .625]], rtol=0, atol=1e-15)
    np.testing.assert_allclose(out["Variance"], expected_cla, rtol=0, atol=1e-15)


@pytest.mark.parametrize("comb", [False, True])
def test_oob_kernel_counts_empty_singleton_and_multiple_oob_trees(comb):
    # The four trees have one, two, zero, and one OOB observations.
    tracking = np.array([[0, 0, 1, 1], [1, 0, 1, 1], [1, 1, 1, 0]],
                        dtype=np.int64)
    fn = _core.Kernel_Self_OOB_Comb if comb else _core.Kernel_Self_OOB
    out = fn(*_roots(comb), np.zeros((3, 1)), np.array([1], dtype=np.int64),
             tracking, 0)
    oob = (tracking == 0).astype(np.int64)
    expected_counts = oob @ oob.T
    np.testing.assert_array_equal(out["N"], expected_counts)
    # Root-only trees place every OOB pair in the same leaf.
    np.testing.assert_array_equal(out["C"], expected_counts)
    np.testing.assert_array_equal(out["Kernel"], (expected_counts > 0).astype(float))


@pytest.mark.parametrize("comb", [False, True])
def test_training_kernel_keeps_inbag_multiplicities(comb):
    tracking = np.array([[2, 0, 1, 0], [0, 2, 0, 1]], dtype=np.int64)
    fn = _core.Kernel_Train_Comb if comb else _core.Kernel_Train
    out = fn(*_roots(comb), np.zeros((1, 1)), np.zeros((2, 1)),
             np.array([1], dtype=np.int64), tracking, 0)
    np.testing.assert_array_equal(out, tracking.sum(axis=1)[None, :])


def _native_weight_fit(model, comb, ncores, *, embedded=False,
                       zero_sample=False, categorical=False, nsplit=0):
    n = 12
    params = _core.CoreParams()
    params.n, params.p = n, 2
    params.ntrees, params.ncores, params.seed = 2, ncores, 1
    params.mtry, params.nsplit = 2, nsplit
    params.nmin = 20 if zero_sample else 1
    params.resample_prob = 1.0
    params.use_obs_w = True
    params.linear_comb = 2 if comb else 1
    params.reinforcement = embedded or comb
    params.embed_ntrees, params.embed_mtry = 50, 2
    params.embed_nmin = 2 if embedded else 20
    params.embed_nsplit = 0
    params.embed_replacement, params.embed_resample_prob = False, .25
    params.embed_protect = 1
    params.obs_track = True

    X = np.column_stack((np.arange(n), np.arange(n)[::-1])).astype(float)
    ncat = np.ones(2, dtype=np.int64)
    weights = np.zeros(n)
    weights[:2] = .5
    if categorical:
        X[:, 0] = np.repeat(np.arange(1, 5), 3)
        X[:, 1] = X[::-1, 0]
        ncat[:] = 4
        weights[:] = 0
        weights[[3, 6]] = .5
    tracking = np.ones((n, 2), dtype=np.int64)
    if zero_sample:
        weights[:] = 0
        weights[0] = 1
        tracking[0, :] = 0

    common = [X, np.arange(1, n + 1, dtype=float), ncat, weights,
              np.full(2, .5), tracking, params]
    if model == "regression":
        fn = _core.RegUniCombForestFit if comb else _core.RegUniForestFit
        key = "NodeAve"
    elif model == "classification":
        fn = _core.ClaUniCombForestFit if comb else _core.ClaUniForestFit
        common[1] = np.arange(n, dtype=np.int64) % 2
        common.insert(3, 2)
        key = "NodeProb"
    else:
        fn = _core.SurvUniCombForestFit if comb else _core.SurvUniForestFit
        common[1] = np.arange(1, n + 1, dtype=np.int64)
        common.insert(2, np.ones(n, dtype=np.int64))
        key = "NodeHaz"

    out = fn(*common)
    forest = out["FittedForest"]
    for node_weight in forest["NodeWeight"]:
        assert np.all(node_weight > 0), (model, comb, ncores, "node mass")
    for tree in forest[key]:
        values = tree if key == "NodeHaz" else [tree]
        for node_values in values:
            assert np.all(np.isfinite(node_values)), (model, comb, ncores, key)


def _check_weight_boundaries(ncores):
    """Run inside a child because the old OpenMP code can abort the process."""
    for model in ("regression", "classification", "survival"):
        for comb in (False, True):
            with pytest.raises(RuntimeError, match="sampled tree has zero total observation weight"):
                _native_weight_fit(model, comb, ncores, zero_sample=True)
            # This is the same supported embedded route tested in current R.
            if not (model == "survival" and comb):
                with pytest.raises(RuntimeError, match="sampled tree has zero total observation weight"):
                    _native_weight_fit(model, comb, ncores, embedded=True)
            for nsplit in (0, 3):
                _native_weight_fit(model, comb, ncores, nsplit=nsplit)
                if model != "survival":
                    _native_weight_fit(model, comb, ncores,
                                       categorical=True, nsplit=nsplit)


@pytest.mark.parametrize("ncores", [1, 2])
def test_zero_weight_samples_raise_catchable_errors_and_positive_nodes_stay_finite(ncores):
    path = str(Path(__file__).resolve())
    repo = str(Path(__file__).resolve().parents[1])
    code = (f"import runpy, sys; sys.path.insert(0, {repo!r}); "
            f"ns = runpy.run_path({path!r}); ns['_check_weight_boundaries']({ncores})")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
