"""Deterministic algebra fixtures matching current RLT statistical corrections.

These isolate prediction and covariance from tree fitting and Monte Carlo
coverage. The independent R fixture is test-statistical-corrections.R.
"""
from types import SimpleNamespace
import warnings

import numpy as np
import pytest
from scipy.stats import norm

from rlt import RLT_surv, get_surv_band
from rlt._core import cindex, mc_band, rlt_tp_smooth


def _root_model(hazards, comb=False):
    b, t = hazards.shape
    model = RLT_surv(n_estimators=b, n_jobs=1, linear_comb=2 if comb else 1)
    model.forest_ = {
        "SplitVar": [np.array([[-1, 0]] if comb else [-1], dtype=np.int64) for _ in range(b)],
        "SplitValue": [np.zeros(1) for _ in range(b)],
        "LeftNode": [np.zeros(1, dtype=np.int64) for _ in range(b)],
        "RightNode": [np.zeros(1, dtype=np.int64) for _ in range(b)],
        "NodeWeight": [np.ones(1) for _ in range(b)],
        "NodeHaz": [[np.r_[0., row]] for row in hazards],
    }
    if comb:
        model.forest_["SplitLoad"] = [np.zeros((1, 2)) for _ in range(b)]
    model.n_features_in_ = 2
    model.ncat_ = np.ones(2, dtype=np.int64)
    model.timepoints_ = np.arange(1, t + 1, dtype=float)
    model.nfail_ = t
    model.params_ = SimpleNamespace(var_mode=1)
    model._cat_shift_cols_ = []
    model.obstrack_ = np.vstack([
        [0, 0, 0, 0, 2, 2, 2, 2],
        [0, 0, 1, 1, 0, 0, 1, 1],
        [0, 1, 0, 1, 0, 1, 0, 2],
        np.ones((max(2*t, 4) - 3, b), dtype=np.int64),
    ]).astype(np.int64)
    return model


def _cov_reference(pred, counts, mode):
    b = len(pred)
    pc = pred - pred.mean(axis=0)
    if mode == "matched":
        delta = pred[:b//2] - pred[b//2:]
        raw = (delta.T @ delta - pc.T @ pc) / b
    elif mode == "ij":
        counts = np.maximum(counts, 0).astype(float)
        nc = counts - counts.mean(axis=1, keepdims=True)
        influence = nc @ pc / b
        count_var = (np.mean(counts**2, axis=1) - np.mean(counts, axis=1)**2).sum()
        raw = influence.T @ influence - count_var * pc.T @ pc / b**2
    else:
        oob = counts == 0
        oob = oob[oob.sum(axis=1) > 0]
        jack = np.array([pred[row].mean(axis=0) for row in oob])
        delta = jack - pred.mean(axis=0)
        nv = len(oob)
        raw = (nv - 1) / nv * delta.T @ delta - (np.e - 1) * nv / b**2 * pc.T @ pc
    vals, vecs = np.linalg.eigh((raw + raw.T) / 2)
    return (vecs * np.maximum(vals, 1e-6)) @ vecs.T


@pytest.mark.parametrize("comb", [False, True])
@pytest.mark.parametrize("mode", ["matched", "ij", "jack"])
@pytest.mark.parametrize("grid", [0, 1, 2, 8, 15])
def test_full_tree_chf_is_selected_before_covariance(comb, mode, grid):
    h = (np.outer([.02, .03, .04, .05, .15, .16, .17, .18], np.arange(1, 13)/12)
         + np.outer([.01, .02, .01, .02, .01, .02, .01, .03], np.arange(12, 0, -1)/12))
    model = _root_model(h, comb)
    x = np.ones((2, 2))
    out = model._predict_raw(x, var_est=True, var_mode=mode,
                             band_grid_size=grid, keep_all=True)
    plain = model._predict_raw(x, band_grid_size=grid)
    _, ix = model._grid_and_mapping(grid)
    tree_chf = np.cumsum(h, axis=1)[:, ix]
    expected_surv = (np.cumprod(1-h, axis=1)[:, ix].mean(axis=0) if comb
                     else np.exp(-tree_chf.mean(axis=0)))
    for subject in range(2):
        np.testing.assert_allclose(out["CHF"][subject], tree_chf.mean(axis=0), atol=1e-12)
        np.testing.assert_allclose(out["Survival"][subject], expected_surv, atol=1e-12)
        np.testing.assert_allclose(out["Hazard"][subject], h[:, ix].mean(axis=0), atol=1e-12)
        np.testing.assert_allclose(out["AllHazard"][subject], h[:, ix], atol=1e-12)
        np.testing.assert_allclose(out["Cov"][subject],
                                   _cov_reference(tree_chf, model.obstrack_, mode), atol=1e-12)
    for key in ("CHF", "Survival", "Hazard"):
        np.testing.assert_array_equal(out[key], plain[key])
    assert "AllHazard" not in plain


@pytest.mark.parametrize("comb", [False, True])
def test_root_curve_endpoint_survives_grid_reduction(comb):
    model = _root_model(np.tile(1 / np.arange(6, 0, -1), (8, 1)), comb)
    for grid in (0, 2):
        out = model._predict_raw(np.ones((1, 2)), band_grid_size=grid)
        assert out["CHF"][0, -1] == pytest.approx(2.45)
        assert out["Survival"][0, -1] == pytest.approx(0. if comb else np.exp(-2.45))


@pytest.mark.parametrize("mode", ["matched", "ij", "jack"])
def test_constant_tree_covariance_retains_target_eigenvalue_floor(mode):
    model = _root_model(np.full((8, 6), .1))
    out = model._predict_raw(np.ones((1, 2)), var_est=True, var_mode=mode,
                             band_grid_size=2)
    np.testing.assert_allclose(out["Cov"][0], np.eye(2)*1e-6, atol=1e-14)


def test_reduced_grid_uses_R_type1_quantiles():
    model = _root_model(np.full((8, 12), .1))
    np.testing.assert_array_equal(model._grid_and_mapping(5)[0], [1, 4, 7, 10, 12])


class _BandFixture:
    def __init__(self, covariance, chf=None):
        self.covariance = np.asarray(covariance, dtype=float)
        self.timepoints_ = np.arange(1, len(covariance)+1, dtype=float)
        self.chf = (np.arange(1, len(covariance)+1)/3 if chf is None else np.asarray(chf))
        self.calls = 0

    def _grid_and_mapping(self, size):
        return self.timepoints_, np.arange(len(self.timepoints_))

    def predict_details(self, x, **kwargs):
        self.calls += 1
        return {"CHF": self.chf[None], "Cov": self.covariance[None],
                "timepoints": self.timepoints_,
                "Survival": np.zeros((1, len(self.chf)))}


def test_band_uses_CHF_reference_from_one_prediction_call():
    model = _BandFixture(np.eye(3)*.01, [.1, .5, 1.2])
    band = get_surv_band(model, None, nsim=100, alpha=[.05, .1])
    assert model.calls == 1
    assert band["lower"].shape == (1, 2, 3)
    np.testing.assert_allclose(np.sqrt(band["lower"]*band["upper"]),
                               np.broadcast_to(np.exp(-model.chf), (1, 2, 3)))
    np.testing.assert_allclose(band["marsd"], [[.1, .1, .1]])


@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.parametrize("mode", ["fixed", "proportion"])
def test_short_smoothed_grid_falls_back_to_naive(n, mode):
    model = _BandFixture(np.diag(np.array([.01, .09])[:n]))
    naive = get_surv_band(model, None, nsim=100)
    with pytest.warns(UserWarning, match="using approach = 'naive'"):
        smooth = get_surv_band(model, None, nsim=100, approach="smoothed",
                               k_rank=0, k_mode=mode)
    np.testing.assert_array_equal(smooth["lower"], naive["lower"])
    np.testing.assert_array_equal(smooth["upper"], naive["upper"])
    np.testing.assert_array_equal(smooth["marsd"], naive["marsd"])


def test_smoothing_rank_cap_and_zero_variance_diagnostics():
    model = _BandFixture(np.eye(4)*.01)
    band = get_surv_band(model, None, approach="smoothed", k_rank=99, nsim=20)
    assert band["diagnostics"][0]["var_explained"]["k_rank"] == 4
    zero = _BandFixture(np.zeros((4, 4)))
    band = get_surv_band(zero, None, approach="smoothed", k_mode="proportion", nsim=20)
    diag = band["diagnostics"][0]
    assert diag["eig"]["k_used"] == 0
    assert np.isnan(diag["eig"]["var_explained"])
    assert np.isnan(diag["var_explained"]["var_explained"])
    np.testing.assert_allclose(band["lower"], np.exp(-zero.chf)[None])
    np.testing.assert_array_equal(band["lower"], band["upper"])


def test_smoothed_band_simulates_raw_covariance(monkeypatch):
    import rlt.bands as bands
    t = np.arange(6.)
    sigma = .01*np.exp(-np.abs(np.subtract.outer(t, t)))
    model = _BandFixture(sigma)
    seen = []

    def capture(scales, covariance, alphas, nsim, seed):
        seen.append(np.array(covariance))
        return np.zeros((len(alphas), len(scales)))

    monkeypatch.setattr(bands, "mc_band", capture)
    band = get_surv_band(model, None, approach="smoothed", k_rank=2, nsim=20)
    assert len(seen) == 1
    np.testing.assert_array_equal(seen[0], sigma)
    assert np.isfinite(band["marsd"]).all()


@pytest.mark.parametrize("kwargs,match", [
    ({"nsim": 0}, "nsim"), ({"nsim": 2.5}, "nsim"),
    ({"i": 1.5}, "i"), ({"i": -1}, "i"), ({"seed": -1}, "seed"),
    ({"alpha": np.nan}, "alpha"), ({"alpha": []}, "alpha"),
    ({"alpha": [[.05]]}, "alpha"), ({"alpha": .6}, "alpha"),
    ({"approach": "bad"}, "approach"), ({"k_mode": "bad"}, "k_mode"),
    ({"approach": "smoothed", "k_rank": -1}, "k_rank"),
    ({"approach": "smoothed", "k_rank": 1.5}, "k_rank"),
    ({"k_mode": "proportion", "k_prop": np.nan}, "k_prop"),
    ({"k_mode": "proportion", "k_prop": 0}, "k_prop"),
])
def test_invalid_band_controls_fail_before_prediction(kwargs, match):
    model = _BandFixture(np.eye(3))
    with pytest.raises(ValueError, match=match):
        get_surv_band(model, None, **kwargs)
    assert model.calls == 0


def test_MC_accepts_singular_covariance_with_correct_marginal_scaling():
    alpha = np.array([.05, .1])
    width = mc_band(np.array([2., 2.]), np.full((2, 2), 4.), alpha, 50000, 42)
    np.testing.assert_allclose(width[:, 0], width[:, 1], atol=1e-12)
    np.testing.assert_allclose(width[:, 0], 2*norm.ppf(1-alpha/2), atol=.1, rtol=0)
    zero = mc_band(np.ones(2), np.zeros((2, 2)), alpha, 20, 42)
    np.testing.assert_array_equal(zero, np.zeros((2, 2)))


def test_MC_rejects_indefinite_covariance():
    with pytest.raises(ValueError, match="positive semidefinite"):
        mc_band(np.ones(2), np.array([[1., 2.], [2., 1.]]), np.array([.05]), 20, 0)


def test_Python_smoother_is_symmetric_and_uses_actual_time_coordinates():
    t = np.arange(1., 13.)
    sigma = np.exp(-np.abs(np.subtract.outer(t, t))/3)*np.sqrt(np.outer(t, t))/20
    regular = rlt_tp_smooth(sigma, t, 6, -1.)
    irregular = rlt_tp_smooth(sigma, t**3, 6, -1.)
    assert np.isfinite(regular).all() and np.isfinite(irregular).all()
    np.testing.assert_allclose(regular, regular.T, atol=1e-12)
    np.testing.assert_allclose(irregular, irregular.T, atol=1e-12)
    assert not np.allclose(regular, irregular)


def test_nonfinite_risk_returns_nan_and_event1_means_failure():
    assert cindex(np.array([1., 2.]), np.array([1, 0]), np.array([2., 1.])) == 1
    assert np.isnan(cindex(np.array([1., 2.]), np.array([1, 1]), np.array([np.nan, 1.])))


def test_all_categorical_LC_keeps_categorical_codes_out_of_loadings():
    x = np.column_stack([np.tile([0, 1, 2], 8), np.tile([2, 0, 1], 8)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = RLT_surv(n_estimators=6, min_samples_leaf=2, linear_comb=2,
                         categorical_features=[0, 1], random_state=4, n_jobs=1).fit(
                             x, (np.arange(1., 25.), np.ones(24)))
    assert all(np.asarray(tree)[0, 0] == -1 for tree in model.forest_["SplitVar"])
    assert all(not np.any(load) for load in model.forest_["SplitLoad"])


def test_band_requires_a_fitted_forest():
    from sklearn.exceptions import NotFittedError

    with pytest.raises(NotFittedError):
        get_surv_band(RLT_surv(), np.zeros((1, 2)))
