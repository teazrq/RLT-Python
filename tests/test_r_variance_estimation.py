"""Port of RLT testthat/test-variance-estimation.R.

Param map: param.control list(var.mode=...) -> var_mode kwarg ("matched"/
"ij"/"jack"; R's "IJ" upper-case maps to lower-case "ij"), ncores->n_jobs,
predict(var.est=TRUE) -> .predict_var(X) (regression) or RLT_cla.predict_var
(classification), both returning (prediction/prob, variance) with negative
estimates cleaned to NaN.

Former API gaps (now ported): classification predict variance (matched/IJ/
jack via predict_var), VarVI surfacing (fit.var_vi_).
"""
import numpy as np
import pytest

from rlt import RLT_reg, RLT_cla
from helpers import generate_simple_regression, generate_classification_data


@pytest.fixture(scope="module")
def d_reg():
    return generate_simple_regression(n=80, p=10)


@pytest.fixture(scope="module")
def d_cla():
    return generate_classification_data(n=80, p=10)


# --- Regression matched --------------------------------------------------------

def test_matched_regression_predict_returns_variance(d_reg):
    fit = RLT_reg(n_estimators=30, var_mode="matched", n_jobs=2, random_state=1).fit(
        d_reg["X"], d_reg["y"])
    pred, var = fit.predict_var(d_reg["X"])
    assert np.asarray(var).dtype.kind == "f"
    assert np.asarray(var).shape[0] == d_reg["n"]


def test_matched_regression_variance_non_negative_or_nan(d_reg):
    fit = RLT_reg(n_estimators=30, var_mode="matched", n_jobs=2, random_state=1).fit(
        d_reg["X"], d_reg["y"])
    _, var = fit.predict_var(d_reg["X"])
    assert np.all((var >= 0) | np.isnan(var))


def test_matched_regression_forces_subsample_prob_half(d_reg):
    # R: resample.replace == 0 and resample.prob == 0.5
    fit = RLT_reg(n_estimators=30, var_mode="matched", n_jobs=2, random_state=1).fit(
        d_reg["X"], d_reg["y"])
    assert fit.params_.replacement is False
    assert fit.params_.resample_prob == 0.5


def test_matched_regression_varvi_returned(d_reg):
    # R: fit$VarVI, length p. Surfaced by the Python estimator layer since
    # the VarVI port (core computes it when importance && var_mode == 1).
    fit = RLT_reg(n_estimators=30, importance="permute", var_mode="matched",
                  n_jobs=2, random_state=1).fit(d_reg["X"], d_reg["y"])
    assert np.asarray(fit.var_vi_).shape == (d_reg["p"],)


def test_matched_regression_varvi_numeric(d_reg):
    # R: expect_type(fit$VarVI, "double"). Ported: float array, length p,
    # finite (matched U-statistic variance estimates).
    fit = RLT_reg(n_estimators=30, importance="permute", var_mode="matched",
                  n_jobs=2, random_state=1).fit(d_reg["X"], d_reg["y"])
    vvi = np.asarray(fit.var_vi_)
    assert vvi.dtype.kind == "f"
    assert vvi.shape == (d_reg["p"],)
    assert np.all(np.isfinite(vvi))


def test_matched_regression_predict_without_var_est_has_no_variance(d_reg):
    # R: predict() without var.est -> pred$Variance is NULL. Python:
    # .predict() returns a plain prediction array.
    fit = RLT_reg(n_estimators=30, var_mode="matched", n_jobs=2, random_state=1).fit(
        d_reg["X"], d_reg["y"])
    out = fit.predict(d_reg["X"])
    assert isinstance(out, np.ndarray)
    assert out.ndim == 1


# --- Classification matched ----------------------------------------------------

def test_matched_classification_predict_returns_variance_matrix(d_cla):
    # R: predict(var.est=TRUE) returns an n x nclass Variance matrix.
    # Ported: RLT_cla.predict_var returns (prob, variance).
    fit = RLT_cla(n_estimators=30, var_mode="matched", n_jobs=2,
                  random_state=1).fit(d_cla["X"], d_cla["y"])
    prob, var = fit.predict_var(d_cla["X"])
    assert prob.shape == (d_cla["n"], fit.classes_.shape[0])
    assert var.shape == (d_cla["n"], fit.classes_.shape[0])


def test_matched_classification_variance_non_negative_or_nan(d_cla):
    # R: Variance >= 0 or NA (clean.variance). Ported.
    fit = RLT_cla(n_estimators=30, var_mode="matched", n_jobs=2,
                  random_state=1).fit(d_cla["X"], d_cla["y"])
    _, var = fit.predict_var(d_cla["X"])
    assert np.all((var >= 0) | np.isnan(var))


def test_matched_classification_varvi_returned(d_cla):
    # R: fit$VarVI, length p. Ported (was API gap).
    fit = RLT_cla(n_estimators=30, importance="permute", var_mode="matched",
                  n_jobs=2, random_state=1).fit(d_cla["X"], d_cla["y"])
    assert np.asarray(fit.var_vi_).shape == (d_cla["p"],)


# --- Regression IJ -------------------------------------------------------------

def test_ij_regression_fit_stores_obstrack(d_reg):
    # R: "ObsTrack" %in% names(fit)
    fit = RLT_reg(n_estimators=30, var_mode="ij", n_jobs=2, random_state=1).fit(
        d_reg["X"], d_reg["y"])
    assert hasattr(fit, "obstrack_")
    assert fit.obstrack_.shape == (d_reg["n"], 30)


def test_ij_regression_predict_returns_variance(d_reg):
    fit = RLT_reg(n_estimators=30, var_mode="ij", n_jobs=2, random_state=1).fit(
        d_reg["X"], d_reg["y"])
    pred, var = fit.predict_var(d_reg["X"])
    assert np.asarray(var).dtype.kind == "f"
    assert np.asarray(var).shape[0] == d_reg["n"]


def test_ij_regression_variance_non_negative_or_nan(d_reg):
    fit = RLT_reg(n_estimators=30, var_mode="ij", n_jobs=2, random_state=1).fit(
        d_reg["X"], d_reg["y"])
    _, var = fit.predict_var(d_reg["X"])
    assert np.all((var >= 0) | np.isnan(var))


# --- Regression jack -----------------------------------------------------------

def test_jack_regression_fit_stores_obstrack(d_reg):
    fit = RLT_reg(n_estimators=30, var_mode="jack", n_jobs=2, random_state=1).fit(
        d_reg["X"], d_reg["y"])
    assert hasattr(fit, "obstrack_")
    assert fit.obstrack_.shape == (d_reg["n"], 30)


def test_jack_regression_predict_returns_variance(d_reg):
    fit = RLT_reg(n_estimators=30, var_mode="jack", n_jobs=2, random_state=1).fit(
        d_reg["X"], d_reg["y"])
    pred, var = fit.predict_var(d_reg["X"])
    assert np.asarray(var).dtype.kind == "f"
    assert np.asarray(var).shape[0] == d_reg["n"]


def test_jack_regression_variance_non_negative_or_nan(d_reg):
    fit = RLT_reg(n_estimators=30, var_mode="jack", n_jobs=2, random_state=1).fit(
        d_reg["X"], d_reg["y"])
    _, var = fit.predict_var(d_reg["X"])
    assert np.all((var >= 0) | np.isnan(var))


# --- Classification IJ/jack ------------------------------------------------------

def test_ij_classification_fit_stores_obstrack(d_cla):
    fit = RLT_cla(n_estimators=30, var_mode="ij", n_jobs=2, random_state=1).fit(
        d_cla["X"], d_cla["y"])
    assert hasattr(fit, "obstrack_")
    assert fit.obstrack_.shape == (d_cla["n"], 30)


def test_ij_classification_predict_returns_variance_matrix(d_cla):
    # R: IJ variance on classification predict. Ported via predict_var.
    fit = RLT_cla(n_estimators=30, var_mode="ij", n_jobs=2,
                  random_state=1).fit(d_cla["X"], d_cla["y"])
    prob, var = fit.predict_var(d_cla["X"])
    assert var.shape == (d_cla["n"], fit.classes_.shape[0])
    assert np.all((var >= 0) | np.isnan(var))


def test_jack_classification_predict_returns_variance_matrix(d_cla):
    # R: jackknife variance on classification predict. Ported via predict_var.
    fit = RLT_cla(n_estimators=30, var_mode="jack", n_jobs=2,
                  random_state=1).fit(d_cla["X"], d_cla["y"])
    prob, var = fit.predict_var(d_cla["X"])
    assert var.shape == (d_cla["n"], fit.classes_.shape[0])
    assert np.all((var >= 0) | np.isnan(var))


# --- Error handling -------------------------------------------------------------

def test_var_est_without_var_mode(d_reg):
    # Match R: uncertainty needs compatible fitted sampling or tracking.
    fit = RLT_reg(n_estimators=30, n_jobs=2, random_state=1).fit(d_reg["X"], d_reg["y"])
    with pytest.raises(ValueError, match="variance mode"):
        fit.predict_var(d_reg["X"])


def test_ij_forest_predict_uses_stored_var_mode(d_reg):
    fit = RLT_reg(n_estimators=30, var_mode="ij", n_jobs=2, random_state=1).fit(
        d_reg["X"], d_reg["y"])
    pred, var = fit.predict_var(d_reg["X"])
    assert np.asarray(var).shape[0] == d_reg["n"]
    assert not np.isnan(var).all()


def test_ij_forest_can_predict_with_jackknife_variance(d_reg):
    fit = RLT_reg(n_estimators=30, var_mode="ij", n_jobs=2, random_state=1).fit(
        d_reg["X"], d_reg["y"])
    _, var = fit.predict_var(d_reg["X"], var_mode="jack")
    assert np.asarray(var).dtype.kind == "f"
    assert np.asarray(var).shape[0] == d_reg["n"]


def test_jack_forest_can_predict_with_ij_variance(d_reg):
    fit = RLT_reg(n_estimators=30, var_mode="jack", n_jobs=2, random_state=1).fit(
        d_reg["X"], d_reg["y"])
    _, var = fit.predict_var(d_reg["X"], var_mode="ij")
    assert np.asarray(var).dtype.kind == "f"
    assert np.asarray(var).shape[0] == d_reg["n"]


def test_clean_variance_floors_negative_to_nan(d_reg):
    # R: clean.variance floors negative variance to NA; predict_var already
    # applies this (negative entries replaced by NaN).
    fit = RLT_reg(n_estimators=30, var_mode="matched", n_jobs=2, random_state=1).fit(
        d_reg["X"], d_reg["y"])
    _, var = fit.predict_var(d_reg["X"])
    assert np.all((var >= 0) | np.isnan(var))


def test_ij_mode_does_not_compute_varvi(d_reg):
    # R: fit$VarVI is NULL for IJ
    fit = RLT_reg(n_estimators=30, importance="permute", var_mode="ij",
                  n_jobs=2, random_state=1).fit(d_reg["X"], d_reg["y"])
    assert not hasattr(fit, "var_vi_")


def test_jack_mode_does_not_compute_varvi(d_reg):
    # R: fit$VarVI is NULL for jack
    fit = RLT_reg(n_estimators=30, importance="permute", var_mode="jack",
                  n_jobs=2, random_state=1).fit(d_reg["X"], d_reg["y"])
    assert not hasattr(fit, "var_vi_")
