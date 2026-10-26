"""Public band workflows and frozen current-R/mgcv numerical references."""
from copy import deepcopy
import json
from pathlib import Path
import pickle
import warnings

import numpy as np
import pytest
from scipy.stats import norm

from rlt import RLT_surv, get_surv_band
from rlt import _core
from rlt._surv_smoother import _GaussianGCV, _tensor_setup, smooth_covariance


FIXTURES = Path(__file__).parent / "fixtures"


def _prediction(n=2, t=6):
    x = np.arange(t)
    covariance = .02 * np.exp(-abs(x[:, None] - x[None, :]) / 2)
    return {"CHF": np.arange(1., n * t + 1).reshape(n, t) / 10,
            "Cov": np.stack([covariance * (i + 1) for i in range(n)]),
            "timepoints": np.arange(1., t + 1) ** 3,
            "var_mode": "IJ"}


def test_cached_prediction_subject_alpha_axes_and_nonmutation():
    prediction = _prediction()
    original = deepcopy(prediction)
    for value in prediction.values():
        if isinstance(value, np.ndarray):
            value.flags.writeable = False
    for approach in ("naive", "smoothed"):
        band = get_surv_band(prediction, i=2, alpha=[.05, .1], approach=approach,
                             k_rank=2, nsim=500, seed=127)
        assert band["lower"].shape == (1, 2, 6)
        np.testing.assert_array_equal(band["subjects"], [2])
        np.testing.assert_array_equal(band["alpha"], [.05, .1])
        np.testing.assert_array_equal(band["timepoints"], prediction["timepoints"])
        np.testing.assert_allclose(np.sqrt(band["lower"] * band["upper"]),
            np.broadcast_to(np.exp(-prediction["CHF"][1]), (1, 2, 6)))
        for name in ("CHF", "Cov", "timepoints"):
            np.testing.assert_array_equal(prediction[name], original[name])
        band["timepoints"][0] = -100
        np.testing.assert_array_equal(prediction["timepoints"], original["timepoints"])


@pytest.mark.parametrize("mode", ["IJ", "jack"])
def test_tracking_only_model_band_override_and_saved_result_need_no_recompute(mode, monkeypatch, tmp_path):
    u = np.arange(1., 33.)
    X = np.column_stack((np.sin(u), np.cos(u / 3)))
    model = RLT_surv(n_estimators=16, min_samples_leaf=3, mtry=2,
                     resample_track=True, random_state=13, n_jobs=1).fit(
                         X, (1 + u % 11, (u % 4 != 0).astype(int)))
    prediction = model.predict_details(X[:3], var_est=True, var_mode=mode,
                                        band_grid_size=5)
    direct = get_surv_band(model, X[:3], var_mode=mode, band_grid_size=5,
                           nsim=100, seed=31)

    def forbidden(*args, **kwargs):
        raise AssertionError("cached band recomputed a forest prediction")

    monkeypatch.setattr(model, "predict_details", forbidden)
    path = tmp_path / "survival-prediction.pkl"
    path.write_bytes(pickle.dumps(prediction))
    del model, prediction
    cached = get_surv_band(pickle.loads(path.read_bytes()), nsim=100, seed=31)
    for name in ("lower", "upper", "timepoints", "marsd"):
        np.testing.assert_array_equal(cached[name], direct[name])


@pytest.mark.parametrize("change,match", [
    ({"Cov": np.eye(6)}, "dimensions"),
    ({"CHF": np.zeros((0, 6)), "Cov": np.zeros((0, 6, 6))}, "dimensions"),
    ({"timepoints": np.arange(5)}, "dimensions"),
    ({"timepoints": np.array([1., 2, 3, np.nan, 5, 6])}, "finite"),
    ({"timepoints": np.array([1., 2, 3, 3, 5, 6])}, "increasing"),
    ({"CHF": np.full((2, 6), np.nan)}, "finite"),
])
def test_cached_prediction_rejects_incompatible_fields(change, match):
    prediction = _prediction()
    prediction.update(change)
    with pytest.raises(ValueError, match=match):
        get_surv_band(prediction, nsim=10)


def test_cached_covariance_and_grid_cannot_be_overridden_or_reinterpreted():
    with pytest.raises(TypeError, match="saved prediction dictionary"):
        get_surv_band((np.ones((2, 6)), np.ones((2, 6, 6))))
    for kwargs in ({"X": np.ones((2, 2))}, {"var_mode": "jack"}, {"band_grid_size": 3}):
        with pytest.raises(ValueError, match="saved prediction"):
            get_surv_band(_prediction(), **kwargs)
    prediction = _prediction()
    prediction.pop("Cov")
    with pytest.raises(ValueError, match="covariance is unavailable"):
        get_surv_band(prediction)


def test_smoothed_band_is_invariant_to_time_relabeling():
    prediction = _prediction(t=12)
    regular = deepcopy(prediction)
    regular["timepoints"] = np.arange(1., 13.)
    a = get_surv_band(prediction, approach="smoothed", k_rank=6, nsim=100)
    b = get_surv_band(regular, approach="smoothed", k_rank=6, nsim=100)
    for field in ("lower", "upper", "marsd"):
        np.testing.assert_array_equal(a[field], b[field])


def _oracle():
    return np.load(FIXTURES / "survival_smoother_mgcv_1_9_4.npz"), json.loads(
        (FIXTURES / "survival_smoother_mgcv_1_9_4.json").read_text())


def test_frozen_R_cr_tensor_design_penalties_and_fixed_smoothing_parameters():
    oracle, _ = _oracle()
    setup = _tensor_setup(12, 6)
    B, transform, _, roots, _, _ = setup
    Z = np.array([np.outer(B[i], B[j]).ravel() for j in range(12) for i in range(12)])
    np.testing.assert_allclose(Z @ transform, oracle["design"], rtol=1e-11, atol=1e-12)
    for j, root in enumerate(roots):
        np.testing.assert_allclose((root @ root.T)[1:, 1:], oracle[f"penalty{j+1}"],
                                   rtol=1e-11, atol=1e-12)
    problem = _GaussianGCV(oracle["toeplitz_n12_k6_raw"], setup)
    state = problem.fit(np.log([.1, .3]))
    full = transform @ state["coefficients"]
    np.testing.assert_allclose(B @ full.reshape(6, 6) @ B.T,
                               oracle["fixed_sp_surface"], rtol=1e-10, atol=1e-12)


def test_default_optimizer_matches_frozen_R_over_rank_scale_and_singular_cases():
    oracle, manifest = _oracle()
    controls = manifest["optimizer_roundoff_controls"]
    for record in manifest["cases"]:
        assert record["status"] == "ok", record
        tag = record["id"]
        raw = oracle[f"{tag}_raw"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            fitted, info = smooth_covariance(raw, int(record["k"]))
        scale = max(np.max(abs(raw)), np.finfo(float).tiny)
        # Three near-stationary cases have independently measured same-R
        # sensitivity to a single ULP in the input, using exact binary doubles.
        # Use that reference envelope only for automatic parameter selection;
        # the fixed-parameter solve below keeps its stricter numerical gate.
        reference = [r for r in controls if r["case"] == tag]
        surface_roundoff = max((r["optimized_surface_delta_scaled"] for r in reference), default=0)
        np.testing.assert_allclose(fitted / scale, oracle[f"{tag}_smooth"] / scale,
                                   rtol=1e-10, atol=1e-12+surface_roundoff, err_msg=tag)
        assert np.isfinite(info["sp"]).all() and np.all(info["sp"] > 0), tag
        # Constant/zero responses have flat GCV: the surface is identified,
        # but smoothing parameters and their EDF can depend on rounding.
        if tag not in ("zero", "constant"):
            edf_R = float(record["edf"])
            edf_roundoff = max((abs(r["edf"]-edf_R) for r in reference), default=0)
            assert info["edf"] == pytest.approx(edf_R, rel=1e-9, abs=1e-9+edf_roundoff), tag
            assert info["rank"] == int(record["rank"]), tag
            assert info["iterations"] == int(record["iterations"]), tag
            if not tag.startswith("toeplitz_n3_"):
                np.testing.assert_allclose(info["sp"], [float(record["sp1"]), float(record["sp2"])],
                                           rtol=2e-6, atol=1e-13, err_msg=tag)
            score_R = float(record["gcv"]) / scale ** 2
            score_budget = 64*info["rank"]*np.finfo(float).eps*(1+abs(score_R))
            assert abs(info["gcv"] / scale ** 2-score_R) <= score_budget, tag
            # The RMS gradient is the previous-step gradient used by magic's
            # stopping rule, rather than a newly evaluated final gradient.
            assert info["rms_gradient"] == pytest.approx(
                float(record["rms_gradient"]), rel=1e-4, abs=1e-10*scale**2), tag


def test_full_corpus_at_R_selected_parameters_keeps_strict_surface_gate():
    oracle, manifest = _oracle()
    for record in manifest["cases"]:
        tag = record["id"]
        raw = oracle[f"{tag}_raw"]
        n = len(raw)
        k = min(max(int(record["k"]), 3), n)
        setup = _tensor_setup(n, k)
        problem = _GaussianGCV(raw, setup)
        state = problem.fit(np.log([float(record["sp1"]), float(record["sp2"])]))
        coefficients = setup[1] @ state["coefficients"]
        fit = setup[0] @ coefficients.reshape(k, k) @ setup[0].T
        scale = max(np.max(abs(raw)), np.finfo(float).tiny)
        np.testing.assert_allclose((fit+fit.T)/2/scale, oracle[f"{tag}_smooth"]/scale,
                                   rtol=1e-10, atol=1e-11, err_msg=tag)


def test_optimizer_roundoff_envelopes_have_binary_reread_and_ULP_controls():
    _, manifest = _oracle()
    controls = manifest["optimizer_roundoff_controls"]
    assert len(controls) == 18
    for row in controls:
        if row["variant"] in ("base", "reread"):
            assert row["input_delta_scaled"] == 0
            assert row["optimized_surface_delta_scaled"] == 0
        else:
            assert row["input_delta_scaled"] <= np.finfo(float).eps
            assert row["fixed_sp_surface_delta_scaled"] <= 32*np.finfo(float).eps


def test_smoothed_final_scales_match_frozen_R_postprocessing():
    oracle, manifest = _oracle()
    for record in manifest["cases"]:
        tag = record["id"]
        raw = oracle[f"{tag}_raw"]
        n = len(raw)
        prediction = {"CHF": np.linspace(.1, 1, n)[None], "Cov": raw[None],
                      "timepoints": np.arange(1., n+1) ** 3}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            band = get_surv_band(prediction, approach="smoothed", k_rank=int(record["k"]), nsim=5)
        expected = oracle[f"{tag}_band_sd"]
        actual = band["diagnostics"][0]["diag_sd"]
        for j, field in enumerate(("raw_sd", "smoothK_sd", "final_sd", "resid_sd"), 1):
            # Positive-spectrum residuals can be at roundoff scale. Their
            # square roots need an absolute tolerance proportional to sqrt(eps).
            scale = max(np.max(abs(raw)), np.finfo(float).eps) ** .5
            np.testing.assert_allclose(actual[field] / scale, expected[:, j] / scale,
                                       rtol=1e-8, atol=3e-7, err_msg=f"{tag}/{field}")


def test_GCV_gradient_and_hessian_agree_with_finite_differences():
    oracle, _ = _oracle()
    problem = _GaussianGCV(oracle["toeplitz_n12_k6_raw"], _tensor_setup(12, 6))
    x = np.array([-2., -.5])
    state = problem.fit(x, derivatives=True)
    h = 1e-4
    finite_grad = np.empty(2)
    finite_hess = np.empty((2, 2))
    for j in range(2):
        dx = np.eye(2)[j] * h
        plus, minus = problem.fit(x+dx, derivatives=True), problem.fit(x-dx, derivatives=True)
        finite_grad[j] = (plus["score"]-minus["score"])/(2*h)
        finite_hess[:, j] = (plus["gradient"]-minus["gradient"])/(2*h)
    np.testing.assert_allclose(state["gradient"], finite_grad, rtol=1e-5, atol=1e-10)
    np.testing.assert_allclose(state["hessian"], finite_hess, rtol=1e-5, atol=1e-10)


def test_shared_MC_postprocessing_matches_frozen_R_Gaussian_draws():
    oracle, _ = _oracle()
    actual = _core._mc_band_from_draws(oracle["mc_scales"], oracle["mc_draws"], np.array([.05, .1, .5]))
    np.testing.assert_allclose(actual, oracle["mc_widths"], rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("scale", [1e-16, 1., 1e12])
def test_MC_uses_R_scale_relative_PSD_domain(scale):
    with pytest.raises(ValueError, match="positive semidefinite"):
        _core.mc_band(np.ones(2), scale*np.array([[1., 2], [2, 1]]), np.array([.05]), 10, 1)
    # A roundoff-sized negative eigenvalue is tolerated at each scale.
    result = _core.mc_band(np.ones(2), scale*np.diag([1., -1e-18]), np.array([.05]), 10, 1)
    assert np.isfinite(result).all()
    zero = _core.mc_band(np.ones(2), np.zeros((2, 2)), np.array([.05]), 10, 1)
    np.testing.assert_array_equal(zero, [[0., 0.]])


@pytest.mark.parametrize("seed", [19, 53, 271, 929])
@pytest.mark.parametrize("dimension", [1, 3])
def test_MC_critical_value_distribution_against_analytic_independent_normals(seed, dimension):
    nsim = 20000
    alpha = np.array([.05, .2])
    width = _core.mc_band(np.ones(dimension), np.eye(dimension), alpha, nsim, seed)
    cdf = (2*norm.cdf(width[:, 0])-1)**dimension
    p = 1-alpha
    # Six binomial standard errors gives a prespecified allowance for all
    # independent seeds/dimensions/levels, not same-seed R/Python equality.
    assert np.all(abs(cdf-p) <= 6*np.sqrt(p*(1-p)/nsim))
