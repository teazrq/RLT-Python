"""Public input and sklearn contracts shared with the corrected R wrapper."""

import numpy as np
import pytest
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rlt import ImportanceTable, RLT_cla, RLT_reg, RLT_surv
from rlt import _params


MODELS = [RLT_reg, RLT_cla, RLT_surv]


def data(model, n=36):
    X = np.random.default_rng(9).normal(size=(n, 3))
    if model is RLT_cla:
        y = np.where(X[:, 0] > 0, "high", "low")
    elif model is RLT_surv:
        y = (np.arange(1., n + 1), np.ones(n))
    else:
        y = X[:, 0] + X[:, 1]
    return X, y


def fitted(model, **kwargs):
    X, y = data(model)
    m = model(n_estimators=8, n_jobs=1, random_state=23, **kwargs).fit(X, y)
    return m, X, y


@pytest.mark.parametrize("model", MODELS)
def test_categories_checked_before_native_fit_and_prediction(model):
    X, y = data(model)
    X[:, 0] = np.arange(len(X)) % 2
    m = model(n_estimators=4, n_jobs=1, random_state=1, categorical_features=[0])
    bad = X.copy()
    bad[0, 0] = -2
    with pytest.raises(ValueError, match="non-negative"):
        m.fit(bad, y)
    m.fit(X, y)
    for value in [-1, -2, .5, 2, 100]:
        bad = X[:2].copy()
        bad[0, 0] = value
        for method in [m.predict, m.forest_kernel]:
            with pytest.raises(ValueError, match="integer training codes"):
                method(bad)
        with pytest.raises(ValueError, match="integer training codes"):
            m.forest_kernel(X[:2], bad)
    assert m.predict(X[:2]).shape[0] == 2


def test_category_mask_shape_and_one_level_codes():
    X, y = data(RLT_reg)
    with pytest.raises(ValueError, match="one-dimensional"):
        RLT_reg(categorical_features=[[True, False, False]]).fit(X, y)
    X[:, 0] = 0
    m = RLT_reg(n_estimators=4, categorical_features=[0], n_jobs=1).fit(X, y)
    bad = X[:1].copy()
    bad[0, 0] = 1
    with pytest.raises(ValueError, match="integer training codes"):
        m.predict(bad)


@pytest.mark.parametrize("model", MODELS)
def test_named_features_align_predictions_and_both_kernel_inputs(model):
    pd = pytest.importorskip("pandas")
    X, y = data(model)
    frame = pd.DataFrame(X, columns=["age", "dose", "noise"])
    m = model(n_estimators=8, n_jobs=1, random_state=1, importance="permute").fit(frame, y)
    assert m.feature_names_in_.tolist() == list(frame.columns)
    assert m.importance_table().variable == list(frame.columns)
    query = frame.iloc[:4]
    reordered = query[["noise", "age", "dose"]].assign(extra="ignored")
    expected = m.predict(query)
    with pytest.warns(UserWarning, match="matching by name"):
        if model is RLT_cla:
            np.testing.assert_array_equal(m.predict(reordered), expected)
        else:
            np.testing.assert_allclose(m.predict(reordered), expected)
    expected_kernel = m.forest_kernel(query, query)
    with pytest.warns(UserWarning, match="matching by name"):
        np.testing.assert_array_equal(m.forest_kernel(reordered, query), expected_kernel)
    with pytest.warns(UserWarning, match="matching by name"):
        np.testing.assert_array_equal(m.forest_kernel(query, reordered), expected_kernel)
    with pytest.raises(ValueError, match="missing features"):
        m.predict(query[["age", "noise"]])
    for width in [2, 4]:
        with pytest.raises(ValueError, match="features"):
            m.predict(np.zeros((2, width)))


def test_duplicate_names_are_accepted_only_in_original_order():
    pd = pytest.importorskip("pandas")
    X, y = data(RLT_reg)
    frame = pd.DataFrame(X, columns=["same", "same", "other"])
    m = RLT_reg(n_estimators=4, n_jobs=1).fit(frame, y)
    np.testing.assert_allclose(m.predict(frame), m.predict(frame.to_numpy()))
    with pytest.raises(ValueError, match="ambiguous duplicate"):
        m.predict(frame.iloc[:, [2, 0, 1]])


@pytest.mark.parametrize("checker", [_params.check_obs_weight, _params.check_var_prob])
def test_weights_finite_sparse_and_scale_invariant(checker):
    np.testing.assert_array_equal(checker([1e308, 1e308, 0], 3), [.5, .5, 0])
    np.testing.assert_allclose(checker([1e-300, 2e-300, 0], 3), [1 / 3, 2 / 3, 0])
    for value in [np.nan, np.inf, -np.inf]:
        with pytest.raises(ValueError, match="finite"):
            checker([1, value, 0], 3)
    with pytest.raises(ValueError, match="positive weight"):
        checker([0, 0, 0], 3)


def test_large_finite_weights_fit_the_same_forest():
    X, y = data(RLT_reg)
    kwargs = dict(n_estimators=8, n_jobs=1, random_state=4)
    ordinary = RLT_reg(**kwargs).fit(X, y, sample_weight=np.ones(len(X)))
    large = RLT_reg(**kwargs).fit(X, y, sample_weight=np.full(len(X), 1e308))
    np.testing.assert_array_equal(ordinary.predict(X), large.predict(X))


@pytest.mark.parametrize("model", MODELS)
def test_variance_requires_compatible_sampling_and_tracking(model):
    m, X, _ = fitted(model)
    for mode, message in [(None, "variance mode"), ("none", "variance mode"),
                          ("matched", "requires fitting"), ("ij", "tracking"), ("jack", "tracking")]:
        with pytest.raises(ValueError, match=message):
            m.predict_var(X[:2], var_mode=mode)
    tracked, X, _ = fitted(model, resample_track=True)
    for mode in ["ij", "jack"]:
        prediction, variance = tracked.predict_var(X[:2], var_mode=mode)
        assert prediction.shape[0] == variance.shape[0] == 2
    with pytest.raises(ValueError, match="requires fitting"):
        tracked.predict_var(X[:2], var_mode="matched")


@pytest.mark.parametrize("model", MODELS)
def test_refit_removes_optional_state_and_old_feature_names(model):
    pd = pytest.importorskip("pandas")
    X, y = data(model)
    m = model(n_estimators=8, n_jobs=1, random_state=2, var_mode="matched").fit(
        pd.DataFrame(X, columns=["a", "b", "c"]), y)
    assert hasattr(m, "varimp_") and hasattr(m, "var_vi_") and hasattr(m, "obstrack_")
    m.set_params(var_mode="none", importance="none", resample_track=False)
    new_X, new_y = data(model, 30)
    m.fit(new_X[:, :2], new_y)
    for attr in ["varimp_", "var_vi_", "obstrack_", "feature_names_in_"]:
        assert not hasattr(m, attr)
    assert m.n_features_in_ == 2
    with pytest.raises(ValueError, match="tracking|resample_track"):
        m.forest_kernel(new_X[:, :2], oob=True)


def test_sklearn_estimator_identity_and_pipeline_cv():
    assert is_classifier(RLT_cla())
    assert is_regressor(RLT_reg())
    m = RLT_cla(n_estimators=4, n_jobs=1, random_state=3)
    assert clone(m).get_params() == m.get_params()
    X, y = data(RLT_cla)
    scores = cross_val_score(make_pipeline(StandardScaler(), m), X, y, cv=2)
    assert scores.shape == (2,) and np.isfinite(scores).all()


@pytest.mark.parametrize("kwargs", [
    {"n_estimators": 0}, {"mtry": 0}, {"mtry": 4}, {"min_samples_leaf": 0},
    {"nsplit": -1}, {"resample_prob": 0}, {"resample_prob": 1.1},
    {"importance": 3}, {"var_mode": 4}, {"linear_comb": 0},
    {"alpha": np.nan}, {"embed_mtry": np.inf},
    {"reinforcement": True, "embed_protect": 0},
    {"linear_comb": 2, "embed_protect": 0},
])
@pytest.mark.parametrize("model", MODELS)
def test_invalid_parameters_rejected_before_native_fitting(model, kwargs):
    X, y = data(model)
    with pytest.raises(ValueError):
        model(**kwargs).fit(X, y)


def test_embedded_parameters_use_R_bounds_without_changing_constructor_values():
    m = RLT_reg(linear_comb=10, embed_ntrees=-2, embed_mtry=10,
                embed_nmin=-3, embed_nsplit=-1, embed_resample_prob=4,
                embed_mute=8, embed_protect=9, embed_threshold=-1, random_state=1)
    cp = m._core_params(36, 3)
    assert (cp.linear_comb, cp.embed_ntrees, cp.embed_mtry, cp.embed_nmin, cp.embed_nsplit) == (3, 1, 3, 1, 0)
    assert (cp.embed_resample_prob, cp.embed_mute, cp.embed_protect, cp.embed_threshold) == (.99, 3, 3, 0)
    assert m.linear_comb == 10 and m.embed_ntrees == -2
    assert RLT_reg().n_estimators == 500


@pytest.mark.parametrize("value", [-1, 1.5, np.nan, np.inf, True, None, "2", [2]])
def test_survival_grid_sizes_are_exact_nonnegative_integer_scalars(value):
    X, y = data(RLT_surv)
    with pytest.raises(ValueError):
        RLT_surv(time_grid_size=value).fit(X, y)
    m, _, _ = fitted(RLT_surv)
    with pytest.raises(ValueError):
        m.predict(X[:2], band_grid_size=value)


def test_survival_keeps_all_failure_times_and_uses_R_type1_evaluation_grid():
    X, y = data(RLT_surv, 20)
    for size in [1, 2, 19]:
        with pytest.raises(ValueError, match="cannot reduce"):
            RLT_surv(time_grid_size=size).fit(X, y)
    m = RLT_surv(n_estimators=4, n_jobs=1, time_grid_size=20).fit(X, y)
    np.testing.assert_array_equal(m.timepoints_, np.arange(1, 21))
    grid, mapping = m._grid_and_mapping(2)
    np.testing.assert_array_equal(grid, [1, 20])
    np.testing.assert_array_equal(mapping, [0, 19])
    np.testing.assert_array_equal(m._grid_and_mapping(1)[0], [1])
    assert m._grid_and_mapping(25)[0].size == 20
    with pytest.raises(ValueError, match="inconsistent lengths"):
        m.fit(X, (y[0], y[1][:-1]))


def test_classifier_labels_and_column_vectors():
    X, y = data(RLT_cla)
    kwargs = dict(n_estimators=4, n_jobs=1, random_state=3)
    expected = RLT_cla(**kwargs).fit(X, y).predict(X)
    with pytest.warns(UserWarning, match="column-vector"):
        m = RLT_cla(**kwargs).fit(X, y.reshape(-1, 1))
    np.testing.assert_array_equal(m.predict(X), expected)
    for value in [np.nan, np.inf, -np.inf]:
        bad = np.arange(len(X), dtype=float) % 2
        bad[0] = value
        with pytest.raises(ValueError):
            RLT_cla(**kwargs).fit(X, bad)
    with pytest.raises(ValueError):
        RLT_cla(**kwargs).fit(X, np.column_stack([y, y]))
    with pytest.raises(ValueError):
        RLT_cla(**kwargs).fit(X, np.arange(len(X)) + .25)
    missing = y.astype(object)
    missing[0] = None
    with pytest.raises(ValueError):
        RLT_cla(**kwargs).fit(X, missing)


def test_zero_vi_variance_significance_matches_R():
    table = ImportanceTable(["positive", "negative", "zero"], [1., -1., 0.], [0., 0., 0.])
    assert table.sig == ["***", "***", ""]


def test_matched_tracking_uses_R_seed_offset():
    # Current R: gen_ms_obs_track_mat_cpp(6L, 3L, 4L, seed + 1L), seed=23.
    # Generated with RLT 6.1.1 from the corrected local R source.
    expected = np.array([[0, 1, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0],
                         [0, 0, 1, 1], [1, 1, 0, 0], [1, 0, 0, 1]])
    X = np.arange(18.).reshape(6, 3)
    m = RLT_reg(n_estimators=4, var_mode="matched", random_state=23, n_jobs=1).fit(X, np.arange(6.))
    np.testing.assert_array_equal(m.obstrack_, expected)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("failure", ["response", "weights", "native"])
@pytest.mark.filterwarnings("ignore:Column .* has few unique values:UserWarning")
def test_failed_refit_preserves_feature_maps_and_class_labels(model, failure, monkeypatch):
    pd = pytest.importorskip("pandas")
    import rlt.estimators as estimators

    X, y = data(model)
    X[:, 0] = np.arange(len(X)) % 2
    frame = pd.DataFrame(X, columns=["group", "dose", "noise"])
    m = model(n_estimators=8, n_jobs=1, random_state=3, categorical_features=[0]).fit(frame, y)
    expected = m.predict(frame)
    forest, params = m.forest_, m.params_
    names, ncat, cat_cols = m.feature_names_in_.copy(), m.ncat_.copy(), list(m._cat_shift_cols_)
    classes = m.classes_.copy() if model is RLT_cla else None

    changed = pd.DataFrame(np.column_stack([X[:, 0], np.arange(len(X)) % 2, X[:, 2]]),
                           columns=["new_group", "new_dose", "new_noise"])
    changed_y = np.where(np.arange(len(X)) % 2, "new_high", "new_low") if model is RLT_cla else y
    m.set_params(categorical_features=[1])
    kwargs = {}
    error = ValueError
    if failure == "response":
        changed_y = (y[0][:-1], y[1][:-1]) if model is RLT_surv else changed_y[:-1]
    elif failure == "weights":
        kwargs["sample_weight"] = np.zeros(len(X))
    else:
        function = {RLT_reg: "RegUniForestFit", RLT_cla: "ClaUniForestFit",
                    RLT_surv: "SurvUniForestFit"}[model]

        def native_failure(*args):
            raise RuntimeError("controlled native fitting failure")

        monkeypatch.setattr(estimators, function, native_failure)
        error = RuntimeError
    with pytest.raises(error):
        m.fit(changed, changed_y, **kwargs)
    assert m.forest_ is forest and m.params_ is params
    np.testing.assert_array_equal(m.feature_names_in_, names)
    np.testing.assert_array_equal(m.ncat_, ncat)
    assert m._cat_shift_cols_ == cat_cols
    if classes is not None:
        np.testing.assert_array_equal(m.classes_, classes)
    if model is RLT_cla:
        np.testing.assert_array_equal(m.predict(frame), expected)
    else:
        np.testing.assert_allclose(m.predict(frame), expected)


@pytest.mark.parametrize("seed", [2**53 + 1, np.int64(2**53 + 1), np.uint64(2**63 + 1)])
def test_integral_random_state_is_not_rounded_through_float(seed):
    cp = RLT_reg(random_state=seed)._core_params(36, 3)
    assert cp.seed == int(seed)


@pytest.mark.parametrize("reinforcement", [False, True])
def test_linear_combinations_preserve_explicit_reinforcement_flag(reinforcement):
    m, _, _ = fitted(RLT_reg, linear_comb=2, reinforcement=reinforcement,
                     linear_comb_method="naive")
    assert m.params_.linear_comb == 2
    assert m.params_.reinforcement is reinforcement
