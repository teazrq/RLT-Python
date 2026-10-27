"""Public fit contracts added by the corrected-R alignment work."""
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import DataConversionWarning

from rlt import RLT_cla, RLT_reg, RLT_surv
from rlt import _params


MODELS = [RLT_reg, RLT_cla, RLT_surv]


def data(model, n=48):
    X = np.random.default_rng(1701).normal(size=(n, 4))
    if model is RLT_cla:
        y = np.where(np.arange(n) % 2, "a", "z")
    elif model is RLT_surv:
        y = (np.arange(1., n + 1), (np.arange(n) % 4 != 0).astype(int))
    else:
        y = X[:, 0] + X[:, 1] ** 2
    return X, y


def model_fit(model, X, y, **kwargs):
    return model(n_estimators=8, n_jobs=1, random_state=19, **kwargs).fit(X, y)


@pytest.mark.parametrize("linear_comb", [1, 2])
@pytest.mark.parametrize("code,name", [(1, "logrank"), (2, "suplogrank"), (3, "coxgrad")])
def test_survival_numeric_split_rules_select_the_named_algorithm(code, name, linear_comb):
    X, y = data(RLT_surv)
    integer = model_fit(RLT_surv, X, y, split_rule=code, linear_comb=linear_comb)
    floating = model_fit(RLT_surv, X, y, split_rule=float(code), linear_comb=linear_comb)
    named = model_fit(RLT_surv, X, y, split_rule=name, linear_comb=linear_comb)
    assert integer.params_.split_rule == floating.params_.split_rule == code
    np.testing.assert_array_equal(integer.predict(X), named.predict(X))
    np.testing.assert_array_equal(floating.predict(X), named.predict(X))


@pytest.mark.parametrize("model,codes", [(RLT_reg, range(1, 5)), (RLT_cla, range(1, 5)), (RLT_surv, range(1, 3))])
def test_integral_double_lc_codes_match_integer_codes(model, codes):
    X, y = data(model)
    for code in codes:
        integer = model_fit(model, X, y, linear_comb=2, linear_comb_method=code)
        floating = model_fit(model, X, y, linear_comb=2, linear_comb_method=float(code))
        assert floating.params_.linear_comb_method == code
        np.testing.assert_array_equal(integer.predict(X), floating.predict(X))


@pytest.mark.parametrize("value", [np.nan, np.inf, 1.5, True, np.bool_(False)])
@pytest.mark.parametrize("parameter", ["split_rule", "linear_comb_method"])
def test_invalid_nonintegral_method_codes_are_rejected(value, parameter):
    X, y = data(RLT_surv)
    with pytest.raises(ValueError, match=parameter):
        model_fit(RLT_surv, X, y, linear_comb=2, **{parameter: value})


@pytest.mark.parametrize("parameter", ["split_rule", "linear_comb_method"])
@pytest.mark.parametrize("value", ["unknown", 99, 0])
def test_unknown_method_codes_warn_and_fall_back(parameter, value):
    X, y = data(RLT_surv)
    with pytest.warns(UserWarning, match=parameter):
        fit = model_fit(RLT_surv, X, y, linear_comb=2, **{parameter: value})
    assert getattr(fit.params_, parameter) == 1


@pytest.mark.parametrize("name", ["reinforcement", "resample_replace"])
@pytest.mark.parametrize("value", [.5, -1, 2, np.inf, "true"])
def test_logical_flags_reject_noncanonical_numeric_coercions(name, value):
    with pytest.raises(ValueError, match=name):
        RLT_reg(**{name: value})._core_params(48, 4)


@pytest.mark.parametrize("name,native_name", [("resample_track", "obs_track"), ("embed_resample_replace", "embed_replacement")])
@pytest.mark.parametrize("value", [.5, -1, 2])
def test_embedded_and_tracking_flags_preserve_finite_nonzero_semantics(name, native_name, value):
    assert getattr(RLT_reg(**{name: value})._core_params(48, 4), native_name)
    with pytest.raises(ValueError, match=name):
        RLT_reg(**{name: np.inf})._core_params(48, 4)


@pytest.mark.parametrize("model", MODELS)
def test_omitted_tree_default_is_effective_only_and_clone_preserves_request(model):
    plain = model(random_state=3)
    reinforced = model(reinforcement=True, random_state=3)
    assert plain.n_estimators is reinforced.n_estimators is None
    assert plain._core_params(48, 4).ntrees == 500
    assert reinforced._core_params(48, 4).ntrees == 100
    assert clone(reinforced).get_params()["n_estimators"] is None
    assert not hasattr(clone(reinforced), "forest_")
    assert model(n_estimators=17, reinforcement=True)._core_params(48, 4).ntrees == 17
    assert model(n_estimators=17, var_mode="matched")._core_params(48, 4).ntrees == 16


@pytest.mark.parametrize("model", MODELS)
def test_omitted_seed_uses_global_numpy_rng_and_explicit_seeds_do_not(model):
    X, y = data(model)
    state = np.random.get_state()
    try:
        np.random.seed(71)
        first = model(n_estimators=4, n_jobs=1).fit(X, y)
        np.random.seed(71)
        second = model(n_estimators=4, n_jobs=1).fit(X, y)
        third = model(n_estimators=4, n_jobs=1).fit(X, y)
        assert first.params_.seed == second.params_.seed != third.params_.seed
        np.testing.assert_array_equal(first.predict(X), second.predict(X))
        before = np.random.get_state()
        model(n_estimators=4, random_state=2**53 + 1)._core_params(48, 4)
        after = np.random.get_state()
        assert before[0] == after[0] and before[2:] == after[2:]
        np.testing.assert_array_equal(before[1], after[1])
    finally:
        np.random.set_state(state)


@pytest.mark.parametrize("factory", [lambda: np.random.default_rng(3), lambda: np.random.RandomState(3)])
def test_supplied_rng_advances_one_effective_seed_per_fit(factory):
    rng, expected = factory(), factory()
    draw = (lambda: expected.integers(0, 2**31 - 1)) if isinstance(expected, np.random.Generator) else (lambda: expected.randint(0, 2**31 - 1))
    fit = RLT_reg(random_state=rng)
    assert fit._core_params(48, 4).seed == draw()
    assert fit._core_params(48, 4).seed == draw()


@pytest.mark.parametrize("model", [RLT_reg, RLT_surv])
def test_response_matrix_is_not_silently_flattened(model):
    X, y = data(model)
    malformed = np.arange(48.).reshape(24, 2)
    if model is RLT_surv:
        y = (malformed, np.ones(48))
    else:
        y = malformed
    with pytest.raises(ValueError):
        model_fit(model, X, y)
    if model is RLT_surv:
        with pytest.raises(ValueError):
            model_fit(model, X, (np.arange(48.), np.ones((24, 2))))
        structured = np.zeros((24, 2), dtype=[("time", float), ("event", int)])
        with pytest.raises(ValueError):
            model_fit(model, X, structured)


@pytest.mark.parametrize("model", [RLT_reg, RLT_surv])
def test_response_one_column_convenience_warns_and_preserves_rows(model):
    X, y = data(model)
    columns = tuple(v[:, None] for v in y) if model is RLT_surv else y[:, None]
    with pytest.warns(DataConversionWarning):
        fitted = model_fit(model, X, columns)
    np.testing.assert_array_equal(fitted.predict(X), model_fit(model, X, y).predict(X))


def feature_frame(n=48):
    pd = pytest.importorskip("pandas")
    labels = np.resize(["z", "a", "m"], n)
    numeric = np.resize([10, 2, 1], n)
    return pd.DataFrame({
        "nominal": pd.Categorical(labels, categories=["z", "a", "m", "unused"]),
        "ordered": pd.Categorical(labels, categories=["m", "z", "a"], ordered=True),
        "character": labels,
        "numeric_labels": pd.Categorical(numeric, categories=[10, 2, 1]),
        "continuous": np.random.default_rng(3).normal(size=n),
    })


@pytest.mark.parametrize("model", MODELS)
def test_feature_dictionaries_preserve_r_rank_and_nominal_meanings(model):
    frame = feature_frame()
    _, y = data(model)
    fitted = model_fit(model, frame, y, resample_track=True)
    assert fitted.ncat_.tolist() == [4, 1, 1, 3, 1]
    assert [None if entry is None else entry["kind"] for entry in fitted.feature_schema_] == ["categorical", "ordered", "string", "categorical", None]
    assert fitted.feature_schema_[0]["levels"].tolist() == ["z", "a", "m", "unused"]
    assert fitted.feature_schema_[3]["levels"].tolist() == [10, 2, 1]
    codes = fitted._check_X(frame)
    np.testing.assert_array_equal(codes[:, 0], np.arange(48) % 3)
    np.testing.assert_array_equal(codes[:, 1], np.resize([1, 2, 0], 48))
    np.testing.assert_array_equal(codes[:, 2], np.resize([2, 0, 1], 48))
    np.testing.assert_array_equal(codes[:, 3], np.arange(48) % 3)
    manual = codes.copy()
    manual[:, [1, 2]] += 1
    expected = model_fit(model, manual, y, categorical_features=[0, 3], resample_track=True)
    np.testing.assert_array_equal(fitted.predict(frame), expected.predict(manual))
    np.testing.assert_array_equal(fitted.predict(frame), fitted.predict(codes))
    # Both cross-kernel inputs must use the same fitted mappings.
    np.testing.assert_array_equal(fitted.forest_kernel(frame[:5]), expected.forest_kernel(manual[:5]))
    np.testing.assert_array_equal(fitted.forest_kernel(frame[:5], frame[5:11]), expected.forest_kernel(manual[:5], manual[5:11]))
    np.testing.assert_array_equal(fitted.forest_kernel(frame[:5], frame, vs_train=True), expected.forest_kernel(manual[:5], manual, vs_train=True))
    np.testing.assert_array_equal(fitted.forest_kernel(frame, oob=True)["N"], expected.forest_kernel(manual, oob=True)["N"])
    reordered = frame.copy()
    reordered["nominal"] = reordered["nominal"].cat.reorder_categories(["unused", "m", "a", "z"])
    reordered = reordered[list(frame.columns)[::-1]]
    with pytest.warns(UserWarning, match="matching by name"):
        np.testing.assert_array_equal(fitted.predict(frame), fitted.predict(reordered))


@pytest.mark.parametrize("column", ["nominal", "ordered", "character", "numeric_labels"])
def test_unknown_and_missing_feature_labels_rejected_in_prediction_and_kernels(column):
    pd = pytest.importorskip("pandas")
    frame = feature_frame()
    _, y = data(RLT_reg)
    fitted = model_fit(RLT_reg, frame, y)
    unknown = frame.copy()
    unknown[column] = pd.Categorical(["unknown"] * len(frame))
    with pytest.raises(ValueError, match="unknown label"):
        fitted.predict(unknown)
    with pytest.raises(ValueError, match="unknown label"):
        fitted.forest_kernel(frame, unknown)
    missing = frame.copy()
    missing.loc[0, column] = None
    with pytest.raises(ValueError, match="missing"):
        fitted.predict(missing)
    with pytest.raises(ValueError, match="missing"):
        model_fit(RLT_reg, missing, y)


def test_typed_numeric_category_labels_are_distinct_from_plain_numeric_codes():
    frame = feature_frame()
    _, y = data(RLT_reg)
    fitted = model_fit(RLT_reg, frame, y)
    codes = fitted._check_X(frame)
    plain = frame.copy()
    plain["numeric_labels"] = np.asarray(plain["numeric_labels"])
    with pytest.raises(ValueError, match="integer training codes"):
        fitted.predict(plain)
    assert np.array_equal(fitted.predict(frame), fitted.predict(codes.tolist()))
    continuous = model_fit(RLT_reg, frame, y, categorical_features=[])
    assert continuous.ncat_.tolist() == [1] * frame.shape[1]
    assert continuous.feature_schema_[0]["nominal"] is False


@pytest.mark.parametrize("model", MODELS)
def test_numeric_frame_training_rejects_new_character_labels_without_a_dictionary(model):
    pd = pytest.importorskip("pandas")
    X, y = data(model)
    frame = pd.DataFrame(X, columns=list("abcd"))
    fitted = model_fit(model, frame, y)
    labels = frame.copy()
    labels["a"] = labels["a"].astype(str)
    with pytest.raises(ValueError, match="was numerical during training"):
        fitted.predict(labels)
    with pytest.raises(ValueError, match="was numerical during training"):
        fitted.forest_kernel(frame, labels)
    # Preserve the existing numeric-string array conversion as a Python
    # convenience. An accepted training input must remain usable for prediction.
    strings = X.astype(str)
    converted = model_fit(model, strings, y)
    np.testing.assert_array_equal(converted.predict(strings), converted.predict(X))


def test_declared_nominal_cardinality_limit_and_ordered_unused_levels():
    pd = pytest.importorskip("pandas")
    X, y = data(RLT_reg)
    for count in (1, 53, 54):
        frame = pd.DataFrame({"x": pd.Categorical(np.zeros(len(X), dtype=int), categories=range(count)), "z": X[:, 1]})
        if count <= 53:
            assert model_fit(RLT_reg, frame, y).ncat_[0] == count
        else:
            with pytest.raises(ValueError, match="53 categories"):
                model_fit(RLT_reg, frame, y)
            frame["x"] = frame["x"].cat.as_ordered()
            assert model_fit(RLT_reg, frame, y).ncat_[0] == 1


@pytest.mark.parametrize("container", ["categorical", "series", "index"])
def test_declared_class_order_unused_levels_and_balanced_ties(container):
    pd = pytest.importorskip("pandas")
    X, _ = data(RLT_cla)
    y = pd.Categorical(["z", "a"] * 24, categories=["z", "a", "unused"])
    if container == "series":
        y = pd.Series(y)
    elif container == "index":
        y = pd.CategoricalIndex(y)
    with pytest.warns(UserWarning, match="unused declared classes"):
        fitted = model_fit(RLT_cla, X, y, min_samples_leaf=100, resample_replace=False, resample_prob=1)
    assert fitted.classes_.tolist() == ["z", "a", "unused"]
    np.testing.assert_array_equal(fitted.predict_proba(X), np.tile([.5, .5, 0], (48, 1)))
    assert np.all(fitted.predict(X) == "z")


def test_float_categorical_classes_and_single_observed_class_remain_declared():
    pd = pytest.importorskip("pandas")
    X, _ = data(RLT_cla)
    y = pd.Categorical(np.resize([.75, .25], 48), categories=[.75, .25])
    fitted = model_fit(RLT_cla, X, y)
    assert fitted.classes_.tolist() == [.75, .25]
    assert fitted.predict(X).dtype.kind == "f"
    with pytest.raises(ValueError, match="classification labels"):
        model_fit(RLT_cla, X, np.asarray(y))
    with pytest.warns(UserWarning, match="unused"):
        single = model_fit(RLT_cla, X, pd.Categorical(["z"] * 48, categories=["a", "z"]))
    assert single.classes_.tolist() == ["a", "z"]
    assert np.all(single.predict(X) == "z")
    missing = pd.Categorical(["z"] * 47 + [None], categories=["a", "z"])
    with pytest.raises(ValueError, match="missing classification labels"):
        model_fit(RLT_cla, X, missing)


@pytest.mark.parametrize("categories", [[.75, .25], [.25, .75]])
@pytest.mark.parametrize("container", ["categorical", "series", "index", "array", "column"])
def test_float_categorical_score_matches_accuracy_with_weights_and_unseen_truth(categories, container):
    pd = pytest.importorskip("pandas")
    X, _ = data(RLT_cla)
    y = pd.Categorical(np.resize([.75, .25], len(X)), categories=categories)
    fitted = model_fit(RLT_cla, X, y, min_samples_leaf=100,
                       resample_replace=False, resample_prob=1)
    truth = np.asarray(y).copy()
    truth[::5] = .5  # A label absent from the fitted classes counts as incorrect.
    prediction = fitted.predict(X)
    weights = np.arange(1., len(X) + 1)
    supplied = {
        "categorical": lambda: pd.Categorical(truth, categories=[.5, .25, .75]),
        "series": lambda: pd.Series(pd.Categorical(truth, categories=[.5, .25, .75])),
        "index": lambda: pd.CategoricalIndex(truth, categories=[.5, .25, .75]),
        "array": lambda: truth,
        "column": lambda: truth[:, None],
    }[container]()
    assert fitted.score(X, supplied) == pytest.approx(np.mean(truth == prediction))
    assert fitted.score(X, supplied, sample_weight=weights) == pytest.approx(
        np.average(truth == prediction, weights=weights))
    assert fitted.score(X, y) == pytest.approx(np.mean(np.asarray(y) == prediction))


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf, None, "pandas_missing"])
def test_float_categorical_score_rejects_missing_or_nonfinite_truth(invalid):
    pd = pytest.importorskip("pandas")
    X, _ = data(RLT_cla)
    y = pd.Categorical(np.resize([.75, .25], len(X)))
    fitted = model_fit(RLT_cla, X, y)
    truth = np.asarray(y, dtype=object).copy()
    truth[0] = pd.NA if isinstance(invalid, str) else invalid
    with pytest.raises(ValueError, match="missing or nonfinite"):
        fitted.score(X, truth)


def test_float_categorical_score_preserves_target_and_weight_validation():
    pd = pytest.importorskip("pandas")
    X, _ = data(RLT_cla)
    y = pd.Categorical(np.resize([.75, .25], len(X)))
    fitted = model_fit(RLT_cla, X, y)
    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        fitted.score(X, y[:-1])
    with pytest.raises(ValueError, match="1d array"):
        fitted.score(X, np.column_stack([y, y]))
    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        fitted.score(X, y, sample_weight=np.ones(len(X) - 1))
    # Weight validation belongs to sklearn: releases differ in the exception
    # type for 2D weights and whether NaN weights raise or return NaN.
    from sklearn.metrics import accuracy_score

    for weights in (np.ones((len(X), 2)), np.full(len(X), np.nan)):
        try:
            expected = accuracy_score(np.asarray(y) == .75, fitted.predict(X) == .75,
                                      sample_weight=weights)
        except (ValueError, TypeError) as error:
            with pytest.raises(type(error)):
                fitted.score(X, y, sample_weight=weights)
        else:
            np.testing.assert_equal(fitted.score(X, y, sample_weight=weights), expected)
    for dtype in (complex, object):
        truth = np.asarray(y, dtype=dtype).copy()
        truth[0] = .25 + 1j
        with pytest.raises(ValueError, match="real classification labels|Complex data"):
            fitted.score(X, truth)


@pytest.mark.parametrize("labels", [["a", "z"], [0, 1], [0., 1.]])
def test_ordinary_classifier_score_keeps_mixin_path(labels, monkeypatch):
    from sklearn.base import ClassifierMixin
    X, _ = data(RLT_cla)
    y = np.resize(labels, len(X))
    fitted = model_fit(RLT_cla, X, y)
    calls = []
    original = ClassifierMixin.score

    def observed(self, *args, **kwargs):
        calls.append(True)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ClassifierMixin, "score", observed)
    weights = np.arange(1., len(X) + 1)
    assert fitted.score(X, y, sample_weight=weights) == pytest.approx(
        np.average(y == fitted.predict(X), weights=weights))
    assert calls == [True]


@pytest.mark.parametrize("explicit_cv", [False, True])
def test_float_categorical_default_scoring_works_with_kfold(explicit_cv):
    pd = pytest.importorskip("pandas")
    from sklearn.model_selection import KFold, cross_val_score
    X, _ = data(RLT_cla)
    y = pd.Categorical(np.where(X[:, 0] > 0, .25, .75), categories=[.75, .25])
    estimator = RLT_cla(n_estimators=8, random_state=19, n_jobs=1)
    folds = KFold(3, shuffle=True, random_state=12) if explicit_cv else KFold(3)
    scores = cross_val_score(estimator, X, y, cv=folds if explicit_cv else 3,
                             error_score="raise")
    expected = []
    for train, test in folds.split(X, y):
        fitted = clone(estimator).fit(X[train], y[train])
        expected.append(np.mean(fitted.predict(X[test]) == np.asarray(y[test])))
    np.testing.assert_array_equal(scores, expected)


def signed_preset(n=48, trees=8):
    counts = np.full((n, trees), -1, dtype=np.int64)
    counts[:4] = 2
    counts[4:16] = 0
    return counts


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("linear_comb", [1, 2])
def test_public_signed_presets_preserve_repeats_and_negative_exclusions(model, linear_comb):
    X, y = data(model)
    preset = signed_preset()
    fitted = model(n_estimators=8, n_jobs=1, random_state=19, min_samples_leaf=100,
                   linear_comb=linear_comb).fit(X, y, resample_preset=preset)
    np.testing.assert_array_equal(fitted.obstrack_, preset)
    assert fitted.params_.obs_track
    for weight in fitted.forest_["NodeWeight"]:
        assert weight[0] == 8
    kernel = fitted.forest_kernel(X, oob=True)
    assert kernel["N"][4, 5] == 8
    assert kernel["N"][16, 17] == 0
    assert kernel["N"][0, 4] == 0
    before = fitted.obstrack_.copy()
    preset[:] = 0
    np.testing.assert_array_equal(fitted.obstrack_, before)
    if model is RLT_reg:
        np.testing.assert_allclose(fitted.predict(X), np.mean(y[:4]))


@pytest.mark.parametrize("mode", ["matched", "ij", "jack"])
@pytest.mark.parametrize("model", MODELS)
def test_variance_modes_replace_custom_presets_before_validating_ignored_matrix(mode, model):
    X, y = data(model)
    with pytest.warns(UserWarning, match="resample_preset is ignored"):
        fitted = model(n_estimators=8, n_jobs=1, random_state=19, var_mode=mode).fit(
            X, y, resample_preset="ignored malformed input")
    expected = model_fit(model, X, y, var_mode=mode)
    np.testing.assert_array_equal(fitted.obstrack_, expected.obstrack_)
    np.testing.assert_array_equal(fitted.predict(X), expected.predict(X))


@pytest.mark.parametrize("change", ["shape", "nan", "inf", "fraction", "negative_fraction", "large", "uint64", "zero", "excluded", "mass", "bool", "string"])
def test_invalid_preset_counts_are_rejected_before_native_use(change):
    counts = signed_preset().astype(float)
    if change == "shape":
        counts = counts[:, :-1]
    elif change in ("nan", "inf", "fraction", "negative_fraction", "large"):
        counts[0, 0] = {"nan": np.nan, "inf": np.inf, "fraction": 1.9, "negative_fraction": -.5, "large": 2**31}[change]
    elif change == "uint64":
        counts = np.full((48, 8), np.iinfo(np.uint64).max, dtype=np.uint64)
    elif change == "zero":
        counts[:, 0] = 0
    elif change == "excluded":
        counts[:, 0] = -1
    elif change == "mass":
        counts[0, 0] = 49
    elif change == "bool":
        counts = np.ones((48, 8), dtype=bool)
    else:
        counts = counts.astype(str)
    with pytest.raises(ValueError, match="resample_preset"):
        _params.check_resample_preset(counts, 48, 8)


@pytest.mark.parametrize("model", MODELS)
def test_failed_refit_keeps_old_schema_labels_and_predictions_on_validation_and_native_failure(model):
    pd = pytest.importorskip("pandas")
    frame = feature_frame()
    _, y = data(model)
    fitted = model_fit(model, frame, y)
    prediction = fitted.predict(frame)
    old_schema, old_forest = fitted.feature_schema_, fitted.forest_
    old_classes = getattr(fitted, "classes_", None)
    changed = frame.rename(columns={"nominal": "new_name"}).copy()
    changed["nominal"] = pd.Categorical(["new"] * len(frame))
    response = np.array(["other_a", "other_b"] * 24) if model is RLT_cla else y
    short = (y[0][:-1], y[1][:-1]) if model is RLT_surv else response[:-1]
    with pytest.raises(ValueError):
        fitted.fit(changed, short)
    with pytest.raises(RuntimeError, match="weight|mass"):
        fitted.fit(changed, response, sample_weight=np.r_[1., np.zeros(47)])
    assert fitted.feature_schema_ is old_schema and fitted.forest_ is old_forest
    assert getattr(fitted, "classes_", None) is old_classes
    np.testing.assert_array_equal(fitted.predict(frame), prediction)


@pytest.mark.parametrize("model", [RLT_reg, RLT_cla])
def test_keep_all_predict_var_keeps_two_item_return_and_points_to_details(model):
    X, y = data(model)
    fitted = model_fit(model, X, y, var_mode="matched")
    with pytest.warns(FutureWarning, match="predict_details"):
        result = fitted.predict_var(X[:3], keep_all=True)
    assert len(result) == 2


@pytest.mark.parametrize("model", MODELS)
def test_native_oob_counts_distinguish_excluded_and_eligible_rows(model):
    X, y = data(model)
    fitted = model(n_estimators=8, n_jobs=1, random_state=19).fit(X, y, resample_preset=signed_preset())
    np.testing.assert_array_equal(fitted.oob_counts_, np.r_[np.zeros(4), np.full(12, 8), np.zeros(32)])
    fitted.set_params(resample_replace=False, resample_prob=1)
    fitted.fit(X, y)
    assert not hasattr(fitted, "oob_counts_")
