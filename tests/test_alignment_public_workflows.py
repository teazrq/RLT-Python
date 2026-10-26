"""Public prediction, inspection, persistence and metric contracts."""

import pickle
import subprocess
import sys

import joblib
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone

from rlt import RLT_cla, RLT_reg, RLT_surv, cindex


def data():
    rng = np.random.default_rng(142)
    x = rng.normal(size=(48, 4))
    y = 2 * x[:, 0] + x[:, 1] + rng.normal(size=48) * .2
    return x, y


def fit_model(family, linear_comb, **kwargs):
    x, y = data()
    cls = {"reg": RLT_reg, "cla": RLT_cla, "surv": RLT_surv}[family]
    target = y if family == "reg" else (
        np.where(y > 0, "yes", "no") if family == "cla" else
        (np.exp(-y / 3), np.arange(48) % 3 != 0))
    params = dict(n_estimators=12, min_samples_leaf=4, mtry=4,
                  linear_comb=linear_comb, random_state=91, n_jobs=1,
                  resample_track=True)
    params.update(kwargs)
    return cls(**params).fit(x, target), x, target


@pytest.mark.parametrize("family", ["reg", "cla", "surv"])
@pytest.mark.parametrize("linear_comb", [1, 3])
def test_complete_public_predictions(family, linear_comb):
    model, x, _ = fit_model(family, linear_comb)
    detail = model.predict_details(x[:3], var_est=True, var_mode="ij", keep_all=True)
    assert detail["var_mode"] == "ij"
    if family == "surv":
        assert_allclose(detail["Survival"], model.predict(x[:3]))
        assert_array_equal(detail["timepoints"], model.timepoints_)
        assert detail["Cov"].shape == (3, len(model.timepoints_), len(model.timepoints_))
        assert detail["AllHazard"].shape == (3, 12, len(model.timepoints_))
        reduced = model.predict_details(x[:1], keep_all=True, band_grid_size=5)
        assert reduced["AllHazard"].shape == (1, 12, len(reduced["timepoints"]))
        assert_allclose(model.predict_all_trees(x[:1], band_grid_size=5), reduced["AllHazard"])
    else:
        assert_array_equal(detail["Prediction"], model.predict(x[:3]))
        assert np.all((detail["Variance"] >= 0) | np.isnan(detail["Variance"]))
        assert_allclose(detail["PredictionAll"], model.predict_all_trees(x[:3]))
        if family == "cla":
            assert_array_equal(detail["classes"], model.classes_)
            assert detail["PredictionAll"].shape == (3, 12, 2)
            assert_allclose(detail["PredictionAll"].mean(axis=1), model.predict_proba(x[:3]))
        else:
            assert_allclose(detail["PredictionAll"].mean(axis=1), detail["Prediction"])


@pytest.mark.parametrize("family", ["reg", "cla", "surv"])
@pytest.mark.parametrize("linear_comb", [1, 3])
def test_fresh_process_pickle_and_joblib(tmp_path, family, linear_comb):
    model, x, y = fit_model(family, linear_comb)
    requested = model.get_params(deep=False)
    expected = model.predict_details(x[:3], var_est=True, var_mode="ij", keep_all=True)
    paths = []
    for extension in ("pickle", "joblib"):
        path = tmp_path / ("model." + extension)
        if extension == "pickle":
            path.write_bytes(pickle.dumps(model))
        else:
            joblib.dump(model, path)
        paths.append(str(path))
    input_path = tmp_path / "inputs.pkl"
    input_path.write_bytes(pickle.dumps((x, y, expected, requested, model.__getstate__()["_native_params_state"])))
    script = '''
import sys, pickle, joblib, numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone
x,y,expected,requested,params = pickle.load(open(sys.argv[1], 'rb'))
for path in sys.argv[2:]:
    m = pickle.load(open(path, 'rb')) if path.endswith('pickle') else joblib.load(path)
    assert m.get_params(deep=False) == requested
    assert m.__getstate__()['_native_params_state'] == params
    actual = m.predict_details(x[:3], var_est=True, var_mode='ij', keep_all=True)
    assert actual.keys() == expected.keys()
    for key in actual:
        if key in ('classes', 'var_mode') or (key == 'Prediction' and hasattr(m, 'classes_')):
            assert_array_equal(actual[key], expected[key])
        else:
            assert_allclose(actual[key], expected[key], rtol=0, atol=0, equal_nan=True)
    assert not hasattr(clone(m), 'forest_')
    before_refit = m.predict(x)
    m.fit(x, y)
    assert_array_equal(m.predict(x), before_refit)
'''
    subprocess.run([sys.executable, "-c", script, str(input_path), *paths], check=True, timeout=60)
    assert not hasattr(clone(model), "forest_")


@pytest.mark.parametrize("rng", [np.random.RandomState(13), np.random.default_rng(13)])
def test_serialization_preserves_rng_and_exact_wide_seed(rng):
    model, x, y = fit_model("reg", 1, random_state=rng)
    before = pickle.dumps(rng)
    restored = pickle.loads(pickle.dumps(model))
    assert pickle.dumps(rng) == before
    assert pickle.dumps(restored.random_state) == before
    model.fit(x, y)
    restored.fit(x, y)
    assert model.params_.seed == restored.params_.seed
    assert_array_equal(model.predict(x), restored.predict(x))
    wide, _, _ = fit_model("reg", 1, random_state=2**53 + 123)
    assert pickle.loads(pickle.dumps(wide)).params_.seed == 2**53 + 123


@pytest.mark.parametrize("family", ["reg", "cla", "surv"])
@pytest.mark.parametrize("linear_comb", [1, 3])
def test_tree_payload_copy_and_reconstructed_predictions(family, linear_comb):
    model, x, _ = fit_model(family, linear_comb)
    raw = model.get_one_tree(0)
    payload = {"reg": "NodeAve", "cla": "NodeProb", "surv": "NodeHaz"}[family]
    assert payload in raw
    assert ("SplitLoad" in raw) == (linear_comb > 1)
    leaf = []
    for row in x[:4]:
        node = 0
        while True:
            ids = np.atleast_1d(raw["SplitVar"][node])
            if ids[0] < 0:
                break
            loads = raw["SplitLoad"][node] if linear_comb > 1 else np.ones(1)
            active = loads != 0
            value = np.dot(row[ids[active]], loads[active])
            node = int(raw["RightNode" if value > raw["SplitValue"][node] else "LeftNode"][node])
        leaf.append(node)
    prediction = np.asarray([raw[payload][i] for i in leaf])
    if family == "surv":
        prediction = prediction[:, 1:]  # native hazards include time-zero slot
    assert_allclose(prediction, model.predict_all_trees(x[:4])[:, 0])
    before = model.predict(x[:4]).copy()
    if family == "surv":
        raw[payload][leaf[0]][:] = 99
    else:
        raw[payload][:] = 99
    if linear_comb > 1:
        raw["SplitLoad"][:] = 99
    assert_array_equal(model.predict(x[:4]), before)
    summary = model.tree_summary(0)
    assert summary["Depth"][0] == 0
    assert len(summary["Node"]) == len(raw["NodeWeight"])
    for node in range(len(raw["NodeWeight"])):
        if summary["Prediction"][node] is not None:
            assert summary["Split"][node] == "leaf"
            if family == "surv":
                assert len(summary["Prediction"][node]) == len(model.timepoints_)
                assert_allclose(summary["Prediction"][node], model.get_one_tree(0)["NodeHaz"][node][1:])
    for bad in (-1, 12, 1.5, True):
        with pytest.raises((ValueError, IndexError)):
            model.get_one_tree(bad)


def test_oob_labels_never_decode_unavailable_rows():
    x, y = data()
    preset = np.zeros((48, 4), dtype=int)
    preset[:20] = 1
    preset[20:25] = -3
    model = RLT_cla(n_estimators=4, min_samples_leaf=50, n_jobs=1, random_state=3).fit(
        x, np.where(y > 0, "yes", "no"), resample_preset=preset)
    result = model.oob_details()
    assert_array_equal(result["counts"], (preset == 0).sum(axis=1))
    assert result["Prediction"][:25].tolist() == [None] * 25
    assert np.isnan(result["Prob"][:25]).all()
    assert set(result["Prediction"][25:]).issubset({"no", "yes"})
    result["counts"][:] = 999
    assert not np.any(model.oob_counts_ == 999)
    full = RLT_reg(n_estimators=4, resample_replace=False, resample_prob=1,
                   n_jobs=1, random_state=3).fit(x, y, resample_preset=preset)
    with pytest.raises(ValueError, match="not computed"):
        full.oob_details()


def test_public_concordance_validation_and_ties():
    assert cindex([1, 2, 3], [1, 1, 1], [3, 2, 1]) == 1
    assert cindex([1, 2, 3], [1, 1, 1], [1, 2, 3]) == 0
    assert cindex([1, 2, 3], [1, 1, 1], [1, 1, 1]) == .5
    assert cindex([1, 1], [1, 1], [2, 2]) == 1
    assert cindex([1, 1], [1, 1], [2, 1]) == .5
    assert cindex([1, 1], [1, 0], [2, 1]) == 1
    assert cindex([1, 1], [1, 0], [1, 2]) == 0
    assert cindex([1, 1], [1, 0], [1, 1]) == .5
    assert np.isnan(cindex([1, 1], [0, 0], [1, 2]))
    assert np.isnan(cindex([1, 2], [0, 0], [1, 2]))
    assert np.isnan(cindex([1, 2], [1, 1], [1, np.inf]))
    for time, event, risk in (([1, np.nan], [1, 1], [1, 2]),
                              ([1, 2], [.5, 1], [1, 2]),
                              ([1, 2], [1], [1, 2]),
                              ([[1, 2]], [1, 1], [1, 2])):
        with pytest.raises(ValueError):
            cindex(time, event, risk)
