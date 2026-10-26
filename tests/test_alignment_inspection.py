"""Category routing and labeled, signed-preset model persistence."""

import pickle
import subprocess
import sys
import warnings

import joblib
import numpy as np
import pytest

from rlt import RLT_reg, RLT_cla, RLT_surv


@pytest.mark.parametrize("cls,linear_comb", [(RLT_reg, 1), (RLT_reg, 2),
                                            (RLT_cla, 1), (RLT_cla, 2),
                                            (RLT_surv, 1)])
def test_category_summary_partitions_reconstruct_native_tree_predictions(cls, linear_comb):
    X = np.column_stack([np.roll(np.repeat(np.arange(4), 16), k)
                         for k in (0, 8, 4)]).astype(float)
    signal = np.isin(X[:, 0], [0, 2])
    y = 3. + 10 * signal if cls is RLT_reg else (
        np.where(signal, "yes", "no") if cls is RLT_cla else
        (np.where(signal, 1., 3.), np.ones(64)))
    model = cls(n_estimators=4, mtry=3, min_samples_leaf=4,
                linear_comb=linear_comb, categorical_features=[0, 1, 2],
                embed_ntrees=24, embed_nmin=4, embed_protect=2,
                random_state=31, n_jobs=1).fit(X, y)
    # Retain the original public keyword as well as positional access.
    raw = model.get_one_tree(tree_id=0)
    summary = model.tree_summary(0)
    category_nodes = 0
    for node, variable in enumerate(raw["SplitVar"]):
        variable = int(np.atleast_1d(variable)[0])
        if variable < 0:
            continue
        category_nodes += 1
        expected = [k for k in range(4)
                    if (int(raw["SplitValue"][node]) >> (k + 1)) & 1]
        assert summary["Value"][node] == expected
        assert "right partition" in summary["Split"][node]
    assert category_nodes > 0, "the fixture must exercise categorical splitting"
    leaves = []
    for row in X:
        node = 0
        while int(np.atleast_1d(raw["SplitVar"][node])[0]) >= 0:
            variable = int(np.atleast_1d(raw["SplitVar"][node])[0])
            right = bool((int(raw["SplitValue"][node]) >> (int(row[variable]) + 1)) & 1)
            node = int(raw["RightNode" if right else "LeftNode"][node])
        leaves.append(node)
    payload = "NodeAve" if cls is RLT_reg else "NodeProb" if cls is RLT_cla else "NodeHaz"
    expected = np.asarray([raw[payload][i] for i in leaves])
    if cls is RLT_surv:
        expected = expected[:, 1:]
    np.testing.assert_allclose(expected, model.predict_all_trees(X)[:, 0], rtol=0, atol=0)


@pytest.mark.parametrize("cls", [RLT_reg, RLT_cla, RLT_surv])
@pytest.mark.parametrize("linear_comb", [1, 2])
def test_rank_summary_thresholds_route_in_public_zero_based_coordinates(cls, linear_comb):
    pd = pytest.importorskip("pandas")
    u = np.arange(64)
    frame = pd.DataFrame({
        "rank": pd.Categorical(np.resize(["low", "high", "medium", "top"], 64),
                               categories=["low", "medium", "high", "top"], ordered=True),
        "text": np.resize(["c", "a", "d", "b", "e"], 64),
        "numeric": np.sin(u * .37),
    })
    rank = np.asarray(frame["rank"].cat.codes)
    text = np.searchsorted(np.unique(frame["text"]), frame["text"])
    encoded = np.column_stack((rank, text, frame["numeric"]))
    signal = 1.8 * rank + .7 * text + .05 * frame["numeric"]
    y = signal if cls is RLT_reg else (
        np.where(signal > np.median(signal), "yes", "no") if cls is RLT_cla else
        (np.exp(-signal / 3), (u % 4 != 0).astype(int)))
    model = cls(n_estimators=2, mtry=3, min_samples_leaf=4,
                linear_comb=linear_comb, linear_comb_method="naive", embed_nmin=100,
                resample_replace=False, resample_prob=1,
                random_state=73, n_jobs=1).fit(frame, y)
    raw, summary = model.get_one_tree(0), model.tree_summary(0)
    rank_nodes, multi_nodes, leaves = 0, 0, []
    for node, variable in enumerate(raw["SplitVar"]):
        ids = np.atleast_1d(variable)
        if ids[0] < 0:
            continue
        loads = raw["SplitLoad"][node] if linear_comb > 1 else np.ones(1)
        active = [(int(j), float(w)) for j, w in zip(ids, loads) if w != 0]
        shifted = [(j, w) for j, w in active if j < 2]
        if shifted:
            rank_nodes += 1
            assert "code(" in summary["Split"][node]
        multi_nodes += len(active) > 1
        np.testing.assert_allclose(summary["Value"][node],
            raw["SplitValue"][node] - sum(w for _, w in shifted), rtol=0, atol=0)
    assert rank_nodes > 0
    if linear_comb > 1:
        assert multi_nodes > 0
    for row in encoded:
        node = 0
        while np.atleast_1d(raw["SplitVar"][node])[0] >= 0:
            ids = np.atleast_1d(raw["SplitVar"][node])
            loads = raw["SplitLoad"][node] if linear_comb > 1 else np.ones(1)
            active = loads != 0
            right = np.dot(row[ids[active]], loads[active]) > summary["Value"][node]
            node = int(raw["RightNode" if right else "LeftNode"][node])
        leaves.append(node)
    payload = "NodeAve" if cls is RLT_reg else "NodeProb" if cls is RLT_cla else "NodeHaz"
    expected = np.asarray([raw[payload][i] for i in leaves])
    if cls is RLT_surv:
        expected = expected[:, 1:]
    np.testing.assert_allclose(expected, model.predict_all_trees(frame)[:, 0], rtol=0, atol=0)


_RESTORE_LABELED_MODEL = '''
import pickle, sys, warnings, joblib, numpy as np
for path in sys.argv[1:]:
    args=pickle.load(open(path,'rb')) if path.endswith('.pickle') else joblib.load(path)
    model=args['model']; assert 'params_' in model.__dict__
    model.__getstate__(); assert 'params_' in model.__dict__
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        actual=model.predict_details(args['query'],keep_all=True)
    assert actual.keys()==args['expected'].keys()
    for key,value in actual.items():
        np.testing.assert_array_equal(value,args['expected'][key])
    oob=model.oob_details()
    np.testing.assert_array_equal(oob['counts'],(args['counts']==0).sum(axis=1))
    assert not oob['available'][:26].any() and oob['available'][26:].all()
    if hasattr(model,'classes_'):
        assert model.classes_.tolist()==['z','a','unused']
        assert oob['Prediction'][:26].tolist()==[None]*26
        assert np.isnan(oob['Prob'][:26]).all()
    else:
        key='Hazard' if hasattr(model,'timepoints_') else 'Prediction'
        assert np.isnan(oob[key][:26]).all()
    np.testing.assert_array_equal(model.obstrack_,args['counts'])
    schema=model.feature_schema_
    assert schema[0]['levels'].tolist()==[20,10,40,30,50]
    assert schema[1]['kind']=='ordered'
    assert schema[1]['levels'].tolist()==['low','medium','high']
    assert schema[2]['kind']=='string' and schema[2]['levels'].tolist()==['a','b','c']
'''


@pytest.mark.parametrize("cls", [RLT_reg, RLT_cla, RLT_surv])
@pytest.mark.parametrize("linear_comb", [1, 2])
def test_fresh_process_preserves_label_dictionaries_class_order_and_signed_presets(tmp_path, cls, linear_comb):
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({
        "group": pd.Categorical(np.resize([20, 10, 40, 30], 64),
                                categories=[20, 10, 40, 30, 50]),
        "rank": pd.Categorical(np.resize(["low", "high", "medium"], 64),
                               categories=["low", "medium", "high"], ordered=True),
        "text": np.resize(["c", "a", "b"], 64),
    })
    counts = np.zeros((64, 4), dtype=int)
    counts[:6], counts[6:26] = -3, 1
    y = np.arange(64, dtype=float) if cls is RLT_reg else (
        pd.Categorical(np.resize(["z", "a"], 64), categories=["z", "a", "unused"])
        if cls is RLT_cla else (np.arange(1., 65), np.arange(64) % 4 != 0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = cls(n_estimators=4, mtry=3, linear_comb=linear_comb,
                    embed_nmin=100, random_state=19, n_jobs=1).fit(
                        frame, y, resample_preset=counts)
    query = frame.iloc[[31, 3, 55, 1]][["text", "rank", "group"]].copy()
    query["group"] = query["group"].cat.reorder_categories([50, 40, 30, 10, 20])
    query["rank"] = query["rank"].cat.reorder_categories(["high", "medium", "low"])
    with pytest.warns(UserWarning, match="matching by name"):
        expected = model.predict_details(query, keep_all=True)
    bundle = dict(model=model, query=query, expected=expected, counts=counts)
    path = tmp_path / "model.pickle"
    path.write_bytes(pickle.dumps(bundle))
    joblib_path = path.with_suffix(".joblib")
    joblib.dump(bundle, joblib_path)
    result = subprocess.run([sys.executable, "-c", _RESTORE_LABELED_MODEL,
                             str(path), str(joblib_path)], capture_output=True,
                            text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
