# Reuse predictions and inspect a fitted model

Use `predict_details` when you need predictions together with class labels,
uncertainty, or individual trees. The usual `predict`, `predict_proba`, and
two-item `predict_var` methods remain available.

```python
import numpy as np
from rlt import RLT_cla

rng = np.random.default_rng(12)
X = rng.normal(size=(80, 4))
y = np.where(X[:, 0] + X[:, 1] > 0, "positive", "negative")
model = RLT_cla(n_estimators=60, resample_track=True,
                random_state=12, n_jobs=1).fit(X, y)

result = model.predict_details(X[:3], var_est=True, var_mode="ij", keep_all=True)
assert np.array_equal(result["Prediction"], model.predict(X[:3]))
assert result["PredictionAll"].shape == (3, 60, 2)
assert np.array_equal(result["classes"], model.classes_)
```

The last axis of `PredictionAll` follows `classes`; its entries are class
probabilities. Regression returns an `(n, trees)` array. Survival returns
`AllHazard` with shape `(n, trees, times)`, together with `timepoints`.
`predict_all_trees` provides these arrays directly for all three estimators.
Negative regression or class-probability variance estimates are reported as
NaN, as in R.

## Check which rows have OOB predictions

Some training rows have no out-of-bag predictions. Use the availability mask
before summarizing OOB accuracy or interpreting labels.

```python
oob = model.oob_details()
available = oob["available"]
oob_accuracy = np.mean(oob["Prediction"][available] == y[available])
assert oob["counts"].shape == (80,)
```

Unavailable class labels are `None`, and unavailable numeric outputs are
NaN. The helper raises an error when OOB computation was disabled during
fitting. The older `oob_*` arrays remain available, but their placeholder
values should not be interpreted without the mask.

## Save a model and use it again

Pickle and joblib preserve the forest, effective parameters, category
dictionaries, class order, survival grid, and retained sampling counts.
Load model files only from sources you trust. Keep the package version and
numerical environment with a saved model for reproducibility.

```python
import pickle

saved = pickle.dumps(model)
restored = pickle.loads(saved)
assert np.array_equal(restored.predict(X[:3]), model.predict(X[:3]))
```

For a file, use `joblib.dump(model, path)` and `joblib.load(path)`. Saving
does not advance a supplied NumPy random generator. `sklearn.base.clone`
continues to return an unfitted estimator with the requested constructor
parameters.

## Keep labels when fitting data frames

Pandas is optional. Unordered categorical columns use categorical splits.
Ordered categoricals and string columns use numerical ranks, matching R's
encoding. Prediction reuses the training dictionary and matches named
columns, so changing their order does not change their meaning.

```python
import pandas as pd

frame = pd.DataFrame({
    "group": pd.Categorical(np.where(X[:, 0] > 0, "high", "low"),
                            categories=["low", "high", "unused"]),
    "score": X[:, 1],
})
labels = pd.Categorical(y, categories=["positive", "negative"])
labeled = RLT_cla(n_estimators=40, random_state=12, n_jobs=1).fit(frame, labels)
assert labeled.classes_.tolist() == ["positive", "negative"]
reordered = frame[["score", "group"]].copy()
reordered["group"] = reordered["group"].cat.reorder_categories(
    ["unused", "high", "low"])
assert np.array_equal(labeled.predict(frame), labeled.predict(reordered))
```

Declared response levels determine probability-column order and tie
breaking, including unused levels. Ordinary label arrays retain sorted
observed class order. Unknown predictor labels and missing values are
rejected. Plain numeric arrays can still use zero-based categorical codes
with `categorical_features`.

For a fitted labeled predictor, a plain numeric prediction column means
zero-based training codes. Keep a pandas categorical dtype when supplying
numeric labels such as 10, 2 and 1, so they are matched as labels rather
than mistaken for codes.

## Supply sampling counts

Pass `resample_preset` to `fit` to use a matrix of shape `(rows, trees)`.
Positive integers are in-bag multiplicities, zero means OOB, and negative
integers exclude a row from both groups. Each column needs positive total
mass no greater than the number of training rows.

```python
from rlt import RLT_reg

preset = np.zeros((80, 4), dtype=int)
for tree in range(4):
    preset[tree:tree + 40, tree] = 1
preset[79, :] = -1
custom = RLT_reg(n_estimators=4, random_state=12, n_jobs=1).fit(
    X, X[:, 0] + X[:, 1], resample_preset=preset)
assert np.array_equal(custom.obstrack_, preset)
assert not custom.oob_details()["available"][79]
```

Matched, IJ, and jackknife fit modes replace a supplied preset with their
sampling scheme and warn. As in R, prediction-time IJ/jackknife calculations
on an ordinary tracked fit treat negative counts as zero. Their treatment
therefore differs from fitting and OOB-kernel exclusion. Also, nominal
full-sample settings can disable OOB predictions and importance even if a
preset contains zeros.

## Inspect a tree or compute concordance

`tree_summary(0)` returns node depth, named splits, thresholds or category
partitions, node weights, and terminal predictions. `print_tree(0)` displays
the same information. `get_one_tree(0)` returns independent copies of all
raw arrays, including class probabilities, survival node hazards, and
linear-combination loadings. Tree indices start at zero.

```python
tree = model.tree_summary(0)
assert tree["Depth"][0] == 0
assert "NodeProb" in model.get_one_tree(0)

from rlt import cindex
assert cindex([1, 2, 3], [1, 1, 1], [3, 2, 1]) == 1.0
```

For `cindex(time, event, risk)`, higher risk means earlier failure. Times
must be finite and events must be exactly zero or one. No comparable pairs
or nonfinite risk scores produce NaN.

Concordance follows RLT's tie rules: equal risks receive half credit for
distinct comparable times. Two events at the same time receive full credit
for equal risks and half credit otherwise. At an event/censoring time tie,
the observed event is treated as earlier; two censored ties are omitted.
