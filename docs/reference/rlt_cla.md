# RLT_cla

RLT classification forest, a sklearn-compatible estimator porting the RLT R
package's `RLT(model = "classification")`.

`RLT_cla` grows a random forest of classification trees. Ordinary integer
or string label arrays use sorted observed classes. A pandas categorical
response retains its declared levels and order, including unused levels
and explicitly categorical numeric labels such as 0.25 and 0.75.
`classes_` determines probability-column order and tie breaking; `predict`
returns labels in their original type.

```python
import numpy as np
from rlt import RLT_cla

rng = np.random.default_rng(42)
X = rng.normal(size=(100, 5))
y = rng.binomial(1, 1 / (1 + np.exp(-2 * X[:, 0])))

model = RLT_cla(n_estimators=500, random_state=42)
model.fit(X, y)                # string labels fine
prob = model.predict_proba(X)  # (n, n_classes)
```

## Constructor

```python
RLT_cla(
    n_estimators=None,
    mtry=None,
    min_samples_leaf=5,
    nsplit=0,
    resample_replace=True,
    resample_prob=None,
    importance="none",
    resample_track=False,
    var_mode="none",
    categorical_features=None,
    linear_comb=1,
    linear_comb_method="default",
    alpha=0,
    reinforcement=False,
    embed_ntrees=50,
    embed_mtry=0.5,
    embed_nmin=5,
    embed_nsplit=3,
    embed_resample_replace=True,
    embed_resample_prob=0.9,
    embed_mute=0,
    embed_protect=None,
    embed_threshold=0.25,
    n_jobs=-1,
    verbose=0,
    random_state=None,
)
```

### Parameters

See [`RLT_reg`](rlt_reg.md) for the shared parameters. The loading method
has classification-specific choices:

| Parameter | Type / values | Default | Description |
|---|---|---|---|
| `linear_comb_method` | "default", "lda", "naive", "random", "logistic" or code 1 to 4 | "default" | How combination loadings are computed ("lda" = code 1 is the default). Integral doubles are accepted. Unknown names/out-of-range codes warn and reset to `lda`; fractional and nonfinite codes are rejected. |
| `alpha` | float | 0 | Clamped to `[0, 0.5]` at fit, like R. |
| `categorical_features` | bool mask / int indices / None | None | See [`RLT_reg`](rlt_reg.md); same behavior for all three estimators. |

## Methods

| Method | Returns | Description |
|---|---|---|
| `fit(X, y, sample_weight=None, var_prob=None, *, resample_preset=None)` | `self` | Grow the forest, optionally with signed sampling counts. |
| `predict(X)` | labels | Predicted class labels (original label type). |
| `predict_proba(X)` | `(n, n_classes)` ndarray | Class-membership probabilities. |
| `predict_log_proba(X)` | `(n, n_classes)` ndarray | Log-probabilities. |
| `predict_details(X, *, var_est=False, var_mode=None, keep_all=False)` | dict | Decoded labels, class order, probabilities, and optional variance and per-tree probabilities. |
| `predict_all_trees(X)` | `(n, trees, classes)` ndarray | Per-tree probabilities in `classes_` order. |
| `oob_details()` | dict | Labeled OOB predictions with counts and availability mask. |
| `tree_summary(tree=0)` / `print_tree(tree=0)` | dict | Readable splits and terminal class probabilities. |
| `predict_var(X, var_mode=None, ...)` | `(prob, variance)` | Class probabilities plus their `(n, n_classes)` variance. Requires a fitted variance mode or an explicit compatible prediction-time mode; negative estimates become NaN. |
| `importance_table()` | `ImportanceTable` | Variable-importance summary (Variable / VI, plus SD / Z / Sig when fitted with `importance != "none"` and `var_mode="matched"`); same as `rlt.importance(model)`. |
| `forest_kernel(X1, X2=None, vs_train=False, oob=False)` | integer ndarray, or dict with `oob=True` | Forest similarity kernel. |
| `get_one_tree(tree_id)` | dict of ndarrays | Raw arrays of tree `tree_id` (0-indexed). |
| `score(X, y, sample_weight=None)` | float | Mean accuracy, including declared numeric categories, with optional sample weights. |

## Fitted attributes

| Attribute | Shape | Description |
|---|---|---|
| `classes_` | `(n_classes,)` | Declared pandas categorical levels, including unused levels, or sorted observed labels for ordinary arrays. |
| `forest_` | dict | The full fitted forest. |
| `oob_class_` | `(n,)` | Out-of-bag predicted classes (integer codes). |
| `oob_prob_` | `(n, n_classes)` | Out-of-bag class probabilities. |
| `oob_counts_` | `(n,)` | Number of eligible OOB predictions per row; use `oob_details()` to mask unavailable rows. |
| `oob_error_` | float | Out-of-bag misclassification rate. |
| `feature_importances_` | `(p,)` | Variable importance (when `importance != "none"`). |
| `var_vi_` | `(p,)` | Per-variable variance of the VI estimates (when `importance != "none"` and `var_mode="matched"`). |
| `obstrack_` | `(n, trees)` | Signed sampling-count matrix, with `trees = params_.ntrees` (when tracked). |
| `n_features_in_` | int | Number of predictors. |

## Examples

```python
import numpy as np
from rlt import RLT_cla

rng = np.random.default_rng(0)
X = rng.normal(size=(150, 5))
prob = 1 / (1 + np.exp(-(2 * X[:, 0] - 1)))
y = rng.binomial(1, prob)

model = RLT_cla(n_estimators=500, importance="permute", random_state=0)
model.fit(X, y)
model.predict_proba(X[:3])     # (3, 2)
model.predict(X[:3])           # e.g. array([1, 0, 1])

# String labels
y_str = np.array(["neg", "pos"])[y]
m2 = RLT_cla(n_estimators=100, random_state=0).fit(X, y_str)
m2.classes_                    # array(['neg', 'pos'])
```

## See also

- [Classification tutorial](../tutorials/classification-tutorial.md)
- [`RLT_reg`](rlt_reg.md) for the full parameter reference
