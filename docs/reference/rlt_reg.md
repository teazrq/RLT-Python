# RLT_reg

`RLT_reg` fits regression forests with axis-aligned or linear-combination
splits. Embedded forests can rank predictors for reinforcement and
linear-combination splitting. The estimator follows the sklearn interface.

```python
import numpy as np
from rlt import RLT_reg

rng = np.random.default_rng(42)
X = rng.normal(size=(100, 5))
y = X[:, 0] + X[:, 1] + rng.normal(size=100)

model = RLT_reg(n_estimators=500, random_state=42)
model.fit(X, y)
pred = model.predict(X)
```

## Constructor

```python
RLT_reg(
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

| Parameter | Type / values | Default | Description |
|---|---|---|---|
| `n_estimators` | int or None | None | Effective count is 100 with reinforcement, 500 otherwise, matching R. |
| `mtry` | int or None | None | Variables considered per split; None → max(1, p // 2). |
| `min_samples_leaf` | int | 5 | Node-size stopping threshold (`nmin`): nodes with at most this many observations are not split; child nodes may be smaller. |
| `nsplit` | int | 0 | Random cut points per variable; 0 = consider all (best split). |
| `resample_replace` | bool | True | Resample with replacement (bootstrap). |
| `resample_prob` | float or None | None | Resampling proportion; None → 1.0 if replace else 0.8. |
| `importance` | "none", "permute", "distribute" | "none" | Variable-importance strategy. |
| `resample_track` | bool | False | Retain sampling counts for training/OOB kernels and prediction-time IJ/jack. Active variance modes also retain counts. |
| `var_mode` | "none", "matched", "ij", "jack" | "none" | Variance-estimation mode. |
| `categorical_features` | bool mask (length p), int indices, or None | None | Nominal-column override. None infers unordered pandas categoricals. Ordered categoricals and string columns retain rank dictionaries. Explicit numeric nominal columns use zero-based integer codes `0..k-1`; nominal predictors support at most 53 levels. |
| `linear_comb` | int | 1 | Maximum predictors per combination, capped at `p`; 1 = axis-aligned. A combination split may use fewer predictors. Values above 1 select combination splitting independently of `reinforcement`. |
| `linear_comb_method` | "default", "sir", "naive", "lm", "pca" or code 1 to 4 | "default" | How combination loadings are computed ("sir" = code 4 is the regression default). Integral doubles are accepted. Unknown names/out-of-range codes warn and reset to `naive`; fractional and nonfinite codes are rejected. |
| `alpha` | float | 0 | Minimum child proportion used in split searches, clamped to `[0, 0.5]` at fit. |
| `reinforcement` | bool | False | Use embedded importance to choose axis-aligned split variables. See the interaction with `linear_comb` below. |
| `n_jobs` | int | -1 | OpenMP threads (-1 or 0 = all cores). |
| `verbose` | int | 0 | Verbosity level. |
| `random_state` | int, Generator, RandomState, or None | None | Explicit native master seed; None draws from NumPy's global RNG. Supplied RNG objects advance. Repeat with the same data, effective parameters, and numerical environment. |

### Variance settings

For all three estimators, fitting with `var_mode` adjusts the sampling settings
after parameter defaults are resolved:

- `"matched"` uses paired half-samples without replacement, setting
  `resample_prob=0.5` and `resample_replace=False`. An odd tree count is reduced
  to the preceding even count; at least two trees are required.
- `"ij"` and `"jack"` set `resample_replace=True` and keep the resolved
  `resample_prob`. These modes estimate prediction variance, not variable
  importance variance.

All active variance modes retain sampling counts and change `importance="none"`
to `"distribute"`. An explicitly selected `"permute"` or `"distribute"` method
is retained. These modes supply their own sampling scheme and ignore a supplied
`resample_preset`. Inspect `model.params_` after fitting for the effective
settings.

### Embedded-model controls

Embedded forests rank predictors within the current node. With
`linear_comb=1`, `reinforcement=True` selects this ranking for split-variable
selection. With `linear_comb>1`, regression and classification use embedded
ranking in nodes with at least `3 * embed_nmin` observations, even when
`reinforcement=False`; smaller nodes use direct screening. Survival
combination splits screen `mtry` candidates without an embedded forest.
Setting an `embed_*` parameter alone does not enable either splitting mode.

The following controls are shared by the estimators. Here `n` is the total
training sample size and `p` is the number of predictors. Proportions for
embedded sampling and candidate selection refer to the current node and
its available predictors.

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `embed_ntrees` | int | 50 | Number of trees in each embedded forest, with a minimum of 1. |
| `embed_mtry` | float | 0.5 | Candidates per embedded split: a proportion of available predictors below 1, or a count at least 1. The resulting count is truncated to an integer. Values are clamped to `[0, p]`. |
| `embed_nmin` | int | 5 | Embedded-tree node-size stopping threshold, with a minimum of 1. Nodes with at most this many observations stop; child nodes can be smaller. Also sets the `3 * embed_nmin` switch described above. |
| `embed_nsplit` | int | 3 | Random cut points per candidate in the embedded forest; 0 considers all cut points. Values are clamped to be nonnegative. |
| `embed_resample_replace` | bool | True | Whether each embedded tree samples with replacement. |
| `embed_resample_prob` | float | 0.9 | Fraction of the current node's observations sampled for each embedded tree, clamped to `[0, 0.99]`. |
| `embed_mute` | float | 0 | Requested removal of low-ranked predictors from descendant candidate sets: a proportion below 1, or a count at least 1. Zero disables muting; protected predictors are retained, so fewer may be removed. Values are clamped to `[0, p]`. |
| `embed_protect` | int or None | None | Maximum number of top-ranked predictors added to the protected set for axis-aligned reinforcement or regression combinations, subject to `embed_threshold`. None uses `ceil(log(n))`, with the natural logarithm. Explicit values are clamped to `[0, p]`; a positive value is required for reinforcement or combination splits. Classification combinations instead protect their threshold-qualified candidate pool. |
| `embed_threshold` | float | 0.25 | Importance cutoff relative to the best embedded importance, clamped to `[0, 1]`. It controls entry into the protected set and also screens classification combination candidates in large nodes. It does not filter regression combination candidates. |

## Methods

| Method | Returns | Description |
|---|---|---|
| `fit(X, y, sample_weight=None, var_prob=None, *, resample_preset=None)` | `self` | Grow the forest with observation weights, variable-sampling probabilities, or signed sampling counts. |
| `predict(X)` | `(n,)` ndarray | Regression predictions. |
| `predict_var(X, var_mode=None, ...)` | `(pred, var)` | Predictions plus variance; requires `var_mode != "none"` at fit (or explicit compatible mode). Negative variance estimates are set to NaN. |
| `predict_all_trees(X)` | `(n, trees)` ndarray | Per-tree predictions; `trees = params_.ntrees`. |
| `predict_details(X, *, var_est=False, var_mode=None, keep_all=False)` | dict | Complete predictions, optional variance and per-tree results, and effective uncertainty mode. |
| `oob_details()` | dict | OOB predictions, counts, and availability mask. |
| `tree_summary(tree=0)` / `print_tree(tree=0)` | dict | Readable node depths, named splits, weights, and terminal predictions. |
| `importance_table()` | `ImportanceTable` | Variable-importance summary (Variable / VI, plus SD / Z / Sig when fitted with `importance != "none"` and `var_mode="matched"`); same as `rlt.importance(model)`. |
| `forest_kernel(X1, X2=None, vs_train=False, oob=False)` | integer ndarray, or dict with `oob=True` | Leaf-sharing counts; `vs_train=True` includes training in-bag multiplicities, while OOB mode returns pairwise normalized values and eligibility counts. Training modes require original training rows in their original order. See [Forest Kernel](../articles/feature-kernel.md). |
| `get_one_tree(tree_id)` | dict of ndarrays | Raw arrays of tree `tree_id` (0-indexed). |
| `score(X, y)` | float | Returns R² (sklearn convention). |

## Fitted attributes

| Attribute | Shape | Description |
|---|---|---|
| `forest_` | dict | The full fitted forest (split variables/values, loadings, node weights/averages), one array per tree. |
| `oob_prediction_` | `(n,)` | Out-of-bag training predictions. |
| `oob_counts_` | `(n,)` | Number of OOB predictions per row; use `oob_details()` to mask unavailable rows. |
| `oob_error_` | float | Out-of-bag MSE. |
| `feature_importances_` | `(p,)` | Variable importance (when `importance != "none"`). |
| `var_vi_` | `(p,)` | Per-variable variance of the VI estimates (when `importance != "none"` and `var_mode="matched"`). |
| `obstrack_` | `(n, trees)` | Signed sampling-count matrix, with `trees = params_.ntrees` (when tracked). |
| `ncat_` | `(p,)` | Per-variable category count (1 = continuous). |
| `n_features_in_` | int | Number of predictors. |

OOB attributes are absent when fitting disables OOB computation. Use
`oob_details()` to mask rows without eligible trees before interpreting
stored predictions or computing an error.

## Examples

```python
import numpy as np
from rlt import RLT_reg

rng = np.random.default_rng(0)
X = rng.normal(size=(100, 5))
y = X[:, 0] + X[:, 1] + rng.normal(size=100)

# Plain forest
m = RLT_reg(n_estimators=500, random_state=0).fit(X, y)
m.oob_error_

# Reinforcement learning trees with combinations of up to 2 variables
m2 = RLT_reg(
    n_estimators=100,
    linear_comb=2,
    linear_comb_method="sir",
    reinforcement=True,
    random_state=0,
).fit(X, y)
m2.oob_error_
```

## See also

- [Regression tutorial](../tutorials/regression-tutorial.md)
- [Linear combinations](../articles/linear-combinations.md)
- Zhu, R., Zeng, D., & Kosorok, M. R. (2015). Reinforcement Learning Trees. JASA.
