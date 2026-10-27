# RLT_surv

RLT survival forest, a sklearn-compatible estimator porting the RLT R
package's `RLT(model = "survival")` for right-censored time-to-event data.

`RLT_surv` grows a forest of survival trees and predicts individual
survival functions, cumulative hazards, and optional cumulative-hazard
covariances for simultaneous confidence bands.

```python
import numpy as np
from rlt import RLT_surv

rng = np.random.default_rng(42)
n = 200
X = rng.normal(size=(n, 5))
t_event = rng.exponential(1 / np.exp(X[:, 0]))
t_censor = rng.uniform(0, 3, size=n)

y = np.empty(n, dtype=[("event", "?"), ("time", float)])  # sksurv convention
y["event"] = t_event <= t_censor
y["time"] = np.minimum(t_event, t_censor)

model = RLT_surv(split_rule="logrank", random_state=42)
model.fit(X, y)
S = model.predict_survival_function(X)   # (n, T) survival curves
model.score(X, y)                        # fitted-data c-index, not validation
```

## Response format

`y` must be either

- a structured array with fields `event` (0/1 or bool; 1 = event observed)
  and `time` (the scikit-survival convention), or
- a `(time, event)` pair of arrays.

`fit()` validates the response and raises `ValueError` for NaN/Inf in
`time` (`"NA/Inf not permitted in y"`), NaN in `event`
(`"NA not permitted in censor"`), or event values outside {0, 1}
(`"censor must be 0 or 1"`). Note that NaN in `event` can only arrive
through a floating event array; a structured bool `event` field cannot
hold NaN. Both tuple components and structured fields must be vectors;
one-column inputs are accepted with a warning.

## Constructor

```python
RLT_surv(
    n_estimators=None,
    mtry=None,
    min_samples_leaf=5,
    nsplit=0,
    resample_replace=True,
    resample_prob=None,
    importance="none",
    resample_track=False,
    var_mode="none",
    split_rule="logrank",
    categorical_features=None,
    linear_comb=1,
    linear_comb_method="default",
    alpha=0,
    reinforcement=False,
    time_grid_size=0,
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

See [`RLT_reg`](rlt_reg.md) for shared parameters, with these
survival-specific choices:

| Parameter | Type / values | Default | Description |
|---|---|---|---|
| `split_rule` | "logrank", "suplogrank", "coxgrad", or codes 1, 2, 3 | "logrank" | Splitting criterion for survival trees. Integral numeric codes such as 2.0 are also accepted. |
| `time_grid_size` | int | 0 | Fitting retains every unique failure time. Zero or a size at least as large as the full grid is accepted; smaller positive values are rejected. Use `band_grid_size` when predicting on a reduced grid. |
| `linear_comb_method` | "default", "coxph", "naive" or code 1 to 2 | "default" | Local Cox coefficients (`"coxph"`, code 1, the default) or weighted survival score statistics (`"naive"`, code 2). Integral doubles are accepted. Unknown names/out-of-range codes warn and reset to `coxph`; fractional and nonfinite codes are rejected. |
| `alpha` | float | 0 | Clamped to `[0, 0.5]` at fit, like R. |
| `categorical_features` | bool mask / int indices / None | None | See [`RLT_reg`](rlt_reg.md); same behavior for all three estimators. |

`linear_comb=k` allows up to `k` variables in a split direction; some
splits use fewer. All three survival split rules support observation
weights. Weighted `logrank` and `suplogrank` account for weights in their
split-score variance calculation. `sample_weight` must be finite and
nonnegative with positive total mass, and does not change tree sampling.
Individual zero weights are allowed, but a sampled outer or embedded tree
with zero total weight raises an error.

`var_mode="matched"` uses paired half-samples without replacement. It
enforces `resample_prob=0.5` and `resample_replace=False`, reduces an odd
tree count to the preceding even number, and requires at least two trees.
The effective count is `model.params_.ntrees`. See [`RLT_reg`](rlt_reg.md)
for the shared variance-mode and observation-tracking settings.

## Methods

| Method | Returns | Description |
|---|---|---|
| `fit(X, y, sample_weight=None, var_prob=None, *, resample_preset=None)` | `self` | Grow the forest with optional observation weights, variable-sampling probabilities and signed sampling counts. Observation weights do not replace the chosen split rule. |
| `predict(X, band_grid_size=0)` | `(n, T)` ndarray | Default survival prediction: `S(t \| x)`. |
| `predict_survival_function(X, band_grid_size=0)` | `(n, T)` ndarray | Survival probabilities `S(t \| x)`. |
| `predict_cumulative_hazard_function(X, band_grid_size=0)` | `(n, T)` ndarray | Cumulative hazard `H(t \| x)`. |
| `predict_risk(X)` | `(n,)` ndarray | Risk score (cumulative hazard summed over the grid). |
| `predict_details(X, *, var_est=False, var_mode=None, keep_all=False, band_grid_size=0)` | dict | Hazard increments, CHF, survival, exact timepoints, and optional CHF covariance and per-tree hazards. |
| `predict_all_trees(X, band_grid_size=0)` | `(n, trees, times)` ndarray | Per-tree hazard increments. |
| `oob_details()` | dict | OOB hazards on the full training grid, with counts and availability mask. |
| `tree_summary(tree=0)` / `print_tree(tree=0)` | dict | Readable splits and terminal hazard vectors. |
| `predict_var(X, var_mode=None, band_grid_size=0)` | `(S, cov)` | Survival curves plus per-subject CHF covariance of shape `(n, T, T)`; requires a fitted variance mode or an explicit compatible prediction-time mode. |
| `importance_table()` | `ImportanceTable` | Variable-importance summary (Variable / VI, plus SD / Z / Sig when fitted with `importance != "none"` and `var_mode="matched"`); same as `rlt.importance(model)`. |
| `score(X, y)` | float | Harrell's c-index (higher is better). |
| `forest_kernel(X1, X2=None, vs_train=False, oob=False)` | integer ndarray, or dict with `oob=True` | Forest similarity kernel. |
| `get_one_tree(tree_id)` | dict | Copied raw arrays and nested node-hazard vectors of tree `tree_id` (zero-based). |

`band_grid_size` restricts returned predictions and covariance matrices to
a reduced empirical-quantile grid. Hazard increments are accumulated on
the full failure-time grid before selecting the requested times. Reducing
the output grid therefore does not change the cumulative hazard or
survival estimate at a retained time.

Ordinary forests return `exp(-CHF)` as survival. Linear combination
forests return the mean of tree product-limit survival curves, which can
differ from `exp(-CHF)`. In both cases `predict_var` returns CHF covariance,
not covariance of the returned survival estimate.

## Fitted attributes

| Attribute | Shape | Description |
|---|---|---|
| `forest_` | dict | The full fitted forest. |
| `timepoints_` | `(T,)` | Sorted unique failure-time grid. |
| `nfail_` | int | Number of failure times. |
| `oob_hazard_` | `(n, T)` | Out-of-bag hazard increments on the full training failure grid. |
| `oob_counts_` | `(n,)` | Number of eligible OOB predictions per row; use `oob_details()` to mask unavailable rows. |
| `oob_error_` | float | One minus concordance using the sum of OOB cumulative hazards over the full grid as risk, restricted to rows with OOB predictions. |
| `feature_importances_` | `(p,)` | Decrease in OOB concordance after perturbing each variable (when `importance != "none"`), rather than a decrease in the split statistic. |
| `var_vi_` | `(p,)` | Per-variable variance of the VI estimates (when `importance != "none"` and `var_mode="matched"`). |
| `obstrack_` | `(n, trees)` | Signed sampling-count matrix, with `trees = params_.ntrees` (when tracked). |
| `n_features_in_` | int | Number of predictors. |

## Examples

```python
import numpy as np
from rlt import RLT_surv, get_surv_band

rng = np.random.default_rng(0)
n, p = 200, 5
X = rng.normal(size=(n, p))
t_event = rng.exponential(1 / np.exp(X[:, 0]))
t_censor = rng.uniform(0, 3, size=n)

y = np.empty(n, dtype=[("event", "?"), ("time", float)])
y["event"] = t_event <= t_censor
y["time"] = np.minimum(t_event, t_censor)

# Variance-enabled forest and simultaneous bands
model = RLT_surv(n_estimators=1000, var_mode="matched",
                 resample_prob=0.5, resample_replace=False, random_state=0)
model.fit(X, y)

# Training rows illustrate returned arrays, not predictive validation.
prediction = model.predict_details(X[:2], var_est=True)
S, cov = prediction["Survival"], prediction["Cov"]
reference = np.exp(-prediction["CHF"])
band = get_surv_band(prediction, alpha=0.05, approach="smoothed")
band["lower"].shape, band["upper"].shape  # (2, T) each
```

The band uses `reference` and CHF covariance. Its nominal coverage is
specified separately for each target over the saved time grid, with no
joint guarantee across targets or coverage between grid points. It is
not a prediction interval for an event time. See
[`get_surv_band`](get_surv_band.md) for the LC survival distinction.

For LC tree inspection, `tree_summary()` gives readable split directions.
Raw `SplitVar` and `SplitLoad` arrays from `get_one_tree()` include every
node. Internal nodes satisfy `SplitVar[:, 0] >= 0`; use that mask for both
arrays when selecting internal-node coefficients.

## See also

- [Survival tutorial](../tutorials/survival-tutorial.md)
- [Survival confidence bands](../articles/confidence-interval.md)
- [`get_surv_band`](get_surv_band.md)
