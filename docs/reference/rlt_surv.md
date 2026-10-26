# RLT_surv

RLT survival forest — sklearn-compatible estimator porting the RLT R
package's `RLT(model = "survival")` for right-censored time-to-event data.

`RLT_surv` grows a forest of survival trees and predicts individual
survival functions, cumulative hazards, and (with `var_mode`) their
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
model.score(X, y)                        # Harrell's c-index
```

## Response format

`y` must be either

- a structured array with fields `event` (0/1 or bool; 1 = event observed)
  and `time` — the scikit-survival convention, or
- a `(time, event)` pair of arrays.

`fit()` validates the response and raises `ValueError` for NaN/Inf in
`time` (`"NA/Inf not permitted in y"`), NaN in `event`
(`"NA not permitted in censor"`), or event values outside {0, 1}
(`"censor must be 0 or 1"`). Note that NaN in `event` can only arrive
via the tuple path — a structured bool `event` field cannot hold NaN.

## Constructor

```python
RLT_surv(
    n_estimators=500,
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

All parameters match [`RLT_reg`](rlt_reg.md) — see that page for the full
table — with these survival-specific additions:

| Parameter | Type / values | Default | Description |
|---|---|---|---|
| `split_rule` | "logrank", "suplogrank", "coxgrad" | "logrank" | Splitting criterion for survival trees. |
| `time_grid_size` | int | 0 | Fitting retains every unique failure time. Zero or a size at least as large as the full grid is accepted; smaller positive values are rejected. Use `band_grid_size` when predicting on a reduced grid. |
| `linear_comb_method` | "default", "coxph", "naive" or int code 1–2 | "default" | How combination loadings are computed ("coxph" = code 1 is the survival default). Unrecognized names or codes emit a `UserWarning` and reset to code 1 (`coxph`). |
| `alpha` | float | 0 | Clamped to `[0, 0.5]` at fit, like R. |
| `categorical_features` | bool mask / int indices / None | None | See [`RLT_reg`](rlt_reg.md); same behavior for all three estimators. |

## Methods

| Method | Returns | Description |
|---|---|---|
| `fit(X, y, sample_weight=None, var_prob=None)` | `self` | Grow the forest. `sample_weight` is used by the `coxgrad` split rule. |
| `predict(X, band_grid_size=0)` | `(n, T)` ndarray | Default survival prediction: `S(t \| x)`. |
| `predict_survival_function(X, band_grid_size=0)` | `(n, T)` ndarray | Survival probabilities `S(t \| x)`. |
| `predict_cumulative_hazard_function(X, band_grid_size=0)` | `(n, T)` ndarray | Cumulative hazard `H(t \| x)`. |
| `predict_risk(X)` | `(n,)` ndarray | Risk score (cumulative hazard summed over the grid). |
| `predict_var(X, var_mode=None, band_grid_size=0)` | `(S, cov)` | Survival curves plus per-subject covariance of the CHF, `cov` shape `(n, T, T)`; requires `var_mode != "none"` at fit. |
| `importance_table()` | `ImportanceTable` | Variable-importance summary (Variable / VI, plus SD / Z / Sig when fitted with `importance != "none"` and `var_mode="matched"`); same as `rlt.importance(model)`. |
| `score(X, y)` | float | Harrell's c-index (higher is better). |
| `forest_kernel(X1, X2=None, vs_train=False)` | integer ndarray | Forest similarity kernel. |
| `get_one_tree(tree_id)` | dict of ndarrays | Raw arrays of tree `tree_id` (0-indexed). |

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
| `oob_hazard_` | `(n, T)` | Out-of-bag predicted cumulative hazard. |
| `oob_error_` | float | Out-of-bag error (1 − c-index style ranking error used by the core). |
| `feature_importances_` | `(p,)` | Variable importance (when `importance != "none"`). |
| `var_vi_` | `(p,)` | Per-variable variance of the VI estimates (when `importance != "none"` and `var_mode="matched"`). |
| `obstrack_` | `(n, n_estimators)` | Inbag-count matrix (when tracked). |
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
model = RLT_surv(n_estimators=1000, var_mode="matched", random_state=0)
model.fit(X, y)
S, cov = model.predict_var(X[:2])
band = get_surv_band(model, X[:2], alpha=0.05, approach="smoothed")
band["lower"].shape, band["upper"].shape  # (2, T) each
```

## See also

- [Survival tutorial](../tutorials/survival-tutorial.md)
- [Survival confidence bands](../articles/confidence-interval.md)
- [`get_surv_band`](get_surv_band.md)
