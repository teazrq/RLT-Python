# RLT Survival Analysis Tutorial

This vignette introduces survival analysis with **rlt** (Reinforcement
Learning Trees). RLT survival forests estimate individual survival, hazard,
and cumulative hazard functions via ensemble tree methods. Key features
include:

- **Three split rules**: logrank (default), suplogrank, and coxgrad.
- **Linear combination (LC) splits**: combine multiple variables into a single split direction.
- **Variance estimation**: matched-sample U-statistic, infinitesimal jackknife (IJ), and jackknife.
- **Confidence bands**: `get_surv_band()` provides naive or smoothed simultaneous bands for survival curves.
- **Tree inspection**: `get_one_tree()` inspects individual tree structures.

The examples below use small simulated datasets so that all code runs quickly.

## Simulated data

We simulate data from a proportional hazards model with exponential event
times. The first two predictors carry signal; the rest are noise. About 30%
of observations are censored.

```python
import numpy as np
from rlt import RLT_surv

rng = np.random.default_rng(42)
n = 200
p = 5
X = rng.normal(size=(n, p))

beta = np.array([0.8, 0.5, 0, 0, 0])
hazard = np.exp(X @ beta)
surv_time = rng.exponential(1.0 / hazard)
censor_time = rng.uniform(0, 3, size=n)

time = np.minimum(surv_time, censor_time)
event = (surv_time <= censor_time).astype(int)

np.bincount(event)  # 0 = censored, 1 = observed event
```

Responses follow the scikit-survival convention — a structured array with
`event` and `time` fields (`1` = event observed, `0` = censored):

```python
y = np.empty(n, dtype=[("event", "?"), ("time", "<f8")])
y["event"] = event.astype(bool)
y["time"] = time
```

### Response validation

`fit()` validates the response before growing the forest and fails fast
with a clear message. Missing or invalid values raise a `ValueError`:

```python
# NaN/Inf in `time` → "NA/Inf not permitted in y"
y_bad_time = y.copy()
y_bad_time["time"][0] = np.nan

# NaN in `event` → "NA not permitted in censor"
# (only via the (time, event) tuple path — the structured bool
#  field cannot hold NaN)
event_nan = np.concatenate([[np.nan], np.ones(n - 1)])

# event values outside {0, 1} → "censor must be 0 or 1"
event_bad = np.full(n, 2.0)

for y_try in (y_bad_time, (y["time"], event_nan), (y["time"], event_bad)):
    try:
        RLT_surv(n_estimators=20, n_jobs=1, random_state=0).fit(X, y_try)
    except ValueError as err:
        print("ValueError:", err)
```

```
ValueError: NA/Inf not permitted in y
ValueError: NA not permitted in censor
ValueError: censor must be 0 or 1
```

Clean your data first — impute, drop, or re-encode before fitting.

## Basic usage

Fit a survival forest with `RLT_surv`. By default `split_rule="logrank"`.

```python
fit = RLT_surv(
    n_estimators=100,
    min_samples_leaf=5,
    n_jobs=1,
    random_state=42,
)
fit.fit(X, y)
```

Predict on new data (or the training data) to obtain survival curves,
hazards, and cumulative hazards:

```python
S = fit.predict_survival_function(X[:5])        # S(t), shape (5, T)
chf = fit.predict_cumulative_hazard_function(X[:5])  # H(t), shape (5, T)

# T is the number of unique failure times
fit.timepoints_.shape
```

Each component is an `n x T` matrix, where `T` is the number of unique
failure times (`fit.timepoints_`). The hazard increment `h(t)` can be
recovered from the cumulative hazard:

```python
hazard = np.diff(chf, prepend=0.0, axis=1)  # h(t) at each failure time
```

Plot the predicted survival curve for the first subject:

```python
import matplotlib.pyplot as plt

plt.step(fit.timepoints_, S[0], where="post")
plt.xlabel("Time")
plt.ylabel("Survival Probability")
plt.title("Predicted Survival Curve (Subject 1)")
plt.show()
```

## Split rules

`rlt` provides three splitting criteria for survival trees:

| Rule | How it scores a split |
|------|----------------------|
| `logrank` | Standard log-rank test statistic (default) |
| `suplogrank` | Maximum of the standardized log-rank process over time |
| `coxgrad` | Gradient of Cox partial likelihood; supports observation weights |

Fit the three rules on the same data and compare out-of-bag error
estimates. The error is `1 - C`, using a risk score that sums the OOB
cumulative hazards over the full failure-time grid. Lower values are
better; this sum is not a time integral.

```python
fit_lr = RLT_surv(n_estimators=100, split_rule="logrank",
                  n_jobs=1, random_state=42)
fit_slr = RLT_surv(n_estimators=100, split_rule="suplogrank",
                   n_jobs=1, random_state=42)
fit_cg = RLT_surv(n_estimators=100, split_rule="coxgrad",
                  n_jobs=1, random_state=42)

fit_lr.fit(X, y); fit_slr.fit(X, y); fit_cg.fit(X, y)

print(f"logrank:     {fit_lr.oob_error_:.5f}")
print(f"suplogrank:  {fit_slr.oob_error_:.5f}")
print(f"coxgrad:     {fit_cg.oob_error_:.5f}")
```

```
logrank:     0.34945
suplogrank:  0.35366
coxgrad:     0.40004
```

These values describe this simulated dataset and do not establish a
general ranking of the split rules. `logrank` is the default; use
`coxgrad` when supplying observation weights as below.

## Observation weights

Observation weights are passed via the `sample_weight` argument of `fit`.
For survival forests, weights are **not** used by `logrank` or
`suplogrank` (due to the difficulty of weighted variance estimation for the
test statistic), but they **are** used by `coxgrad`.

```python
w = rng.uniform(size=n)
fit_w = RLT_surv(n_estimators=100, split_rule="coxgrad",
                 n_jobs=1, random_state=1)
fit_w.fit(X, y, sample_weight=w)
```

## Linear combination splits

When `linear_comb > 1`, each split uses a linear combination of
`linear_comb` variables instead of a single variable. For survival
forests, the available methods are:

- `"coxph"` (default): coefficients from a local Cox model fit.
- `"naive"`: simple correlation-based direction.

```python
fit_lc = RLT_surv(
    n_estimators=100,
    split_rule="logrank",
    linear_comb=3,
    linear_comb_method="coxph",
    n_jobs=1,
    random_state=1,
)
fit_lc.fit(X, y)
```

Predictions from LC forests have the same structure as standard forests:

```python
S_lc = fit_lc.predict_survival_function(X[:5])
S_lc.shape  # (5, T)
```

## Variable importance

Set `importance="permute"` to compute variable importance. The importance
measure for survival forests is based on the decrease in the splitting
criterion (logrank, suplogrank, or coxgrad).

```python
fit_imp = RLT_surv(n_estimators=100, importance="permute",
                   n_jobs=1, random_state=1)
fit_imp.fit(X, y)

np.round(fit_imp.feature_importances_, 3)
```

```
array([ 0.05 ,  0.036, -0.004, -0.003,  0.004])
```

The first two variables carry the signal and clearly stand out.

## Variance estimation and confidence bands

`rlt` supports three variance estimation strategies for survival
predictions:

- **`"matched"`**: matched-sample U-statistic decomposition. Requires an even number of trees and subsampling without replacement at 50% (automatically adjusted).
- **`"ij"`**: infinitesimal jackknife.
- **`"jack"`**: jackknife variance.

Enable variance estimation during fitting via `var_mode`, then request
covariance matrices at prediction time with `predict_var()`:

```python
fit_var = RLT_surv(n_estimators=1000, var_mode="matched",
                   n_jobs=1, random_state=42)
fit_var.fit(X, y)

# Predict with variance estimation
Sv, cov = fit_var.predict_var(X[:3])

# cov is an (n, T, T) array: covariance of the cumulative hazard over time
print(Sv.shape, cov.shape)  # (3, 131) (3, 131, 131)
```

The marginal variances and critical values needed for bands are computed
internally by `get_surv_band()` (next section). Reliable variance
estimation typically requires many trees (e.g., 1,000+).

## Confidence bands with `get_surv_band()`

Given a variance-enabled fitted model, `get_surv_band()` computes
simultaneous confidence bands for the survival function. Two approaches
are available:

- **`"naive"`**: uses the full covariance matrix with a Monte Carlo critical value.
- **`"smoothed"`**: GAM-smoothed low-rank covariance plus eigenvalue-ratio weighted residual correction.

```python
from rlt import get_surv_band

# Naive band for the first test subject
band_naive = get_surv_band(fit_var, X, i=1, alpha=0.05,
                           approach="naive", nsim=5000)

# Smoothed band
band_smooth = get_surv_band(fit_var, X, i=1, alpha=0.05,
                            approach="smoothed", nsim=5000, k_rank=10)
```

Plot survival curve with the naive band:

```python
t = band_naive["timepoints"]
S1 = fit_var.predict_survival_function(X[:1])[0]

plt.step(t, S1, where="post", color="black", label="Estimate")
plt.step(t, band_naive["lower"][0], where="post", color="blue",
         linestyle="--", label="95% Band")
plt.step(t, band_naive["upper"][0], where="post", color="blue",
         linestyle="--")
plt.xlabel("Time"); plt.ylabel("Survival")
plt.title("Survival Curve with 95% Confidence Band")
plt.legend(loc="upper right")
plt.show()
```

You can also request all subjects at once with `i=0` (the default).

## Reducing the time grid for bands

For large datasets, the full set of failure times can make covariance
matrices unwieldy. Use `band_grid_size` in `predict_var()` (or pass it to
`get_surv_band()`) to evaluate variance on a reduced quantile-based grid:

```python
S3, cov3 = fit_var.predict_var(X[:3], band_grid_size=50)
S3.shape  # (3, 50) — at most 50 time points
```

## Inspecting individual trees

Use `get_one_tree()` to inspect the structure of any tree in the fitted
forest. It returns the raw arrays of the tree (0-indexed):

```python
tree = fit.get_one_tree(0)
tree.keys()
```

```
dict_keys(['SplitVar', 'SplitValue', 'LeftNode', 'RightNode', 'NodeWeight'])
```

```python
n_nodes = len(tree["SplitVar"])
n_splits = int((tree["SplitVar"] >= 0).sum())
print(f"{n_nodes} nodes, {n_splits} splits")
```

```
121 nodes, 60 splits
```

A node `i` is internal when `SplitVar[i] >= 0`: the split sends points with
`X[SplitVar[i]] <= SplitValue[i]` to `LeftNode[i]` and the rest to
`RightNode[i]`. Terminal nodes have `SplitVar[i] == -1`.

For LC forests, `get_one_tree()` also gives access to the linear
combination loadings, which live in `fit.forest_["SplitLoad"]` (one row of
`linear_comb` coefficients per internal node, aligned with the internal
`SplitVar` entries — see the
[Single Tree](../articles/feature-single-tree.md) article):

```python
loadings = np.asarray(fit_lc.forest_["SplitLoad"][0])
loadings.shape  # (n_internal_nodes, linear_comb)
```

## Summary

- Fit a survival forest with `RLT_surv().fit(X, y)`.
- Predict with `predict_survival_function()` / `predict_cumulative_hazard_function()` to obtain `S(t)` and `H(t)`.
- Choose `split_rule` among `"logrank"`, `"suplogrank"`, and `"coxgrad"`.
- Use `sample_weight` with `split_rule="coxgrad"` for weighted splits.
- Enable LC splits via `linear_comb=k, linear_comb_method="coxph"`.
- Request variable importance with `importance="permute"` and inspect via `feature_importances_`.
- Estimate prediction variance with `var_mode="matched"` / `"ij"` / `"jack"`, then call `predict_var()`.
- Build confidence bands with `get_surv_band(model, X, approach="naive" | "smoothed")`.
- Inspect trees with `get_one_tree(i)`.
