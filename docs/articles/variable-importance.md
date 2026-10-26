# Variable Importance (Permutation and Distributed)

This page demonstrates how to obtain and visualize variable importance
(VI) from `rlt` — the Python port of the RLT R package's variable
importance article.

## Data

We generate continuous and categorical predictors with a continuous outcome.
Each forest declares the last five columns in `categorical_features`.

```python
import numpy as np

# (Optional) For reproducibility in this tutorial only.
rng = np.random.default_rng(1)

# ---- Generate a small synthetic dataset ----
trainn = 80
testn = 20
n = trainn + testn
p = 10

# Continuous + categorical predictors (last half as categorical)
X1 = rng.normal(size=(n, p // 2))
X2 = (rng.uniform(size=(n, p // 2)) * 3).astype(int)  # integers 0, 1, 2
X = np.hstack([X1, X2])

# Continuous outcome with a simple signal + noise
y = (
    1
    + X[:, 1:6].sum(axis=1)
    + 2 * np.isin(X[:, p // 2], [1, 2])
    + rng.normal(size=n)
)

# Train / test split
trainX, trainY = X[:trainn], y[:trainn]
testX, testY = X[trainn:], y[trainn:]
```

## Option A - Permutation Importance

Set `importance="permute"` to enable permutation-based VI during training.

```python
from rlt import RLT_reg

# Minimal, sensible defaults
ntrees = 200
nmin = 5
mtry = p // 2
samplereplace = True
sampleprob = 0.80
rule = "best"
nsplit = 0 if rule == "best" else 3

fit_perm = RLT_reg(
    n_estimators=ntrees,
    mtry=mtry,
    min_samples_leaf=nmin,
    resample_prob=sampleprob,
    resample_replace=samplereplace,
    nsplit=nsplit,
    importance="permute",  # permutation-based VI
    categorical_features=range(p // 2, p),
    n_jobs=1,
    random_state=1,
)
fit_perm.fit(trainX, trainY)

# VI vector lives in feature_importances_
np.round(fit_perm.feature_importances_, 3)
```

```
array([ 7.500e-02,  5.440e-01,  1.290e-01,  5.340e-01,  2.100e-02,
        3.994e+00, -5.400e-02,  3.000e-03, -1.600e-02,  4.100e-02])
```

```python
import matplotlib.pyplot as plt

# Simple visualization
fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(np.arange(p), fit_perm.feature_importances_)
ax.set_title("Permutation Variable Importance (rlt)")
ax.set_ylabel("Importance")
ax.set_xlabel("Predictors")
plt.show()
```

## Option B - Distributed Assignment Importance

This configuration evaluates importance on out-of-bag observations.
Use `importance="distribute"` for distributed assignment
importance. Unlike permutation importance, distributed importance works by
probabilistically routing OOB observations through the tree when a split
on the target variable is encountered. This requires sufficient OOB
samples per tree — avoid very high `resample_prob` with
`resample_replace=False`, which leaves too few OOB observations.

```python
fit_dist = RLT_reg(
    n_estimators=ntrees,
    mtry=mtry,
    min_samples_leaf=nmin,
    nsplit=nsplit,
    resample_prob=0.632,      # ~63.2% in-bag, ~36.8% OOB
    resample_replace=False,   # without replacement
    importance="distribute",  # distributed assignment VI
    categorical_features=range(p // 2, p),
    n_jobs=1,
    random_state=1,
)
fit_dist.fit(trainX, trainY)

# VI vector lives in feature_importances_
np.round(fit_dist.feature_importances_, 3)
```

```
array([-5.700e-02,  6.810e-01,  8.200e-02,  5.580e-01, -4.200e-02,
        3.949e+00, -1.000e-03, -8.500e-02, -3.800e-02,  2.100e-02])
```

```python
fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(np.arange(p), fit_dist.feature_importances_)
ax.set_title("Distributed Variable Importance (rlt)")
ax.set_ylabel("Importance")
ax.set_xlabel("Predictors")
plt.show()
```

The simulated signal uses continuous predictors at indices 1 through 4
and the categorical predictor at index 5. The other predictors are noise;
their estimated importance can still vary above or below zero.

## Variance of variable importance (matched forests)

VI estimates are themselves noisy. Fitting with `var_mode="matched"`
(mixed with either importance option) makes the forest also return the
per-variable variance of the VI estimates, so you can judge which
importances are actually distinguishable from noise — the Python port of
R's `importance(fit)` output.

```python
fit_vi = RLT_reg(
    n_estimators=ntrees,
    mtry=mtry,
    min_samples_leaf=nmin,
    importance="distribute",
    var_mode="matched",   # per-variable variance of VI
    categorical_features=range(p // 2, p),
    n_jobs=1,
    random_state=1,
)
fit_vi.fit(trainX, trainY)

# (p,) variance of each VI estimate
np.round(fit_vi.var_vi_, 4)
```

```
array([ 5.00e-04,  1.07e-02,  1.39e-02,  3.79e-02, -2.72e-02,  5.44e-01,
       -5.50e-03,  3.80e-03, -1.00e-04,  1.33e-02])
```

The friendlier view is `importance_table()`, which prints `Variable`,
`VI`, and — when matched variance is available — the standard deviation
`SD = sqrt(VarVI)`, the standardized statistic `Z = VI / SD`, and a
significance code (`***` for |Z| ≥ 2.58, `**` for ≥ 1.96, `*` for
≥ 1.64). The layout mirrors R's `print.importance.RLT`:

```python
fit_vi.importance_table()
```

```
Variable             VI           SD          Z  Sig
----------------------------------------------------------
V1              -0.0209     0.021899      -0.95
V2               0.5996     0.103274       5.81  ***
V3               0.0609     0.117999       0.52
V4               0.5832     0.194568       3.00  ***
V5              -0.0230           NA         NA
V6               3.8730     0.737591       5.25  ***
V7              -0.0231           NA         NA
V8              -0.0973     0.061325      -1.59
V9              -0.0141           NA         NA
V10             -0.0504     0.115454      -0.44

Note: 3 variable(s) with negative variance estimate (SD, Z shown as NA)
```

Two things to note in this output:

- **Negative variance estimates.** The matched U-statistic variance can
  come out negative in a finite forest. Exactly like R, such variables
  get `NA` for `SD` and `Z` and an empty significance code, and the
  table appends a trailing `Note` counting them. A negative variance
  estimate does not by itself show that a variable is noise.
- **Without matched variance**, the table simply has no `SD` / `Z` /
  `Sig` columns — `importance_table()` works for any
  `importance != "none"` fit and prints just `Variable` and `VI`.

The same table is available through the module-level function
`rlt.importance(model)`, mirroring R's `importance(fit)` idiom (the
estimator attribute `.importance` holds the constructor parameter
string, so the method could not take that name):

```python
import rlt

rlt.importance(fit_vi)   # same table as fit_vi.importance_table()
```

!!! note
    `var_mode="matched"` implies the matched-forest preset: subsampling
    without replacement at 50% and an even tree count. When importance is
    omitted, distributed importance is enabled automatically. Both
    `importance="permute"` and `importance="distribute"` can be requested
    explicitly; the selected importance measure is retained.
