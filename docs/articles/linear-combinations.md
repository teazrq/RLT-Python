# RLT with Linear Combinations

Linear-combination splits allow each internal node to split on a weighted
combination of several variables rather than a single predictor. This can
help when the signal is carried by a direction such as `X1 + X2` instead of
one coordinate at a time.

The example below uses regression. The same idea can also be applied to
classification and survival forests by using `RLT_cla` or `RLT_surv` and
choosing a model-appropriate `linear_comb_method`.

## Regression Example

```python
import numpy as np

rng = np.random.default_rng(3)

n = 160
p = 6
X = rng.normal(size=(n, p))

y = 1 + X[:, 0] + X[:, 1] + 0.5 * X[:, 2] + rng.normal(size=n, scale=0.5)

train_id = np.arange(120)
trainX, trainY = X[:120], y[:120]
testX, testY = X[120:], y[120:]
```

Fit a regression forest that considers linear combinations of up to three
variables at a split. Here `linear_comb_method="sir"` selects sliced inverse
regression (SIR), which uses the response to estimate a predictor direction.
The tree then searches for a cutpoint along that direction. With `nsplit=3`,
it compares three random candidate cutpoints; `nsplit=0` would search all
available cutpoints.

```python
from rlt import RLT_reg

fit_lc = RLT_reg(
    n_estimators=200,
    min_samples_leaf=5,
    mtry=3,
    nsplit=3,
    linear_comb=3,
    linear_comb_method="sir",
    n_jobs=1,
    random_state=3,
)
fit_lc.fit(trainX, trainY)

print(fit_lc)
```

```
RLT_reg(linear_comb=3, linear_comb_method='sir', mtry=3, n_estimators=200,
        n_jobs=1, nsplit=3, random_state=3)
```

Linear combinations do not require reinforcement. This fit uses the default
`reinforcement=False`.

## Prediction

```python
test_pred = fit_lc.predict(testX)

print(f"Test MSE: {np.mean((test_pred - testY) ** 2):.4f}")
print(f"Maximum combination size: {fit_lc.linear_comb}")
```

```
Test MSE: 0.2817
Maximum combination size: 3
```

The held-out mean squared error summarizes prediction error on the 40
observations excluded from fitting. It describes this simulated dataset and
fit. The stored combination setting confirms the maximum of three variables,
not the number used at every split.

## Method names and integer codes

`linear_comb_method` also accepts the integer method codes used by the R
package's core, which is useful when translating R code or tuning over methods
programmatically. The names and codes are equivalent:

| Model | Names | Codes |
|---|---|---|
| `RLT_reg` | `naive`, `lm`, `pca`, `sir` | 1, 2, 3, 4 |
| `RLT_cla` | `lda`, `naive`, `random`, `logistic` | 1, 2, 3, 4 |
| `RLT_surv` | `coxph`, `naive` | 1, 2 |

```python
fit_code = RLT_reg(
    n_estimators=200,
    min_samples_leaf=5,
    mtry=3,
    nsplit=3,
    linear_comb=3,
    linear_comb_method=4,  # integer code for "sir"
    n_jobs=1,
    random_state=3,
)
fit_code.fit(trainX, trainY)

# identical forest to linear_comb_method="sir"
np.array_equal(fit_code.predict(testX), fit_lc.predict(testX))  # True
```

Like R, an unrecognized method name (or an out-of-range integer) does
not error: it emits a `UserWarning` and resets to code 1, which is `naive` for
regression, `lda` for classification, `coxph` for survival:

```python
fit_bad = RLT_reg(
    n_estimators=200,
    mtry=3,
    nsplit=3,
    linear_comb=3,
    linear_comb_method="ridge",  # not a valid method name
    n_jobs=1,
    random_state=3,
)
fit_bad.fit(trainX, trainY)
# UserWarning: linear_comb_method not recognized. Use 'naive', 'lm',
#              'pca', or 'sir'. Resetting to naive

fit_bad.params_.linear_comb_method  # 1
```

## Other Model Types

For classification, use `RLT_cla` with methods such as `"lda"`, `"naive"`,
`"random"`, or `"logistic"`.

For survival, use `RLT_surv` with methods such as `"coxph"` or `"naive"`.
