# RLT Regression Tutorial

This page shows how to fit and predict a regression model with `rlt` — the
Python port of the RLT R package's regression vignette.

## Prerequisites

Install `rlt` (`pip install rlt-forests`), then import it:

```python
from rlt import RLT_reg
```

## Data

We generate continuous and categorical predictors with a continuous outcome.

```python
import numpy as np

# (Optional) For reproducibility in this tutorial only.
# Detailed notes on random seeds are in the Seed & Reproducibility article.
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

## Fit

Declare the last five columns in `categorical_features` to use categorical splits.

```python
from rlt import RLT_reg

ntrees = 200
nmin = 5
mtry = p // 2
samplereplace = True
sampleprob = 0.80
rule = "best"
nsplit = 0 if rule == "best" else 3  # 0 = consider all cut points
importance = "permute"

fit = RLT_reg(
    n_estimators=ntrees,
    mtry=mtry,
    min_samples_leaf=nmin,
    resample_prob=sampleprob,
    resample_replace=samplereplace,
    nsplit=nsplit,
    importance=importance,
    categorical_features=range(p // 2, p),
    n_jobs=1,
    random_state=1,
)
fit.fit(trainX, trainY)
```

## Predict

```python
pred = fit.predict(testX)

# OOB (out-of-bag) predictions and error are computed during the fit
oob_pred = fit.oob_prediction_
test_pred = pred
```

## Evaluate

```python
mse_oob = float(np.mean((oob_pred - trainY) ** 2))
mse_test = float(np.mean((test_pred - testY) ** 2))

# A compact summary
print(f"OOB MSE:  {mse_oob:.4f}")
print(f"Test MSE:  {mse_test:.4f}")
```

```
OOB MSE:  2.4167
Test MSE:  4.6647
```

Each OOB prediction uses only trees that omitted that training observation.
The test MSE evaluates the separate held-out observations.

## Inspect

```python
print(fit)
```

```
RLT_reg(categorical_features=range(5, 10), importance='permute', mtry=5,
        n_estimators=200, n_jobs=1, random_state=1, resample_prob=0.8)
```

Fitted attributes worth knowing:

```python
fit.n_features_in_          # number of predictors
fit.oob_error_              # out-of-bag MSE (also = MSE of oob_prediction_)
fit.oob_prediction_         # (n,) OOB training predictions
fit.feature_importances_    # (p,) variable importance
fit.get_one_tree(0)         # raw arrays of tree 0 — see Single Tree article
```
