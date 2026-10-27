# RLT Regression Tutorial

Use a regression forest to predict a continuous response from numeric and
categorical predictors. We simulate a known signal, fit on 80 observations,
and evaluate predictions on 20 held-out observations.

## Prerequisites

Install the current source as described in [Get Started](get-started.md), then import it:

```python
from rlt import RLT_reg
```

## Data

The first five predictors are numeric and the last five contain integer-coded
categories. The response combines predictors at indices 1 through 5, an
additional effect for categories 1 and 2 of predictor 5, and independent
Gaussian noise. Python column indices start at zero.

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

Declare the last five columns in `categorical_features` to use categorical
splits. The forest uses 200 trees and samples five candidate predictors at
each node with `mtry=p // 2`. Setting `nsplit=0` searches all available
cutpoints for numeric candidates. The explicit `random_state` controls fitting
randomness; `rng` above controls data generation. See
[Seeds & Reproducibility](../articles/feature-seed.md) for details.

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

print("Observed  Predicted")
for observed, predicted in zip(testY[:5], test_pred[:5]):
    print(f"{observed:8.3f}  {predicted:9.3f}")
```

```
Observed  Predicted
   4.242      3.456
   4.893      4.464
  -1.416      1.433
   6.339      4.525
   5.782      5.120
```

`predict()` returns one numeric prediction per held-out row, estimating the
conditional mean response at those predictor values. Individual responses
also contain noise, so predictions need not match them exactly.

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
The test MSE evaluates the separate held-out observations. Both are mean
squared errors, with smaller values indicating more accurate predictions.
Their difference reflects the different evaluation samples and trees used;
the test estimate is noisy with only 20 observations.

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
fit.get_one_tree(0)         # raw arrays of tree 0; see Single Tree article
```

To assess which predictors the forest relies on, continue to
[Variable Importance](../articles/variable-importance.md).
