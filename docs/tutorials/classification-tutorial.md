# RLT Classification Tutorial

Use a classification forest to predict class labels and probabilities from
numeric and categorical predictors. We simulate a binary response, fit on
80 observations, and evaluate predictions on 20 held-out observations.

## Prerequisites

Install the current source as described in [Get Started](get-started.md), then import it:

```python
from rlt import RLT_cla
```

## Data Preparation

The first five predictors are numeric and the last five contain integer-coded
categories. The probability of class `1` increases with the first numeric
predictor; the other predictors have no effect on the simulated response.

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
X2 = (rng.uniform(size=(n, p // 2)) * 10).astype(int)  # integers 0-9
X = np.hstack([X1, X2])

# Binary outcome via a simple logistic signal
def sigmoid(x):
    return np.exp(x) / (1 + np.exp(x))

prob = sigmoid(-0.5 + 2 * X[:, 0])  # signal in the first continuous feature
y = rng.binomial(1, p=prob)

# Train / test split
trainX, trainY = X[:trainn], y[:trainn]
testX, testY = X[trainn:], y[trainn:]
```

## Fit a classification model

Declare the last five columns in `categorical_features` to use categorical
splits. The forest uses 200 trees and samples five candidate predictors at
each node with `mtry=p // 2`. Setting `nsplit=0` searches all available
cutpoints for numeric candidates. The explicit `random_state` controls fitting
randomness; `rng` above controls data generation. See
[Seeds & Reproducibility](../articles/feature-seed.md) for details.

```python
from rlt import RLT_cla

ntrees = 200
nmin = 5
mtry = p // 2
samplereplace = True
sampleprob = 0.80
rule = "best"
nsplit = 0 if rule == "best" else 3
importance = "permute"

fit = RLT_cla(
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
# Predicted class labels (original label type: 0/1 here)
labels = fit.predict(testX)

# Class-membership probabilities, shape (n, n_classes)
proba = fit.predict_proba(testX)

# Decode the OOB class codes using the training label dictionary
oob_labels = fit.classes_[fit.oob_class_]

print("Probability columns:", fit.classes_)
print("Observed  Predicted  P(0)   P(1)")
for observed, predicted, row in zip(testY[:5], labels[:5], proba[:5]):
    print(f"{observed:8d}  {predicted:9d}  {row[0]:.3f}  {row[1]:.3f}")
```

```
Probability columns: [0 1]
Observed  Predicted  P(0)   P(1)
       0          0  0.772  0.228
       0          1  0.344  0.656
       0          0  0.882  0.118
       1          1  0.392  0.608
       0          0  0.809  0.191
```

`predict()` returns labels in their original type. Probability column `j`
corresponds to `fit.classes_[j]`, so the two columns here give probabilities
of classes `0` and `1`. Each row sums to one, and the predicted label has
the largest probability. `classes_` is the class order to use for both
`predict_proba()` and `oob_prob_`. For an ordinary label array it contains
sorted unique labels; for a pandas categorical response it preserves the
declared category order, including unused categories.

## Evaluate

```python
acc_oob = float(np.mean(oob_labels == trainY))
acc_test = float(np.mean(labels == testY))

# A compact summary
print(f"OOB accuracy:  {acc_oob:.4f}")
print(f"Test accuracy:  {acc_test:.4f}")
```

```
OOB accuracy:  0.8000
Test accuracy:  0.7500
```

Each OOB class prediction uses only trees that omitted that training observation.
Test accuracy evaluates the separate held-out observations. Accuracy is the
fraction of correctly classified observations, so larger values indicate
better classification. With only 20 test observations, each additional correct
classification changes test accuracy by 0.05.

## Inspect the fitted object

```python
print(fit)
```

```
RLT_cla(categorical_features=range(5, 10), importance='permute', mtry=5,
        n_estimators=200, n_jobs=1, random_state=1, resample_prob=0.8)
```

Fitted attributes worth knowing:

```python
fit.classes_          # class labels in probability-column order
fit.n_features_in_    # number of predictors
fit.oob_error_        # out-of-bag misclassification rate
fit.oob_class_        # (n,) OOB predicted classes (integer codes)
fit.oob_prob_         # (n, n_classes) OOB class probabilities
fit.feature_importances_  # (p,) variable importance
```

With the data above: `fit.classes_` is `array([0, 1])`, `fit.n_features_in_`
is `10`, and `fit.oob_error_` is `0.2000`.

String labels work too: `RLT_cla` stores `classes_` and `predict` returns
the original label type:

```python
y_str = np.array(["neg", "pos"])[y]
fit2 = RLT_cla(n_estimators=200, random_state=1,
               categorical_features=range(p // 2, p))
fit2.fit(X, y_str)
fit2.classes_        # array(['neg', 'pos'], dtype='<U3')
fit2.predict(X[:3])  # e.g. array(['pos', 'neg', 'neg'], dtype='<U3')
```

## Prediction variance

Classification forests support variance estimation too, the equivalent
of R's `predict(fit, var.est = TRUE)`. Fit with `var_mode` (see the
[survival tutorial](survival-tutorial.md) for the full variance story),
then call `predict_var()` to get the class probabilities together with
their variance:

```python
fit_v = RLT_cla(
    n_estimators=500,
    var_mode="matched",
    categorical_features=range(p // 2, p),
    n_jobs=1,
    random_state=1,
)
fit_v.fit(trainX, trainY)

probs, variances = fit_v.predict_var(testX)

probs.shape      # (n, n_classes); same values as predict_proba
variances.shape  # (n, n_classes)
```

```python
np.round(variances[:3], 5)
```

```
array([[0.00848, 0.00848],
       [0.01075, 0.01075],
       [0.00958, 0.00958]])
```

With two classes the two variance columns are identical (the second
probability is one minus the first). Note that negative variance
estimates are cleaned to `NaN` before returning; check with
`np.isnan(variances).any()` and drop or inspect those cells rather than
propagating them into downstream summaries. Variance estimation generally
needs more trees than a prediction-only illustration. The 500 trees here
demonstrate the calculation; this count does not establish its accuracy.
