# Get Started with rlt

Install `rlt`, fit a regression forest, and evaluate predictions on observations
held out before fitting. The example also distinguishes an ordinary forest from
reinforcement and linear-combination splits.

## Install the rlt package

These guides describe **version 0.3.0 from the GitHub source**. Use a shared
user-wide or machine-wide Python installation, with Python 3.10 or later.
The source build needs Armadillo, LAPACK, BLAS, OpenMP, and a C++17 compiler.
On Debian/Ubuntu, install the shared system libraries with:

```bash
sudo apt-get install -y libarmadillo-dev liblapack-dev \
    libopenblas-dev libomp-dev g++
```

Install the build tools in your shared Python installation, then build
the current source without an isolated build environment:

```bash
python -m pip install setuptools wheel pybind11
git clone https://github.com/teazrq/RLT-Python
cd RLT-Python
python -m pip install --no-build-isolation .
```

The commands above target Linux. See [platform notes](#platform-notes) for
Windows and macOS. For plotting the examples in the feature articles, install
`matplotlib` in the same shared Python installation.

### Verify the installation

Run this from outside the cloned repository to check the installed package:

```python
import rlt

print(rlt.__version__)
print(rlt.__file__)
```

The version should be `0.3.0`. The path identifies which installation Python
loaded. Different source revisions can share a version number; record
`git rev-parse HEAD` in the checkout when you need to reproduce a source build.

## Fit and evaluate a regression forest

We will predict a continuous response from six numeric predictors. The response
depends on the first two predictors plus independent noise. Reserve 40 rows for
evaluation and use the other 160 to fit the forest:

```python
import numpy as np
from rlt import RLT_reg

rng = np.random.default_rng(1)
X = rng.normal(size=(200, 6))
y = X[:, 0] - X[:, 1] + rng.normal(size=200)

trainX, trainY = X[:160], y[:160]
testX, testY = X[160:], y[160:]
```

The default forest uses one variable per split. Here `mtry=3` samples three
candidate variables at each node, and `nsplit=0` searches all available
cutpoints for those candidates. Each tree makes 128 draws with replacement
from the training set. We use 100 trees for this short example and fix
`random_state` to control fitting randomness.

```python
settings = dict(
    n_estimators=100, mtry=3, nsplit=0,
    resample_replace=True, resample_prob=0.8,
    random_state=1, n_jobs=1,
)
model = RLT_reg(**settings).fit(trainX, trainY)
pred = model.predict(testX)

print(f"OOB MSE: {model.oob_error_:.3f}")
print(f"Test MSE: {np.mean((pred - testY) ** 2):.3f}")
print("Observed  Predicted")
for observed, predicted in zip(testY[:5], pred[:5]):
    print(f"{observed:8.2f}  {predicted:9.2f}")
```

```text
OOB MSE: 1.401
Test MSE: 1.847
Observed  Predicted
    1.47       0.78
    3.20       0.73
    0.50      -0.20
    2.96       2.88
   -0.29       0.25
```

The predictions estimate the mean response at each row's predictor values;
individual responses also contain noise. Both errors are mean squared errors
(MSEs), so smaller values indicate more accurate predictions. Each OOB
prediction uses only trees that omitted that training observation. Test MSE
evaluates the 40 separate observations reserved before fitting. These estimates
can differ in a small sample; this example is not a performance benchmark.

## Try reinforcement or linear-combination splits

Reinforcement fits embedded forests within nodes to guide variable selection.
Keep the data and outer forest settings fixed while requesting it:

```python
reinforced = RLT_reg(reinforcement=True, **settings).fit(trainX, trainY)
reinforced_pred = reinforced.predict(testX)
```

We explicitly set the tree count in both fits because its default is 500 for
an ordinary forest and 100 with reinforcement. This example demonstrates the
control without establishing an accuracy advantage.

Linear-combination splits instead choose a direction formed from up to
`linear_comb` predictors. They do not require reinforcement:

```python
combined = RLT_reg(linear_comb=2, **settings).fit(trainX, trainY)
combined_pred = combined.predict(testX)
```

The [linear-combination guide](../articles/linear-combinations.md) explains
direction estimation. The [embedded-model reference](../reference/rlt_reg.md#embedded-model-controls)
describes variable screening and its controls.

## Categorical predictors

Pandas is optional. In a pandas data frame, unordered `Categorical` columns
automatically use categorical splits, preserving declared levels, including
unused ones. Ordered categoricals and string columns use numerical ranks.
For nominal labels with no meaningful order, use an unordered pandas
`Categorical` column so splits can group levels rather than threshold their ranks.
Prediction matches named columns and reuses the fitted label dictionary;
unknown labels and missing values are rejected.
Install pandas in your shared Python installation with
`python -m pip install pandas` when using this optional input path.

Plain numeric arrays remain supported. Declare their nominal columns with
`categorical_features` and supply **non-negative integer level codes**
(`0, 1, ..., k-1`):

```python
import numpy as np
from rlt import RLT_reg

rng = np.random.default_rng(0)
n = 100
X = np.hstack([
    rng.normal(size=(n, 2)),           # continuous columns 0-1
    rng.integers(0, 3, size=(n, 1)),   # 3-level categorical, column 2
    rng.integers(0, 2, size=(n, 1)),   # binary categorical, column 3
])
y = X[:, 0] + np.isin(X[:, 2], [1, 2]) + rng.normal(size=n)

# Integer indices of the categorical columns...
model = RLT_reg(n_estimators=100, categorical_features=[2, 3],
                random_state=0).fit(X, y)

# ...or an equivalent boolean mask of length p
model = RLT_reg(n_estimators=100,
                categorical_features=[False, False, True, True],
                random_state=0).fit(X, y)

model.ncat_  # array([1, 1, 3, 2]); 1 = continuous, k = k-level categorical
```

Rules to keep in mind:

- Nominal columns in a plain numeric array must use non-negative integer
  codes `0..k-1`; other values raise a `ValueError`.
- A numeric prediction column for a fitted labeled predictor means
  zero-based training codes. Retain a pandas categorical dtype when
  supplying numeric labels such as 10, 2, and 1, so they are matched as
  labels rather than interpreted as codes.
- A categorical column may have at most **53 levels** (`ValueError`
  beyond that, matching the R package). Ordered and string rank encodings
  use numerical splits rather than categorical splits.
- A numerical column with few unique values (up to 10) can trigger a
  `UserWarning` suggesting an explicit categorical declaration. It remains
  numerical unless declared otherwise.

See [Model Workflows](../articles/model-workflows.md) for category reordering
and declared response-class levels.

## Platform notes

The standard build configuration and CI target Linux with GCC and OpenMP.
macOS and Windows builds require platform-specific compiler and linker
configuration; the commands above are not a tested installation recipe for
those platforms. Set `n_jobs` on the estimator to control forest parallelism.

## Randomness and reproducibility

Pass an integer `random_state` for a repeatable forest seed, separately
from the NumPy generator used to simulate data:

```python
from rlt import RLT_reg

model = RLT_reg(n_estimators=200, random_state=42)
model.fit(X, y)
```

The same integer seed, data, parameters, and numerical environment support
repeatable fits. Cross-platform bitwise identity is not promised. With
`random_state=None`, fitting draws from NumPy's global RNG, so
`np.random.seed(...)` controls omitted forest seeds. See the
[Seeds & Reproducibility](../articles/feature-seed.md) article for the full
story, including how seeds interact with data simulation.

## Next steps

See the [Regression](regression-tutorial.md), [Classification](classification-tutorial.md),
and [Survival](survival-tutorial.md) tutorials to get started with modeling.
