# Get Started with rlt

This page helps you install the `rlt` package and points you to the modeling
tutorials. It is the Python equivalent of the R package's "Get Started"
vignette, adapted for the current GitHub source and its compiled core.
Use a shared user-wide or machine-wide Python installation.

## Install the rlt package

The 0.3.0 alignment updates documented here are available from the GitHub
source. This change does not publish a new PyPI release. The source build
needs Armadillo, LAPACK, BLAS, OpenMP, and a C++17 compiler.
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

### Verify the installation

```python
import rlt

print(rlt.__version__)
```

A quick smoke test that also exercises the compiled core:

```python
import numpy as np
from rlt import RLT_reg

rng = np.random.default_rng(0)
X = rng.normal(size=(60, 5))
y = X[:, 0] + rng.normal(size=60)

model = RLT_reg(n_estimators=100, random_state=0)
model.fit(X, y)
print(f"OOB error: {model.oob_error_:.3f}")
```

## Categorical predictors

Pandas is optional. In a pandas data frame, unordered `Categorical` columns
automatically use categorical splits, preserving declared levels, including
unused ones. Ordered categoricals and string columns use numerical ranks.
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
