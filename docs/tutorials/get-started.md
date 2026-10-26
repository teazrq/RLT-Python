# Get Started with rlt

This page helps you install the `rlt` package and points you to the modeling
tutorials. It is the Python equivalent of the R package's "Get Started"
vignette — adapted for `pip`, virtual environments, and the platforms the
C++ core supports.

## Install the rlt package

### Released version

Install from PyPI (system dependencies: Armadillo, LAPACK, BLAS, OpenMP):

```bash
pip install rlt-forests
```

On Debian/Ubuntu the system libraries are:

```bash
sudo apt-get install -y libarmadillo-dev liblapack-dev \
    libopenblas-dev libomp-dev g++
```

### Development version

Install the development version from GitHub:

```bash
git clone https://github.com/teazrq/RLT-Python
cd RLT-Python
pip install .
```

Or, directly from the repository URL:

```bash
pip install git+https://github.com/teazrq/RLT-Python
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

`rlt` does not consume R-style factors or pandas `Categorical` columns
directly. Instead, declare categorical columns with the
`categorical_features` constructor parameter — the sklearn-style
replacement for R's `data.frame` factor path — and pass the columns as
**non-negative integer level codes** (`0, 1, ..., k-1`):

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

model.ncat_  # array([1, 1, 3, 2]) — 1 = continuous, k = k-level categorical
```

Rules to keep in mind:

- Declared columns must be non-negative integer codes `0..k-1` —
  anything else raises a `ValueError`. Encode with
  `pandas.factorize()` or `sklearn.preprocessing.OrdinalEncoder`
  first (both produce 0-based codes by default).
- A categorical column may have at most **53 levels** (`ValueError`
  beyond that, matching the R package).
- If you *forget* to declare a categorical column and it has few unique
  values (≤ 10), `fit()` emits a `UserWarning` hinting that it may be
  categorical — worth a look before you trust continuous splits on it.
- Undeclared columns are treated as continuous, which is safe for
  genuinely numeric low-cardinality data.

## Platform notes

### Linux

Linux is the best-supported platform: install the system packages above and
`pip install` works out of the box. Parallelism uses OpenMP — set the
`OMP_NUM_THREADS` environment variable if you need to limit threads.

### macOS

`pip install rlt-forests` builds the C++ core from source, so you need
AppleClang (via Xcode or the Command Line Tools) plus Homebrew libraries:

```bash
xcode-select --install
brew install armadillo lapack openblas libomp
pip install rlt-forests
```

`libomp` provides OpenMP on macOS (AppleClang does not ship it). Without it
the package still builds, but runs single-threaded.

### Windows

A C++ compiler from Visual Studio Build Tools plus
[vcpkg](https://vcpkg.io/) (`armadillo`, `lapack`, `openblas`) is required
for source builds. Prebuilt wheels are planned; until then Linux and macOS
are the recommended platforms.

## Randomness and reproducibility

Unlike R's `set.seed()`, `rlt` follows the scikit-learn convention: seeds
are passed per estimator through `random_state`:

```python
from rlt import RLT_reg

model = RLT_reg(n_estimators=200, random_state=42)
model.fit(X, y)
```

The same `random_state` + same data + same parameters gives the same
forest on every platform. See the
[Seeds & Reproducibility](../articles/feature-seed.md) article for the full
story, including how seeds interact with data simulation.

## Next steps

See the [Regression](regression-tutorial.md), [Classification](classification-tutorial.md),
and [Survival](survival-tutorial.md) tutorials to get started with modeling.
