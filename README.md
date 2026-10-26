# RLT for Python

[![CI](https://github.com/teazrq/RLT-Python/actions/workflows/ci.yml/badge.svg)](https://github.com/teazrq/RLT-Python/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/rlt-forests.svg)](https://pypi.org/project/rlt-forests/)

`rlt` is a Python port of the [RLT R package](https://github.com/teazrq/RLT)
(Reinforcement Learning Trees, Zhu, Zeng & Kosorok, and successors):
random forests with **embedded-model variable selection** and
**linear-combination splits**, for regression, classification, and
survival analysis. Its estimators use scikit-learn's `fit`, `predict`,
parameter, and pipeline interfaces.

The shared tree algorithms come from RLT source version 6.1.1, with
R/Rcpp interfaces replaced by pybind11. Python has its own estimator and
array APIs. The optional smoothed confidence-band method uses a separate
spline approximation and is not numerically identical to R's `mgcv` smoother.

## Installation

```bash
# system deps: armadillo, lapack, blas, openmp (e.g. apt install
# libarmadillo-dev liblapack-dev libopenblas-dev)
pip install rlt-forests
```

From source:

```bash
git clone https://github.com/teazrq/RLT-Python
cd RLT-Python
pip install .
```

## Quickstart

### Regression

```python
import numpy as np
from rlt import RLT_reg

rng = np.random.default_rng(42)
X = rng.normal(size=(200, 6))
y_reg = X[:, 0] - X[:, 1] + rng.normal(size=200)
model = RLT_reg(n_estimators=100, importance="permute", random_state=42)
model.fit(X, y_reg)
pred = model.predict(X)
model.oob_error_               # out-of-bag MSE
model.feature_importances_
```

These predictions illustrate returned values on training rows. Use the
stored OOB error or separate held-out data to evaluate prediction accuracy.

### Classification

```python
from rlt import RLT_cla

y_class = np.where(y_reg > 0, "positive", "negative")
model = RLT_cla(n_estimators=100, random_state=42)
model.fit(X, y_class)          # string labels are supported
prob = model.predict_proba(X)  # (n, n_classes)
```

### Survival

```python
import numpy as np
from rlt import RLT_surv

event_time = rng.exponential(1 / np.exp(X[:, 0]))
censor_time = rng.uniform(0, 3, size=len(X))
y_surv = np.empty(len(X), dtype=[("event", "?"), ("time", float)])
y_surv["event"] = event_time <= censor_time
y_surv["time"] = np.minimum(event_time, censor_time)
model = RLT_surv(n_estimators=100, split_rule="logrank", random_state=42)
model.fit(X, y_surv)
S = model.predict_survival_function(X)   # (n, T) survival curves
model.score(X, y_surv)                   # in-sample concordance, not OOB
```

### Reinforcement and linear combination splits

```python
from rlt import RLT_reg

# embedded-model variable selection + linear combination splits
model = RLT_reg(
    n_estimators=100,        # R default under reinforcement
    linear_comb=2,           # combine up to 2 variables per split
    linear_comb_method="sir",  # or naive / lm / pca
    reinforcement=True,      # embedded random-forest pre-screening
    random_state=42,
)
model.fit(X, y_reg)
```

Reinforcement uses embedded forests to guide variable selection. Linear
combination splits choose a predictor direction and do not require
reinforcement. This example demonstrates the controls without making an
accuracy comparison.

### Variance estimation & confidence bands

```python
model = RLT_reg(n_estimators=2000, var_mode="matched")  # or "ij"/"jack"
model.fit(X, y_reg)
X_query = X[:5]
pred, var = model.predict_var(X_query)

model = RLT_surv(n_estimators=2000, var_mode="matched")
model.fit(X, y_surv)
S, cov = model.predict_var(X_query)  # cov is on the cumulative-hazard scale

from rlt import get_surv_band
band = get_surv_band(model, X_query, alpha=0.05, approach="smoothed")
band["lower"], band["upper"]  # simultaneous survival bands
```

The bands target the conditional survival curve separately for each query
point over the returned time grid. Their reference is `exp(-CHF)`. For
linear combination forests, the returned survival estimate instead averages
tree product-limit curves; its sampling variance is not estimated by this
CHF-based calculation. These are nominal bands, not event-time prediction
intervals.

### Forest similarity kernel

```python
K = model.forest_kernel(X_query)          # raw leaf-sharing tree counts
Kc = model.forest_kernel(X_query, X)
Kt = model.forest_kernel(X_query, X, vs_train=True)  # in-bag multiplicity

# OOB self-kernel (RLT >= 6.1.0): co-occurrence counted only from trees
# where both observations were excluded from fitting that tree
out = model.forest_kernel(X, oob=True)
out["Kernel"]   # (n, n) normalized co-occurrence in [0, 1]
out["N"], out["C"]  # both-OOB counts / both-OOB same-leaf counts
```

Training and OOB kernels require retained resampling records, enabled by
`resample_track=True` or a variance mode. Supply the original training rows
in their original order. Ordinary counts are not normalized prediction weights.

## Parameter reference

| Python (sklearn-style) | R (RLT) | Default | Notes |
|---|---|---|---|
| `n_estimators` | `ntrees` | 500 | R defaults to 100 under reinforcement; set explicitly for comparisons. |
| `mtry` | `mtry` | max(1, p//2) | |
| `min_samples_leaf` | `nmin` | 5 | |
| `nsplit` | `nsplit` | 0 | random cut points; 0 = all (best split) |
| `resample_replace` | `resample.replace` | True | |
| `resample_prob` | `resample.prob` | 1.0 / 0.8 | |
| `importance` | `importance` | "none" | "permute" or "distribute" |
| `resample_track` | `resample.track` | False | needed for kernels/IJ |
| `var_mode` | `var.mode` | "none" | "matched", "ij", "jack" |
| `linear_comb` | `linear.comb` | 1 | >1 = linear-combination splits |
| `linear_comb_method` | `linear.comb.method` | model-specific | reg: sir/naive/lm/pca; cla: lda/naive/random/logistic; surv: coxph/naive |
| `reinforcement` | `reinforcement` | False | embedded-model splitting |
| `split_rule` | `split.rule` | model-specific | surv: logrank/suplogrank/coxgrad |
| `embed_*` | `param.control$embed.*` | see R docs | embedded forest controls |
| `n_jobs` | `ncores` | -1 | OpenMP threads |
| `random_state` | `seed` | None | Integer forest seed, or a NumPy RandomState. |

For a fixed dataset, integer `random_state` supplies the C++ forest seed,
corresponding to R's `RLT(seed=...)`. Data generation uses a separate RNG.
Reproducibility also depends on the implementation and numerical toolchain;
cross-platform bitwise identity is not promised.

## Development

```bash
python -m pip install --no-build-isolation -e . pytest scikit-learn threadpoolctl
python -m pytest tests/
```

Use a shared user-wide or machine-wide Python installation. Install
`setuptools` and `pybind11` there before building without isolation.
The source build configuration and CI currently target Linux; Windows and
macOS installation support should not be inferred from the estimator API.

The C++ core under `src/cpp/` is regenerated from an explicitly selected R
checkout by `tools/vendor_core.py`, which preserves Python's compatibility
layer. The parity tests use `Rscript` and an RLT installation in a shared R
library. Set `RLT_RSCRIPT` if that executable is not on PATH. Compare the
same source revisions and parameter values in both languages.

## License

GPL (>= 3), inherited from the RLT R package.

## Citation

If you use RLT methods, please cite the corresponding R-side papers:
Zhu, R., Zeng, D., & Kosorok, M. R. (2015) *Reinforcement Learning
Trees*, JASA; and follow-ups on variance estimation and survival bands.
