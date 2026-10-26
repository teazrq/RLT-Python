# rlt — Reinforcement Learning Trees for Python

[![CI](https://github.com/teazrq/RLT-Python/actions/workflows/ci.yml/badge.svg)](https://github.com/teazrq/RLT-Python/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/rlt-forests.svg)](https://pypi.org/project/rlt-forests/)

`rlt` is a Python port of the [RLT R package](https://cran.r-project.org/package=RLT)
(Reinforcement Learning Trees, Zhu, Zeng & Kosorok, and successors) —
random forests with **embedded-model variable selection** and
**linear-combination splits**, for regression, classification, and
survival analysis. It is **fully scikit-learn compatible**: if you know
how to use `RandomForestRegressor`, you already know how to use `rlt`.

The ~19k-line OpenMP/Armadillo C++ compute core is the exact core of the
R package (vendored, R/Rcpp-free), exposed through pybind11 — so results
are R-grade fast and parallel.

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
from rlt import RLT_reg

model = RLT_reg(n_estimators=500, random_state=42)
model.fit(X, y)                # sklearn API
pred = model.predict(X)
model.oob_error_               # out-of-bag MSE
model.feature_importances_
```

### Classification

```python
from rlt import RLT_cla

model = RLT_cla(n_estimators=500)
model.fit(X, y)                # string labels fine
prob = model.predict_proba(X)  # (n, n_classes)
```

### Survival

```python
import numpy as np
from rlt import RLT_surv

y = np.empty(n, dtype=[("event", "?"), ("time", float)])  # sksurv convention
y["event"], y["time"] = event, time
model = RLT_surv(split_rule="logrank")
model.fit(X, y)
S = model.predict_survival_function(X)   # (n, T) survival curves
model.score(X, y)                        # Harrell's c-index
```

### Reinforcement learning trees (the point of the package)

```python
from rlt import RLT_reg

# embedded-model variable selection + linear combination splits
model = RLT_reg(
    n_estimators=100,        # R default under reinforcement
    linear_comb=2,           # combine 2 variables per split
    linear_comb_method="sir",  # or naive / lm / pca
    reinforcement=True,      # embedded random-forest pre-screening
    random_state=42,
)
model.fit(X, y)
```

On linear signal DGPs, `linear_comb=2` roughly halves the OOB error of
the axis-aligned forest (see `tests/`).

### Variance estimation & confidence bands

```python
model = RLT_reg(n_estimators=2000, var_mode="matched")  # or "ij"/"jack"
pred, var = model.predict_var(X)

model = RLT_surv(n_estimators=2000, var_mode="matched")
S, cov = model.predict_var(X_test)

from rlt import get_surv_band
band = get_surv_band(model, X_test, alpha=0.05, approach="smoothed")
band["lower"], band["upper"]  # simultaneous survival bands
```

### Forest similarity kernel

```python
K = model.forest_kernel(X_test)          # (n, n) co-occurrence counts
Kc = model.forest_kernel(X_test, X_train)
Kt = model.forest_kernel(X_test, X_train, vs_train=True)  # needs resample_track=True

# OOB self-kernel (RLT >= 6.1.0): co-occurrence counted only from trees
# where both observations are out-of-bag — unbiased for DoF estimation
out = model.forest_kernel(X_train, oob=True)   # needs resample_track=True
out["Kernel"]   # (n, n) normalized co-occurrence in [0, 1]
out["N"], out["C"]  # both-OOB counts / both-OOB same-leaf counts
```

## Parameter reference

| Python (sklearn-style) | R (RLT) | Default | Notes |
|---|---|---|---|
| `n_estimators` | `ntrees` | 500 (100 if reinforcement) | |
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
| `random_state` | `seed` | None | numpy seeding |

RNG note: trees are seeded through Python's `random_state` (numpy), then
per-tree seeds derive inside C++ — reproducible across platforms, not
bit-compatible with R.

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e . pytest scikit-learn threadpoolctl
pytest tests/
```

The C++ core under `src/cpp/` is regenerated idempotently from the
pristine R sources by `tools/vendor_core.py` (strips Rcpp, swaps the RNG,
adds the compat header). The `smoke` and differential-vs-R scripts used
during development live in the commit history.

## License

GPL (>= 3), inherited from the RLT R package.

## Citation

If you use RLT methods, please cite the corresponding R-side papers:
Zhu, R., Zeng, D., & Kosorok, M. R. (2015) *Reinforcement Learning
Trees*, JASA; and follow-ups on variance estimation and survival bands.
