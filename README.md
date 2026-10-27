# RLT for Python

[![CI](https://github.com/teazrq/RLT-Python/actions/workflows/ci.yml/badge.svg)](https://github.com/teazrq/RLT-Python/actions/workflows/ci.yml)

Fit random forests for continuous responses, class labels, and right-censored
survival times with a scikit-learn-style interface. `rlt` brings the tree
algorithms of the [RLT R package](https://github.com/teazrq/RLT) to Python.

The default forest uses one variable at each split. Optional reinforcement uses
embedded forests to guide variable selection. Linear-combination splits choose
a direction formed from several predictors and do not require reinforcement.

Start with the [installation and first-analysis guide](https://teazrq.github.io/RLT-Python/tutorials/get-started/),
or choose a [modeling tutorial](https://teazrq.github.io/RLT-Python/#choose-your-analysis).

## Installation

These guides describe **version 0.3.0 from the GitHub source**. Use a shared
user-wide or machine-wide Python installation. On Linux, install the system
libraries and then build the source:

```bash
# Debian/Ubuntu system libraries and C++ compiler
sudo apt-get install -y libarmadillo-dev liblapack-dev libopenblas-dev libomp-dev g++
python -m pip install setuptools wheel pybind11
git clone https://github.com/teazrq/RLT-Python
cd RLT-Python
python -m pip install --no-build-isolation .
```

Python 3.10 or later is required. The standard build targets Linux; Windows and
macOS need additional compiler and linker configuration. See the installation
guide above for verification and optional plotting and pandas dependencies.

## Quickstart

### Regression

```python
import numpy as np
from rlt import RLT_reg

rng = np.random.default_rng(1)
X = rng.normal(size=(200, 6))
y_reg = X[:, 0] - X[:, 1] + rng.normal(size=200)
trainX, testX = X[:160], X[160:]
model = RLT_reg(
    n_estimators=100, mtry=3, nsplit=0,
    resample_replace=True, resample_prob=0.8,
    random_state=1, n_jobs=1,
).fit(trainX, y_reg[:160])
pred = model.predict(testX)
print(f"OOB MSE: {model.oob_error_:.3f}")
print(f"Test MSE: {np.mean((pred - y_reg[160:]) ** 2):.3f}")
```

```text
OOB MSE: 1.401
Test MSE: 1.847
```

Both results are mean squared prediction errors, so smaller is better. Each
OOB prediction uses only trees that omitted that training row; test MSE uses
40 separate held-out rows. This small example demonstrates evaluation rather
than establishing a performance advantage.

### Classification

```python
from rlt import RLT_cla

y_class = np.where(y_reg > 0, "positive", "negative")
model = RLT_cla(n_estimators=100, random_state=42)
model.fit(trainX, y_class[:160])  # string labels are supported
prob = model.predict_proba(testX)
print(model.classes_)            # labels in probability-column order
print(np.round(prob[:3], 3))     # one row per held-out observation
```

Each row assigns probabilities to the labels in `model.classes_`. See the
[classification tutorial](https://teazrq.github.io/RLT-Python/tutorials/classification-tutorial/)
for decoded predictions and OOB/test accuracy.

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
model.fit(trainX, y_surv[:160])
S = model.predict_survival_function(testX)  # (40, T) survival curves
timepoints = model.timepoints_             # time coordinates of S's columns
print(f"Test concordance: {model.score(testX, y_surv[160:]):.3f}")
```

`S[i, j]` estimates survival beyond `timepoints[j]` for test row `i`.
Concordance measures risk ordering among comparable pairs; larger is better.
The [survival tutorial](https://teazrq.github.io/RLT-Python/tutorials/survival-tutorial/)
shows and interprets a fitted curve.

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
| `n_estimators` | `ntrees` | None | 100 under reinforcement, 500 otherwise, matching R. |
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
| `embed_*` | `param.control$embed.*` | see [reference](https://teazrq.github.io/RLT-Python/reference/rlt_reg/#embedded-model-controls) | embedded forest controls |
| `n_jobs` | `ncores` | -1 | OpenMP threads |
| `random_state` | `seed` | None | Integer forest seed, NumPy RandomState/Generator, or NumPy global RNG when omitted. |

For a fixed dataset, integer `random_state` supplies the C++ forest seed,
corresponding to R's `RLT(seed=...)`. Data generation uses a separate RNG.
Reproducibility also depends on the implementation and numerical toolchain;
cross-platform bitwise identity is not promised.

Use [`predict_details`, `oob_details`, and model persistence](https://teazrq.github.io/RLT-Python/articles/model-workflows/)
to retain labels, uncertainty, per-tree predictions, and sampling metadata.
All three estimators accept signed `resample_preset` counts at fit time and
support optional pandas category dictionaries. See [changes](CHANGELOG.md)
for revised defaults and validation rules.

## Development

The shared tree algorithms come from the corrected RLT 6.1.1 source, with
R/Rcpp interfaces replaced by pybind11. Python has its own estimator and array
APIs. The [comparison guide](https://teazrq.github.io/RLT-Python/reference/r-parity/)
documents source matching, smoother implementation, numerical agreement, and
random-stream limits.

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

For documentation changes, run `python tools/verify_examples.py` and
`python tools/check_expected_outputs.py` from the checkout. The first command
saves figures under `build/verify_figs/`. Copy the regenerated PNGs referenced
by the changed pages into `docs/assets/figures/`, then run
`python -m mkdocs build --strict` and inspect the pages and figures. Install
`matplotlib`, `pandas`, `mkdocs`, and `mkdocs-material` in the shared Python
installation for these checks. The example tools import the checkout;
installation verification must run separately from outside it.

## License

GPL (>= 3), inherited from the RLT R package.

## Citations

Cite the papers for the methods used in your analysis:

- **Reinforcement learning trees:** Zhu, R., Zeng, D., and Kosorok, M. R.
  (2015). [*Reinforcement Learning Trees*](https://doi.org/10.1080/01621459.2015.1036994).
  *Journal of the American Statistical Association*, 110(512), 1770-1784.
- **Matched-sample variance estimation:** Xu, T., Zhu, R., and Shao, X.
  (2024). [*On variance estimation of random forests with Infinite-order U-statistics*](https://doi.org/10.1214/24-EJS2247).
  *Electronic Journal of Statistics*, 18(1), 2135-2207.
- **Survival confidence bands:** Formentini, S. E., Liang, W., and Zhu, R.
  (2022). [*Confidence Band Estimation for Survival Random Forests*](https://arxiv.org/abs/2204.12038).
  *arXiv preprint*, arXiv:2204.12038.

The survival paper develops the cumulative-hazard covariance and Gaussian
simulation framework for the bands. See the
[`get_surv_band` reference](https://teazrq.github.io/RLT-Python/reference/get_surv_band/)
for the current smoothing and covariance handling.
