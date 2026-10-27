# RLT for Python

[![CI](https://github.com/teazrq/RLT-Python/actions/workflows/ci.yml/badge.svg)](https://github.com/teazrq/RLT-Python/actions/workflows/ci.yml)
[![Docs](https://github.com/teazrq/RLT-Python/actions/workflows/docs.yml/badge.svg)](https://github.com/teazrq/RLT-Python/actions/workflows/docs.yml)
[![PyPI](https://img.shields.io/pypi/v/rlt-forests.svg)](https://pypi.org/project/rlt-forests/)

`rlt` is the Python port of the [RLT R package](https://cran.r-project.org/package=RLT)
(Reinforcement Learning Trees, Zhu, Zeng & Kosorok, and successors) —
random forests with **embedded-model variable selection** and
**linear-combination splits**, for regression, classification, and survival
analysis. Its estimators use scikit-learn's `fit`, `predict`, parameter,
and pipeline interfaces.

The shared tree algorithms come from RLT source version 6.1.1, with
R/Rcpp interfaces replaced by pybind11. Python has its own array and
estimator APIs. Smoothed confidence bands use a narrow Python implementation
of R's `mgcv` 1.9-4 tensor smoother, with no R process needed during prediction.
The [comparison guide](reference/r-parity.md) describes tested numerical
agreement and random-stream limits.

---

## Features

<div class="grid cards" markdown>

-   :material-tree:{ .lg .middle } **Three forest families**

    ---

    `RLT_reg`, `RLT_cla`, and `RLT_surv` cover regression, classification,
    and right-censored survival analysis — all with the same sklearn-style
    `fit` / `predict` interface.

    [:octicons-arrow-right-24: Tutorials](tutorials/regression-tutorial.md)

-   :material-brain:{ .lg .middle } **Reinforcement learning trees**

    ---

    Embedded random-forest pre-screening at each node selects which variables
    to consider, and splits can be **linear combinations** of several
    variables (`linear_comb > 1`) instead of axis-aligned.

    [:octicons-arrow-right-24: Linear combinations](articles/linear-combinations.md)

-   :material-chart-line:{ .lg .middle } **Variance estimation & bands**

    ---

    Matched-sample U-statistic, infinitesimal jackknife, and jackknife
    variances for predictions; simultaneous confidence bands for survival
    curves via `get_surv_band()`.

    [:octicons-arrow-right-24: Confidence bands](articles/confidence-interval.md)

-   :material-scale-balance:{ .lg .middle } **Variable importance**

    ---

    Permutation importance and distributed-assignment importance, computed
    out-of-bag during the fit.

    [:octicons-arrow-right-24: Variable importance](articles/variable-importance.md)

-   :material-graph-outline:{ .lg .middle } **Forest similarity kernel**

    ---

    Terminal-node co-occurrence counts between any two sets of points —
    the random-forest kernel useful for smoothing, representation learning,
    and covariate-distance tasks.

    [:octicons-arrow-right-24: Forest kernel](articles/feature-kernel.md)

-   :material-magnify-expand:{ .lg .middle } **Full tree inspection**

    ---

    Every tree's split variables, split values, loadings, node weights, and
    node predictions are exposed as numpy arrays via `get_one_tree(i)`.

    [:octicons-arrow-right-24: Single tree](articles/feature-single-tree.md)

</div>

## Installation

```bash
# system deps: armadillo, lapack, blas, openmp
# (e.g. apt install libarmadillo-dev liblapack-dev libopenblas-dev)
pip install rlt-forests
```

From source:

```bash
git clone https://github.com/teazrq/RLT-Python
cd RLT-Python
pip install .
```

See [Get Started](tutorials/get-started.md) for platform notes.

## Quickstart

=== "Regression"

    ```python
    import numpy as np
    from rlt import RLT_reg

    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, 10))
    y = X[:, 0] + X[:, 1] + rng.normal(size=100)

    model = RLT_reg(n_estimators=500, importance="permute", random_state=42)
    model.fit(X, y)                # sklearn API
    pred = model.predict(X)
    model.oob_error_               # out-of-bag MSE
    model.feature_importances_
    ```

=== "Classification"

    ```python
    import numpy as np
    from rlt import RLT_cla

    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, 10))
    y = rng.binomial(1, 1 / (1 + np.exp(-(2 * X[:, 0]))))  # signal in X0

    model = RLT_cla(n_estimators=500, random_state=42)
    model.fit(X, y)                # string labels fine
    prob = model.predict_proba(X)  # (n, n_classes)
    labels = model.predict(X)
    ```

=== "Survival"

    ```python
    import numpy as np
    from rlt import RLT_surv

    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 5))
    t_event = rng.exponential(1 / np.exp(X[:, 0]))
    t_censor = rng.uniform(0, 3, size=200)

    y = np.empty(200, dtype=[("event", "?"), ("time", float)])  # sksurv convention
    y["event"] = t_event <= t_censor
    y["time"] = np.minimum(t_event, t_censor)

    model = RLT_surv(split_rule="logrank", random_state=42)
    model.fit(X, y)
    S = model.predict_survival_function(X)   # (n, T) survival curves
    model.score(X, y)                        # Harrell's c-index
    ```

=== "Reinforcement learning trees"

    ```python
    import numpy as np
    from rlt import RLT_reg

    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, 10))
    y = X[:, 0] + X[:, 1] + rng.normal(size=100)

    # embedded-model variable selection + linear combination splits
    model = RLT_reg(
        n_estimators=100,          # R default under reinforcement
        linear_comb=2,             # combine 2 variables per split
        linear_comb_method="sir",  # or naive / lm / pca
        reinforcement=True,        # embedded random-forest pre-screening
        random_state=42,
    )
    model.fit(X, y)
    ```

    On linear-signal DGPs, `linear_comb=2` roughly halves the OOB error of
    the axis-aligned forest (see `tests/`).

=== "Variance & confidence bands"

    ```python
    import numpy as np
    from rlt import RLT_reg, RLT_surv, get_surv_band

    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, 5))
    y = X[:, 0] + rng.normal(size=100)

    model = RLT_reg(n_estimators=2000, var_mode="matched", random_state=1)
    model.fit(X, y)
    pred, var = model.predict_var(X)

    t_event = rng.exponential(1 / np.exp(X[:, 0]))
    t_censor = rng.uniform(0, 3, size=100)
    ys = np.empty(100, dtype=[("event", "?"), ("time", float)])
    ys["event"] = t_event <= t_censor
    ys["time"] = np.minimum(t_event, t_censor)

    smodel = RLT_surv(n_estimators=2000, var_mode="matched", random_state=1)
    smodel.fit(X, ys)
    S, cov = smodel.predict_var(X)

    band = get_surv_band(smodel, X[:3], alpha=0.05, approach="smoothed")
    band["lower"], band["upper"]  # simultaneous survival bands
    ```

=== "Forest similarity kernel"

    ```python
    import numpy as np
    from rlt import RLT_reg

    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, 5))
    y = X[:, 0] + X[:, 1] + rng.normal(size=100)
    X_test = rng.normal(size=(10, 5))

    model = RLT_reg(n_estimators=500, resample_track=True, random_state=1)
    model.fit(X, y)
    K = model.forest_kernel(X_test)               # (n, n) co-occurrence counts
    Kc = model.forest_kernel(X_test, X)           # cross kernel
    Kt = model.forest_kernel(X_test, X, vs_train=True)  # needs resample_track=True

    # OOB self-kernel (RLT >= 6.1.0): co-occurrence only from trees where
    # both observations were excluded from fitting that tree
    oob = model.forest_kernel(X, oob=True)
    oob["Kernel"]  # (n, n) normalized co-occurrence in [0, 1]
    ```

## Where to go next

- The [Get Started](tutorials/get-started.md) page covers installation details.
- Tutorials: [regression](tutorials/regression-tutorial.md),
  [classification](tutorials/classification-tutorial.md),
  [survival](tutorials/survival-tutorial.md).
- Feature articles: [variable importance](articles/variable-importance.md),
  [linear combinations](articles/linear-combinations.md),
  [survival confidence bands](articles/confidence-interval.md),
  [forest kernel](articles/feature-kernel.md),
  [single tree](articles/feature-single-tree.md), and
  [seeds & reproducibility](articles/feature-seed.md).
- The [API reference](reference/rlt_reg.md) documents every public class and
  function.

## Citation

If you use RLT methods, please cite the corresponding R-side papers:
Zhu, R., Zeng, D., & Kosorok, M. R. (2015) *Reinforcement Learning Trees*,
JASA; and follow-ups on variance estimation and survival bands.

## License

GPL (>= 3), inherited from the RLT R package.
