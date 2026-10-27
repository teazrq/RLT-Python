# RLT for Python

Fit random forests for continuous responses, class labels, and right-censored
survival times with a scikit-learn-style interface. `rlt` brings the tree
algorithms of the [RLT R package](https://github.com/teazrq/RLT) to Python.

For installation, examples, and the R interface, see the
[R version documentation](https://teazrq.github.io/RLT/).

The default forest uses one variable at each split. Optional **reinforcement**
uses embedded forests to guide variable selection. **Linear-combination splits**
split along a direction formed from several predictors and can be used with or
without reinforcement.

[Get started: install and fit a forest](tutorials/get-started.md){ .md-button .md-button--primary }

## Install from source

These guides describe the GitHub source version, **0.3.0**. Install it from
[the source repository](https://github.com/teazrq/RLT-Python), using the
[installation instructions](tutorials/get-started.md#install-the-rlt-package)
for shared Python dependencies and system libraries. The standard build targets
Linux; Windows and macOS need additional compiler and linker configuration.

## Predict a continuous response

After installation, fit on 160 simulated observations and predict the 40 held
out before fitting. The response depends on the first two predictors plus noise.

```python
import numpy as np
from rlt import RLT_reg

rng = np.random.default_rng(1)
X = rng.normal(size=(200, 6))
y = X[:, 0] - X[:, 1] + rng.normal(size=200)

model = RLT_reg(
    n_estimators=100, mtry=3, nsplit=0,
    resample_replace=True, resample_prob=0.8,
    random_state=1, n_jobs=1,
).fit(X[:160], y[:160])
pred = model.predict(X[160:])

print(f"OOB MSE: {model.oob_error_:.3f}")
print(f"Test MSE: {np.mean((pred - y[160:]) ** 2):.3f}")
```

```text
OOB MSE: 1.401
Test MSE: 1.847
```

Both numbers are mean squared prediction errors; smaller is better. Each
out-of-bag (OOB) prediction uses only trees that omitted that training row.
Test MSE evaluates the separate 40-row test set. Their difference here is one
small-sample result, not evidence that one evaluation method is generally better.
The [first-analysis walkthrough](tutorials/get-started.md#fit-and-evaluate-a-regression-forest)
also shows individual predictions and explains the fitting controls.

## Choose your analysis

| Task | Start here | Main result |
|---|---|---|
| Predict a continuous response | [Regression](tutorials/regression-tutorial.md) | Predicted response means and OOB/test MSE |
| Predict a class | [Classification](tutorials/classification-tutorial.md) | Class labels and probabilities in `classes_` order |
| Model a censored event time | [Survival](tutorials/survival-tutorial.md) | Conditional survival curves, cumulative hazards, and OOB concordance error |

## Understand and inspect a fitted forest

- [Variable importance](articles/variable-importance.md): measure the change in
  OOB predictive loss when a variable is perturbed.
- [Linear combinations](articles/linear-combinations.md): allow up to several
  predictors in a split direction.
- [Survival confidence bands](articles/confidence-interval.md): construct nominal
  simultaneous bands separately for each target curve over its returned time grid.
- [Forest kernels](articles/feature-kernel.md): inspect which observations share
  terminal nodes, with separate ordinary, training, and OOB calculations.
- [Individual trees](articles/feature-single-tree.md): read named splits and
  terminal predictions with `tree_summary()`.
- [Model workflows](articles/model-workflows.md): work with labeled data, inspect
  OOB availability, request prediction uncertainty, and save fitted models.

See [reproducibility](articles/feature-seed.md) to control simulation and forest
randomness, or [comparing Python and R](reference/r-parity.md) for numerical
agreement, source matching, and interface differences.

## Citations and license

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
[`get_surv_band` reference](reference/get_surv_band.md)
for the current smoothing and covariance handling.

RLT for Python is distributed under GPL (>= 3), inherited from the R package.

[![CI](https://github.com/teazrq/RLT-Python/actions/workflows/ci.yml/badge.svg)](https://github.com/teazrq/RLT-Python/actions/workflows/ci.yml)
[![Docs](https://github.com/teazrq/RLT-Python/actions/workflows/docs.yml/badge.svg)](https://github.com/teazrq/RLT-Python/actions/workflows/docs.yml)
