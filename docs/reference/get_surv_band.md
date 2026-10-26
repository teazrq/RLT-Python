# get_surv_band

Nominal simultaneous bands for the conditional survival curve at each
query point, using a saved survival prediction or a fitted
[`RLT_surv`](rlt_surv.md). Coverage
is specified separately for each curve over the returned time grid.
These are not prediction intervals for an individual's event time.

The reference curve is `exp(-CHF)`, and the covariance is on the CHF
scale. For linear combination forests, the returned survival estimate is
the mean of tree product-limit curves; this band does not directly
estimate uncertainty in that different estimator.

```python
import numpy as np
from rlt import RLT_surv, get_surv_band

rng = np.random.default_rng(42)
n = 200
X = rng.normal(size=(n, 5))
t_event = rng.exponential(1 / np.exp(X[:, 0]))
t_censor = rng.uniform(0, 3, size=n)
y = np.empty(n, dtype=[("event", "?"), ("time", float)])
y["event"] = t_event <= t_censor
y["time"] = np.minimum(t_event, t_censor)

model = RLT_surv(n_estimators=1000, var_mode="matched", random_state=42)
model.fit(X, y)

X_query = X[:3]
prediction = model.predict_details(X_query, var_est=True)
band = get_surv_band(prediction, i=0, alpha=0.05, approach="smoothed")
band["lower"], band["upper"]  # (n, T) band limits
```

## Signature

```text
get_surv_band(
    model,          # saved prediction dictionary or fitted RLT_surv
    X=None,         # query covariates when supplying a model
    i=0,            # subject index: 0 = all, 1-based like R when positive
    alpha=0.05,     # level or array of levels
    approach="naive",
    nsim=5000,      # MC simulations for the critical value
    k_rank=10,      # rank truncation (smoothed approach)
    k_mode="fixed", # "fixed" or "proportion"
    k_prop=0.99,    # cumulative eigenvalue proportion (k_mode="proportion")
    seed=0,         # RNG seed for the MC band (deterministic given seed)
    band_grid_size=0,  # optional reduced time grid (see predict_var)
    var_mode=None,  # compatible override when supplying a model
)
```

### Parameters

| Parameter | Type / values | Default | Description |
|---|---|---|---|
| `model` | dict or fitted `RLT_surv` | required | Saved result with `CHF`, `Cov`, and `timepoints`, or a model supporting the requested covariance. |
| `X` | array-like `(n, p)` | None | Query covariates when supplying a model; omit with a saved prediction. |
| `i` | int | 0 | Subject selector: `0` = all subjects, `k` (1-based, like R) = subject `k` only. |
| `alpha` | float or array | 0.05 | Significance level(s); each must be in `[0, 0.5]`. |
| `approach` | "naive", "smoothed" | "naive" | Band construction; see below. |
| `nsim` | int | 5000 | Monte Carlo simulations for the critical value. |
| `k_rank` | int | 10 | Rank of the low-rank covariance truncation (smoothed). |
| `k_mode` | "fixed", "proportion" | "fixed" | Choose rank directly or by cumulative eigenvalue proportion. |
| `k_prop` | float | 0.99 | Target cumulative eigenvalue proportion when `k_mode="proportion"`. |
| `seed` | int | 0 | Seed for the Monte Carlo critical value. |
| `band_grid_size` | int | 0 | Select a reduced grid during model prediction (0 = full grid). Saved results use their stored grid. |
| `var_mode` | None, "matched", "IJ", "jack" | None | Prediction-time override for a model. Matched requires a matched fit; IJ/jack require retained resampling counts. Saved covariance cannot be overridden. |

### Approaches

- **"naive"**: marginal SDs `sqrt(diag(Cov))`; Monte Carlo band with the
  full covariance matrix.
- **"smoothed"**: natural cubic regression spline tensor smoothing of the
  covariance surface, rank-`k` PSD truncation, positive-spectrum residual
  correction, and eigenvalue-ratio weights. The narrow Gaussian smoother
  follows R's `mgcv` 1.9-4 `gam(z ~ te(i, j, k))` calculation on grid
  indices, including two curvature penalties, GCV selection, and the
  `magic` Newton/descent search. It runs through NumPy and SciPy without
  requiring R. Relabeling the time coordinates preserves the marginal
  scales. Grids with fewer than three times use the naive calculation
  with a warning.

This replaces the earlier Python actual-time B-spline approximation.
Smoothed band values can change when upgrading. Frozen R fixtures check
the basis, penalties, fitted covariance, and final scales across small
grids, ranks, covariance scales, and singular cases. Near a GCV minimum,
floating-point rounding can change the final parameter selection and its
convergence flag. Tests measure this sensitivity against R itself and
retain stricter checks with fixed smoothing parameters. For flat GCV
objectives, smoothing parameters and EDF are not uniquely determined by
the fitted surface.

Both methods simulate from the supplied CHF covariance. Smoothing changes
the marginal standardizing scales. `nsim` controls Monte Carlo precision
of the critical value; increasing it does not establish coverage.
Monte Carlo draws use Python's seeded stream and need not match R for
the same integer seed. Covariances must be positive semidefinite; only
negative eigenvalues within R/Armadillo's scale-relative roundoff tolerance
are clipped.

### Returns

A dict with:

| Key | Shape | Description |
|---|---|---|
| `lower` / `upper` | `(n_selected, T)` or `(n_selected, n_alpha, T)` | Limits for one or multiple alpha levels, respectively. |
| `timepoints` | `(T,)` | The time grid used. |
| `diagnostics` | list of dicts | Per-subject SDs, variance-explained, rank/eigenvalue info. |
| `marsd` | `(n_selected, T)` | Final marginal standardizing scales. |
| `alpha` | `(n_alpha,)` | Requested noncoverage levels. |
| `subjects` | `(n_selected,)` | Selected 1-based subject numbers in the prediction. |

Smoothed diagnostics include the selected smoothing parameters, EDF,
GCV score, numerical rank, iterations, and convergence flag under
`diagnostics[j]["smoother"]`. A flat or non-improving GCV search should
be assessed alongside the fitted scales; its smoothing parameters alone
do not measure uncertainty.

## Examples

```python
import numpy as np
from rlt import RLT_surv, get_surv_band

rng = np.random.default_rng(0)
n = 200
X = rng.normal(size=(n, 5))
t_event = rng.exponential(1 / np.exp(X[:, 0]))
t_censor = rng.uniform(0, 3, size=n)
y = np.empty(n, dtype=[("event", "?"), ("time", float)])
y["event"] = t_event <= t_censor
y["time"] = np.minimum(t_event, t_censor)

model = RLT_surv(n_estimators=1000, var_mode="matched", random_state=0)
model.fit(X, y)

# Naive band for the first subject
X_query = X[:3]
band_naive = get_surv_band(model, X_query, i=1, alpha=0.05,
                           approach="naive", nsim=5000)

# Smoothed band with rank chosen by eigenvalue proportion
prediction = model.predict_details(X_query, var_est=True, band_grid_size=20)
band_smooth = get_surv_band(prediction, i=0, alpha=[0.05, 0.10],
                            approach="smoothed", nsim=5000,
                            k_mode="proportion", k_prop=0.95)

# A tracking-only fit can request IJ covariance at prediction time.
tracked = RLT_surv(n_estimators=1000, resample_track=True, random_state=0)
tracked.fit(X, y)
band_ij = get_surv_band(tracked, X_query, var_mode="IJ",
                        band_grid_size=20, nsim=1000)
```

A saved dictionary uses arrays with exact axes: `CHF` is `(n, T)`,
`Cov` is `(n, T, T)`, and `timepoints` is `(T,)`. The helper validates
their dimensions and finite values, preserves the stored time vector,
and does not modify the input arrays or recompute forest predictions.
Select the grid and covariance mode before caching the result.

## See also

- [Survival confidence bands article](../articles/confidence-interval.md)
- [`RLT_surv`](rlt_surv.md)
