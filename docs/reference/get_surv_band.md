# get_surv_band

Nominal simultaneous bands for the conditional survival curve at each
query point, using a variance-enabled [`RLT_surv`](rlt_surv.md). Coverage
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
band = get_surv_band(model, X_query, i=0, alpha=0.05, approach="smoothed")
band["lower"], band["upper"]  # (n, T) band limits
```

## Signature

```python
get_surv_band(
    model,          # fitted RLT_surv with var_mode != "none"
    X_query,        # covariates for three query points
    i=0,            # subject index: 0 = all, 1-based like R when positive
    alpha=0.05,     # level or array of levels
    approach="naive",
    nsim=5000,      # MC simulations for the critical value
    k_rank=10,      # rank truncation (smoothed approach)
    k_mode="fixed", # "fixed" or "proportion"
    k_prop=0.99,    # cumulative eigenvalue proportion (k_mode="proportion")
    seed=0,         # RNG seed for the MC band (deterministic given seed)
    band_grid_size=0,  # optional reduced time grid (see predict_var)
)
```

### Parameters

| Parameter | Type / values | Default | Description |
|---|---|---|---|
| `model` | fitted `RLT_surv` | — | Must have been fitted with `var_mode != "none"`. |
| `X` | array-like `(n, p)` | — | Covariates to compute bands for. |
| `i` | int | 0 | Subject selector: `0` = all subjects, `k` (1-based, like R) = subject `k` only. |
| `alpha` | float or array | 0.05 | Significance level(s); each must be in `[0, 0.5]`. |
| `approach` | "naive", "smoothed" | "naive" | Band construction; see below. |
| `nsim` | int | 5000 | Monte Carlo simulations for the critical value. |
| `k_rank` | int | 10 | Rank of the low-rank covariance truncation (smoothed). |
| `k_mode` | "fixed", "proportion" | "fixed" | Choose rank directly or by cumulative eigenvalue proportion. |
| `k_prop` | float | 0.99 | Target cumulative eigenvalue proportion when `k_mode="proportion"`. |
| `seed` | int | 0 | Seed for the Monte Carlo critical value. |
| `band_grid_size` | int | 0 | Pass a reduced time grid through to `predict_var` (0 = full grid). |

### Approaches

- **"naive"**: marginal SDs `sqrt(diag(Cov))`; Monte Carlo band with the
  full covariance matrix.
- **"smoothed"**: tensor-product penalized B-spline smoothing of the
  covariance surface, rank-`k` PSD truncation, positive-spectrum residual
  correction, and eigenvalue-ratio weights. Python's smoother uses actual
  failure times and an approximate GCV smoothing choice. It is an
  alternative to R's index-coordinate `mgcv` GAM, so the two smoothed
  bands need not agree numerically. Grids too short for smoothing fall
  back to the raw marginal scales with a warning.

Both methods simulate from the supplied CHF covariance. Smoothing changes
the marginal standardizing scales. `nsim` controls Monte Carlo precision
of the critical value; increasing it does not establish coverage.

### Returns

A dict with:

| Key | Shape | Description |
|---|---|---|
| `lower` / `upper` | `(n_selected, T)` or `(n_selected, n_alpha, T)` | Limits for one or multiple alpha levels, respectively. |
| `timepoints` | `(T,)` | The time grid used. |
| `diagnostics` | list of dicts | Per-subject SDs, variance-explained, rank/eigenvalue info. |
| `marsd` | `(n_selected, T)` | Final marginal standardizing scales. |

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
band_smooth = get_surv_band(model, X_query, i=0, alpha=0.05,
                            approach="smoothed", nsim=5000,
                            k_mode="proportion", k_prop=0.95)
```

## See also

- [Survival confidence bands article](../articles/confidence-interval.md)
- [`RLT_surv`](rlt_surv.md)
