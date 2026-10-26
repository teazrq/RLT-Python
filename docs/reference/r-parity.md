# Comparing Python and R results

The Python package shares RLT's tree algorithms and adds a scikit-learn
interface. The current native source was synchronized with the corrected
RLT 6.1.1 source tree. It includes the fixes for IJ count moments,
weighted fitting, OOB kernels, and survival prediction grids.

For a numerical comparison, use the same input data, column order,
explicit forest parameters, and integer forest seed. Python's
`random_state` corresponds to `RLT(seed=...)`. Seeding the surrounding
NumPy and R data generators does not produce the same simulated data.
Source revisions and numerical libraries also matter; a shared package
version number alone does not establish an exact source match.

The tests compare fitted predictions and tracking records with an
installed R package, and use small saved-tree fixtures to check variance
and survival calculations directly. They check numerical behavior, not
statistical coverage or a general predictive advantage.

## Interface differences

| Python | R |
|---|---|
| Separate `RLT_reg`, `RLT_cla`, and `RLT_surv` estimators | `RLT(..., model=...)` |
| 500 trees by default, including reinforcement | 500 ordinarily; 100 when reinforcement is explicitly enabled |
| Numeric category codes starting at 0, declared with `categorical_features` | Factor columns or the documented numeric category representation |
| Regression predictions have shape `(n,)`; class predictions use the original labels | Regression predictions have shape `n x 1`; class predictions are factors |
| Survival covariance has shape `(n, T, T)` | CHF covariance has shape `T x T x n` |
| Separate `predict_var` method, which requires a supported uncertainty setup | `predict(..., var.est=TRUE)` |

Both implementations preserve the full failure-time grid while fitting.
Prediction on a reduced grid selects from complete cumulative-hazard
curves. Ordinary survival forests return `exp(-CHF)`; linear combination
forests average the tree product-limit survival curves. Covariance
remains on the CHF scale; confidence bands use `exp(-CHF)` as their
survival reference in both cases.

## Smoothed confidence bands

Python's `approach="smoothed"` is an alternative approximation. It fits
a penalized B-spline surface on the actual failure times with an
approximate GCV smoothing choice. R uses an `mgcv` GAM on grid-index
coordinates. Their smoothed marginal scales and band limits need not
agree. The Python method is retained explicitly as an alternative, not
presented as an exact implementation of the R smoother.

Both approaches use the supplied CHF covariance for Gaussian simulation;
smoothing changes marginal standardizing scales. Bands have nominal
simultaneous scope over the returned grid separately for each query
point. They do not provide joint coverage across subjects or predict an
individual's event time.

## Re-running comparisons

Install the intended R checkout in a shared R library. Put `Rscript` on
PATH or set `RLT_RSCRIPT` to its executable path, then run the Python
tests. Use `R_LIBS_USER` to select an existing shared user library when
needed. R comparisons are skipped when Rscript is unavailable or
`RLT_SKIP_R_TESTS` is set; an ordinary Python-only CI pass therefore does
not establish cross-language agreement.
