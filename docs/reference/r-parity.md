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
| `n_estimators=None` resolves to 500 ordinarily or 100 with reinforcement | 500 ordinarily; 100 when reinforcement is explicitly enabled |
| Optional pandas categorical, ordered, and string columns; zero-based numeric category codes also supported | Factor, ordered-factor, character columns, or numerical training codes |
| Regression predictions have shape `(n,)`; class predictions use the original labels | Regression predictions have shape `n x 1`; class predictions are factors |
| Survival covariance has shape `(n, T, T)` | CHF covariance has shape `T x T x n` |
| Separate `predict_var` method, which requires a supported uncertainty setup | `predict(..., var.est=TRUE)` |

`predict_details` provides complete public predictions, including class
order, per-tree arrays, the exact survival grid, and uncertainty mode.
Models support pickle/joblib, signed sampling presets, and labeled OOB
inspection with explicit availability. See [model workflows](../articles/model-workflows.md).

Python deliberately rejects fractional sampling counts and noncanonical
primary boolean options instead of silently converting them. It also
rejects missing or infinite concordance times. Declared categorical
response levels retain their order; ordinary label arrays use sorted
observed labels. Character sorting can depend on R's locale, so use
explicit category levels for comparisons involving arbitrary text.

Both implementations preserve the full failure-time grid while fitting.
Prediction on a reduced grid selects from complete cumulative-hazard
curves. Ordinary survival forests return `exp(-CHF)`; linear combination
forests average the tree product-limit survival curves. Covariance
remains on the CHF scale; confidence bands use `exp(-CHF)` as their
survival reference in both cases.

## Smoothed confidence bands

Python's `approach="smoothed"` follows the narrow Gaussian `mgcv` 1.9-4
calculation used by RLT: natural cubic regression spline tensor margins
on grid indices, integrated-curvature penalties, two GCV-selected
smoothing parameters, and the `magic` Newton/descent search. It runs
without R. Frozen R fixtures compare the design, penalties, fixed and
optimized fits, and final marginal scales across grid sizes, ranks,
covariance scales, and singular cases. This replaces the earlier Python
actual-time B-spline approximation.

This is algorithmic alignment with the pinned smoother, with numerical
tolerances for floating-point libraries. It does not imply bit-identical
outputs across all BLAS/LAPACK versions or unique smoothing parameters
when GCV is flat. Exact binary input and one-ULP perturbation controls
show that R's final automatic selection and convergence flag can change
near a numerical GCV minimum. Tests use that measured reference
sensitivity for those cases, retain stricter fixed-parameter checks, and
include a frozen independent holdout corpus. Diagnostics include the
selected parameters, EDF, and convergence information. Different Gaussian random streams also mean
that equal integer seeds need not produce equal Monte Carlo band limits.

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
