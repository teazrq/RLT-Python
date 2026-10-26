# R functionality and validation

Python provides the supported RLT workflows through separate regression,
classification, and survival estimators. The table identifies public entry
points and the tests that exercise them. A passing case establishes the
tested behavior; it does not establish bitwise identity across platforms
or statistical confidence-band coverage.

| R workflow | Python entry point | Validation in `tests/` |
|---|---|---|
| Regression, classification, right-censored survival | `RLT_reg`, `RLT_cla`, `RLT_surv` | `test_r_parity.py`, `test_alignment_r_workflows.py` |
| Ordinary and reinforced fitting; embedded controls | Constructor controls, `fit` | `test_alignment_api.py`, regression/classification/survival LC tests, core parity tests |
| Linear-combination methods and survival split rules | `linear_comb_method`, `split_rule` | `test_alignment_api.py`, `test_alignment_native_corrections.py` |
| Observation and variable weights | `fit(sample_weight=..., var_prob=...)` | Native correction tests, core parity tests, `test_parity_core_corrections.py` |
| Labeled and numerical predictors; declared class order | Optional pandas inputs, `categorical_features`, `classes_` | `test_alignment_api.py`, `test_alignment_r_workflows.py` |
| Bootstrap, subsampling and custom signed sampling | Sampling constructor controls, `fit(resample_preset=...)` | API tests, R workflow comparisons, native OOB-count tests |
| Predictions, probabilities and hazards | `predict`, `predict_proba`, `predict_details` | R workflow comparisons, `test_alignment_public_workflows.py` |
| Individual-tree predictions and exact survival grids | `predict_all_trees`, `predict_details(keep_all=True)` | Public workflow tests, `test_parity_survival_corrections.py` |
| OOB prediction and errors | `oob_details`, `oob_*` attributes | Public workflow tests, native OOB-count tests, R workflow comparisons |
| Matched, IJ and jackknife uncertainty | `predict_var`, `predict_details(var_est=True)` | `test_r_uncertainty_parity.py`, core parity tests |
| Permutation and distributed importance; matched VI variance | `importance_table`, `rlt.importance` | `test_r_importance.py`, R workflow and core parity comparisons |
| Self, cross, training and OOB kernels | `forest_kernel` | `test_alignment_r_workflows.py`, `test_kernel_oob.py` |
| Raw trees and readable node summaries | `get_one_tree`, `tree_summary`, `print_tree` | `test_alignment_inspection.py`, public workflow tests |
| Standalone concordance | `rlt.cindex` | Public workflow tests, native concordance fixtures |
| Bands from saved CHF and covariance | `get_surv_band(prediction)` | Survival alignment fixtures and cached-result tests |
| Naive and R-compatible smoothed bands | `get_surv_band(approach=...)` | Frozen mgcv 1.9-4 fixtures, common-draw and Monte Carlo tests |
| Fitted-model summary and persistence | `summary`, pickle, joblib | Public workflow and independent inspection tests, including fresh-process reload |
| Reproducible seeds and parallel execution | `random_state`, `n_jobs` | API seed tests, native bounded subprocess tests, core parity comparisons |

Quantile forests are outside this comparison because they are not a
supported public workflow of the current R package.

The R reference revision and source fingerprints are recorded in
`tools/r-reference.json`. The CI job for Python 3.12 installs that exact R
revision and runs live comparisons. Other supported Python jobs run the
ordinary suite and saved numerical fixtures, with live-R skips visible in
their logs. Locally, set `RLT_RSCRIPT` and, if needed, `R_LIBS_USER` to shared
installations before running `python -m pytest tests/ -q -rs`.

See [Python and R differences](r-parity.md) for array layouts, validation
choices, category ordering, survival estimands, and random-stream limits.
