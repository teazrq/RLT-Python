# Changes

## 0.3.0

- Align omitted tree counts with R: 100 with reinforcement, 500 otherwise.
  `n_estimators=None` retains the omitted value until fitting. Explicit
  counts remain unchanged, subject to matched sampling's even-count rule.
- Support pandas categorical predictors and responses with fitted label
  dictionaries and declared class order, while keeping pandas optional.
- Add signed `resample_preset` fitting, complete `predict_details` outputs,
  per-tree predictions across all families, labeled OOB outputs with an
  availability mask, standalone `cindex`, and readable tree inspection.
- Make fitted estimators saveable with pickle and joblib. Extracted trees
  are independent copies and include model-specific node payloads.
- Accept numeric survival split-rule codes and integral numeric LC methods.
  Reject response matrices, fractional sampling counts, and noncanonical
  primary boolean options. Omitted seeds follow NumPy's global RNG.
- Correct shared native handling of insufficient variable-sampling support,
  zero-weight SIR slices, and unweighted LC supremum-logrank splitting.
- Allow survival bands from saved CHF/covariance predictions and compatible
  prediction-time uncertainty modes. Use the R-compatible `mgcv` 1.9-4
  tensor smoother for `approach="smoothed"`, replacing the previous
  approximation. See the confidence-band reference
  for simulation and coverage limitations.
