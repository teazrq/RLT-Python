# Random Seed and Reproducibility

This page explains how to control randomness in your analyses with `rlt`'s
`random_state`, the Python counterpart of RLT's `seed` argument.
Data simulation has a separate NumPy seed.

## Where to set the seed

With `rlt`, randomness lives in two places: your data simulation (numpy)
and the forest's internal resampling (`random_state` on the estimator).

- For **fully reproducible data + model**, seed your numpy Generator at the
  top of the script and pass a fixed `random_state` to each estimator.
- If your data are fixed and only modeling randomness matters, only the
  estimator's `random_state` needs to be fixed.

**Prerequisites** — See [Get Started](../tutorials/get-started.md).

## Demonstration — same seed, same results (regression)

We run the same pipeline twice with the same seed and compare outputs.

```python
import numpy as np
from rlt import RLT_reg

# Shared forest configuration
ntrees = 200
nmin = 5
mtry = 5
samplereplace = True
sampleprob = 0.80
nsplit = 0

def run_pipeline(seed):
    """Simulate data and fit a forest, all from one seed."""
    # Simulate data using this run's seed.
    rng = np.random.default_rng(seed)

    # Small dataset (~100 obs)
    trainn = 80
    testn = 20
    n = trainn + testn
    p = 10
    X1 = rng.normal(size=(n, p // 2))
    X2 = (rng.uniform(size=(n, p // 2)) * 3).astype(int)  # integers 0,1,2
    X = np.hstack([X1, X2])

    y = (
        1
        + X[:, 1:6].sum(axis=1)
        + 2 * np.isin(X[:, p // 2], [1, 2])
        + rng.normal(size=n)
    )

    trainX, trainY = X[:trainn], y[:trainn]
    testX, testY = X[trainn:], y[trainn:]

    # Fit
    fit = RLT_reg(
        n_estimators=ntrees,
        mtry=mtry,
        min_samples_leaf=nmin,
        resample_prob=sampleprob,
        resample_replace=samplereplace,
        nsplit=nsplit,
        importance="permute",
        categorical_features=range(p // 2, p),
        n_jobs=1,
        random_state=seed,
    )
    fit.fit(trainX, trainY)
    pred = fit.predict(testX)

    return (
        float(np.mean((fit.oob_prediction_ - trainY) ** 2)),
        float(np.mean((pred - testY) ** 2)),
        pred,
    )

# ---------- Run A (seed = 1) ----------
mse_oob_A, mse_test_A, pred_A = run_pipeline(1)

# ---------- Run B (same seed = 1) ----------
mse_oob_B, mse_test_B, pred_B = run_pipeline(1)

# ---------- Summary for same-seed runs ----------
print(f"A OOB MSE:  {mse_oob_A:.6f}")
print(f"A test MSE:  {mse_test_A:.6f}")
print(f"B OOB MSE:  {mse_oob_B:.6f}")
print(f"B test MSE:  {mse_test_B:.6f}")
print(f"Same-seed predictions identical: {np.array_equal(pred_A, pred_B)}")
```

```
A OOB MSE:  2.416684
A test MSE:  4.664737
B OOB MSE:  2.416684
B test MSE:  4.664737
Same-seed predictions identical: True
```

## Demonstration — different seed, potentially different results

Now we change the seed and rerun the same pipeline once.

```python
# ---------- Run C (seed = 2) ----------
mse_oob_C, mse_test_C, pred_C = run_pipeline(2)

print(f"C OOB MSE:  {mse_oob_C:.6f}")
print(f"C test MSE:  {mse_test_C:.6f}")
print(f"Diff-seed predictions equal to Run A: {np.array_equal(pred_C, pred_A)}")
```

```
C OOB MSE:  2.777294
C test MSE:  2.199118
Diff-seed predictions equal to Run A: False
```

## Tips

- Choose any integer you like for the seed; the specific value doesn't matter—consistency does.
- Keep one seeded `np.random.default_rng(...)` near the top of your script to make the whole workflow reproducible.
- The same pattern works for classification and survival: place the data-simulation seed before model fitting and give each estimator a fixed `random_state`.
- `random_state` also accepts a NumPy `RandomState` or `Generator` instance; a fresh seed is drawn from it for each forest. Use a separate generator for data simulation.
- When `random_state=None`, fitting draws from NumPy's global RNG, so
  `np.random.seed(...)` controls omitted forest seeds. This does not make
  NumPy's surrounding random stream match R's.

## Cross-language note

In this Python port, `random_state` maps directly to the C++ master seed
(the same argument R's `RLT()` calls `seed`), and numpy handles your data
simulation separately. For cross-language comparisons, supply the same
data, forest seed, and explicit parameter values. Matching a surrounding
`set.seed(...)` call to a NumPy data-generation seed does not produce the
same data. Numerical libraries and source revisions can also affect
results; cross-platform bitwise identity is not promised.
