# Random Forest Kernel

This page shows how to compute and visualize the random forest kernel
induced by an `rlt` forest via `forest_kernel()` — the Python port of the
RLT R package's forest kernel article.

## Example 1 - Axis-aligned

We build a regression forest using "best" splits and compute kernel
weights.

```python
import numpy as np
from rlt import RLT_reg

# ---- Generate a small dataset (about 100 obs) ----
rng = np.random.default_rng(1)
n = 120
p = 5
X = rng.uniform(size=(n, p))
y = X[:, 0] + X[:, 1] + rng.normal(size=n)

# Train the forest (axis-aligned "best" splits)
fit_rect = RLT_reg(
    n_estimators=300, mtry=p, min_samples_leaf=5,
    nsplit=0,  # "best" splits: consider all cut points
    resample_prob=0.8, resample_replace=True,
    importance="permute",
    resample_track=True,
    n_jobs=1,
    random_state=1,
)
fit_rect.fit(X, y)

# Choose a target point
newX = np.full((1, p), 0.5)
newX[0, 1] = 0.3

# Forest kernel weights between newX and all X
KW_rect = fit_rect.forest_kernel(newX, X)

# A simple size mapping for visualization (normalized)
size_scale = 10 * np.sqrt(KW_rect[0] / np.sqrt((KW_rect ** 2).sum()))
```

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# X0 vs X1 (both are signal features)
axes[0, 0].scatter(X[:, 0], X[:, 1], s=8, color="deepskyblue")
axes[0, 0].scatter(X[:, 0], X[:, 1], s=size_scale, color="darkorange",
                   alpha=0.6, linewidths=0.5, edgecolors="black")
axes[0, 0].scatter(newX[0, 0], newX[0, 1], marker="x", s=100,
                   color="black", linewidths=3)
axes[0, 0].set_title("Rectangular kernel - X1 vs X2")

# X0 vs X2 (X3 is noise)
axes[0, 1].scatter(X[:, 0], X[:, 2], s=8, color="deepskyblue")
axes[0, 1].scatter(X[:, 0], X[:, 2], s=size_scale, color="darkorange",
                   alpha=0.6, linewidths=0.5, edgecolors="black")
axes[0, 1].scatter(newX[0, 0], newX[0, 2], marker="x", s=100,
                   color="black", linewidths=3)
axes[0, 1].set_title("Rectangular kernel - X1 vs X3")

# Compute kernel inside the original training forest (vs_train=True)
KW_rect_train = fit_rect.forest_kernel(newX, X, vs_train=True)
size_scale_train = 10 * np.sqrt(
    KW_rect_train[0] / np.sqrt((KW_rect_train ** 2).sum()))

axes[1, 0].scatter(X[:, 0], X[:, 1], s=8, color="deepskyblue")
axes[1, 0].scatter(X[:, 0], X[:, 1], s=size_scale_train, color="darkorange",
                   alpha=0.6, linewidths=0.5, edgecolors="black")
axes[1, 0].scatter(newX[0, 0], newX[0, 1], marker="x", s=100,
                   color="black", linewidths=3)
axes[1, 0].set_title("Rectangular kernel (vs_train=True) - X1 vs X2")

# Another projection
axes[1, 1].scatter(X[:, 1], X[:, 2], s=8, color="deepskyblue")
axes[1, 1].scatter(X[:, 1], X[:, 2], s=size_scale, color="darkorange",
                   alpha=0.6, linewidths=0.5, edgecolors="black")
axes[1, 1].scatter(newX[0, 1], newX[0, 2], marker="x", s=100,
                   color="black", linewidths=3)
axes[1, 1].set_title("Rectangular kernel - X2 vs X3")

plt.tight_layout()
plt.show()
```

Note how the kernel concentrates on points close to the target along the
**signal** variables (X1, X2) but spreads over the whole range of the
noise variable (X3) — the forest has learned which dimensions matter.

## Example 2 - Linear-combination

We build a regression forest using random linear-combination splits and
compute kernel weights.

```python
rng = np.random.default_rng(1)
n = 120
p = 3
X = rng.uniform(size=(n, p))
y = X[:, 0] + X[:, 2] + 0.3 * rng.normal(size=n)

fit_lc = RLT_reg(
    n_estimators=300, mtry=p, min_samples_leaf=5,
    nsplit=3,
    resample_prob=0.9, resample_replace=False,
    linear_comb=2,
    linear_comb_method="sir",
    importance="permute",
    n_jobs=1,
    random_state=1,
)
fit_lc.fit(X, y)

newX = np.full((1, p), 0.5)
newX[0, 1] = 0.3
KW_lc = fit_lc.forest_kernel(newX, X)
size_scale_lc = 10 * np.sqrt(KW_lc[0] / np.sqrt((KW_lc ** 2).sum()))
```

```python
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(X[:, 0], X[:, 2], s=8, color="deepskyblue")
ax.scatter(X[:, 0], X[:, 2], s=size_scale_lc, color="darkorange",
           alpha=0.6, linewidths=0.5, edgecolors="black")
ax.scatter(newX[0, 0], newX[0, 2], marker="x", s=100, color="black",
           linewidths=3)
ax.set_title("Linear-combination kernel - X1 vs X3")
plt.show()
```

## Example 3 - OOB self-kernel

The plain self-kernel counts leaf co-occurrence over **all** trees, but
in-bag observations influence the tree structure (and thus their own leaf
assignment) — this self-contamination biases the kernel when it is used
for degrees-of-freedom estimation. The OOB self-kernel (ported from RLT
6.1.0's `forest.kernel(oob = TRUE)`) counts co-occurrence only from trees
where **both** observations are out-of-bag. It requires
`resample_track=True` at fit time and the original training data as `X1`.

```python
import numpy as np
from rlt import RLT_reg

rng = np.random.default_rng(1)
n, p = 120, 5
X = rng.normal(size=(n, p))
y = X[:, 0] + X[:, 1] + rng.normal(size=n)

fit = RLT_reg(
    n_estimators=300, nsplit=2,
    resample_track=True,  # required for the OOB kernel
    n_jobs=1, random_state=1,
)
fit.fit(X, y)

out = fit.forest_kernel(X, oob=True)
K, N, C = out["Kernel"], out["N"], out["C"]
# K = C / N where N > 0, and 0 otherwise
```

`N[i, j]` counts trees where observations `i` and `j` are both
out-of-bag; `C[i, j]` counts those trees where they also land in the same
terminal node; `Kernel = C / N` normalizes them into `[0, 1]`. Pairs that
are never OOB together (`N = 0`) get `Kernel = 0`.

```python
print(K.shape)    # (120, 120)
print(N[0, 1], C[0, 1])   # 48 1
print(np.trace(K))        # 120.0 — diagonal is 1 wherever N > 0
```

Use `N` to distinguish an unsupported pair (`N = 0`) from an observed
zero leaf-sharing count. These matrices describe the fitted forest;
using them in a degrees-of-freedom estimator requires that estimator's
own assumptions.

## Example 4 - Linear-combination (SIR, no embedded model)

Fit a model using linear-combination splits without an embedded forest;
variables are ranked by marginal screening. This corresponds to R's
`MASS::mvrnorm` example, replaced here by `numpy`.

```python
n = 300
p = 5
Sigma = np.full((p, p), 0.3)
np.fill_diagonal(Sigma, 1.0)
Sigma[0, 4] = Sigma[4, 0] = 0.9
Sigma[0, 2] = Sigma[2, 0] = Sigma[4, 2] = Sigma[2, 4] = -0.3

X = rng.multivariate_normal(np.zeros(p), Sigma, size=n)
y = 1 + X[:, 0] + X[:, 2] + 0.5 * X[:, 4] ** 2 + rng.normal(size=n)
w = rng.uniform(size=n)

RLTfit = RLT_reg(
    n_estimators=100, min_samples_leaf=20, mtry=3,
    nsplit=3,
    resample_prob=0.8, resample_replace=False,
    linear_comb=3,
    linear_comb_method="sir",
    importance="permute",
    n_jobs=1,
    random_state=1,
)
RLTfit.fit(X, y, sample_weight=w)

pred = RLTfit.predict(X)
print(f"Training MSE: {np.mean((pred - y) ** 2):.4f}")
```

```
Training MSE: 0.9073
```

```python
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(pred, y, s=12, alpha=0.7)
ax.set_xlabel("Predicted")
ax.set_ylabel("Observed")
ax.set_title("SIR linear-combination forest: predicted vs observed")
plt.show()
```
