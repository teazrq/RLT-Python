# Single Tree

This page shows how to inspect a single tree grown inside an `rlt` forest
using `get_one_tree()`. For a readable table, use `print_tree(0)` or retrieve
its columns with `tree_summary(0)`.

## Data

We generate continuous and categorical predictors with a continuous outcome.

```python
import numpy as np

# (Optional) For reproducibility in this tutorial only.
rng = np.random.default_rng(1)

# ---- Generate a small synthetic dataset ----
trainn = 80
testn = 20
n = trainn + testn
p = 10

# Continuous + categorical predictors (last half as categorical)
X1 = rng.normal(size=(n, p // 2))
X2 = (rng.uniform(size=(n, p // 2)) * 3).astype(int)  # integers 0, 1, 2
X = np.hstack([X1, X2])

# Continuous outcome with a simple signal + noise
y = (
    1
    + X[:, 1:6].sum(axis=1)
    + 2 * np.isin(X[:, p // 2], [1, 2])
    + rng.normal(size=n)
)

# Train / test split
trainX, trainY = X[:trainn], y[:trainn]
testX, testY = X[trainn:], y[trainn:]
```

## Fit

```python
from rlt import RLT_reg

ntrees = 200
nmin = 5
mtry = p // 2
samplereplace = True
sampleprob = 0.80
rule = "best"
nsplit = 0 if rule == "best" else 3
importance = "permute"

fit = RLT_reg(
    n_estimators=ntrees,
    mtry=mtry,
    min_samples_leaf=nmin,
    resample_prob=sampleprob,
    resample_replace=samplereplace,
    nsplit=nsplit,
    importance=importance,
    n_jobs=1,
    random_state=1,
)
fit.fit(trainX, trainY)
```

## Inspect one tree

Use `get_one_tree(tree_id)` to inspect a single tree.
Choose `tree_id` from 0 to `n_estimators - 1` (0-indexed, unlike R's
1-based indexing). Here we take the first tree:

```python
tree = fit.get_one_tree(0)
tree.keys()
```

```
dict_keys(['SplitVar', 'SplitValue', 'LeftNode', 'RightNode', 'NodeWeight', 'NodeAve'])
```

The arrays are independent copies aligned by node index. Classification
also returns `NodeProb`; survival returns the nested `NodeHaz` arrays;
linear-combination trees include `SplitLoad`. Editing an extracted tree
does not change predictions from the fitted model.

| Array | Meaning |
|-------|---------|
| `SplitVar` | Variable index split at this node; -1 for terminal nodes |
| `SplitValue` | Threshold: points with `x[SplitVar] <= SplitValue` go left |
| `LeftNode` / `RightNode` | Child node indices |
| `NodeWeight` | Number of training observations reaching the node |
| `NodeAve` | Average response in the node (regression forests) |

```python
n_nodes = len(tree["SplitVar"])
n_splits = int((tree["SplitVar"] >= 0).sum())
print(f"{n_nodes} nodes, {n_splits} internal splits")
```

```
45 nodes, 22 internal splits
```

A compact text preview of the tree, in the spirit of R's `get.one.tree()`:

```python
def print_tree_preview(tree, max_lines=14):
    lines = []
    for i, var in enumerate(tree["SplitVar"]):
        if var < 0:
            lines.append(
                f"node {i:>3} | terminal          "
                f"| n = {int(tree['NodeWeight'][i]):>4} "
                f"| mean = {tree['NodeAve'][i]:8.3f}"
            )
        else:
            lines.append(
                f"node {i:>3} | X[{var}] <= {tree['SplitValue'][i]:7.3f} "
                f"| n = {int(tree['NodeWeight'][i]):>4} "
                f"| mean = {tree['NodeAve'][i]:8.3f}"
            )
    for line in lines[:max_lines]:
        print(line)
    if len(lines) > max_lines:
        print("\n... output truncated ...")

print_tree_preview(tree)
```

```
node   0 | X[7] <=   0.500 | n =   64 | mean =    0.000
node   1 | X[3] <=   1.885 | n =   19 | mean =    0.000
node   2 | X[3] <=   0.519 | n =   45 | mean =    0.000
node   3 | X[1] <=  -0.383 | n =   18 | mean =    0.000
node   4 | terminal          | n =    1 | mean =    8.422
node   5 | terminal          | n =    4 | mean =    2.055
node   6 | X[5] <=   0.500 | n =   14 | mean =    0.000
node   7 | terminal          | n =    1 | mean =    1.812
node   8 | X[2] <=  -0.267 | n =   13 | mean =    0.000
node   9 | X[5] <=   1.500 | n =    7 | mean =    0.000
node  10 | X[2] <=  -0.083 | n =    6 | mean =    0.000
node  11 | terminal          | n =    1 | mean =    2.658
node  12 | X[0] <=  -1.095 | n =    6 | mean =    0.000
node  13 | terminal          | n =    2 | mean =    4.480

... output truncated ...
```

For linear-combination forests, the loadings of every internal split live
in `fit.forest_["SplitLoad"][tree_id]` — one row of `linear_comb`
coefficients per internal node (in node order, matching the internal
`SplitVar` rows):

```python
fit_lc = RLT_reg(n_estimators=50, mtry=3, nsplit=3,
                 linear_comb=3, linear_comb_method="sir",
                 n_jobs=1, random_state=1)
fit_lc.fit(trainX, trainY)

loadings = np.asarray(fit_lc.forest_["SplitLoad"][0])
loadings.shape  # (n_internal_nodes, linear_comb)

np.round(loadings[0], 3)  # coefficients of the first split
```

```
(45, 3)
array([0.735, 0.445, 0.512])
```
