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
    categorical_features=range(p // 2, p),
    n_jobs=1,
    random_state=1,
)
fit.fit(trainX, trainY)
```

## Inspect one tree

Use `get_one_tree(tree_id)` to inspect a single tree.
Choose `tree_id` from 0 to `fit.params_.ntrees - 1` (0-indexed, unlike R's
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
| `SplitValue` | Numerical threshold or encoded categorical partition; use `tree_summary` to decode it |
| `LeftNode` / `RightNode` | Child node indices |
| `NodeWeight` | Number of in-bag observations at the node, including repeats |
| `NodeAve` | Terminal-node response mean; internal-node entries are placeholders |

```python
n_nodes = len(tree["SplitVar"])
n_splits = int((tree["SplitVar"] >= 0).sum())
print(f"{n_nodes} nodes, {n_splits} internal splits")
```

```
35 nodes, 17 internal splits
```

A compact preview from the public summary shows numerical thresholds and
categorical right partitions:

```python
def print_tree_preview(summary, max_lines=14):
    for i in summary["Node"][:max_lines]:
        split, value = summary["Split"][i], summary["Value"][i]
        if split == "leaf":
            detail = f"leaf | mean = {float(summary['Prediction'][i]):.3f}"
        elif isinstance(value, list):
            detail = split + " {" + ", ".join(map(str, value)) + "}"
        else:
            detail = f"{split} <= {value:.3f}"
        print(f"node {i:>3} | n = {int(summary['n'][i]):>4} | {detail}")
    if len(summary["Node"]) > max_lines:
        print("\n... output truncated ...")

print_tree_preview(fit.tree_summary(0))
```

```
node   0 | n =   64 | X[7] in right partition {0}
node   1 | n =   45 | 1 * X[3] <= 0.519
node   2 | n =   19 | 1 * X[1] <= -0.383
node   3 | n =   34 | 1 * X[1] <= 0.665
node   4 | n =   11 | 1 * X[2] <= -0.096
node   5 | n =   29 | X[5] in right partition {2}
node   6 | n =    5 | leaf | mean = -0.007
node   7 | n =   17 | X[9] in right partition {2}
node   8 | n =   12 | 1 * X[4] <= -0.464
node   9 | n =    5 | leaf | mean = 0.003
node  10 | n =   12 | 1 * X[0] <= -0.231
node  11 | n =    5 | leaf | mean = 1.795
node  12 | n =    7 | 1 * X[1] <= -0.641
node  13 | n =    4 | leaf | mean = 0.825

... output truncated ...
```

For linear-combination forests, the loadings of every internal split live
in `fit.forest_["SplitLoad"][tree_id]`: one row of `linear_comb`
coefficients per node, with split loadings at internal nodes aligned with
the `SplitVar` rows:

```python
fit_lc = RLT_reg(n_estimators=50, mtry=3, nsplit=3,
                 linear_comb=3, linear_comb_method="sir",
                 categorical_features=range(p // 2, p),
                 n_jobs=1, random_state=1)
fit_lc.fit(trainX, trainY)

loadings = np.asarray(fit_lc.forest_["SplitLoad"][0])
loadings.shape  # (n_nodes, linear_comb)

np.round(loadings[0], 3)  # coefficients of the first split
```

```
(41, 3)
array([0.294, 0.956, 0.   ])
```
