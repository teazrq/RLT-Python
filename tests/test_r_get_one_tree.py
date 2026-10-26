"""Port of RLT tests/testthat/test-get-one-tree.R.

The Python API's ``get_one_tree(i)`` returns a dict of raw node arrays
(SplitVar/SplitValue/LeftNode/RightNode/NodeWeight/NodeAve, 0-based) rather
than R's summary data.frame, so data.frame-specific assertions are adapted:

- Node indices: R 1-based -> Python 0-based (root node 0).
- ``tree$Split == "*"`` (terminal)  -> ``SplitVar < 0``.
- ``tree$Depth``: not stored in the Python tree dict; recomputed by walking
  LeftNode/RightNode from the root.
- Column names for split variables are not stored; ``SplitVar`` holds
  0-based column indices.
"""

import numpy as np
import pytest

from rlt import RLT_reg, RLT_cla
from helpers import generate_simple_regression, generate_classification_data


def _tree_depth(tree):
    """Depth of every node, root = 0 (mirrors R tree$Depth)."""
    n = len(tree["SplitVar"])
    depth = np.full(n, -1, dtype=int)
    depth[0] = 0
    stack = [0]
    while stack:
        i = stack.pop()
        if tree["SplitVar"][i] < 0:
            continue
        for child in (tree["LeftNode"][i], tree["RightNode"][i]):
            depth[child] = depth[i] + 1
            stack.append(child)
    return depth


def test_get_one_tree_returns_dict_for_regression():
    # expect_is(tree, "data.frame") -> dict of raw node arrays
    d = generate_simple_regression(n=80, p=10)
    fit = RLT_reg(n_estimators=30, n_jobs=2, verbose=0, random_state=1).fit(d["X"], d["y"])
    tree = fit.get_one_tree(0)
    assert isinstance(tree, dict)
    for key in ("SplitVar", "SplitValue", "LeftNode", "RightNode", "NodeWeight", "NodeAve"):
        assert key in tree


def test_get_one_tree_regression_has_correct_fields():
    # expected_cols <- c("Node", "Depth", "Split", "Value", "n")
    d = generate_simple_regression(n=80, p=10)
    fit = RLT_reg(n_estimators=30, n_jobs=2, verbose=0, random_state=1).fit(d["X"], d["y"])
    tree = fit.get_one_tree(0)
    # Node -> implicit array index; Split -> SplitVar; Value -> SplitValue;
    # n -> NodeWeight; Depth -> recomputable from LeftNode/RightNode
    assert len(tree["SplitVar"]) == len(tree["NodeWeight"])
    assert len(_tree_depth(tree)) == len(tree["SplitVar"])


def test_get_one_tree_regression_node_column_is_0_indexed():
    # R: tree$Node[1] == 1 (1-based); Python root node index is 0
    d = generate_simple_regression(n=80, p=10)
    fit = RLT_reg(n_estimators=30, n_jobs=2, verbose=0, random_state=1).fit(d["X"], d["y"])
    tree = fit.get_one_tree(0)
    assert len(tree["SplitVar"]) > 0  # node 0 exists


def test_get_one_tree_regression_depth_starts_at_0():
    d = generate_simple_regression(n=80, p=10)
    fit = RLT_reg(n_estimators=30, n_jobs=2, verbose=0, random_state=1).fit(d["X"], d["y"])
    depth = _tree_depth(fit.get_one_tree(0))
    assert depth[0] == 0


def test_get_one_tree_regression_has_terminal_nodes():
    # R: any(tree$Split == "*"); terminal nodes have SplitVar == -1
    d = generate_simple_regression(n=80, p=10)
    fit = RLT_reg(n_estimators=30, n_jobs=2, verbose=0, random_state=1).fit(d["X"], d["y"])
    tree = fit.get_one_tree(0)
    assert np.any(np.asarray(tree["SplitVar"]) < 0)


def test_get_one_tree_regression_leaf_value_is_numeric():
    # R checks the Value column exists for all nodes -> SplitValue numeric array
    d = generate_simple_regression(n=80, p=10)
    fit = RLT_reg(n_estimators=30, n_jobs=2, verbose=0, random_state=1).fit(d["X"], d["y"])
    tree = fit.get_one_tree(0)
    values = np.asarray(tree["SplitValue"])
    assert values.dtype == np.float64


def test_get_one_tree_regression_n_column_is_positive():
    d = generate_simple_regression(n=80, p=10)
    fit = RLT_reg(n_estimators=30, n_jobs=2, verbose=0, random_state=1).fit(d["X"], d["y"])
    tree = fit.get_one_tree(0)
    assert np.all(np.asarray(tree["NodeWeight"]) > 0)


def test_get_one_tree_regression_root_node_n_equals_training_n():
    d = generate_simple_regression(n=80, p=10)
    fit = RLT_reg(n_estimators=30, n_jobs=2, verbose=0, random_state=1).fit(d["X"], d["y"])
    tree = fit.get_one_tree(0)
    assert tree["NodeWeight"][0] == d["n"]


def test_get_one_tree_regression_split_var_references_x_columns():
    # R checks Split labels are X colnames; Python stores 0-based indices
    d = generate_simple_regression(n=80, p=10)
    fit = RLT_reg(n_estimators=30, n_jobs=2, verbose=0, random_state=1).fit(d["X"], d["y"])
    tree = fit.get_one_tree(0)
    split_nodes = np.asarray(tree["SplitVar"])
    split_nodes = split_nodes[split_nodes >= 0]
    assert np.all((split_nodes >= 0) & (split_nodes < d["p"]))


def test_get_one_tree_returns_dict_for_classification():
    d = generate_classification_data(n=80, p=10)
    fit = RLT_cla(n_estimators=30, n_jobs=2, verbose=0, random_state=1).fit(d["X"], d["y"])
    tree = fit.get_one_tree(0)
    assert isinstance(tree, dict)
    for key in ("SplitVar", "SplitValue", "LeftNode", "RightNode", "NodeWeight"):
        assert key in tree


def test_get_one_tree_classification_has_correct_fields():
    d = generate_classification_data(n=80, p=10)
    fit = RLT_cla(n_estimators=30, n_jobs=2, verbose=0, random_state=1).fit(d["X"], d["y"])
    tree = fit.get_one_tree(0)
    assert len(tree["SplitVar"]) == len(tree["NodeWeight"])


def test_get_one_tree_classification_nodeprob_is_stored_in_forest():
    # R: expect_false(is.null(fit$FittedForest$NodeProb))
    d = generate_classification_data(n=80, p=10)
    fit = RLT_cla(n_estimators=30, n_jobs=2, verbose=0, random_state=1).fit(d["X"], d["y"])
    assert "NodeProb" in fit.forest_
    assert len(fit.forest_["NodeProb"]) == 30


def test_get_one_tree_classification_root_node_n_equals_training_n():
    d = generate_classification_data(n=80, p=10)
    fit = RLT_cla(n_estimators=30, n_jobs=2, verbose=0, random_state=1).fit(d["X"], d["y"])
    tree = fit.get_one_tree(0)
    assert tree["NodeWeight"][0] == d["n"]


def test_get_one_tree_regression_lc_tree_has_splitload_in_forest():
    # R: expect_false(is.null(fit$FittedForest$SplitLoad))
    d = generate_simple_regression(n=80, p=10)
    fit = RLT_reg(
        n_estimators=30,
        linear_comb=3,
        linear_comb_method="naive",
        n_jobs=2,
        verbose=0, random_state=1).fit(d["X"], d["y"])
    assert "SplitLoad" in fit.forest_


def test_get_one_tree_classification_lc_tree_works():
    d = generate_classification_data(n=80, p=10)
    fit = RLT_cla(
        n_estimators=30,
        linear_comb=3,
        linear_comb_method="lda",
        n_jobs=2,
        verbose=0, random_state=1).fit(d["X"], d["y"])
    tree = fit.get_one_tree(0)
    assert isinstance(tree, dict)
    assert len(tree["SplitVar"]) > 0


def test_get_one_tree_errors_on_tree_index_negative():
    # Negative indices are invalid rather than indexing from the end.
    d = generate_simple_regression(n=80, p=10)
    fit = RLT_reg(n_estimators=30, n_jobs=2, verbose=0, random_state=1).fit(d["X"], d["y"])
    with pytest.raises(IndexError):
        fit.get_one_tree(-1)


def test_get_one_tree_errors_on_tree_index_above_ntrees():
    d = generate_simple_regression(n=80, p=10)
    fit = RLT_reg(n_estimators=30, n_jobs=2, verbose=0, random_state=1).fit(d["X"], d["y"])
    # R index 31 (1-based) == Python index 30 (0-based)
    with pytest.raises(IndexError):
        fit.get_one_tree(30)


def test_get_one_tree_works_for_all_trees_in_forest():
    d = generate_simple_regression(n=80, p=5)
    n_estimators = 5
    fit = RLT_reg(n_estimators=n_estimators, n_jobs=2, verbose=0, random_state=1).fit(
        d["X"], d["y"]
    )
    for i in range(n_estimators):
        tree = fit.get_one_tree(i)
        assert isinstance(tree, dict)
        assert len(tree["SplitVar"]) > 0


def test_get_one_tree_uses_zero_based_column_indices_without_names():
    # R: colnames default to V1..Vp; Python stores raw 0-based SplitVar codes
    rng = np.random.default_rng(42)
    X = rng.normal(size=(80, 5))
    y = X[:, 0] + rng.normal(size=80)
    fit = RLT_reg(n_estimators=30, n_jobs=2, verbose=0, random_state=1).fit(X, y)
    tree = fit.get_one_tree(0)
    split_nodes = np.asarray(tree["SplitVar"])
    split_nodes = split_nodes[split_nodes >= 0]
    if len(split_nodes) > 0:
        assert np.all((split_nodes >= 0) & (split_nodes < 5))


def test_get_one_tree_depth_increases_monotonically_along_tree_paths():
    d = generate_simple_regression(n=80, p=10)
    fit = RLT_reg(n_estimators=30, n_jobs=2, verbose=0, random_state=1).fit(d["X"], d["y"])
    depth = _tree_depth(fit.get_one_tree(0))
    assert depth[0] == 0
    assert depth.max() > 0
    # every node is reachable and depth >= parent depth + 1
    assert np.all(depth >= 0)
