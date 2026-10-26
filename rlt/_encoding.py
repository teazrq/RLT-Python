"""Optional data-frame label encoding and reusable fitted feature schemas."""
from __future__ import annotations

import warnings

import numpy as np


def _pandas(value):
    # Array-only use does not import or require pandas.
    if type(value).__module__.split(".")[0] == "pandas":
        import pandas as pd
        return pd
    return None


def categorical_target(y):
    """Return declared class levels and codes, or None for ordinary labels."""
    pd = _pandas(y)
    if pd is None or not isinstance(getattr(y, "dtype", None), pd.CategoricalDtype):
        return None
    if np.ndim(y) != 1:
        raise ValueError("y must be a one-dimensional classification target")
    categorical = y.array if isinstance(y, pd.Series) else y
    levels = np.asarray(categorical.categories).copy()
    codes = np.asarray(categorical.codes, dtype=np.int64).copy()
    if np.any(codes < 0):
        raise ValueError("y contains missing classification labels")
    if levels.size < 2:
        raise ValueError("y must have at least 2 declared classes")
    if np.unique(codes).size < levels.size:
        warnings.warn("y has unused declared classes; retaining all class levels",
                      UserWarning, stacklevel=3)
    return levels, codes


def _nominal_mask(value, p, inferred):
    if value is None:
        return inferred.copy()
    if isinstance(value, (str, bytes)):
        raise ValueError("categorical_features must be None, a boolean mask, or integer column indices")
    mask = np.asarray(value)
    if mask.ndim != 1:
        raise ValueError("categorical_features must be a one-dimensional mask or index array")
    if mask.dtype == bool:
        if mask.size != p:
            raise ValueError(f"categorical_features boolean mask must have length n_features={p}, got {mask.size}")
        return mask.copy()
    if mask.size and not np.issubdtype(mask.dtype, np.integer):
        raise ValueError("categorical_features must be None, a boolean mask, or integer column indices")
    result = np.zeros(p, dtype=bool)
    if mask.size:
        indices = np.asarray(mask, dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= p):
            raise ValueError(f"categorical_features indices must be in [0, n_features={p})")
        result[indices] = True
    return result


def _columns(X):
    pd = _pandas(X)
    if pd is not None and isinstance(X, pd.DataFrame):
        if not all(X.shape):
            raise ValueError("X must be a nonempty two-dimensional array")
        return [X.iloc[:, j] for j in range(X.shape[1])], pd
    values = np.asarray(X)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2 or not all(values.shape):
        raise ValueError("X must be a nonempty two-dimensional array")
    return [values[:, j] for j in range(values.shape[1])], None


def fit_features(X, categorical_features):
    """Encode training columns without changing estimator state."""
    columns, pd = _columns(X)
    schema = []
    encoded = []
    inferred = np.zeros(len(columns), dtype=bool)
    for j, column in enumerate(columns):
        entry = None
        values = np.asarray(column)
        if pd is not None:
            if np.any(pd.isna(column)):
                raise ValueError(f"X column {j} contains missing values")
            if isinstance(column.dtype, pd.CategoricalDtype):
                kind = "ordered" if column.dtype.ordered else "categorical"
                levels = np.asarray(column.cat.categories).copy()
                values = np.asarray(column.cat.codes, dtype=np.float64)
                inferred[j] = kind == "categorical"
                entry = {"kind": kind, "levels": levels, "nlevels": len(levels)}
            elif all(isinstance(v, str) for v in values):
                levels = np.unique(values)
                table = {label: k for k, label in enumerate(levels)}
                values = np.asarray([table[v] for v in values], dtype=np.float64)
                entry = {"kind": "string", "levels": levels, "nlevels": len(levels)}
        if np.iscomplexobj(values):
            raise ValueError("X must contain real values")
        values = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("X contains NA/Inf")
        encoded.append(values)
        schema.append(entry)
    nominal = _nominal_mask(categorical_features, len(columns), inferred)
    ncat = np.ones(len(columns), dtype=np.int64)
    for j, values in enumerate(encoded):
        entry = schema[j]
        if nominal[j]:
            if np.any(values < 0) or np.any(values != np.floor(values)):
                raise ValueError(f"categorical column {j} must contain non-negative integer level codes (0, 1, ..., k-1)")
            count = entry["nlevels"] if entry is not None else int(values.max()) + 1
            if count > 53:
                raise ValueError("cannot handle categorical predictors with more than 53 categories")
            ncat[j] = count
            if entry is None:
                entry = {"kind": "codes", "levels": None, "nlevels": count}
                schema[j] = entry
        elif entry is None and np.unique(values).size <= 10:
            warnings.warn(f"Column {j} has few unique values; if it is categorical, pass categorical_features to treat it as such.",
                          UserWarning, stacklevel=3)
        if entry is not None:
            entry["nominal"] = bool(nominal[j])
    return np.column_stack(encoded), ncat, tuple(schema)


def transform_features(X, schema):
    """Map labels to fitted zero-based codes, accepting explicit numeric codes."""
    columns, pd = _columns(X)
    if len(columns) != len(schema):
        raise ValueError(f"X has {len(columns)} features; expected {len(schema)} from training")
    encoded = []
    for j, (column, entry) in enumerate(zip(columns, schema)):
        values = np.asarray(column)
        typed = pd is not None and isinstance(column.dtype, pd.CategoricalDtype)
        if pd is not None and np.any(pd.isna(column)):
            raise ValueError(f"X column {j} contains missing values")
        labels = typed or all(isinstance(v, str) for v in values)
        if entry is not None and labels:
            levels = entry["levels"]
            if levels is None:
                raise ValueError(f"X column {j} requires integer training codes")
            table = {label: k for k, label in enumerate(levels)}
            try:
                values = np.asarray([table[v] for v in values], dtype=np.float64)
            except (KeyError, TypeError) as exc:
                raise ValueError(f"X column {j} contains an unknown label") from exc
        elif entry is None and (typed or (pd is not None and labels)):
            raise ValueError(f"X column {j} was numerical during training; supply numerical values")
        if np.iscomplexobj(values):
            raise ValueError("X must contain real values")
        values = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("X contains NA/Inf")
        if entry is not None:
            count = entry["nlevels"]
            if np.any(values != np.floor(values)) or np.any(values < 0) or np.any(values >= count):
                raise ValueError(f"categorical column {j} must contain integer training codes in [0, {count})")
        encoded.append(values)
    return np.column_stack(encoded)
