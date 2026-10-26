"""Public fitted-model outputs and portable estimator state."""

from copy import deepcopy
from numbers import Integral

import numpy as np
from sklearn.utils.validation import check_is_fitted

from ._core import CoreParams


class RLTInspectionMixin:
    """Inspection shared by regression, classification and survival forests."""

    def __getstate__(self):
        state = super().__getstate__().copy()
        params = state.pop("params_", None)
        if params is not None:
            # Read the binding's writable scalar properties, including the
            # integer seed. This does not advance a constructor RNG object.
            state["_native_params_state"] = {
                name: getattr(params, name) for name in dir(type(params))
                if not name.startswith("_")
                and isinstance(getattr(type(params), name), property)
            }
        return state

    def __setstate__(self, state):
        state = state.copy()
        values = state.pop("_native_params_state", None)
        if values is not None:
            params = CoreParams()
            for name, value in values.items():
                setattr(params, name, value)
            state["params_"] = params
        super().__setstate__(state)

    def predict_details(self, X, *, var_est=False, var_mode=None,
                        keep_all=False, ncores=0, band_grid_size=0):
        """Return predictions and their labels, grid and uncertainty mode.

        Classification includes decoded ``Prediction`` and ``classes``.
        ``PredictionAll`` contains per-tree class probabilities with axes
        (subject, tree, class), or regression predictions (subject, tree).
        Survival returns ``Hazard``, ``CHF``, ``Survival`` and ``timepoints``;
        optional ``Cov`` has axes (subject, time, time), and ``AllHazard``
        has axes (subject, tree, time). Covariance is for cumulative hazard.
        Set ``var_est=True`` to request uncertainty; ``var_mode`` selects
        a compatible fitted or prediction-time method.
        """
        check_is_fitted(self, "forest_")
        args = dict(var_est=var_est, var_mode=var_mode,
                    keep_all=keep_all, ncores=ncores)
        if hasattr(self, "timepoints_"):
            args["band_grid_size"] = band_grid_size
        elif band_grid_size != 0:
            raise ValueError("band_grid_size is only available for survival")
        result = dict(self._predict_raw(X, **args))
        if hasattr(self, "classes_"):
            result["Prediction"] = self.classes_[np.asarray(result["Prediction"], dtype=int)]
            result["classes"] = self.classes_.copy()
        if "Variance" in result:
            variance = np.array(result["Variance"], dtype=float, copy=True)
            variance[variance < 0] = np.nan
            result["Variance"] = variance
        if hasattr(self, "timepoints_"):
            result["timepoints"] = self._grid_and_mapping(band_grid_size)[0].copy()
        mode = self._resolve_var_mode(var_est, var_mode)
        result["var_mode"] = ("none", "matched", "ij", "jack")[mode]
        return result

    def predict_all_trees(self, X, *, band_grid_size=0):
        """Per-tree probabilities or hazards, with tree axis second.

        Class order is ``classes_``. Survival time order is the grid in
        ``predict_details(X, band_grid_size=...)``.
        """
        result = self.predict_details(X, keep_all=True, band_grid_size=band_grid_size)
        return result["AllHazard" if hasattr(self, "timepoints_") else "PredictionAll"]

    def oob_details(self):
        """Return OOB outputs, row counts and an explicit availability mask.

        Missing numeric predictions are NaN; missing class labels are None.
        Negative signed sampling counts are excluded, not OOB. Nominal
        full-sample fitting may disable OOB computation even with a preset.
        Survival ``Hazard`` uses the full training failure-time grid.
        """
        check_is_fitted(self, "forest_")
        if not hasattr(self, "oob_counts_"):
            raise ValueError("OOB outputs were not computed for this fit")
        counts = self.oob_counts_.copy()
        available = counts > 0
        result = {"counts": counts, "available": available,
                  "Error": self.oob_error_}
        if hasattr(self, "oob_prob_"):
            labels = np.full(counts.shape, None, dtype=object)
            labels[available] = self.classes_[self.oob_class_[available]]
            result["Prediction"] = labels
            result["classes"] = self.classes_.copy()
            result["Prob"] = self.oob_prob_.copy()
            result["Prob"][~available] = np.nan
        else:
            key, attribute = (("Hazard", "oob_hazard_") if hasattr(self, "timepoints_")
                              else ("Prediction", "oob_prediction_"))
            result[key] = getattr(self, attribute).copy()
            result[key][~available] = np.nan
            if hasattr(self, "timepoints_"):
                result["timepoints"] = self.timepoints_.copy()
        return result

    def get_one_tree(self, tree_id=0):
        """Copy all raw arrays of a zero-based tree, including node payloads.

        Survival ``NodeHaz`` remains a list of possibly empty node arrays.
        A terminal hazard array includes the native initial zero slot before
        the values on ``timepoints_``; ``tree_summary`` omits that slot.
        Editing this result cannot change the fitted model.
        """
        check_is_fitted(self, "forest_")
        tree = tree_id
        if isinstance(tree, (bool, np.bool_)) or not isinstance(tree, Integral):
            raise ValueError("tree must be a zero-based integer index")
        if not 0 <= tree < len(self.forest_["SplitVar"]):
            raise IndexError("tree index is outside the fitted forest")
        return {key: deepcopy(value[int(tree)]) for key, value in self.forest_.items()}

    def tree_summary(self, tree=0):
        """Return aligned columns describing splits and terminal predictions.

        Node and feature indices are zero-based. Numeric splits send values
        <= the threshold left; categorical partitions list labels sent right.
        ``code(name)`` denotes a fitted zero-based rank in ``feature_schema_``;
        thresholds account for the native core's one-based encoding.
        Node weights include repeated in-bag observations. Leaf predictions
        are a scalar, class probabilities in ``classes_`` order, or hazard
        increments on ``timepoints_`` (without the native initial zero slot).
        """
        raw = self.get_one_tree(tree)
        variables = np.asarray(raw["SplitVar"])
        comb = variables.ndim == 2
        terminal = (variables[:, 0] if comb else variables) < 0
        count = len(terminal)
        depth = np.zeros(count, dtype=int)
        queue = [0]
        for node in queue:
            if not terminal[node]:
                for key in ("LeftNode", "RightNode"):
                    child = int(raw[key][node])
                    depth[child] = depth[node] + 1
                    queue.append(child)
        names = getattr(self, "feature_names_in_",
                        np.array([f"X[{i}]" for i in range(self.n_features_in_)]))
        splits, values, predictions = [], [], []
        payload = next(key for key in ("NodeAve", "NodeProb", "NodeHaz") if key in raw)
        for node in range(count):
            prediction = deepcopy(raw[payload][node]) if terminal[node] else None
            if terminal[node] and payload == "NodeHaz":
                prediction = prediction[1:]
            predictions.append(prediction)
            if terminal[node]:
                splits.append("leaf")
                values.append(None)
                continue
            ids = variables[node] if comb else np.array([variables[node]])
            loads = raw["SplitLoad"][node] if comb else np.ones(1)
            active = [(int(j), float(w)) for j, w in zip(ids, loads) if w != 0 and j >= 0]
            j = active[0][0]
            cut = float(raw["SplitValue"][node])
            if self.ncat_[j] > 1 and len(active) == 1:
                schema = self.feature_schema_[j]
                levels = schema["levels"]
                if levels is None:
                    levels = np.arange(self.ncat_[j])
                right = [deepcopy(label) for k, label in enumerate(levels)
                         if (int(cut) >> (k + 1)) & 1]
                splits.append(str(names[j]) + " in right partition")
                values.append(right)
            else:
                terms = []
                offset = 0.0
                for j, w in active:
                    encoded = self.feature_schema_[j] is not None
                    label = f"code({names[j]})" if encoded else str(names[j])
                    terms.append(f"{w:.6g} * {label}")
                    if encoded:
                        offset += w
                splits.append(" + ".join(terms))
                values.append(cut - offset)
        return {"Node": np.arange(count), "Depth": depth, "Split": splits,
                "Value": values, "n": raw["NodeWeight"], "Prediction": predictions}

    def print_tree(self, tree=0):
        """Print a readable tree and return its summary columns."""
        result = self.tree_summary(tree)
        print("Node  Depth  Weight  Split  Value  Terminal prediction")
        for i in result["Node"]:
            print(f"{i}  {result['Depth'][i]}  {result['n'][i]:.6g}  "
                  f"{result['Split'][i]}  {result['Value'][i]}  {result['Prediction'][i]}")
        return result

    def summary(self):
        """Describe the effective fitted model without changing its parameters."""
        check_is_fitted(self, "forest_")
        family = "survival" if hasattr(self, "timepoints_") else (
            "classification" if hasattr(self, "classes_") else "regression")
        return (f"RLT {family}: {self.params_.ntrees} trees, "
                f"{self.n_features_in_} features, linear_comb={self.params_.linear_comb}, "
                f"seed={self.params_.seed}")
