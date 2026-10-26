"""sklearn-compatible estimators for RLT forests.

Provides RLT_reg, RLT_cla, and RLT_surv — Python
ports of the RLT R package's RegForest / ClaForest / SurvForest.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, column_or_1d

from . import _encoding, _params
from ._inspection import RLTInspectionMixin
from .bands import get_surv_band
from .importance import ImportanceTable, build_importance_table as _importance_table
from ._core import (
    ClaUniCombForestFit,
    ClaUniCombForestPred,
    ClaUniForestFit,
    ClaUniForestPred,
    RegUniCombForestFit,
    RegUniCombForestPred,
    RegUniForestFit,
    RegUniForestPred,
    SurvUniCombForestFit,
    SurvUniCombForestPred,
    SurvUniForestFit,
    SurvUniForestPred,
    cindex,
    gen_ms_obs_track_mat,
)

# ----------------------------------------------------------------------------
# shared helpers
# ----------------------------------------------------------------------------

_LC_METHOD_REG = {"default": 4, "naive": 1, "lm": 2, "pca": 3, "sir": 4}
_LC_METHOD_CLA = {"default": 1, "lda": 1, "naive": 2, "random": 3, "logistic": 4}
_LC_METHOD_SURV = {"default": 1, "coxph": 1, "naive": 2}
_SPLIT_RULES_SURV = {"default": 1, "logrank": 1, "suplogrank": 2, "coxgrad": 3}


def _response_vector(value, name):
    result = column_or_1d(value, warn=True)
    if np.iscomplexobj(result):
        raise ValueError(f"{name} must contain real values")
    return np.asarray(result, dtype=np.float64)


def _validate_surv_y(y):
    """Accept structured array with (event, time) fields or a (time, event)
    pair; return (time, event) float/int arrays."""
    if isinstance(y, np.ndarray) and y.dtype.names is not None:
        names = set(y.dtype.names)
        if {"event", "time"}.issubset(names):
            event = _response_vector(y["event"], "event")
            time = _response_vector(y["time"], "time")
            if time.size != event.size:
                raise ValueError("survival time and event have inconsistent lengths")
            return time, event
        raise ValueError(
            "survival y must be a structured array with fields 'event' "
            "and 'time' (scikit-survival convention)"
        )
    if isinstance(y, (tuple, list)) and len(y) == 2:
        time = _response_vector(y[0], "time")
        event = _response_vector(y[1], "event")
        if time.size != event.size:
            raise ValueError("survival time and event have inconsistent lengths")
        return time, event
    raise ValueError(
        "survival y must be a structured array with fields 'event'/'time' "
        "or a (time, event) pair of arrays"
    )


class _BaseRLT(RLTInspectionMixin, BaseEstimator):
    """Shared machinery for RLT estimators."""

    def _check_X(self, X, *, reset=False):
        names = getattr(X, "columns", None)
        names = None if names is None else np.asarray([str(c) for c in names], dtype=object)
        shape = np.shape(X)
        if len(shape) == 1:
            X = np.asarray(X).reshape(-1, 1)
            shape = X.shape
        if len(shape) != 2 or not all(shape):
            raise ValueError("X must be a nonempty two-dimensional array")
        if not reset and hasattr(self, "n_features_in_"):
            p = self.n_features_in_
            fitted_names = getattr(self, "feature_names_in_", None)
            if names is not None and fitted_names is not None:
                if not np.array_equal(names, fitted_names):
                    matching = [name for name in names if name in fitted_names]
                    if len(set(fitted_names)) != p or len(set(matching)) != len(matching):
                        raise ValueError("X has ambiguous duplicate feature names; supply the original training order")
                    if any(name not in names for name in fitted_names):
                        raise ValueError("X is missing features from the training data")
                    order = [list(names).index(name) for name in fitted_names]
                    warnings.warn("X feature names do not match training data; matching by name", UserWarning, stacklevel=3)
                    X = X.iloc[:, order] if hasattr(X, "iloc") else np.asarray(X)[:, order]
            if np.shape(X)[1] != p:
                raise ValueError(f"X has {np.shape(X)[1]} features; expected {p} from training")
        if not reset and hasattr(self, "feature_schema_"):
            return _encoding.transform_features(X, self.feature_schema_)
        if np.iscomplexobj(X):
            raise ValueError("X must contain real values")
        X = np.asarray(X, dtype=np.float64)
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NA/Inf")
        if not reset and hasattr(self, "n_features_in_"):
            for j in self._cat_shift_cols_:
                col = X[:, j]
                if np.any(col != np.floor(col)) or np.any(col < 0) or np.any(col >= self.ncat_[j]):
                    raise ValueError(f"categorical column {j} must contain integer training codes in [0, {self.ncat_[j]})")
        return X

    def _resolve_seed(self):
        # Explicit integers are native master seeds without transformation.
        # Numerical reproducibility also depends on the build and environment.
        rs = self.random_state
        if rs is None:
            return int(check_random_state(None).randint(0, 2**31 - 1))
        if isinstance(rs, (int, np.integer)):
            return int(rs)
        if isinstance(rs, np.random.Generator):
            return int(rs.integers(0, 2**31 - 1))
        # RandomState instances: draw one seed
        return int(check_random_state(rs).randint(0, 2**31 - 1))

    def _shift_categorical(self, X, cols=None):
        """Shift declared categorical columns to the core's 1-based level
        codes (R's data.matrix convention: codes 1..k; category 0 is a
        phantom that never occurs)."""
        if cols is None:
            cols = getattr(self, "_cat_shift_cols_", None)
        if not cols:
            return X
        X = X.copy()
        for j in cols:
            X[:, j] = X[:, j] + 1.0
        return X

    def _prepare_fit_inputs(self, X):
        names = getattr(X, "columns", None)
        names = None if names is None else np.asarray([str(c) for c in names], dtype=object)
        X, ncat, schema = _encoding.fit_features(X, self.categorical_features)
        n, p = X.shape
        # All saved dictionaries use public zero-based codes, including ranks.
        cat_cols = [j for j, entry in enumerate(schema) if entry is not None]
        return X, ncat, n, p, names, cat_cols, schema

    def _replace_fit_metadata(self, names, cat_cols, schema):
        """Install new metadata only after native fitting has succeeded."""
        for name in ("varimp_", "var_vi_", "obstrack_", "oob_prediction_",
                     "oob_class_", "oob_prob_", "oob_hazard_", "oob_error_", "oob_counts_"):
            self.__dict__.pop(name, None)
        self._cat_shift_cols_ = cat_cols
        self.feature_schema_ = schema
        if names is None:
            self.__dict__.pop("feature_names_in_", None)
        else:
            self.feature_names_in_ = names

    def _resolve_lc_method(self, codes, valid_names, valid_codes, model_label,
                           reset_name):
        """Mirror R's linear.comb.method handling: accept names or integer
        codes; unrecognized values warn and reset to code 1 (R: RegForest.r /
        ClaForest.r / SurvForest.r)."""
        return self._method_code(self.linear_comb_method, codes, valid_names,
                                 "linear_comb_method", reset_name)

    @staticmethod
    def _method_code(v, codes, valid_names, name, reset_name):
        if isinstance(v, str):
            code = codes.get(v.lower())
            if code is None:
                warnings.warn(
                    f"{name} not recognized. Use {valid_names}. "
                    f"Resetting to {reset_name}",
                    UserWarning, stacklevel=3)
                code = 1
        elif isinstance(v, (bool, np.bool_)):
            raise ValueError(f"{name} must be a name or integral numeric code, not a boolean")
        elif isinstance(v, (int, float, np.integer, np.floating)):
            number = _params._number(v, name)
            if number != int(number):
                raise ValueError(f"{name} must be an integral numeric code")
            iv = int(number)
            if iv in codes.values():
                code = iv
            else:
                warnings.warn(
                    f"{name} integer must be one of "
                    f"{sorted(set(codes.values()))}. Resetting to 1 ({reset_name})",
                    UserWarning, stacklevel=3)
                code = 1
        else:
            warnings.warn(
                f"{name} must be a string or integral numeric code. "
                f"Resetting to {reset_name}",
                UserWarning, stacklevel=3)
            code = 1
        return int(code)

    def importance_table(self):
        """Variable-importance summary table (port of R's importance.RLT).

        Returns an :class:`rlt.ImportanceTable` with Variable / VI columns,
        plus SD / Z / Sig when the forest was fitted with
        ``importance != 'none'`` and ``var_mode='matched'`` (the core then
        returns the per-variable variance of VI). Negative variance
        estimates yield NaN for SD and Z and an empty significance code,
        exactly like R.

        Also available as the module-level :func:`rlt.importance(model)`,
        mirroring R's ``importance(fit)`` idiom (the estimator attribute
        ``.importance`` holds the constructor parameter string).
        """
        if not hasattr(self, "varimp_"):
            raise RuntimeError(
                "No variable importance in this model. Fit with "
                "importance='permute' or 'distribute'.")
        return _importance_table(self)

    @staticmethod
    def _empty_obstrack():
        return np.zeros((0, 0), dtype=np.int64)

    def _ncores(self, ncores=0):
        if ncores and ncores > 0:
            return int(ncores)
        v = self.n_jobs
        if v is None or v < 0:
            return 0  # 0 = all available cores (checkCores convention)
        return int(v)

    def _resolve_var_mode(self, var_est, var_mode):
        if not var_est:
            return 0
        fitted = int(self.params_.var_mode)
        mode = fitted if var_mode is None else _params._encode(var_mode, _params._VAR_MODE_CODES, "var_mode")
        if mode == 0:
            raise ValueError("predict_var requires a fitted variance mode or an explicit compatible var_mode")
        if mode == 1 and fitted != 1:
            raise ValueError("matched variance requires fitting with var_mode='matched'")
        if mode in (2, 3) and not hasattr(self, "obstrack_"):
            raise ValueError("IJ/jack variance requires resample tracking at fit time")
        return int(mode)

    def _obstrack_or_empty(self):
        return getattr(self, "obstrack_", self._empty_obstrack())

    def _base_args(self):
        """Common forest list prefix for predict calls."""
        f = self.forest_
        args = [f["SplitVar"], f["SplitValue"], f["LeftNode"],
                f["RightNode"], f["NodeWeight"]]
        if "SplitLoad" in f:
            # Comb forests interleave SplitLoad after SplitVar
            args.insert(1, f["SplitLoad"])
        return args

    def _is_comb(self):
        return "SplitLoad" in self.forest_

    def _core_params(self, n, p):
        cp = _params.build_core_params(self, n, p, self._resolve_seed())
        vm = cp.var_mode
        if vm == 1:
            # matched U-statistic preset (mirrors RLT.r): subsample without
            # replacement, prob 0.5, even tree count, distribute importance
            cp.replacement = False
            cp.resample_prob = 0.5
            if cp.ntrees % 2 != 0:
                cp.ntrees = 2 * (cp.ntrees // 2)
            if cp.ntrees < 2:
                raise ValueError(
                    "var_mode='matched' requires at least 2 trees")
            if cp.importance == 0:
                cp.importance = 2
        elif vm in (2, 3):
            # IJ / jackknife work with bootstrap samples
            cp.replacement = True
            if cp.importance == 0:
                cp.importance = 2
        return cp

    def _matched_obstrack(self, cp, n):
        k = int(n * cp.resample_prob)
        return gen_ms_obs_track_mat(n, k, cp.ntrees, cp.seed + 1)

    def _fit_obstrack(self, cp, n, resample_preset):
        if resample_preset is not None and cp.var_mode:
            warnings.warn("resample_preset is ignored because var_mode supplies its sampling scheme",
                          UserWarning, stacklevel=3)
        if cp.var_mode == 1:
            return self._matched_obstrack(cp, n)
        if cp.var_mode or resample_preset is None:
            return self._empty_obstrack()
        counts = _params.check_resample_preset(resample_preset, n, cp.ntrees)
        cp.obs_track = True
        return counts

    def forest_kernel(self, X1, X2=None, vs_train=False, oob=False):
        """Forest similarity kernel (co-occurrence in terminal nodes).

        - ``X2=None``: self-kernel of X1, shape (n1, n1).
        - ``X2`` given, ``vs_train=False``: cross-kernel (n1, n2).
        - ``vs_train=True``: training-process kernel. X2 must be the
          training data and the forest must have been fitted with
          ``resample_track=True``; entries weight by inbag counts.
        - ``oob=True``: OOB self-kernel (RLT >= 6.1.0), with co-occurrence
          counted only from trees where both observations are
          out-of-bag. Each contributing tree was fitted without either
          observation in the pair. Requires
          ``resample_track=True``, ``X2=None`` and X1 to be the original
          training data. Returns a dict with keys ``Kernel`` (normalized
          co-occurrence in [0, 1]), ``N`` (both-OOB tree count) and ``C``
          (both-OOB, same-leaf count).

        Without ``oob``, returns integer counts; divide by ``params_.ntrees``
        for frequencies.
        """
        check_is_fitted(self, "forest_")
        from . import _core
        X1 = self._shift_categorical(self._check_X(X1))
        f = self.forest_
        comb = self._is_comb()

        def prefix():
            args = [f["SplitVar"], f["SplitValue"], f["LeftNode"],
                    f["RightNode"], f["NodeWeight"]]
            if comb:
                args.insert(1, f["SplitLoad"])
            return args

        if X2 is None:
            if oob:
                if not hasattr(self, "obstrack_"):
                    raise ValueError(
                        "oob=True requires resample_track=True at fit time")
                if X1.shape[0] != self.obstrack_.shape[0]:
                    raise ValueError(
                        "oob=True requires X1 to be the original training "
                        "data")
                fn = _core.Kernel_Self_OOB_Comb if comb else _core.Kernel_Self_OOB
                return fn(*prefix(), X1, self.ncat_, self.obstrack_, 0)
            fn = _core.Kernel_Self_Comb if comb else _core.Kernel_Self
            return fn(*prefix(), X1, self.ncat_, 0)

        if oob:
            raise ValueError(
                "oob=True is only supported for the self-kernel "
                "(X2 must be None)")

        X2 = self._shift_categorical(self._check_X(X2))
        if not vs_train:
            fn = _core.Kernel_Cross_Comb if comb else _core.Kernel_Cross
            return fn(*prefix(), X1, X2, self.ncat_, 0)

        if not hasattr(self, "obstrack_"):
            raise ValueError(
                "vs_train=True requires resample_track=True at fit time")
        if X2.shape[0] != self.obstrack_.shape[0]:
            raise ValueError("vs_train=True requires X2 to be the original training data")
        fn = _core.Kernel_Train_Comb if comb else _core.Kernel_Train
        return fn(*prefix(), X1, X2, self.ncat_, self.obstrack_, 0)

    @property
    def feature_importances_(self):
        check_is_fitted(self, "varimp_")
        return np.asarray(self.varimp_)


# ----------------------------------------------------------------------------
# regression
# ----------------------------------------------------------------------------

class RLT_reg(RegressorMixin, _BaseRLT):
    """Reinforcement Learning Trees regression forest.

    Parameters
    ----------
    n_estimators : int or None, default=None
        Number of trees. None resolves to 100 with reinforcement, otherwise 500.
    mtry : int or None, default=None
        Variables considered per split (None -> max(1, p // 2)).
    min_samples_leaf : int, default=5
        Node-size stopping threshold (RLT ``nmin``), not a guaranteed
        minimum number of observations in every terminal node.
    nsplit : int, default=0
        Random cut points per variable (0 = consider all; best split).
    resample_replace : bool, default=True
        Resample with replacement (bootstrap).
    resample_prob : float or None, default=None
        Resampling proportion (None -> 1.0 if replace else 0.8).
    importance : {"none", "permute", "distribute"}, default="none"
        Variable importance strategy. Active variance modes enable "distribute"
        if importance is disabled; an explicit "permute" setting is retained.
    resample_track : bool, default=False
        Keep the n x ntrees inbag-count matrix (required by IJ/jack).
    var_mode : {"none", "matched", "ij", "jack"}, default="none"
        Variance estimation mode.
    linear_comb : int, default=1
        Variables combined per split; 1 = axis-aligned, >1 = linear
        combination splits with embedded variable screening.
    linear_comb_method : str, default="default"
        Regression loadings: "sir" (default), "naive", "lm", "pca".
    alpha : float, default=0
        Minimum child-sample proportion, clamped to [0, 0.5].
    reinforcement : bool, default=False
        Use embedded-model (reinforcement) splitting.
    embed_* : embedded forest controls (see RLT documentation).
    n_jobs : int, default=-1
        OpenMP threads (-1 or 0 = all cores).
    verbose : int, default=0
    random_state : int, RandomState instance or None
        Explicit native master seed. None draws from NumPy's global RNG;
        Generator and RandomState inputs advance their supplied state.
        Numerical reproducibility also depends on the build and environment.
    """

    _tree_extra_keys = ("NodeAve",)

    def __init__(
        self,
        n_estimators=None,
        mtry=None,
        min_samples_leaf=5,
        nsplit=0,
        resample_replace=True,
        resample_prob=None,
        importance="none",
        resample_track=False,
        var_mode="none",
        categorical_features=None,
        linear_comb=1,
        linear_comb_method="default",
        alpha=0,
        reinforcement=False,
        embed_ntrees=50,
        embed_mtry=0.5,
        embed_nmin=5,
        embed_nsplit=3,
        embed_resample_replace=True,
        embed_resample_prob=0.9,
        embed_mute=0,
        embed_protect=None,
        embed_threshold=0.25,
        n_jobs=-1,
        verbose=0,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.mtry = mtry
        self.min_samples_leaf = min_samples_leaf
        self.nsplit = nsplit
        self.resample_replace = resample_replace
        self.resample_prob = resample_prob
        self.importance = importance
        self.resample_track = resample_track
        self.var_mode = var_mode
        self.categorical_features = categorical_features
        self.linear_comb = linear_comb
        self.linear_comb_method = linear_comb_method
        self.alpha = alpha
        self.reinforcement = reinforcement
        self.embed_ntrees = embed_ntrees
        self.embed_mtry = embed_mtry
        self.embed_nmin = embed_nmin
        self.embed_nsplit = embed_nsplit
        self.embed_resample_replace = embed_resample_replace
        self.embed_resample_prob = embed_resample_prob
        self.embed_mute = embed_mute
        self.embed_protect = embed_protect
        self.embed_threshold = embed_threshold
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None, var_prob=None, *, resample_preset=None):
        """Fit a forest; optional signed resampling counts have shape (n, trees).

        Positive counts are in-bag multiplicities, zero is OOB, and negative
        counts exclude a row from both. Active variance modes replace presets.
        """
        X, ncat, n, p, names, cat_cols, schema = self._prepare_fit_inputs(X)
        y = _response_vector(y, "y")
        if y.shape[0] != n:
            raise ValueError("X and y have inconsistent lengths")
        if not np.all(np.isfinite(y)):
            raise ValueError("y contains NA/Inf")

        cp = self._core_params(n, p)
        cp.use_obs_w = sample_weight is not None
        cp.use_var_prob = var_prob is not None

        if cp.linear_comb == 1:
            cp.split_rule = 1
            cp.linear_comb_method = 1
        else:
            cp.linear_comb_method = self._resolve_lc_method(
                _LC_METHOD_REG, "'naive', 'lm', 'pca', or 'sir'", {1, 2, 3, 4},
                "regression", "naive")
            cp.split_rule = 1

        obsw = (_params.check_obs_weight(sample_weight, n)
                if sample_weight is not None else np.full(n, 1.0 / n))
        varp = (_params.check_var_prob(var_prob, p)
                if var_prob is not None else np.full(p, 1.0 / p))

        obstrack = self._fit_obstrack(cp, n, resample_preset)

        if cp.linear_comb > 1:
            out = RegUniCombForestFit(self._shift_categorical(X, cat_cols), y, ncat,
                                      obsw, varp, obstrack, cp)
        else:
            out = RegUniForestFit(self._shift_categorical(X, cat_cols), y, ncat, obsw,
                                  varp, obstrack, cp)

        self._replace_fit_metadata(names, cat_cols, schema)
        self.forest_ = out["FittedForest"]
        self.n_features_in_ = p
        self.ncat_ = ncat
        self.params_ = cp
        if "Prediction" in out:
            self.oob_prediction_ = np.asarray(out["Prediction"])
            self.oob_error_ = float(out["Error"])
        if "VarImp" in out:
            self.varimp_ = np.asarray(out["VarImp"])
        if "VarVI" in out:
            self.var_vi_ = np.asarray(out["VarVI"])
        if "ObsTrack" in out:
            self.obstrack_ = np.asarray(out["ObsTrack"])
        if "OOBCount" in out:
            self.oob_counts_ = np.asarray(out["OOBCount"])
        return self

    def _predict_raw(self, X, var_est=False, var_mode=None, keep_all=False,
                     ncores=0):
        check_is_fitted(self, "forest_")
        X = self._shift_categorical(self._check_X(X))
        vm = self._resolve_var_mode(var_est, var_mode)
        if self._is_comb():
            return RegUniCombForestPred(
                *self._base_args(), self.forest_["NodeAve"], X, self.ncat_,
                self._obstrack_or_empty(), vm, bool(keep_all),
                self._ncores(ncores), int(self.verbose),
            )
        return RegUniForestPred(
            *self._base_args(), self.forest_["NodeAve"], X, self.ncat_,
            self._obstrack_or_empty(), vm, bool(keep_all),
            self._ncores(ncores), int(self.verbose),
        )

    def predict(self, X):
        """Predict regression targets for X."""
        return np.asarray(self._predict_raw(X)["Prediction"])

    def predict_var(self, X, var_mode=None, keep_all=False, ncores=0):
        """Predict with variance; returns (prediction, variance).

        Requires the forest fitted with var_mode != "none" (or an explicit
        compatible var_mode). Negative variance estimates are set to NaN.
        """
        if keep_all:
            warnings.warn("predict_var(keep_all=True) does not return per-tree results; use predict_details(keep_all=True) instead",
                          FutureWarning, stacklevel=2)
        out = self._predict_raw(X, var_est=True, var_mode=var_mode,
                                keep_all=keep_all, ncores=ncores)
        var = np.asarray(out.get("Variance", np.full(len(X), np.nan)))
        var[var < 0] = np.nan
        return np.asarray(out["Prediction"]), var

    def predict_all_trees(self, X):
        """Per-tree predictions; returns array (n, ntrees)."""
        return np.asarray(self._predict_raw(X, keep_all=True)["PredictionAll"])


# ----------------------------------------------------------------------------
# classification
# ----------------------------------------------------------------------------

class RLT_cla(ClassifierMixin, _BaseRLT):
    """Reinforcement Learning Trees classification forest.

    Parameters mirror :class:`RLT_reg`; classification-specific:

    linear_comb_method : str, default="default"
        Loadings: "lda" (default), "naive", "random", "logistic".
    """

    _tree_extra_keys = ()

    def __init__(
        self,
        n_estimators=None,
        mtry=None,
        min_samples_leaf=5,
        nsplit=0,
        resample_replace=True,
        resample_prob=None,
        importance="none",
        resample_track=False,
        var_mode="none",
        categorical_features=None,
        linear_comb=1,
        linear_comb_method="default",
        alpha=0,
        reinforcement=False,
        embed_ntrees=50,
        embed_mtry=0.5,
        embed_nmin=5,
        embed_nsplit=3,
        embed_resample_replace=True,
        embed_resample_prob=0.9,
        embed_mute=0,
        embed_protect=None,
        embed_threshold=0.25,
        n_jobs=-1,
        verbose=0,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.mtry = mtry
        self.min_samples_leaf = min_samples_leaf
        self.nsplit = nsplit
        self.resample_replace = resample_replace
        self.resample_prob = resample_prob
        self.importance = importance
        self.resample_track = resample_track
        self.var_mode = var_mode
        self.categorical_features = categorical_features
        self.linear_comb = linear_comb
        self.linear_comb_method = linear_comb_method
        self.alpha = alpha
        self.reinforcement = reinforcement
        self.embed_ntrees = embed_ntrees
        self.embed_mtry = embed_mtry
        self.embed_nmin = embed_nmin
        self.embed_nsplit = embed_nsplit
        self.embed_resample_replace = embed_resample_replace
        self.embed_resample_prob = embed_resample_prob
        self.embed_mute = embed_mute
        self.embed_protect = embed_protect
        self.embed_threshold = embed_threshold
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None, var_prob=None, *, resample_preset=None):
        """Fit labels, preserving declared pandas categorical class levels.

        ``resample_preset`` accepts signed integer counts as in RLT_reg.fit.
        """
        X, ncat, n, p, names, cat_cols, schema = self._prepare_fit_inputs(X)
        declared = _encoding.categorical_target(y)
        if declared is not None:
            classes, y_int = declared
        else:
            y_raw = column_or_1d(y, warn=True)
            if y_raw.dtype.kind in "fc" and not np.all(np.isfinite(y_raw)):
                raise ValueError("y contains NA/Inf")
            try:
                check_classification_targets(y_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid classification labels: {exc}") from exc
            classes = np.unique(y_raw)
            if classes.shape[0] < 2:
                raise ValueError("y must have at least 2 distinct classes")
            table = {c: i for i, c in enumerate(classes)}
            y_int = np.asarray([table[v] for v in y_raw], dtype=np.int64)
        if y_int.shape[0] != n:
            raise ValueError("X and y have inconsistent lengths")
        nclass = int(classes.shape[0])

        cp = self._core_params(n, p)
        cp.use_obs_w = sample_weight is not None
        cp.use_var_prob = var_prob is not None

        if cp.linear_comb == 1:
            cp.split_rule = 1
            cp.linear_comb_method = 1
        else:
            cp.linear_comb_method = self._resolve_lc_method(
                _LC_METHOD_CLA, "'lda', 'naive', 'random', or 'logistic'",
                {1, 2, 3, 4}, "classification", "lda")
            cp.split_rule = 1

        obsw = (_params.check_obs_weight(sample_weight, n)
                if sample_weight is not None else np.full(n, 1.0 / n))
        varp = (_params.check_var_prob(var_prob, p)
                if var_prob is not None else np.full(p, 1.0 / p))

        obstrack = self._fit_obstrack(cp, n, resample_preset)

        if cp.linear_comb > 1:
            out = ClaUniCombForestFit(self._shift_categorical(X, cat_cols), y_int, ncat,
                                      nclass, obsw, varp, obstrack, cp)
        else:
            out = ClaUniForestFit(self._shift_categorical(X, cat_cols), y_int, ncat,
                                  nclass, obsw, varp, obstrack, cp)

        self._replace_fit_metadata(names, cat_cols, schema)
        self.classes_ = classes
        self.forest_ = out["FittedForest"]
        self.n_features_in_ = p
        self.ncat_ = ncat
        self.params_ = cp
        if "Prediction" in out:
            self.oob_class_ = np.asarray(out["Prediction"]).astype(int)
            self.oob_prob_ = np.asarray(out["Prob"])
            self.oob_error_ = float(out["Error"])
        if "VarImp" in out:
            self.varimp_ = np.asarray(out["VarImp"])
        if "VarVI" in out:
            self.var_vi_ = np.asarray(out["VarVI"])
        if "ObsTrack" in out:
            self.obstrack_ = np.asarray(out["ObsTrack"])
        if "OOBCount" in out:
            self.oob_counts_ = np.asarray(out["OOBCount"])
        return self

    def _predict_raw(self, X, var_est=False, var_mode=None, keep_all=False,
                     ncores=0):
        check_is_fitted(self, "forest_")
        X = self._shift_categorical(self._check_X(X))
        vm = self._resolve_var_mode(var_est, var_mode)
        if self._is_comb():
            return ClaUniCombForestPred(
                *self._base_args(), self.forest_["NodeProb"], X, self.ncat_,
                self._obstrack_or_empty(), vm, bool(keep_all),
                self._ncores(ncores), int(self.verbose),
            )
        return ClaUniForestPred(
            *self._base_args(), self.forest_["NodeProb"], X, self.ncat_,
            self._obstrack_or_empty(), vm, bool(keep_all),
            self._ncores(ncores), int(self.verbose),
        )

    def predict(self, X):
        """Predict class labels for X."""
        out = self._predict_raw(X)
        return self.classes_[np.asarray(out["Prediction"]).astype(int)]

    def predict_proba(self, X):
        """Class probabilities for X; shape (n, n_classes)."""
        return np.asarray(self._predict_raw(X)["Prob"])

    def predict_log_proba(self, X):
        return np.log(self.predict_proba(X))

    def predict_var(self, X, var_mode=None, keep_all=False, ncores=0):
        """Class probabilities with variance; returns (prob, variance).

        Mirrors R's ``predict(fit, var.est = TRUE)`` for classification
        forests: ``prob`` is the (n, n_classes) class-probability matrix
        (same values as :meth:`predict_proba`) and ``variance`` is the
        (n, n_classes) variance of the probability estimates under the
        forest's ``var_mode`` (matched / IJ / jack). Negative variance
        estimates are set to NaN (R's ``clean.variance``).
        """
        if keep_all:
            warnings.warn("predict_var(keep_all=True) does not return per-tree results; use predict_details(keep_all=True) instead",
                          FutureWarning, stacklevel=2)
        out = self._predict_raw(X, var_est=True, var_mode=var_mode,
                                keep_all=keep_all, ncores=ncores)
        prob = np.asarray(out["Prob"])
        var = np.asarray(out.get(
            "Variance", np.full(prob.shape, np.nan)))
        var = var.copy()
        var[var < 0] = np.nan
        return prob, var


# ----------------------------------------------------------------------------
# survival
# ----------------------------------------------------------------------------

class RLT_surv(_BaseRLT):
    """Reinforcement Learning Trees survival forest.

    y must be a structured array with fields ``event`` (0/1) and ``time``
    (scikit-survival convention), or a ``(time, event)`` pair of arrays.

    Survival-specific parameters
    ----------------------------
    split_rule : {"logrank", "suplogrank", "coxgrad"}, default="logrank"
    time_grid_size : int, default=0
        Training failure-grid size check (0 = all). A positive value cannot
        reduce the full grid; use band_grid_size when predicting.
    linear_comb_method : str, default="default"
        Survival loadings: "coxph" (default) or "naive".
    """

    _tree_extra_keys = ()

    def __init__(
        self,
        n_estimators=None,
        mtry=None,
        min_samples_leaf=5,
        nsplit=0,
        resample_replace=True,
        resample_prob=None,
        importance="none",
        resample_track=False,
        var_mode="none",
        split_rule="logrank",
        categorical_features=None,
        linear_comb=1,
        linear_comb_method="default",
        alpha=0,
        reinforcement=False,
        time_grid_size=0,
        embed_ntrees=50,
        embed_mtry=0.5,
        embed_nmin=5,
        embed_nsplit=3,
        embed_resample_replace=True,
        embed_resample_prob=0.9,
        embed_mute=0,
        embed_protect=None,
        embed_threshold=0.25,
        n_jobs=-1,
        verbose=0,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.mtry = mtry
        self.min_samples_leaf = min_samples_leaf
        self.nsplit = nsplit
        self.resample_replace = resample_replace
        self.resample_prob = resample_prob
        self.importance = importance
        self.resample_track = resample_track
        self.var_mode = var_mode
        self.split_rule = split_rule
        self.categorical_features = categorical_features
        self.linear_comb = linear_comb
        self.linear_comb_method = linear_comb_method
        self.alpha = alpha
        self.reinforcement = reinforcement
        self.time_grid_size = time_grid_size
        self.embed_ntrees = embed_ntrees
        self.embed_mtry = embed_mtry
        self.embed_nmin = embed_nmin
        self.embed_nsplit = embed_nsplit
        self.embed_resample_replace = embed_resample_replace
        self.embed_resample_prob = embed_resample_prob
        self.embed_mute = embed_mute
        self.embed_protect = embed_protect
        self.embed_threshold = embed_threshold
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None, var_prob=None, *, resample_preset=None):
        """Fit right-censored outcomes with optional signed resampling counts.

        ``resample_preset`` follows RLT_reg.fit and is replaced by active
        variance-mode sampling. The full training failure grid is retained.
        """
        X, ncat, n, p, names, cat_cols, schema = self._prepare_fit_inputs(X)
        time, event = _validate_surv_y(y)
        if time.shape[0] != n or event.shape[0] != n:
            raise ValueError("X and y have inconsistent lengths")
        if not np.all(np.isfinite(time)):
            raise ValueError("NA/Inf not permitted in y")
        event_f = np.asarray(event, dtype=np.float64)
        if not np.all(np.isfinite(event_f)):
            raise ValueError("NA not permitted in censor")
        if not np.all(np.isin(event_f, (0.0, 1.0))):
            raise ValueError("censor must be 0 or 1")
        event = event_f.astype(np.int64)

        # Fit on the complete sorted failure-time grid.
        timepoints = np.sort(np.unique(time[event == 1]))
        if timepoints.shape[0] == 0:
            raise ValueError("no observed failures: event is all zero")
        tgs = _params.check_grid_size(self.time_grid_size, "time_grid_size")
        if 0 < tgs < timepoints.size:
            raise ValueError("time_grid_size cannot reduce the training failure-time grid; use band_grid_size at prediction time")

        # map observed times onto the 1-based integer grid (R logic)
        y_point = np.empty(n, dtype=np.int64)
        for i in range(n):
            if event[i] == 1:
                y_point[i] = int(np.searchsorted(timepoints, time[i],
                                                 side="left")) + 1
            else:
                y_point[i] = int(np.searchsorted(timepoints, time[i],
                                                 side="right"))
        censor = event.astype(np.int64)

        cp = self._core_params(n, p)
        cp.use_obs_w = sample_weight is not None
        cp.use_var_prob = var_prob is not None

        cp.split_rule = self._method_code(
            self.split_rule, _SPLIT_RULES_SURV, "'logrank', 'suplogrank', or 'coxgrad'",
            "split_rule", "logrank")

        if cp.linear_comb == 1:
            cp.linear_comb_method = 1
        else:
            cp.linear_comb_method = self._resolve_lc_method(
                _LC_METHOD_SURV, "'coxph', or 'naive'", {1, 2},
                "survival", "coxph")

        obsw = (_params.check_obs_weight(sample_weight, n)
                if sample_weight is not None else np.full(n, 1.0 / n))
        varp = (_params.check_var_prob(var_prob, p)
                if var_prob is not None else np.full(p, 1.0 / p))

        obstrack = self._fit_obstrack(cp, n, resample_preset)

        if cp.linear_comb > 1:
            out = SurvUniCombForestFit(self._shift_categorical(X, cat_cols), y_point,
                                       censor, ncat, obsw, varp, obstrack, cp)
        else:
            out = SurvUniForestFit(self._shift_categorical(X, cat_cols), y_point, censor,
                                   ncat, obsw, varp, obstrack, cp)

        self._replace_fit_metadata(names, cat_cols, schema)
        self.forest_ = out["FittedForest"]
        self.n_features_in_ = p
        self.ncat_ = ncat
        self.params_ = cp
        self.timepoints_ = timepoints
        self.nfail_ = int(out["NFail"])
        if "Prediction" in out:
            self.oob_hazard_ = np.asarray(out["Prediction"])
            self.oob_error_ = float(out["Error"])
        if "VarImp" in out:
            self.varimp_ = np.asarray(out["VarImp"])
        if "VarVI" in out:
            self.var_vi_ = np.asarray(out["VarVI"])
        if "ObsTrack" in out:
            self.obstrack_ = np.asarray(out["ObsTrack"])
        if "OOBCount" in out:
            self.oob_counts_ = np.asarray(out["OOBCount"])
        return self

    def _grid_and_mapping(self, band_grid_size):
        original = self.timepoints_
        nfail = original.shape[0]
        new_grid = original
        bgs = _params.check_grid_size(band_grid_size, "band_grid_size")
        if bgs > 0:
            eff = min(bgs, nfail)
            if eff < nfail:
                probs = np.linspace(0.05, 1.0, eff)
                new_grid = np.unique(
                    np.quantile(original, probs, method="inverted_cdf")
                ).astype(np.float64)
        new_grid = np.sort(new_grid)
        mapping = np.searchsorted(original, new_grid, side="left")
        return new_grid, mapping.astype(np.int64)

    def _predict_raw(self, X, var_est=False, var_mode=None, keep_all=False,
                     ncores=0, band_grid_size=0):
        check_is_fitted(self, "forest_")
        X = self._shift_categorical(self._check_X(X))
        vm = self._resolve_var_mode(var_est, var_mode)
        _, mapping = self._grid_and_mapping(band_grid_size)
        if self._is_comb():
            return SurvUniCombForestPred(
                *self._base_args(), self.forest_["NodeHaz"], X, self.ncat_,
                self.nfail_, mapping, self._obstrack_or_empty(), vm,
                bool(keep_all), self._ncores(ncores), int(self.verbose),
            )
        return SurvUniForestPred(
            *self._base_args(), self.forest_["NodeHaz"], X, self.ncat_,
            self.nfail_, mapping, self._obstrack_or_empty(), vm,
            bool(keep_all), self._ncores(ncores), int(self.verbose),
        )

    def predict_survival_function(self, X, band_grid_size=0):
        """Survival probabilities S(t | x); shape (n, T)."""
        out = self._predict_raw(X, band_grid_size=band_grid_size)
        return np.asarray(out["Survival"])

    def predict_cumulative_hazard_function(self, X, band_grid_size=0):
        """Cumulative hazard H(t | x); shape (n, T)."""
        out = self._predict_raw(X, band_grid_size=band_grid_size)
        return np.asarray(out["CHF"])

    def predict(self, X, band_grid_size=0):
        """Default survival prediction: S(t | x), shape (n, T)."""
        return self.predict_survival_function(X, band_grid_size=band_grid_size)

    def predict_risk(self, X):
        """Risk score (cumulative hazard summed over the grid)."""
        chf = self.predict_cumulative_hazard_function(X)
        return chf.sum(axis=1)

    def predict_var(self, X, var_mode=None, band_grid_size=0, ncores=0):
        """Prediction with covariance; returns (survival, cov).

        cov has shape (n, T, T), the per-subject covariance of the CHF.
        Requires a fitted variance mode or an explicit compatible var_mode.
        """
        out = self._predict_raw(X, var_est=True, var_mode=var_mode,
                                band_grid_size=band_grid_size, ncores=ncores)
        return np.asarray(out["Survival"]), np.asarray(out["Cov"])

    def score(self, X, y):
        """Harrell's c-index (higher is better)."""
        time, event = _validate_surv_y(y)
        risk = self.predict_risk(X)
        if time.size != risk.size:
            raise ValueError("X and y have inconsistent lengths")
        if not np.all(np.isfinite(time)) or not np.all(np.isin(event, (0, 1))):
            raise ValueError("survival time must be finite and event must be 0 or 1")
        return float(cindex(time, event, risk))
