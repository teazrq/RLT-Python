"""Live parity for corrected uncertainty against the installed current RLT.

Set RLT_RSCRIPT if Rscript is outside PATH. R uses the caller's shared
library configuration, including R_LIBS_USER. No private R library is made.
"""

import os
import shutil
import subprocess

import numpy as np
import pytest

from rlt import RLT_cla, RLT_reg, RLT_surv


RSCRIPT = os.environ.get("RLT_RSCRIPT") or shutil.which("Rscript")
pytestmark = pytest.mark.skipif(
    RSCRIPT is None or os.environ.get("RLT_SKIP_R_TESTS") is not None,
    reason="R + current RLT not available",
)


def _load_array(directory, name):
    shape = tuple(int(v) for v in (directory / f"{name}.shape").read_text().strip().split(","))
    values = np.loadtxt(directory / f"{name}.txt", ndmin=1)
    assert values.size == np.prod(shape), name
    return values.reshape(shape, order="F")


def _compare(actual, expected, field, *, covariance=False):
    actual = np.asarray(actual)
    assert actual.shape == expected.shape, field
    if field == "VarVI" or field.endswith(("Variance", "Cov")):
        assert np.isfinite(expected).all(), field
        assert np.any(np.abs(expected) > 1e-14), f"{field} fixture needs nonzero uncertainty"
    # Seventeen-digit serialization preserves double precision. Covariance
    # eigendecompositions can differ in last bits across BLAS/LAPACK builds.
    np.testing.assert_allclose(actual, expected,
                               rtol=1e-11 if covariance else 1e-12,
                               atol=1e-12, equal_nan=True, err_msg=field)


@pytest.mark.parametrize("model", ["regression", "classification", "survival"])
@pytest.mark.parametrize("mode", ["matched", "ij"])
@pytest.mark.parametrize("linear_comb", [1, 2])
def test_live_R_uncertainty_matches_tracking_trees_and_predictions(tmp_path, model, mode, linear_comb):
    rng = np.random.default_rng(7021)
    n, p, ntrees, seed = 48, 4, 24, 1237
    X = rng.normal(size=(n, p))
    query = X[:5] + .05
    signal = X[:, 0] + .7 * X[:, 1] + .2 * X[:, 2] * X[:, 3]
    if model == "classification":
        y = (signal + rng.normal(scale=.3, size=n) > 0).astype(np.int64)
    elif model == "survival":
        time = np.maximum(.2, np.round(rng.exponential(np.exp(-.5 * signal)) * 3) / 3)
        event = (np.arange(n) % 4 != 0).astype(np.int64)
        y = (time, event)
    else:
        y = signal + rng.normal(scale=.2, size=n)

    np.savetxt(tmp_path / "X.csv", X, delimiter=",", fmt="%.17g")
    np.savetxt(tmp_path / "query.csv", query, delimiter=",", fmt="%.17g")
    np.savetxt(tmp_path / "y.txt", y[0] if model == "survival" else y, fmt="%.17g")
    if model == "survival":
        np.savetxt(tmp_path / "event.txt", y[1], fmt="%d")

    # Use marginal screening with actual multi-variable naive LC splits,
    # avoiding an unnecessary embedded forest in these uncertainty fixtures.
    cls = {"regression": RLT_reg, "classification": RLT_cla, "survival": RLT_surv}[model]
    fitted = cls(n_estimators=ntrees, min_samples_leaf=4, mtry=3,
                 nsplit=0, var_mode=mode, linear_comb=linear_comb,
                 linear_comb_method="naive", embed_nmin=100,
                 random_state=seed, n_jobs=1).fit(X, y)
    if linear_comb > 1:
        assert any(np.any(np.count_nonzero(np.abs(load) > 1e-12, axis=1) > 1)
                   for load in fitted.forest_["SplitLoad"])

    r_mode = "IJ" if mode == "ij" else "matched"
    root = tmp_path.as_posix()
    script = f'''
library(RLT)
stopifnot(packageVersion("RLT") >= "6.1.1")
root <- "{root}"
write_array <- function(x, name) {{
  shape <- dim(x)
  if (is.null(shape)) shape <- length(x)
  writeLines(paste(shape, collapse=","), file.path(root, paste0(name, ".shape")))
  values <- sprintf("%.17g", as.double(x))
  values[is.na(as.double(x))] <- "nan"
  writeLines(values, file.path(root, paste0(name, ".txt")))
}}
X <- as.matrix(read.table(file.path(root,"X.csv"), sep=","))
query <- as.matrix(read.table(file.path(root,"query.csv"), sep=","))
y <- scan(file.path(root,"y.txt"), quiet=TRUE)
if ("{model}" == "classification") y <- factor(y, levels=c(0,1))
args <- list(x=X, y=y, model="{model}", ntrees={ntrees}L, nmin=4L,
             mtry=3L, nsplit=0L, var.mode="{r_mode}", linear.comb={linear_comb}L,
             linear.comb.method=if ({linear_comb}L > 1L) "naive" else "default",
             param.control=list(embed.nmin=100L),
             ncores=1L, seed={seed}L, verbose=0L)
if ("{model}" == "survival") args$censor <- scan(file.path(root,"event.txt"), quiet=TRUE)
fit <- do.call(RLT, args)
write_array(fit$ObsTrack, "tracking")
if ("{mode}" == "matched") write_array(as.vector(fit$VarVI), "importance_variance")
for (grid in if ("{model}" == "survival") c(0L,4L) else 0L) {{
  pred <- predict(fit, query, var.est=TRUE, keep.all=TRUE,
                  band.grid.size=grid, ncores=1L)
  prefix <- paste0("grid",grid,"_")
  if ("{model}" == "survival") {{
    for (field in c("Hazard","CHF","Survival","timepoints"))
      write_array(pred[[field]], paste0(prefix,field))
    write_array(aperm(pred$Cov,c(3,1,2)), paste0(prefix,"Cov"))
    write_array(aperm(pred$AllHazard,c(3,1,2)), paste0(prefix,"AllHazard"))
  }} else if ("{model}" == "classification") {{
    write_array(as.integer(pred$Prediction)-1L, paste0(prefix,"Prediction"))
    for (field in c("Prob","Variance")) write_array(pred[[field]], paste0(prefix,field))
    write_array(aperm(pred$PredictionAll,c(3,1,2)), paste0(prefix,"PredictionAll"))
  }} else {{
    for (field in c("Prediction","Variance"))
      write_array(as.vector(pred[[field]]), paste0(prefix,field))
    write_array(pred$PredictionAll, paste0(prefix,"PredictionAll"))
  }}
}}
'''
    script_path = tmp_path / "uncertainty.R"
    script_path.write_text(script, encoding="utf-8")
    result = subprocess.run([RSCRIPT, "--vanilla", str(script_path)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr

    # First distinguish RNG/sampling differences from prediction algebra.
    tracking = _load_array(tmp_path, "tracking")
    np.testing.assert_array_equal(fitted.obstrack_, tracking, err_msg="in-bag counts")
    if mode == "matched":
        half = ntrees // 2
        np.testing.assert_array_equal(tracking[:, :half] + tracking[:, half:], np.ones((n, half)))
        assert np.all(tracking.sum(axis=0) == n // 2)
        _compare(fitted.var_vi_, _load_array(tmp_path, "importance_variance"), "VarVI")
    else:
        assert np.any(tracking.mean(axis=1) % 1 != 0), "IJ fixture needs fractional empirical count means"

    for grid in ([0, 4] if model == "survival" else [0]):
        prefix = f"grid{grid}_"
        kwargs = {"band_grid_size": grid} if model == "survival" else {}
        out = fitted._predict_raw(query, var_est=True, keep_all=True, ncores=1, **kwargs)
        if model == "survival":
            # Per-tree hazards expose tree/routing differences before the
            # cumulative-hazard and covariance comparisons.
            fields = ["AllHazard", "Hazard", "CHF", "Survival", "Cov"]
            _compare(fitted._grid_and_mapping(grid)[0],
                     _load_array(tmp_path, prefix + "timepoints"), "timepoints")
        elif model == "classification":
            fields = ["PredictionAll", "Prob", "Prediction", "Variance"]
        else:
            fields = ["PredictionAll", "Prediction", "Variance"]
        for field in fields:
            expected = _load_array(tmp_path, prefix + field)
            if field == "Prediction" and model == "classification":
                np.testing.assert_array_equal(out[field], expected, err_msg=field)
            else:
                _compare(out[field], expected, prefix + field,
                         covariance=field == "Cov")
