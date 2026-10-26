"""Live corrected-R parity for labeled schemas and signed resampling workflows.

Set RLT_RSCRIPT and the shared R library configuration to the reviewed reference.
The package version alone does not identify that corrected source. These tests
make no private R library and skip visibly when R is unavailable or disabled.
"""
import os
import shutil
import subprocess
import warnings

import numpy as np
import pytest

from rlt import RLT_cla, RLT_reg, RLT_surv


RSCRIPT = os.environ.get("RLT_RSCRIPT") or shutil.which("Rscript")
pytestmark = pytest.mark.skipif(
    RSCRIPT is None or os.environ.get("RLT_SKIP_R_TESTS") is not None,
    reason="R + reviewed corrected RLT reference not available",
)


def _load_array(directory, name):
    shape = tuple(map(int, (directory / f"{name}.shape").read_text().strip().split(",")))
    return np.loadtxt(directory / f"{name}.txt", ndmin=1).reshape(shape, order="F")


def _compare(directory, name, actual):
    expected = _load_array(directory, name)
    assert np.shape(actual) == expected.shape, name
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12,
                               equal_nan=True, err_msg=name)


@pytest.mark.parametrize("model", ["regression", "classification", "survival"])
@pytest.mark.parametrize("linear_comb", [1, 2])
@pytest.mark.parametrize("preset", [False, True])
def test_live_R_labeled_schema_and_signed_sampling_workflow(tmp_path, model, linear_comb, preset):
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(1911)
    n, trees = 48, 8
    labels = np.asarray(["z", "a", "m"])
    raw = pd.DataFrame({
        "nominal": labels[rng.integers(0, 3, n)],
        "ordered": labels[rng.integers(0, 3, n)],
        "character": np.asarray(["u", "v", "w"])[rng.integers(0, 3, n)],
        "numeric_labels": np.asarray([10, 2, 1])[rng.integers(0, 3, n)],
        "continuous": rng.normal(size=n),
        "continuous2": rng.normal(size=n),
    })
    frame = raw.copy()
    frame["nominal"] = pd.Categorical(raw["nominal"], categories=["z", "a", "m", "unused"])
    frame["ordered"] = pd.Categorical(raw["ordered"], categories=["m", "z", "a"], ordered=True)
    frame["numeric_labels"] = pd.Categorical(raw["numeric_labels"], categories=[10, 2, 1])
    signal = raw["continuous"].to_numpy() + .7 * raw["continuous2"].to_numpy()
    regression = signal + .2 * (raw["nominal"].to_numpy() == "z")
    classes = pd.Categorical(np.where(signal > 0, "z", "a"), categories=["z", "a", "unused"])
    time = np.maximum(.2, np.round(rng.exponential(np.exp(-.3 * signal)) * 3) / 3)
    event = (np.arange(n) % 4 != 0).astype(int)
    y = {"regression": regression, "classification": classes, "survival": (time, event)}[model]
    counts = np.zeros((n, trees), dtype=np.int64)
    for tree in range(trees):
        counts[(np.arange(12) + tree) % 24, tree] = 2
        counts[32:, tree] = -1
    weights = 1 + np.arange(n) % 5
    raw.to_csv(tmp_path / "X.csv", index=False)
    np.savetxt(tmp_path / "y.csv", np.column_stack([regression, signal > 0, time, event]), delimiter=",", fmt="%.17g")
    np.savetxt(tmp_path / "preset.csv", counts, delimiter=",", fmt="%d")
    np.savetxt(tmp_path / "weights.txt", weights, fmt="%d")
    cls = {"regression": RLT_reg, "classification": RLT_cla, "survival": RLT_surv}[model]
    kwargs = dict(n_estimators=trees, mtry=3, min_samples_leaf=4, n_jobs=1,
                  random_state=1237, linear_comb=linear_comb, linear_comb_method="naive",
                  embed_nmin=100, resample_track=True, importance="permute")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="y has unused declared classes")
        fitted = cls(**kwargs).fit(frame, y, sample_weight=weights,
                                  resample_preset=counts if preset else None)
    # New data reverses declared level and column order. Training mappings win.
    query = frame.iloc[:7].copy()
    query["nominal"] = query["nominal"].cat.reorder_categories(["unused", "m", "a", "z"])
    query = query[list(frame.columns)[::-1]]

    root = tmp_path.as_posix()
    script = f'''
library(RLT)
root <- "{root}"
write_array <- function(x, name) {{
  shape <- dim(x)
  if (is.null(shape)) shape <- length(x)
  writeLines(paste(shape, collapse=","), file.path(root,paste0(name,".shape")))
  values <- sprintf("%.17g", as.double(x))
  values[is.na(as.double(x))] <- "nan"
  writeLines(values, file.path(root,paste0(name,".txt")))
}}
X <- read.csv(file.path(root,"X.csv"), stringsAsFactors=FALSE)
X$nominal <- factor(X$nominal, levels=c("z","a","m","unused"))
X$ordered <- ordered(X$ordered, levels=c("m","z","a"))
X$numeric_labels <- factor(X$numeric_labels, levels=c(10,2,1))
outcomes <- as.matrix(read.table(file.path(root,"y.csv"), sep=","))
y <- outcomes[,1]
if ("{model}" == "classification")
  y <- factor(ifelse(outcomes[,2] == 1,"z","a"), levels=c("z","a","unused"))
if ("{model}" == "survival") y <- outcomes[,3]
args <- list(x=X, y=y, model="{model}", ntrees={trees}L, mtry=3L, nmin=4L,
             seed=1237L, ncores=1L, linear.comb={linear_comb}L,
             linear.comb.method=if ({linear_comb}L > 1L) "naive" else "default", importance="permute",
             obs.w=scan(file.path(root,"weights.txt"),quiet=TRUE),
             param.control=list(resample.track=TRUE, embed.nmin=100L))
if ("{model}" == "survival") args$censor <- outcomes[,4]
if ({str(preset).upper()}) args$resample.preset <- as.matrix(read.table(file.path(root,"preset.csv"),sep=","))
fit <- do.call(RLT,args)
write_array(as.vector(fit$ncat), "ncat")
write_array(data.matrix(X), "encoded")
write_array(fit$ObsTrack, "tracking")
write_array(as.vector(fit$VarImp), "importance")
write_array(fit$Error, "error")
if (!is.null(fit$OOBCount)) write_array(as.vector(fit$OOBCount), "oob_counts")
query <- X[1:7,,drop=FALSE]
query$nominal <- factor(as.character(query$nominal),levels=rev(levels(X$nominal)))
query <- query[,rev(names(query)),drop=FALSE]
grids <- if ("{model}" == "survival") c(0L,4L) else 0L
for (grid in grids) {{
  pred <- predict(fit,query,keep.all=TRUE,band.grid.size=grid,ncores=1L)
  prefix <- paste0("grid",grid,"_")
  if ("{model}" == "survival") {{
    for (field in c("Hazard","CHF","Survival","timepoints")) write_array(pred[[field]],paste0(prefix,field))
    write_array(aperm(pred$AllHazard,c(3,1,2)),paste0(prefix,"AllHazard"))
  }} else if ("{model}" == "classification") {{
    writeLines(as.character(pred$Prediction),file.path(root,paste0(prefix,"labels.txt")))
    writeLines(fit$ylabels,file.path(root,"classes.txt"))
    write_array(pred$Prob,paste0(prefix,"Prob"))
    write_array(aperm(pred$PredictionAll,c(3,1,2)),paste0(prefix,"PredictionAll"))
  }} else {{
    write_array(as.vector(pred$Prediction),paste0(prefix,"Prediction"))
    write_array(pred$PredictionAll,paste0(prefix,"PredictionAll"))
  }}
}}
for (tree in seq_len({trees}L)) {{
  for (field in c("SplitVar","SplitValue","LeftNode","RightNode","NodeWeight")) {{
    value <- fit$FittedForest[[field]][[tree]]
    if (field != "SplitVar" || {linear_comb}L == 1L) value <- as.vector(value)
    write_array(value,paste0("tree",tree-1L,"_",field))
  }}
  if ({linear_comb}L > 1L) write_array(fit$FittedForest$SplitLoad[[tree]],paste0("tree",tree-1L,"_SplitLoad"))
}}
write_array(forest.kernel(fit,X1=query)$Kernel,"kernel_self")
write_array(forest.kernel(fit,X1=query,X2=X[8:13,,drop=FALSE])$Kernel,"kernel_cross")
write_array(forest.kernel(fit,X1=query,X2=X,vs.train=TRUE)$Kernel,"kernel_train")
oob <- forest.kernel(fit,X1=X,oob=TRUE)
for (field in c("Kernel","N","C")) write_array(oob[[field]],paste0("kernel_oob_",field))
'''
    script_path = tmp_path / "workflow.R"
    script_path.write_text(script, encoding="utf-8")
    result = subprocess.run([RSCRIPT, "--vanilla", str(script_path)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    _compare(tmp_path, "ncat", fitted.ncat_)
    _compare(tmp_path, "encoded", fitted._shift_categorical(fitted._check_X(frame)))
    _compare(tmp_path, "tracking", fitted.obstrack_)
    _compare(tmp_path, "importance", fitted.varimp_)
    _compare(tmp_path, "error", [fitted.oob_error_])
    if (tmp_path / "oob_counts.txt").exists():
        _compare(tmp_path, "oob_counts", fitted.oob_counts_)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X feature names do not match")
        for grid in ([0, 4] if model == "survival" else [0]):
            details = fitted.predict_details(query, keep_all=True, band_grid_size=grid)
            prefix = f"grid{grid}_"
            fields = {"regression": ["Prediction", "PredictionAll"],
                      "classification": ["Prob", "PredictionAll"],
                      "survival": ["Hazard", "CHF", "Survival", "AllHazard", "timepoints"]}[model]
            for field in fields:
                _compare(tmp_path, prefix + field, details[field])
            if model == "classification":
                assert details["Prediction"].tolist() == (tmp_path / f"{prefix}labels.txt").read_text().splitlines()
                assert fitted.classes_.tolist() == (tmp_path / "classes.txt").read_text().splitlines()
        for tree in range(trees):
            for field in ("SplitVar", "SplitValue", "LeftNode", "RightNode", "NodeWeight") + (("SplitLoad",) if linear_comb > 1 else ()):
                _compare(tmp_path, f"tree{tree}_{field}", fitted.forest_[field][tree])
        _compare(tmp_path, "kernel_self", fitted.forest_kernel(query))
        _compare(tmp_path, "kernel_cross", fitted.forest_kernel(query, frame.iloc[7:13]))
        _compare(tmp_path, "kernel_train", fitted.forest_kernel(query, frame, vs_train=True))
        oob = fitted.forest_kernel(frame, oob=True)
        for field in ("Kernel", "N", "C"):
            _compare(tmp_path, f"kernel_oob_{field}", oob[field])
    if preset:
        np.testing.assert_array_equal(fitted.obstrack_, counts)
        assert np.all(oob["N"][32:, :] == 0)
