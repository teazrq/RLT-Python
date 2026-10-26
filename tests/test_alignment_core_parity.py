"""Focused live-R coverage of embedded, weighted and singular LC branches."""

import os
import shutil
import subprocess

import numpy as np
import pytest

from rlt import RLT_reg, RLT_cla, RLT_surv


RSCRIPT = os.environ.get("RLT_RSCRIPT") or shutil.which("Rscript")
pytestmark = pytest.mark.skipif(
    RSCRIPT is None or os.environ.get("RLT_SKIP_R_TESTS") is not None,
    reason="R + corrected RLT reference not available",
)


def _data(variant):
    rng = np.random.default_rng(45682)
    X = rng.normal(size=(64, 5))
    X[:, 3] += .35 * X[:, 0]
    signal = 1.5 * X[:, 0] - .7 * X[:, 1] + .25 * X[:, 2] * X[:, 3]
    regression = signal + rng.normal(scale=.3, size=64)
    classification = np.digitize(signal + rng.normal(scale=.4, size=64), [-.7, .7])
    time = np.maximum(.25, np.round(rng.exponential(np.exp(-.4 * signal)) * 4) / 4)
    if variant == "categorical":
        X[:, 0] = np.resize([0., 1., 2., 3.], 64)
        group = np.isin(X[:, 0], [0, 2])
        regression = 3 * group + .1 * X[:, 1]
        classification = group.astype(int)
        time = np.where(group, .5, 4.) + .05 * np.abs(X[:, 1])
    elif variant == "collinear":
        X[:, 1], X[:, 2], X[:, 4] = X[:, 0], 1., 0.
    event = (np.arange(64) % 5 != 0).astype(int)
    return X, regression, classification, time, event


CASES = [("embedded", "regression"), ("embedded", "classification")]
CASES += [(variant, family) for variant in ("categorical", "collinear")
          for family in ("regression", "classification", "survival")]


@pytest.mark.parametrize("variant,family", CASES)
def test_live_R_embedded_weighted_and_rank_deficient_native_branches(tmp_path, variant, family):
    X, regression, classification, time, event = _data(variant)
    query = X[:6].copy()
    cls = {"regression": RLT_reg, "classification": RLT_cla, "survival": RLT_surv}[family]
    response = {"regression": regression, "classification": classification,
                "survival": (time, event)}[family]
    comb = variant != "categorical"
    weights = np.resize([1., 2., 3., 2.], 64)
    var_prob = np.array([1., 2., 4., 2., 1.])
    for name, value in dict(X=X, query=query, y=(time if family == "survival" else response),
                            event=event, weights=weights, var_prob=var_prob).items():
        np.savetxt(tmp_path / (name + ".csv"), value, delimiter=",", fmt="%.17g")
    # The categorical cases also compare one/two-core permutation importance.
    cores = [1, 2] if variant == "categorical" else [1]
    fitted = {}
    for ncores in cores:
        model = cls(n_estimators=18, min_samples_leaf=6, mtry=4, nsplit=0,
                    linear_comb=2 if comb else 1, alpha=.1,
                    categorical_features=[0] if variant == "categorical" else None,
                    embed_ntrees=12, embed_nmin=4 if variant == "embedded" else 100,
                    embed_mtry=.6, embed_nsplit=1, embed_resample_replace=False,
                    embed_resample_prob=.7, embed_mute=.2, embed_protect=2,
                    resample_replace=True, resample_prob=.8, resample_track=True,
                    importance="permute", random_state=8831, n_jobs=ncores)
        kwargs = dict(sample_weight=weights, var_prob=var_prob) if variant == "categorical" else {}
        fitted[ncores] = model.fit(X, response, **kwargs)
        if comb:
            assert any(np.any(np.count_nonzero(np.abs(a) > 1e-12, axis=1) > 1)
                       for a in model.forest_["SplitLoad"])
        else:
            assert any(np.any(a == 0) for a in model.forest_["SplitVar"])
    script = f'''
library(RLT)
root <- "{tmp_path.as_posix()}"
read <- function(name) as.matrix(read.table(file.path(root,paste0(name,'.csv')),sep=','))
X <- read('X'); query <- read('query'); y <- as.vector(read('y'))
if ('{family}'=='classification') y <- factor(y,levels=sort(unique(y)))
if ('{variant}'=='categorical') {{
  levels <- sort(unique(X[,1])); X <- as.data.frame(X); query <- as.data.frame(query)
  X[,1] <- factor(X[,1],levels=levels); query[,1] <- factor(query[,1],levels=levels)
}}
write_array <- function(value,name) {{
  shape <- dim(value); if(is.null(shape)) shape <- length(value)
  writeLines(paste(shape,collapse=','),file.path(root,paste0(name,'.shape')))
  text <- sprintf('%.17g',as.double(value)); text[is.na(as.double(value))] <- 'nan'
  writeLines(text,file.path(root,paste0(name,'.txt')))
}}
for (cores in c({','.join(map(str, cores))})) {{
  args <- list(x=X,y=y,model='{family}',ntrees=18L,nmin=6L,mtry=4L,nsplit=0L,
    linear.comb={2 if comb else 1}L,alpha=.1,importance='permute',
    resample.replace=TRUE,resample.prob=.8,seed=8831L,ncores=cores,
    param.control=list(resample.track=TRUE,embed.ntrees=12L,
      embed.nmin={4 if variant == 'embedded' else 100}L,embed.mtry=.6,embed.nsplit=1L,
      embed.resample.replace=FALSE,embed.resample.prob=.7,embed.mute=.2,embed.protect=2L))
  if ('{family}'=='survival') args$censor <- as.vector(read('event'))
  if ('{variant}'=='categorical') {{args$obs.w <- as.vector(read('weights'));args$var.prob <- as.vector(read('var_prob'))}}
  fit <- do.call(RLT,args); prefix <- paste0(cores,'_')
  write_array(fit$ObsTrack,paste0(prefix,'tracking'))
  write_array(as.vector(fit$VarImp),paste0(prefix,'importance'))
  for (field in setdiff(names(fit$FittedForest),'NodeHaz'))
    for (tree in seq_along(fit$FittedForest[[field]]))
      write_array(fit$FittedForest[[field]][[tree]],paste0(prefix,'tree',tree-1L,'_',field))
  if ('NodeHaz' %in% names(fit$FittedForest))
    for (tree in seq_along(fit$FittedForest$NodeHaz))
      for (node in seq_along(fit$FittedForest$NodeHaz[[tree]]))
        write_array(fit$FittedForest$NodeHaz[[tree]][[node]],paste0(prefix,'tree',tree-1L,'_NodeHaz',node-1L))
  pred <- predict(fit,query,keep.all=TRUE,ncores=cores)
  if ('{family}'=='survival') {{
    for (field in c('Hazard','CHF','Survival')) write_array(pred[[field]],paste0(prefix,field))
    write_array(aperm(pred$AllHazard,c(3,1,2)),paste0(prefix,'AllHazard'))
  }} else if ('{family}'=='classification') {{
    write_array(pred$Prob,paste0(prefix,'Prob'))
    write_array(aperm(pred$PredictionAll,c(3,1,2)),paste0(prefix,'PredictionAll'))
  }} else {{
    write_array(as.vector(pred$Prediction),paste0(prefix,'Prediction'))
    write_array(pred$PredictionAll,paste0(prefix,'PredictionAll'))
  }}
}}
'''
    source = tmp_path / "branches.R"
    source.write_text(script, encoding="utf-8")
    process = subprocess.run([RSCRIPT, "--vanilla", str(source)],
                             capture_output=True, text=True, timeout=60)
    assert process.returncode == 0, process.stdout + process.stderr

    def reference(name):
        shape = tuple(map(int, (tmp_path / (name + ".shape")).read_text().split(",")))
        value = np.loadtxt(tmp_path / (name + ".txt"), ndmin=1) if np.prod(shape) else np.empty(0)
        return value.reshape(shape, order="F")

    for ncores, model in fitted.items():
        prefix = str(ncores) + "_"
        np.testing.assert_array_equal(model.obstrack_, reference(prefix + "tracking"))
        np.testing.assert_allclose(model.varimp_, reference(prefix + "importance"), rtol=1e-12, atol=1e-12)
        for field, trees in model.forest_.items():
            for tree, values in enumerate(trees):
                name = prefix + f"tree{tree}_{field}"
                if field == "NodeHaz":
                    for node, value in enumerate(values):
                        np.testing.assert_allclose(value, reference(name + str(node)).ravel(), rtol=1e-12, atol=1e-12)
                    continue
                target = reference(name)
                # Rcpp wraps native vectors as one-column matrices. The
                # Python binding exposes those same vectors as one dimension.
                if np.asarray(values).ndim == 1:
                    target = target.ravel()
                if variant == "collinear" and family == "survival" and field == "SplitLoad":
                    # Exact duplicate predictors do not identify individual
                    # coefficients. Test their identifiable projection and
                    # all saved routing/prediction fields at strict tolerance.
                    ids = model.forest_["SplitVar"][tree]
                    for node in np.flatnonzero(ids[:, 0] >= 0):
                        np.testing.assert_allclose(X[:, ids[node]] @ values[node],
                            X[:, ids[node]] @ target[node], rtol=1e-12, atol=1e-12)
                elif field in ("SplitVar", "LeftNode", "RightNode"):
                    np.testing.assert_array_equal(values, target)
                else:
                    np.testing.assert_allclose(values, target, rtol=1e-12, atol=1e-12)
        details = model.predict_details(query, keep_all=True)
        for field in ("Hazard", "CHF", "Survival", "AllHazard") if family == "survival" else (
                ("Prob", "PredictionAll") if family == "classification" else ("Prediction", "PredictionAll")):
            np.testing.assert_allclose(details[field], reference(prefix + field), rtol=1e-12, atol=1e-12)
    if len(cores) == 2:
        np.testing.assert_array_equal(fitted[1].obstrack_, fitted[2].obstrack_)
        np.testing.assert_allclose(fitted[1].varimp_, fitted[2].varimp_, rtol=1e-12, atol=1e-12)
