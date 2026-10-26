"""Numerical parity tests against the installed R package.

These compare rlt-python against the installed R RLT on identical data
and identical seeds. They require the *same source version* on both
sides; the comparison uses dedicated temp dirs and Rscript. When R is
not available the tests skip cleanly.
"""
import os
import shutil
import subprocess
import tempfile

import numpy as np
import pytest

from rlt import RLT_cla, RLT_reg, RLT_surv

RSCRIPT = os.environ.get("RLT_RSCRIPT") or shutil.which("Rscript")
HAS_R = RSCRIPT is not None and os.environ.get("RLT_SKIP_R_TESTS") is None

R_PREAMBLE = """library(RLT)
"""

needs_r = pytest.mark.skipif(not HAS_R, reason="R + RLT not available")


def _run_r(script: str):
    with tempfile.NamedTemporaryFile("w", suffix=".R", delete=False,
                                     encoding="utf-8") as f:
        f.write(R_PREAMBLE + script)
        path = f.name
    try:
        out = subprocess.run([RSCRIPT, "--vanilla", path], capture_output=True, text=True,
                             timeout=600)
        if out.returncode != 0:
            raise RuntimeError(out.stderr[-2000:])
        return out.stdout
    finally:
        os.unlink(path)


@needs_r
class TestParityRegression:
    def test_oob_prediction_and_error(self, tmp_path):
        rng = np.random.default_rng(7)
        n, p = 120, 8
        X = rng.normal(size=(n, p))
        y = X[:, 0] + X[:, 1] * X[:, 2] + rng.normal(size=n)
        d = tmp_path
        np.savetxt(d / "X.csv", X, fmt="%.17g", delimiter=",")
        np.savetxt(d / "y.csv", y, fmt="%.17g")
        _run_r(f"""
            X <- as.matrix(read.table("{d.as_posix()}/X.csv", sep = ",", header = FALSE))
            y <- scan("{d.as_posix()}/y.csv", quiet = TRUE)
            fit <- RLT(X, y, ntrees = 50, seed = 42, ncores = 1,
                       model = "regression", verbose = FALSE)
            write(fit$Prediction, file = "{d.as_posix()}/pred.csv", ncolumns = 1)
            write(fit$Error, file = "{d.as_posix()}/err.csv", ncolumns = 1)
        """)
        m = RLT_reg(n_estimators=50, random_state=42, n_jobs=1).fit(X, y)
        R_pred = np.loadtxt(d / "pred.csv")
        assert np.abs(m.oob_prediction_ - R_pred).max() < 1e-6
        assert abs(m.oob_error_ - float(np.loadtxt(d / "err.csv"))) < 1e-6  # R write() ~7 sig digits

    def test_oob_prediction_and_error_embed(self, tmp_path):
        rng = np.random.default_rng(8)
        n, p = 120, 8
        X = rng.normal(size=(n, p))
        y = X[:, 0] + 0.8 * X[:, 1] + rng.normal(size=n)
        d = tmp_path
        np.savetxt(d / "X.csv", X, fmt="%.17g", delimiter=",")
        np.savetxt(d / "y.csv", y, fmt="%.17g")
        _run_r(f"""
            X <- as.matrix(read.table("{d.as_posix()}/X.csv", sep = ",", header = FALSE))
            y <- scan("{d.as_posix()}/y.csv", quiet = TRUE)
            fit <- RLT(X, y, ntrees = 30, seed = 42, ncores = 1,
                       model = "regression", reinforcement = TRUE,
                       param.control = list(embed.ntrees = 25), verbose = FALSE)
            write(fit$Prediction, file = "{d.as_posix()}/pred.csv", ncolumns = 1)
            write(fit$Error, file = "{d.as_posix()}/err.csv", ncolumns = 1)
        """)
        m = RLT_reg(n_estimators=30, reinforcement=True, embed_ntrees=25,
                    random_state=42, n_jobs=1).fit(X, y)
        R_pred = np.loadtxt(d / "pred.csv")
        assert np.abs(m.oob_prediction_ - R_pred).max() < 1e-6
        assert abs(m.oob_error_ - float(np.loadtxt(d / "err.csv"))) < 1e-6  # R write() ~7 sig digits


@needs_r
class TestParityClassification:
    def test_oob_classes_and_error(self, tmp_path):
        rng = np.random.default_rng(99)
        n, p = 120, 8
        X = rng.normal(size=(n, p))
        y = (X[:, 0] + 0.8 * X[:, 1] + 0.5 * rng.normal(size=n) > 0).astype(int)
        d = tmp_path
        np.savetxt(d / "X.csv", X, fmt="%.17g", delimiter=",")
        np.savetxt(d / "y.csv", y, fmt="%d")
        _run_r(f"""
            X <- as.matrix(read.table("{d.as_posix()}/X.csv", sep = ",", header = FALSE))
            y <- factor(scan("{d.as_posix()}/y.csv", quiet = TRUE))
            fit <- RLT(X, y, ntrees = 50, seed = 42, ncores = 1,
                       model = "classification", verbose = FALSE)
            write(as.integer(fit$Prediction), file = "{d.as_posix()}/pred.csv", ncolumns = 1)
            write(fit$Error, file = "{d.as_posix()}/err.csv", ncolumns = 1)
        """)
        m = RLT_cla(n_estimators=50, random_state=42, n_jobs=1).fit(X, y)
        R_pred = np.loadtxt(d / "pred.csv", dtype=int)
        # R factor levels are 1-based
        assert (m.oob_class_ + 1 == R_pred).mean() == 1.0
        # R write() emits ~7 significant digits
        assert abs(m.oob_error_ - float(np.loadtxt(d / "err.csv"))) < 1e-6


@needs_r
class TestParitySurvival:
    def test_oob_error_and_survival_curves(self, tmp_path):
        rng = np.random.default_rng(98)
        n, p = 120, 8
        X = rng.normal(size=(n, p))
        lam = np.exp(X[:, 0] + 0.5 * X[:, 1])
        t = rng.exponential(1.0 / lam)
        ct = rng.exponential(1.0 / np.quantile(lam, 0.45))
        ev = (t <= ct).astype(int)
        tt = np.minimum(t, ct)
        d = tmp_path
        np.savetxt(d / "X.csv", X, fmt="%.17g", delimiter=",")
        np.savetxt(d / "y.csv", tt, fmt="%.17g")
        np.savetxt(d / "c.csv", ev, fmt="%d")
        _run_r(f"""
            X <- as.matrix(read.table("{d.as_posix()}/X.csv", sep = ",", header = FALSE))
            y <- scan("{d.as_posix()}/y.csv", quiet = TRUE)
            cen <- scan("{d.as_posix()}/c.csv", quiet = TRUE)
            fit <- RLT(X, y, censor = cen, ntrees = 50, seed = 42,
                       ncores = 1, model = "survival", verbose = FALSE)
            write(fit$Error, file = "{d.as_posix()}/err.csv", ncolumns = 1)
            pred <- predict(fit, X[1:5, ])
            write(as.vector(pred$Survival), file = "{d.as_posix()}/S.csv", ncolumns = 1)
            write(pred$timepoints, file = "{d.as_posix()}/t.csv", ncolumns = 1)
        """)
        ys = np.empty(n, dtype=[("event", "?"), ("time", "<f8")])
        ys["event"] = ev.astype(bool)
        ys["time"] = tt
        m = RLT_surv(n_estimators=50, random_state=42, n_jobs=1).fit(X, ys)
        assert abs(m.oob_error_ - float(np.loadtxt(d / "err.csv"))) < 1e-6  # R write() ~7 sig digits
        S = m.predict_survival_function(X[:5])
        # R dumps column-major
        R_S = np.loadtxt(d / "S.csv").reshape(-1, 5).T
        assert np.abs(S - R_S).max() < 1e-6
        assert np.allclose(np.loadtxt(d / "t.csv"), m.timepoints_)
