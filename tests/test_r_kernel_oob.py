"""Numerical parity tests for the OOB self-kernel.

These require an RLT installation with the OOB kernel API in a shared R
library. Use the same corrected source revision on both sides. Count
matrices are compared exactly; normalized entries allow serialization
rounding from R's write().
"""
import os
import shutil
import subprocess
import tempfile

import numpy as np
import pytest

from rlt import RLT_reg

RSCRIPT = os.environ.get("RLT_RSCRIPT") or shutil.which("Rscript")
HAS_R = RSCRIPT is not None and os.environ.get("RLT_SKIP_R_TESTS") is None

R_PREAMBLE = """library(RLT)
stopifnot(packageVersion("RLT") >= "6.1.0")
"""

needs_r = pytest.mark.skipif(not HAS_R, reason="R + RLT >= 6.1.1 not available")


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
class TestParityKernelOOB:
    def test_oob_self_kernel_bit_exact(self, tmp_path):
        rng = np.random.default_rng(123)
        n, p = 90, 6
        X = rng.normal(size=(n, p))
        y = X[:, 0] + 0.7 * X[:, 1] + rng.normal(size=n)
        d = tmp_path
        np.savetxt(d / "X.csv", X, fmt="%.17g", delimiter=",")
        np.savetxt(d / "y.csv", y, fmt="%.17g")
        _run_r(f"""
            X <- as.matrix(read.table("{d.as_posix()}/X.csv", sep = ",", header = FALSE))
            y <- scan("{d.as_posix()}/y.csv", quiet = TRUE)
            fit <- RLT(X, y, ntrees = 40, seed = 42, ncores = 1,
                       model = "regression", verbose = FALSE,
                       param.control = list(resample.track = TRUE))
            K <- forest.kernel(fit, X1 = X, oob = TRUE)
            write(as.vector(K$Kernel), file = "{d.as_posix()}/K.csv", ncolumns = 1)
            write(as.vector(K$N), file = "{d.as_posix()}/N.csv", ncolumns = 1)
            write(as.vector(K$C), file = "{d.as_posix()}/C.csv", ncolumns = 1)
        """)
        m = RLT_reg(n_estimators=40, random_state=42, n_jobs=1,
                    resample_track=True).fit(X, y)
        out = m.forest_kernel(X, oob=True)
        # R dumps column-major
        R_K = np.loadtxt(d / "K.csv").reshape(n, n, order="F")
        R_N = np.loadtxt(d / "N.csv").reshape(n, n, order="F")
        R_C = np.loadtxt(d / "C.csv").reshape(n, n, order="F")
        # integer count matrices must match exactly
        np.testing.assert_array_equal(out["N"], R_N)
        np.testing.assert_array_equal(out["C"], R_C)
        # Kernel floats: R write() ~7 significant digits
        assert np.abs(out["Kernel"] - R_K).max() < 1e-6

    def test_oob_self_kernel_comb_bit_exact(self, tmp_path):
        rng = np.random.default_rng(321)
        n, p = 80, 6
        X = rng.normal(size=(n, p))
        y = X[:, 0] * X[:, 1] + rng.normal(size=n)
        d = tmp_path
        np.savetxt(d / "X.csv", X, fmt="%.17g", delimiter=",")
        np.savetxt(d / "y.csv", y, fmt="%.17g")
        _run_r(f"""
            X <- as.matrix(read.table("{d.as_posix()}/X.csv", sep = ",", header = FALSE))
            y <- scan("{d.as_posix()}/y.csv", quiet = TRUE)
            fit <- RLT(X, y, ntrees = 20, seed = 42, ncores = 1,
                       model = "regression", verbose = FALSE,
                       param.control = list(resample.track = TRUE,
                                            linear.comb = 2))
            K <- forest.kernel(fit, X1 = X, oob = TRUE)
            write(as.vector(K$Kernel), file = "{d.as_posix()}/K.csv", ncolumns = 1)
            write(as.vector(K$N), file = "{d.as_posix()}/N.csv", ncolumns = 1)
            write(as.vector(K$C), file = "{d.as_posix()}/C.csv", ncolumns = 1)
        """)
        m = RLT_reg(n_estimators=20, random_state=42, n_jobs=1,
                    resample_track=True, linear_comb=2).fit(X, y)
        out = m.forest_kernel(X, oob=True)
        R_K = np.loadtxt(d / "K.csv").reshape(n, n, order="F")
        R_N = np.loadtxt(d / "N.csv").reshape(n, n, order="F")
        R_C = np.loadtxt(d / "C.csv").reshape(n, n, order="F")
        np.testing.assert_array_equal(out["N"], R_N)
        np.testing.assert_array_equal(out["C"], R_C)
        assert np.abs(out["Kernel"] - R_K).max() < 1e-6
