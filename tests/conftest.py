"""Make the live-R CI contract explicit instead of silently skipping it."""

import os
import shutil
import subprocess

import pytest


def pytest_sessionstart(session):
    if os.environ.get("RLT_REQUIRE_R_TESTS") != "1":
        return
    executable = os.environ.get("RLT_RSCRIPT") or shutil.which("Rscript")
    if not executable or os.environ.get("RLT_SKIP_R_TESTS") is not None:
        raise pytest.UsageError("This run requires live R comparisons; Rscript is unavailable or tests are disabled")
    result = subprocess.run([executable, "--vanilla", "-e", "library(RLT)"],
                            capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise pytest.UsageError("This run requires an installed RLT reference: " + result.stderr[-1500:])


def pytest_report_header(config):
    revision = os.environ.get("RLT_REFERENCE_REVISION")
    return f"Live R source reference: {revision}" if revision else "R reference: local installation, or visible skips when unavailable"
