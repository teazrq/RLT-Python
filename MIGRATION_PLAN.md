# RLT → Python Migration Plan

Historical port plan, retained for provenance. The October 2026 revision
synchronizes the shared core with corrected RLT 6.1.1 sources and repairs
the Python API and survival prediction differences. See
[Comparing Python and R](docs/reference/r-parity.md) for the current
contract and the remaining intentional smoother difference. The old
verification counts and RNG notes below describe the original port.

**Source:** R package `RLT` v6.0.2 (`/home/tez/RLT`, CRAN `teazrq/RLT`)
**Target:** Python package `rlt-forests` (this repo)

## Status — ALL CORE MILESTONES COMPLETE

| Milestone | State |
|---|---|
| M0 scaffolding, build, git | ✅ done |
| M1 core lift (Rcpp stripped, compiles) | ✅ done — 40+ TUs, plain g++/Armadillo/OpenMP |
| M2 regression vertical slice | ✅ done |
| M3 classification + survival | ✅ done |
| M4 reinforcement / linear-comb forests | ✅ done — all three models, differential vs R |
| M5 importance + kernels + survival bands | ✅ done — incl. C++ tensor-product spline |
| M6 docs, CI, PyPI prep | ✅ done — CI green py3.10–3.13, sdist verified from scratch |

Remaining (optional, post-0.2.0): macOS/Windows wheels via cibuildwheel;
`get.one.tree` printing polish; PyPI upload when ready for release.

## Locked design decisions (user, 2026-08-30)

1. **RNG: numpy seeding.** `random_state` → master seed; per-tree seeds
   inside C++ (vendored xoshiro256plus + Lemire). Not bit-compatible with R.
2. **U-statistics: heavy computation in C++.** IJ/jack cores were already
   C++; survival bands got a NEW C++ tensor-product penalized B-spline
   smoother (`src/bindings/rlt_tp_smooth.cpp`) replacing mgcv `te(i,j)`,
   with GCV lambda selection — same statistical pipeline as R.
3. **Full sklearn compatibility.** `RLT_reg`/`RLT_cla`/
   `RLT_surv` on `BaseEstimator`; structured-array survival y;
   `score()` = Harrell's c-index. Verified with `clone`.
4. **Scope: regression, classification, survival only.**

## Architecture

```
src/cpp/        vendored compute core (tools/vendor_core.py regenerates
                idempotently from /home/tez/RLT/src)
src/bindings/   pybind11 layer (_core): 12 fit/pred entry points
                (Uni+Comb x reg/cla/surv), 6 kernels, cindex, mc_band,
                rlt_tp_smooth, matched ObsTrack gen
rlt/            estimators.py (sklearn), bands.py (get_surv_band), _params.py
tests/          22 tests: basic, comb, importance/kernels/bands
.github/        CI: ubuntu matrix py3.10-3.13 (apt armadillo/lapack/blas)
```

## Verification summary

- 22/22 pytest, also in a fresh venv from the sdist
- CI green on Python 3.10/3.11/3.12/3.13 (ubuntu-latest)
- Differential vs R package: OOB errors match to MC noise on reg/cla/surv
  (500 and 2000-tree runs); LC-split loadings recover planted directions
- MC band critical values: py 2.8432 vs R 2.8415 (50k sims, different RNG)
- embed VI recovers {0,1} on y = x0 + x1² DGP; kernel matrix invariants hold

## Known pitfalls (fixed — do not regress)

- **ncat convention: 1 = continuous.** Zeros send splits into the
  categorical path (OOB `goright[x]` reads, silent garbage).
- **embed_protect must be ≥ 1** when reinforcement/Comb is active
  (default None → ceil(log(n)); a 0 underflows `var_id.subvec(0, -1)`).
- Armadillo aux-memory constructor is column-major-only; numpy buffers
  must be copied explicitly (see np2mat).
- Link with `-DARMA_DONT_USE_WRAPPER -llapack -lblas`.
- `cindex_d/cindex_i` live in the vendored core (Stat_Function.cpp).
- RLT's `cindex(pred=risk)`: higher risk = higher event probability.
- mc_band alpha: `sd * q.t()` outer-product shape (1×d per alpha level).
- R var.mode presets mirrored: matched → subsample 0.5/even trees/
  distribute importance; IJ/jack → bootstrap + obs_track.
- sdist needs MANIFEST.in (headers aren't included by default).
- CI expressions: `${{ matrix.python-version }}` (dot, not dash).
