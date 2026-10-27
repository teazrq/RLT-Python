# Survival smoother reference fixtures

The frozen oracle uses corrected RLT revision
`398603bb415dc5a9293396cee6145eeeb446b291`, RLT DLL MD5
`c32d2002806b3b74dfad23fb82c40e4d`, R 4.6.1 and mgcv 1.9-4.
Ordinary Python tests read the NPZ and JSON files and require no R runtime.

To regenerate, first verify the intended R source checkout against the
pinned revision and file hashes, then install that checkout into an
existing shared R library:

```text
python tools/check_r_reference.py <RLT-source-checkout>
R CMD INSTALL --library=<shared-R-library> <RLT-source-checkout>
```

Set `RLT_REFERENCE_LIBRARY` to that shared R library, `RLT_RSCRIPT` to the
shared Rscript executable, and `RLT_REFERENCE_REVISION` to the revision
printed by `python tools/check_r_reference.py --revision`. The revision
is an explicit assertion that this installed package came from the
verified checkout. It is not inferred from a DLL or package version.
The generator requires it to match `tools/r-reference.json`. It prints
the actual loaded RLT native-library path in the generation log and
records its basename, platform, and MD5 in the manifest on Windows,
Linux, or macOS.

Use the shared Python and R installations, create an output directory,
and run these three commands from the repository root:

```text
Rscript tests/fixtures/generate_survival_smoother.R <output-directory>
python tests/fixtures/pack_survival_smoother.py <output-directory>
python tests/fixtures/generate_survival_sensitivity.py <sensitivity-directory>
```

The first command generates all 51 development and frozen holdout cases,
including raw covariance matrices in exact binary-double format. The
second packs the arrays and replaces the manifest. The third regenerates
18 same-R binary-reread and one-ULP controls and merges them into that new
manifest. Run all three commands: packing intentionally removes old
sensitivity controls, which could be stale after a changed input.

Automatic GCV selection is sensitive to score rounding near a numerical
minimum. Tests add only the measured same-R surface/EDF sensitivity
envelope for the three affected cases. Fixed-parameter solves, basis and
penalties retain stricter numerical checks. Numerical rank, iterations,
stopping gradient and GCV checks also remain in place. The twelve holdout
definitions were frozen before this acceptance rule was chosen.

Optimized and fixed-parameter surfaces share a `1e-11` absolute covariance
tolerance after scaling by the largest raw covariance entry, alongside
the `1e-10` relative tolerance. Large smoothing parameters can make the
augmented penalized system poorly conditioned even with a well-conditioned
design. Automatic selection cannot reasonably be required to beat the
precision of its fixed-parameter solver. The measured three-case rounding
envelopes are added only to the automatic-selection gate.

The Gaussian-draw fixture checks the same postprocessing function used by
production band simulation. Equal R/Python integer seeds do not imply
identical random streams. Independent-seed distribution tests provide a
separate Monte Carlo check.
