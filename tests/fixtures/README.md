# Survival smoother reference fixtures

The frozen oracle uses corrected RLT revision
`398603bb415dc5a9293396cee6145eeeb446b291`, RLT DLL MD5
`c32d2002806b3b74dfad23fb82c40e4d`, R 4.6.1 and mgcv 1.9-4.
Ordinary Python tests read the NPZ and JSON files and require no R runtime.

To regenerate, use existing shared R and Python installations. Set
`RLT_REFERENCE_LIBRARY` to the corrected shared R library and
`RLT_RSCRIPT` to the shared Rscript executable. Create an output directory
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

The Gaussian-draw fixture checks the same postprocessing function used by
production band simulation. Equal R/Python integer seeds do not imply
identical random streams. Independent-seed distribution tests provide a
separate Monte Carlo check.
