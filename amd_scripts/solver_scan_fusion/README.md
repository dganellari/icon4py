# Larger target: both vertical solvers

Submit from the icon4py root on Beverin:

```bash
sbatch amd_scripts/solver_scan_fusion/run_amd.sh
```

The two vertical solvers take 1.925 ms, **35.7%** of the latest compiler-fused
regional/120 granule's 5.394 ms. Reducing the solvers' combined time by 14% would
save 5% of the granule. This is a target, not a prediction.

This first structural change calculates the four tridiagonal coefficients inside
the forward sweep, rather than writing full-column arrays and loading them in a
separate kernel. Both predictor and corrector use the same changed routine. The
isolated compiled sweep drops from four two-dimensional temporary arrays to zero;
CPU comparisons for three input sets and both outputs pass with maximum error
0.0, including untouched boundaries. Types, casts and arithmetic order are
preserved. The full solver can still require other intermediate fields, so the
runtime audit checks its net reduction instead of assuming four arrays disappear
from the complete solver.

In the latest regional generated code those four coefficient arrays occupy about
152 MB per solver invocation. That is array capacity, not measured HBM traffic.
An older profile places coefficient preparation plus the forward sweeps at about
10.2% of profiled granule kernel time. This particular intervention addresses
that subset of the 35.7% solver cost; it cannot eliminate the entire solver cost.
The additional arithmetic inside a serial sweep could offset its memory savings.

The job compares:

- A: existing vertical solvers, with compiler-fused theta-rho.
- B: fused-coefficient forward sweeps in both solvers, with **the identical
  theta-rho implementation** used in A.

All other programs, block shapes and four-level blocking remain unchanged.
One regional/120 run measures the incremental improvement. It uses the established
restored-state granule validation and 12 balanced ABBA/BAAB quartets with 12
interleaved A/A controls. It reports both solvers together, theta-rho, total device
time and host wall time. No counter profiling. The four-hour allocation limit is
not an expected runtime.

The original solver modules are untouched. The harness creates separate program
modules that import the new sweep; the existing compiler-fusion patch is loaded
through the same private GT4Py copy already used successfully in job 639200.
Source snapshots, initial/final hashes, output validation and code audits accompany
the result. The code audit requires fewer full-column intermediate arrays in each
solver, no increase in kernel count, and unchanged block configuration.

Results: `amd_scripts/solver_scan_fusion_runs/amd_<job>/TIMING_SUMMARY.md` and JSON;
`STATUS.json` records progress/failure, `COMPLETE` marks all checks passed.
Job 639284 passed GPU validation across 148 state fields and measured an
additional 3.28% granule device-time reduction (2.93% wall time), with theta
fusion fixed in both arms. Both solvers improve 8.33% together. The result passes
the conservative A/A noise screen, though below the 5% target. Full-pipeline
code removes one kernel and a net three coefficient-sized temporary arrays per
solver variant. Review: `../solver_scan_fusion_runs/amd_639284/REVIEW.md`.
