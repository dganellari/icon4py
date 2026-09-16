# Regional dycore: measured results and code review

Updated September 16, 2026. [Start the review here](../review_2026_09_16/README.md).

MI300A's disadvantage depends strongly on the grid: the baseline solve_nonhydro
granule took 1.107× GH200's summed program device time on global/120 and 1.479×
on regional/120. Regional whole-call wall time was 1.339×. These are granule
measurements, not a full timestep or the whole weather model.

Cache and access-pattern measurements suggested where to investigate, but have
not established that cache capacity causes the full vendor gap. AMD's L2 request
counters and NVIDIA's sector counters have different meanings; AMD fabric traffic
is not physical HBM traffic. Occupancy/latency explanations remain open.

## What now works on MI300A regional/120

| Change | Measured granule device-time reduction | Interpretation |
|---|---:|---|
| Compiler theta-rho fusion versus native | **2.13%** | Replaces the manual Python theta rewrite. |
| Compiler theta-rho versus Python fusion | 0.85% observed | Smaller extra granule gain does not clear the conservative noise screen. |
| Solver coefficient fusion, theta fusion enabled in both arms | **3.28% additional** | Independent optimisation; solver time falls 8.33%. |

The compiler version uses five theta kernels rather than six native or seven
with the Python rewrite. It preserves external outputs and avoids the rewrite's
duplicated gradient storage. The solver change computes coefficients inside the
forward sweep, removing one kernel and a net three full-column temporary arrays
per compiled solver variant. Both GPU experiments pass validation across 148
state fields, with zero observed finite-value error and matching nonfinite
patterns. Source checks and paired controls pass.

**Do not add the Python and compiler theta improvements.** They are alternatives.
Solver fusion adds to compiler theta fusion. Combining the independently measured
reductions gives roughly 5.34% less device time versus original, but that remains
a cross-run estimate; no single paired job has measured the complete combination
against original code. The solver job directly measured an additional 2.93%
wall-time reduction as well as its 3.28% device-time reduction.

## Read the work

- [Story, exact measurements, limits and next step](../review_2026_09_16/README.md).
- [Code guide with before/after examples and compiler call path](../review_2026_09_16/CODE_GUIDE.md).
- [Compiler experiment evidence](../review_2026_09_16/evidence/amd_639200/REVIEW.md).
- [Solver experiment evidence](../review_2026_09_16/evidence/amd_639284/REVIEW.md).
- [Historical September 11 narrative](DYCORE_GRANULE_ANALYSIS_2026-09-11.md).

The next step is review, then a direct original-versus-combined comparison on
MI300A and GH200. The new compiler and solver changes have not been timed on
GH200; general enablement also needs global-grid, level-count and precision
coverage. The performance changes remain experimental and opt-in.
