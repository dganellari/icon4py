# Compiler fusion: actual-granule timing on MI300A

From the icon4py root on Beverin:

```bash
sbatch amd_scripts/theta_shared_timing/run_amd.sh
```

One allocation runs two comparisons on the regional grid with 120 levels:

1. Native code (A) versus compiler fusion (B).
2. Our existing Python fusion (A) versus compiler fusion (B).

This measures theta-rho device time, the sum of program device times for the
solve_nonhydro granule, and granule host wall time. It uses the existing granule
input fixture and restored-state validation. Each comparison has 12 balanced
ABBA/BAAB quartets, interleaved with 12 A/A controls, five warmups and ten measured
calls per block. Restoring state and compiling occur outside the timing window.
The second comparison checks whether compiler fusion improves on the optimisation
we already have. It has its own paired baseline; do not subtract timings from the
two separate processes to estimate another contrast.

There is no counter profiling in this job. Compilation has the existing CMake
signal workaround and timeout. The allocation limit is four hours, not a predicted
runtime. Both comparisons share a job-specific build cache to reuse unchanged
granule programs. Each produces numerical checks and generated-code audits.
Expected theta kernel counts: native six, Python fusion seven, compiler fusion
five. Native/compiler must retain four 42122×120 gradient buffers, and all arms
must keep the same AMD block shape and four-level blocking. An unexpected/no-op
compilation fails the audit. Initial model inputs must match between comparisons.

Results go to `amd_scripts/theta_shared_timing_runs/amd_<job>/`:

- `TIMING_SUMMARY.md` and `.json`: both comparisons, intervals and noise diagnostics.
- Each comparison directory: `timing.log`, raw `timing.*.json`, per-program timers,
  generated program sources and `code_audit.json`.
- `STATUS.json`: current phase or failure; `COMPLETE` only after both comparisons
  and final source checks pass.

Positive saved milliseconds mean compiler fusion is faster. Check the A/A controls
and arm-order diagnostics before claiming a benefit; a confidence interval alone
does not remove systematic ordering bias. A result here establishes an MI300A
effect; it does not establish a GH200 speedup or quantify cache causation.

The harness is an isolated copy of the existing causal harness. The job copies
the installed GT4Py package into its output directory and replaces exactly one
module with the already tested shared-output patch. It first checks the installed
module against pinned original/patched hashes and checks the required domain_utils
fix. The pytest process verifies it imported this private copy. Shared-output
splitting stays disabled in A and is enabled only for theta's matching horizontal
domain and split vertical bands in B. The installed compiler and model files are
unchanged. Source hashes and a snapshot accompany the run; avoid changing any
experiment or shared model/compiler sources while it is running.

The previous job 639135 checked three synthetic inputs on saved graphs and collected
no timing. This job compiles from actual model inputs and collects timing.

Local checks: eight targeted tests passed, including saved native/fused graph audits,
arm-specific hook selection, compiler-copy integrity and a fresh-process import
check. Shell syntax and Python syntax checks pass. Full-granule GPU validation
and performance remain pending until this job runs.
