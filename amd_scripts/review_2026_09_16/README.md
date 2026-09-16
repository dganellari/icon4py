# Regional dycore optimisation: review package

**Two changes improve MI300A performance without changing the computed results
in the tested regional/120 configuration.** Compiler fusion reduces granule
device time by 2.13% versus native code. A separate solver change saves another
3.28% with compiler fusion already enabled. Approximately 5.34% combined is a
cross-run estimate; the complete combination has not been timed directly against
unmodified code in one paired run.

Start with this page, then [the code guide](CODE_GUIDE.md). This branch contains
the measured experimental implementations and evidence for review. The two
optimisations are **not enabled by default in the model**.

## The story

Our starting measurement was a grid-dependent performance gap: MI300A took
1.107× GH200's summed program device time on global/120, but 1.479× on the
operational regional/120 grid. Regional whole-call wall time had a smaller ratio,
1.339×. These are measurements of the `solve_nonhydro` granule, not diffusion or
a complete weather-model timestep.

Regional is smaller, but differs in connectivity, boundary rules and active
domains as well as size. Both chips gain L2 hits on regional; GH200 gains more
under its own counters. AMD's all-request and NVIDIA's sector-based counters
have different meanings, and AMD fabric traffic is not physical HBM traffic.
The cache evidence motivated experiments; it did not prove that cache capacity
causes the timing gap. Occupancy and latency-hiding explanations remain open.

We then tested changes to the code, measuring their effect on total runtime.
Changing block traversal and the tested block shapes did not help. Aligning
theta-rho's output domains in Python did help, but that rewrite produced seven
kernels and duplicated some temporary storage. That led to a compiler change
which handles the original Python program directly.

The next target was the vertical solvers, together about 36% of the remaining
device time. They first constructed four coefficient arrays and then read them
in a forward sweep. Calculating the coefficients inside the sweep reduced
intermediate storage and improved both solvers.

## What is actually measured

All rows below are MI300A, regional grid, 120 levels. Percentages mean **less
time**, not throughput speedup. Each row uses its own paired comparison.

| Comparison | Granule device time, A → B | Reduction | Target-program reduction |
|---|---:|---:|---:|
| Native → compiler theta fusion, job 639200 | 5.511207 → 5.393807 ms | **2.13%** | Theta-rho **14.33%** |
| Python theta fusion → compiler theta fusion, same job | 5.405009 → 5.359334 ms | **0.85% observed** | Theta-rho **4.82%** |
| Compiler theta fusion → theta + solver fusion, job 639284 | 5.519307 → 5.338015 ms | **3.28% additional** | Both solvers **8.33%** |

The 0.85% incremental granule result has positive confidence intervals but does
not clear our conservative A/A noise threshold. The 2.13% and additional 3.28%
results do. The solver run also reduces whole-call wall time by **2.93%**.

**Do not add the Python and compiler theta gains:** they are alternatives for
the same calculation. Solver fusion is a separate change, tested with the same
compiler-fused theta implementation in both arms. Applying the two independently
measured reductions multiplicatively gives `1 - (1-0.0213021)*(1-0.0328468)` ≈
5.34%, but the runs used different nodes and that remains an estimate.

Each experiment used 12 balanced ABBA/BAAB quartets, interleaved A/A controls,
and restored initial model state. Validation covered 148 state arrays, with
zero observed finite-value error and matching nonfinite patterns. Source hashes
were unchanged during both jobs. These checks support the tested configuration;
they are not proof for every grid, precision or backend.

## What to review

1. **GT4Py compiler change:** [focused patch and regression tests](patches/03-shared-output-fusion-gt4py.patch).
   Adds an off-by-default option to split producers of external outputs while
   preserving all required stores. Existing DaCe fusion then joins matching
   regions. Native theta has six kernels; the selected compiler result has five.
2. **Icon4Py solver change:** [readable, semantically identical prototype](code/forward_sweep_fused.py).
   Moves coefficient arithmetic into the forward scan, retaining casts and
   operation order. The full solver loses one kernel and a net three
   coefficient-sized temporary arrays per specialization. Its unchanged public
   program interface lets the harness select it without editing the model.
   A [minimal production candidate patch](patches/04-solver-forward-sweep-icon4py.patch)
   is included for review, but is not applied; it retains the old standalone scan API.
3. **Benchmark prerequisites:** explicit grid-level handling in Icon4Py and
   conversion of a GPU scalar before `round()` in GT4Py. These are separate from
   the performance changes: [levels](patches/01-grid-levels-icon4py.patch),
   [scalar conversion](patches/02-domain-scalar-gt4py.patch).
4. **Measurement and controls:** [theta review](evidence/amd_639200/REVIEW.md),
   [solver review](evidence/amd_639284/REVIEW.md), and the retained raw reports.

The executable experiment snapshots are `../theta_shared_timing/` and
`../solver_scan_fusion/`. They intentionally preserve the tested code, including
copies of the measurement harness. Review the focused changes above first;
consolidating the harness is separate cleanup, not part of the measured gain.
The formatted solver copy has the same Python AST as the executed prototype.
Patches are review/apply artefacts; they are not all applied automatically.

## Evidence and reproduction

Run locally, from this directory:

```bash
python3 verify_evidence.py
```

This needs only Python's standard library. It checks artifact checksums, source
consistency, validation status and reconstructs timing summaries from raw samples.
Compressed JSON files are the original reports, not reduced synthetic data.
Absolute paths in reports record where a job ran; the package's relative links
are the portable review entry points.

[BASES.json](BASES.json) pins the measured Icon4Py, GT4Py and DaCe revisions.
The GT4Py patches belong in the GT4Py repository, not Icon4Py. Patch 02 is a
required benchmark fix; patch 03 includes its regression test. Do not apply the
levels patch again to a branch which already contains the fixture change.

Local checks already completed: 44 compiler-related tests passed, 2 expected
failures; compiler pre-commit checks passed; eight targeted theta-harness checks;
six solver-output CPU comparisons and two solver integration tests. The recorded
GPU jobs supply the actual regional correctness and timing evidence. The compiler
tests are included for rerunning with the pinned GT4Py/DaCe environment.

On a suitably configured Beverin checkout, the user can reproduce the measured
experiments with the two wrappers documented in their directories. They require
the matching compiler trees beside Icon4Py, the `venv_mi300` environment, ROCm,
the regional test data and the existing cluster setup. No cluster job is started
by this review package. Do not alter experiment/model/compiler sources during a
run, because the final source audit will reject changes.

## Next step

Review the two small algorithm/compiler changes before doing more tuning. Then
run **original versus both changes together**, with matched controls, on MI300A
and GH200. That measures the combined gain directly and establishes whether the
new changes help or regress NVIDIA. Check global and representative additional
level counts/precision modes before general enablement.

No new capacity claim, vendor-gap closure percentage or general production
default is justified by the current results. The demonstrated outcome is two
compatible MI300A regional optimisations with reproducible evidence.
