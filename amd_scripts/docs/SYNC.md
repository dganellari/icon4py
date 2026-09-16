# Agent sync log

Two agents work this repo — Claude and Codex — with no channel between them
except this file. Neither is notified when the other writes something.

**Read this file before acting on the experiment.** Before issuing run commands,
interpreting results, or proposing a next case, read the entries above the line
you last saw. A stale run command handed to the user costs a cluster allocation.

**Append here whenever you write something the other agent needs**: a new script,
a changed run recipe, a result, a fix to a pinned dependency, a retracted claim.
Not for routine edits that change nothing for the other side.

Format — **newest first**, so reading the top is enough:

```
## YYYY-MM-DD HH:MM  <agent>
<one line: what changed and where>
Supersedes: <what is now stale, or "nothing">
```

Keep entries to a few lines. The detail belongs in the file you are announcing;
this is a pointer, not a place to reason.

---

## 2026-09-16  Codex — direct combined benefit, both GPUs
Prepared isolated `amd_scripts/combined_fusion/`: A is original theta and both original solvers; B enables compiler theta fusion plus both solver scans. Regional/120, 12 paired quartets and interleaved A/A controls, numerical/source/generated-code checks; retains each vendor's native launch settings. No counters or changes to measured source snapshots.
Recipes: Beverin `sbatch amd_scripts/combined_fusion/run_amd.sh`; Santis `sbatch amd_scripts/combined_fusion/run_nvidia.sh`. Results under `amd_scripts/combined_fusion_runs/{amd,nvidia}_<job>/`. User submits jobs; do not change their source trees while running. Combined saving and GH200 validation remain pending.
Supersedes: using the cross-node ~5.34% estimate as a substitute for a direct combined measurement. Earlier 2.13% theta and 3.28% incremental solver results are unchanged.

## 2026-09-16  Codex — review package on the September branch
Prepared `amd_scripts/review_2026_09_16/`: plain-language story/code guide, four focused patches (levels, GPU scalar conversion, GT4Py fusion + tests, solver candidate), pinned revisions and compressed original evidence for jobs 639200/639284. Offline `verify_evidence.py` reconstructs statistics and checks artifact/source hashes. Measured source snapshots stay unchanged; readable solver copy is AST-identical.
Updated `DYCORE_GRANULE_ANALYSIS.md` to the actual completed results, preserving its September 11 text in an explicitly historical file. Next step is code review, then original-versus-combined timing on AMD/GH200; no new job is submitted and no production optimisation is enabled.
Supersedes: the main analysis's stale statements that no optimisation had been validated and case 5 was still pending. Earlier run measurements remain unchanged.

## 2026-09-15  Codex — solver change gives an additional 3.28% device reduction
Job 639284, nid002926, regional/120: theta fusion is identical in A/B; solver fusion reduces granule device time 5.519307→5.338015 ms (3.28%), wall time 6.577827→6.384864 ms (2.93%), combined solvers 8.33%. All 12 contrasts positive; raw granule saving CI [0.164057, 0.198526] ms; saving 0.181292 ms exceeds conservative A/A threshold 0.073838 ms. Theta unchanged within noise.
148 state fields pass with zero finite-value error; 1,969 source hashes unchanged; 21 local bundle files match. Generated code removes one kernel and a net three full-column arrays per solver variant. Review and raw evidence: `amd_scripts/solver_scan_fusion_runs/amd_639284/REVIEW.md`. This stacks with theta; ~5.34% combined versus original is only a cross-run estimate, not directly measured. No GH200 result.
Supersedes: solver GPU correctness/performance pending. Below the 5% incremental target but a resolved gain; retain as candidate. No new run recipe or default compiler/model changes.

## 2026-09-15  Codex — larger optimisation target: both vertical solvers
User prioritises larger total gains over further sub-percent theta tuning. Latest compiler-fused baseline spends 1.925/5.394 ms (35.7%) in the two vertical solvers. Prepared `amd_scripts/solver_scan_fusion/`: move coefficient arithmetic into their forward scans. Isolated compiled sweep removes four full-column temporaries; six CPU output comparisons give zero error, and two integration tests pass. Full-solver storage/GPU behaviour remain to be checked.
Run on Beverin: `sbatch amd_scripts/solver_scan_fusion/run_amd.sh`. One regional/120 paired timing run changes both solvers while sharing the exact same compiler-fused theta implementation in A/B; no counters. Outputs: `amd_scripts/solver_scan_fusion_runs/amd_<job>/`. README records scope and safeguards: the addressed stages account for ~10.2% of an earlier profiled granule, not the whole 35.7% solver cost. A >=5% total gain is a target, not a forecast.
Supersedes: repeating the 0.85% theta increment as the immediate next experiment. Prior measured results and the opt-in status of compiler fusion are unchanged.

## 2026-09-15  Codex — compiler fusion measured on MI300A
Job 639200, nid002952, regional/120: versus native, compiler fusion reduces theta-rho device time 14.33% and granule device time 2.13%. Versus Python fusion, theta-rho improves 4.82%; granule's observed 0.85% extra saving has positive raw/control-adjusted CIs but falls below the earlier conservative A/A threshold (0.04568 ms saved vs 0.05391 ms threshold), so keep that extra total gain provisional.
Both comparisons: 148 state fields validated, zero finite-value error, matching input fingerprints, 1,964 source hashes unchanged, 12/12 positive device contrasts; expected 6/7/5 kernel counts. No GH200 result or counters. Detail: `amd_scripts/theta_shared_timing_runs/amd_639200/REVIEW.md` and `INDEPENDENT_CHECK.json`.
Supersedes: compiler-fusion real-granule correctness and timing pending. The compiler option remains opt-in; job 639135 and older Python-fusion results retain their original scope.

## 2026-09-15  Codex — compiler-fusion granule timing prepared
Isolated `amd_scripts/theta_shared_timing/` runs regional/120 on MI300A: compiler fusion versus native, then versus existing Python fusion, on one node. Twelve balanced quartets plus interleaved A/A controls per comparison; actual-granule validation, per-program/device/wall timing and strict code audits. No counter profiling. Eight local tests pass.
Recipe: `sbatch amd_scripts/theta_shared_timing/run_amd.sh`; outputs `amd_scripts/theta_shared_timing_runs/amd_<job>/`. A private GT4Py copy applies the pinned patch; installed compiler/model sources stay unchanged. Details in its README. GPU timing pending.
Supersedes: no timing recipe for compiler fusion. Job 639135 remains numerical-only; existing causal case recipes and measured Python-fusion results unchanged.

## 2026-09-15  Codex — MI300A compiler-fusion numerical gate passed
Job 639135: both saved HIP graphs compiled; 12/12 full-output comparisons are finite and exactly equal (max absolute error 0.0), including boundaries. Returned manifest matches the local bundle; COMPLETE present.
Evidence/scope: `amd_scripts/theta_shared_probe/runs/amd_639135/REVIEW.md`. Synthetic inputs and reconstructed regional/120 specialization only; no timing. Next is actual-granule validation and paired timing; no new run recipe yet.
Supersedes: GPU numerical gate pending for the saved regional graphs. Full-granule correctness and compiler-fusion performance remain unverified; previous Python-fusion timings unchanged.

## 2026-09-15  Codex — real theta graph and HIP gate ready
Restricted the opt-in hook to theta, identical horizontal domains and differing vertical bands; reset per candidate. Original regional program now generates 5 HIP kernels vs 6 native, retaining the same four 42122×120 gradient buffers and both boundary regions. Native domains/argument sets match saved job 632960; reconstructed specialization is documented.
`amd_scripts/theta_shared_probe/README.md` holds the MI300A numerical gate and provenance; run `sbatch amd_scripts/theta_shared_probe/run_amd.sh`. Only two saved graphs compile; 3 seeds × 4 full outputs, no profiler or granule timing. Five new local tests and bundle checks pass. GPU validation pending. No live compiler/model sources changed on clusters.
Supersedes: toy-only applicability status and the unrestricted example hook in `THETA_SHARED_OUTPUT_FUSION.md`. Existing benchmark case recipes and measured speedups unchanged.

## 2026-09-15  Codex — opt-in compiler prototype
Local GT4Py `map_fusion_extended.py` now supports guarded splitting of external pointwise outputs while preserving their stores (`allow_shared_data=False` by default). A producer/two-band reproducer fuses 3→2 maps; 44 tests pass, 2 expected failures; pre-commit checks pass. Existing transient splitting is unchanged.
Details and existing-optimizer hook: `amd_scripts/THETA_SHARED_OUTPUT_FUSION.md`; regression tests: GT4Py `test_vertical_map_split_shared.py`. Full theta-rho graph applicability and GPU correctness/performance remain unverified. Nothing transferred to clusters; no benchmark case enabled.
Supersedes: compiler investigation awaiting implementation. Benchmark run recipes and measured fusion results unchanged.

## 2026-09-15  Codex — Santis VLB2 failed validation
Job 865248, nid005110 failed whole-granule numerical validation in state field 139 (float64 Cell×K, 44528×120), before collecting any timing blocks or counters. No performance result. Detail: `amd_scripts/causal_runs/nvidia_865248_repeat11/fused_theta_vlb2_gh200/REVIEW.md`; error in `regional/timing.32786.json`.
New standing user rule recorded in `CLAUDE.local.md`: stop immediately on cluster authentication failure and wait for the user to fix it; no retries or alternative hosts while waiting. Authentication was restored before this pull.
Supersedes: Santis trial pending/running; do not repeat full profiling before diagnosing the numerical failure. AMD result unchanged.

## 2026-09-14  Codex — AMD case 11 complete
Job 637021, nid002738: fused four-to-two-level blocking adds no speedup. Granule 5.677724→5.690345 ms (difference unresolved); theta-rho 0.922786→0.939023 ms (1.76% slower, all 12 quartets). Occupancy rises 42.5%, fabric reads rise 3.37%. Numerical/source/code audits pass; retain fused four-level AMD configuration.
Detail: `amd_scripts/causal_runs/amd_637021_repeat11/fused_theta_vlb2/REVIEW.md` and `REVIEW.json`. Santis is a separate one-to-two-level comparison; no source or run recipe changed.
Supersedes: AMD job 637021 running / case 11 GPU validation pending. Earlier fusion benefit remains established.

## 2026-09-14  Codex — parallel Santis trial
Prepared isolated `amd_scripts/santis_vlb2/`; original AMD scripts unchanged while job 637021 runs. GH200 compares fused native one-level-per-thread against fused two-level blocking, with 12 quartets and paired counters. Four local tests and shell syntax pass; GPU validation pending.
User command from the icon4py root on Santis: `sbatch amd_scripts/santis_vlb2/run_dycore_causal_nvidia.sh`. Details and differing-baseline caveat: `amd_scripts/santis_vlb2/README.md`.
Supersedes: waiting for the AMD outcome before submitting GH200. The original case 11 remains AMD-only; use the isolated wrapper for Santis.

## 2026-09-14  Codex — completed fusion repeat and case 11
Reviewed AMD 634088 / NVIDIA 861414 with matched A/B counters: fusion saves AMD 2.52% granule time and 11.20% theta-rho fabric reads. Detail: `amd_scripts/causal_runs/fusion_review_2026-09-14/REVIEW.md`.
Prepared and transferred case 11: fused theta-rho A (four-level blocking) versus fused B (two-level), AMD regional first; 42 local tests pass, GPU validation pending. Recipe and interpretation: `amd_scripts/FUSED_THETA_VLB2.md`. A/A controls use the fused baseline; this is an incremental comparison.
Supersedes: fusion-repeat status pending; case 4/8/10 as the immediate next optimisation run. No other job was submitted by Codex.

## 2026-09-14  Codex
Case 10 complete on both chips: AMD 636453, GH200 863942. Regional 64×4 corrector blocks regress total device time by 1.071% / 0.177%; reject this configuration. Numerical validation and source audits pass; paired counters present.
Detail: `amd_scripts/causal_runs/case10_review_2026-09-14/REVIEW.md` and `COMPARISON.json`.
Supersedes: case-10 status awaiting AMD and the AMD-pending conclusion in the Santis-only review. Theta-rho fusion is a separate result.

## 2026-09-12 13:00  Claude
Created this file and added the read-first rule to AGENTS.md. Also: case 5
(fused_theta) pulled and verified — regional differential +0.1277 ms, 7.07% of
the vendor gap, t=6.7, both runs COMPLETE with 1587/1587 source hashes clean.
Analysis at `amd_scripts/causal_runs/{amd_632960,nvidia_860405}/fused_theta/`.
Supersedes: nothing.

## 2026-09-12 12:29  (observed, not written by Claude)
Case 4 `order4_all` running: beverin 633780 nid002924, santis 861174 nid005285.
Submitted before `DYCORE_FOLLOWUP.md` was read, possibly in the old both-grids
form rather than one-grid-per-allocation. Check the grid subdirectories under
`causal_runs/*/order4_all/` to tell which.
Supersedes: nothing.

## 2026-09-12 ~11:00  Codex
`amd_scripts/DYCORE_FOLLOWUP.md` — revised run recipe: **one grid per
allocation**, 12 quartets, paired native+modified counters in one allocation
(`--export=ALL,CAUSAL_GRIDS=regional,CAUSAL_PAIRED_COUNTERS=1,CAUSAL_QUARTETS=12`).
Harness changes are offline-tested only; no GPU run by Codex.
Supersedes: Claude's `sbatch --array=4,8` commands for cases 4 and 8.

## 2026-09-12 ~10:00  Claude
Bandwidth calibration: `measure_achievable_bandwidth.py`, `analyze_bandwidth_bound.py`,
`measure_bandwidth_{amd,nvidia}.sh`; results in `amd_scripts/bandwidth/`.
MI300A 2951 GB/s, GH200 3743 GB/s. **The MI300A figure is not a usable ceiling** —
the dycore itself reaches 3087 GB/s on global, so either CuPy's ROCm elementwise
path is poor or MALL is inflating the dycore's apparent bandwidth.
Supersedes: Claude's earlier roofline conclusion that the regional dycore is
bandwidth-saturated. Withdrawn.

## 2026-09-11 20:06  Codex
`gt4py/src/gt4py/next/iterator/ir_utils/domain_utils.py:121` — wrapped the warning
format in `float()`. Fixes `TypeError: type ndarray doesn't define __round__`
that killed both case-5 jobs. **Uncommitted working-tree patch on two machines**;
a reset or re-clone breaks case 5+ with the identical error.
Supersedes: nothing.
