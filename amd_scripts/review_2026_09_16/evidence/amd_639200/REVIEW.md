# Compiler fusion timing review — MI300A job 639200

Regional/120, node nid002952. Both comparisons completed. Compiler fusion is a
validated improvement over native code in this run, and speeds theta-rho further
than the existing Python fusion. The small additional whole-granule device gain
against Python fusion remains provisional under our conservative noise screen.

| Comparison | Metric | Baseline ms | Compiler ms | Time reduction |
|---|---|---:|---:|---:|
| Native → compiler | Theta-rho device | 0.872057 | 0.747106 | 14.33% |
| Native → compiler | Granule device | 5.511207 | 5.393807 | 2.13% |
| Native → compiler | Granule wall | 6.412382 | 6.330604 | 1.28% |
| Python fusion → compiler | Theta-rho device | 0.781927 | 0.744248 | 4.82% |
| Python fusion → compiler | Granule device | 5.405009 | 5.359334 | 0.85% observed |
| Python fusion → compiler | Granule wall | 6.314351 | 6.265489 | 0.77% |

Each row compares its own paired arms. The two comparisons ran in separate
processes on the same node; do not combine their absolute times or add percentages.
Device granule time is the sum of program device times for solve_nonhydro, not a
full timestep or diffusion.

## Checks and uncertainty

Both comparisons passed validation across 148 state fields; maximum absolute
error on finite values was 0.0, with matching nonfinite patterns. Initial array
and scalar fingerprints match between comparisons. Rayleigh's restored initial
flag was zero. All 1,964 source hashes match before/after; the 16 transferred
experiment files also match the local bundle.

There are 12 balanced ABBA/BAAB intervention quartets and 12 interleaved A/A
controls per comparison. Device savings are positive in all 12 quartets for
both theta-rho and the granule. No arm-order dependence was detected. I
independently reconstructed block medians, quartet contrasts and matched-control
confidence intervals from raw samples; they reproduce the summaries.

Against native, granule device saving is 0.117400 ms, raw 95% CI
[0.105326, 0.129475]; the control-adjusted CI is [0.106054, 0.146849].
This exceeds the older conservative threshold, |mean A/A bias| + 2 × A/A
quartet standard deviation, of 0.042309 ms.

Against Python fusion, theta-rho saving is 0.037679 ms, raw 95% CI
[0.035035, 0.040322], and exceeds its 0.015519 ms conservative threshold.
The granule saving is 0.045676 ms, raw CI [0.032822, 0.058530]; its matched-control
adjusted CI is also positive, [0.037859, 0.076045]. However, its conservative
threshold is 0.053914 ms, larger than the raw saving. This is encouraging evidence
of an extra total gain, but does not pass every previously used noise criterion.
A repeat is appropriate before claiming that extra 0.85% as established.

## What changed and what this means

Generated code matches the intended intervention: native six kernels, Python
fusion seven, compiler fusion five. Native and compiler retain four 42122×120
gradient buffers; Python fusion has eight buffers over two overlapping cell
ranges. Both compiler variants have two combined interior kernels. All arms use
256×1×1 thread blocks and four-level blocking.

The compiler route improves the original Python program without needing the
manual source rewrite. This is performance evidence for that transformation;
there were no hardware-counter measurements here to separate launch savings,
reuse, latency or register effects. It does not establish cache capacity as the
cause of the regional vendor gap. No GH200 compiler-fusion timing exists yet.

Keep this as the leading compiler candidate. Repeat its incremental comparison
with Python fusion to firm up the small granule effect, then check GH200 before
considering broader enablement. The option remains opt-in.

Evidence: TIMING_SUMMARY.json/.md, each comparison's raw timing report and code
audit, source_hashes.json/.final.json, and INDEPENDENT_CHECK.json in this directory.
