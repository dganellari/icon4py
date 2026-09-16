# Direct combined-fusion measurement

Regional grid, 120 levels, `solve_nonhydro` only. Run the same bundle on
MI300A and GH200. This measures the total benefit directly; it does not add
percentages from different nodes.

- **A:** original theta-rho and original predictor/corrector vertical solvers.
- **B:** compiler-fused theta-rho plus coefficient arithmetic inside both solver scans.
- All other programs, input fields and vendor-specific launch settings stay fixed.
  No manual Python theta rewrite, blocking experiment or counter profiling.

From the Icon4Py checkout, on Beverin:

```bash
sbatch amd_scripts/combined_fusion/run_amd.sh
```

On Santis:

```bash
sbatch amd_scripts/combined_fusion/run_nvidia.sh
```

The user submits jobs. Do not transfer source changes into either checkout
while its job runs. Results go to `amd_scripts/combined_fusion_runs/amd_<job>/`
or `nvidia_<job>/`. Both scripts snapshot the bundle and use a private patched
GT4Py package. Existing production/compiler files are not overwritten.

The validated timing harness supplies 12 balanced paired quartets and 12
interleaved A/A controls, whole-granule numerical validation, explicit 120-level
verification, generated-code checks, input fingerprints and before/after source
hashes. Original and combined arms run in the same process on the same node.
The source pins require the existing GT4Py scalar-conversion fix on both clusters.
A compiler preflight and per-CMake timeout guard against the earlier build hang.

Expected structural checks: theta 6→5 kernels with unchanged gradient buffers;
both solvers lose coefficient-sized intermediate storage and at least one
kernel. AMD retains 256-thread blocks/four-level blocking; NVIDIA retains native
128×2 or 64×1 blocks and no vertical blocking. NVIDIA correctness and the
combined performance on either GPU are still pending. A failed check prevents
COMPLETE; raw logs remain available for diagnosis.

## Reading the result

Require COMPLETE, numerical/source/code audits, and review A/A and order checks
before accepting a speedup. `TIMING_SUMMARY.json` reports device sum, host wall,
theta and combined solver times; raw per-program timings remain available.
Report device and wall savings separately, with uncertainty. The conservative
noise screen is the same as before: mean saving exceeds |A/A mean| + 2×A/A SD,
in addition to positive raw/control-adjusted confidence intervals and no
unresolved order effect.

For each GPU: reduction = 1 − B/A. Compare matching input fingerprints and source
hashes across GPUs before comparing vendor ratios. Original vendor ratio is
AMD_A/GH200_A; combined ratio is AMD_B/GH200_B. Gap closure is
`((AMD_A−AMD_B)−(GH200_A−GH200_B))/(AMD_A−GH200_A)`; report its uncertainty,
not just a point estimate. These separate-node runs measure each vendor's
same-node treatment effect; node-to-node generality needs replication.

Earlier AMD measurements suggest about **5.34% less device time combined**.
That remains a cross-run estimate until this direct comparison finishes.
This run does not measure memory traffic, prove a unique cache mechanism,
or validate performance on the global grid.
