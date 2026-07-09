# MI300A dycore optimization — status: what worked, what didn't

**Updated: 2026-07-09.** Consolidated status of the AMD MI300A dycore optimization effort.
Supersedes the scattered per-topic docs for a top-level view. Numbers are measured on the
`mi300` Slurm partition (gfx942 = MI300A), `icon_benchmark_regional` grid, same-job/same-node
A/B, correctness-validated. See [HARDWARE_REFERENCE](HARDWARE_REFERENCE.md) and
[CLUSTER_NODE_VARIANCE](CLUSTER_NODE_VARIANCE.md) for the run-to-run variance caveat.

> **TL;DR (read this first):** at the **isolated-kernel** (stencil-test) level the wins are real —
> **−10…−17%** from the block config and **−14.3%** from the exner-AoS. But at the **full
> `solve_nonhydro` step** these dilute to **~−2% (inside the run-to-run noise)**, and the exner-AoS
> **does not even apply to the full-model codegen** (fails closed → no effect). So: **keep the
> block config** (real, free, per-kernel gains that just form a small fraction of the step);
> **treat the exner-AoS as a validated proof-of-concept, not a shipped model win.** Everything else
> tried was net-negative or marginal — the heavy kernels are at their HBM bandwidth floor. The
> full-model measurement is in [§ Isolated kernel vs full model](#-isolated-kernel-vs-full-model--the-measured-dilution-2026-07-09).

---

## ✅ Landed (committed)

### 1. Per-program 2D thread-block config
**Repo:** icon4py `amd_profiling_rocm723` — commits `35e4ce0fb` → `1b69c6ad8` → `ccb7fdebd`.
**Where:** `model/common/src/icon4py/model/common/model_options.py`, ROCm branch, `_mi300a_block_2d`.

| kernel | block | gain vs 256×1 default |
|---|---|---|
| `compute_rho_theta_pgrad_and_update_vn` (thetarho) | **(32,16,1)** | −10% (32×8) + 6.5% (512-thr) |
| `compute_horizontal_velocity_quantities_and_fluxes` (hvel) | **(128,2,1)** | −14% |
| `compute_advection_in_horizontal_momentum` (#8 hmom) | **(64,8,1)** | −17% + 6.2% |
| `compute_advection_in_corrector_vertical_momentum` (#7 vmom) | **(128,4,1)** | −12.5% + 6.9% |

**Mechanism:** the hot edge kernels gather `field[E2C[edge], k+ikoffset]` — connectivity
(`E2C[edge]`) and geometry (normals, edge lengths, `pos_on_tplane`) are **K-invariant**, so a 2D
block grouping several K-levels of the *same* edges fetches that data once and reuses it across
the K-group instead of re-scattering per K-level. Both **shape and size** matter: a 512-thread
block holds more K-levels → more reuse (the +6-7% on top of the 256-thread shapes).
**Per-program only** — 256×1 stays the global default (best for the Cell-indexed solver + most
kernels; a global flip regresses the solver ~+10% and divdamp ~+23%). Do **not** combine with
vertical loop blocking (both restructure K → interfere). Detail: [THETARHO_32x8_RESULT](THETARHO_32x8_RESULT.md).

### 2. exner-AoS in-kernel packing (thetarho)
**Repo:** gt4py fork `extend_loopblocking` (sandbox: `gt4py_claude`; your fork: `dganellari/gt4py`) — commit `d53089f`.
**Code (both files in `src/gt4py/next/program_processors/runners/dace/workflow/`):**
- **`exner_aos.py`** — the transform. `apply_exner_aos()` does the whole thing: the `aos[i*3+f]`
  interleaved layout, the coalesced `__repack_exner_aos_kernel` it injects, the `_READ_RE` regex
  that redirects the 18 terrain reads, `_N_REGIONAL` (the hardcoded Cell·K), and the self-check
  (`tot`) that returns `False` → no patch if anything doesn't match.
- **`compilation.py`** — the pipeline hook (search `EXNER_AOS`). After `sdfg.compile()`: if
  `GT4PY_EXNER_AOS=1` and `sdfg.name == exner_aos.PROGRAM`, it calls `apply_exner_aos(build_folder)`
  then `dace.codegen.compiler.configure_and_compile` + `load_precompiled_sdfg` to recompile and
  reload the patched `.so`.

**Enable:** opt-in via `GT4PY_EXNER_AOS=1`. **Result: −14.3%** (1.266 → 1.085 ms), correctness-passing.

thetarho's dominant cost is the doubly-indirect terrain gather of three separate cell fields
(`d2dz2` / `ddz` / `temporal_extrapolation` of the perturbed exner) at the same uncached
`[E2C[edge], k+ikoffset]` address (~10.6% L2), costing three scattered 64-byte line-fetches. The
transform packs them interleaved into one `aos[i*3+f]` so **one fetch serves all three**; a
coalesced repack kernel rebuilds the AoS each call (correct for changing inputs). It's a
**post-codegen** transform (gt4py can't make the pack dim unit-stride at the frontend), is
self-checking and **fails closed** (mismatch → no patch → original correct code). `N` (= Cell·K)
is **hardcoded to the regional grid** (auto-deriving it from the SDFG proved unreliable → OOB),
which is why it's opt-in, not default-on. Detail: `EXNER_AOS_OPTIMIZATION.md` (in the gt4py/sandbox
docs).

> **⚠️ Does NOT apply to the full model (measured 2026-07-09).** In the full `solve_nonhydro`
> compile, thetarho's terrain-read expression differs from the isolated stencil test, so the
> read-redirect regex matches **0 of 18 reads** → the transform fails closed (`MISMATCH`,
> `reads:0`) → **no patch is applied, zero effect in a real run**. The −14.3% above is an
> **isolated-stencil-only** result. Treat this as a validated *mechanism*, not a shipped feature;
> a production version needs a proper SDFG/frontend transform (or a codegen-robust matcher). See
> [§ Isolated kernel vs full model](#-isolated-kernel-vs-full-model--the-measured-dilution-2026-07-09).

> **Overlap note:** exner-AoS and thetarho's 32×16 block attack the *same* gather. When
> `GT4PY_EXNER_AOS=1` the exner-AoS dominates; 32×16 is the best block when it's off. They are not
> additive.

---

## ⚠️ Isolated kernel vs full model — the measured dilution (2026-07-09)

Everything in "Landed" above is an **isolated stencil-test** number: one gt4py program compiled and
benchmarked alone. A full-model A/B on the complete `solve_nonhydro` dycore step (the integration
benchmark `test_benchmark_solve_nonhydro`) shows how much of that survives into a real run.

**Full `solve_nonhydro` step — same node, 1300+ rounds, `pytest-benchmark`:**

| config | median | IQR | StdDev |
|---|---|---|---|
| baseline — `256×1` all kernels, `GT4PY_EXNER_AOS=0` | **8.223 ms** | 0.40 | 2.94 |
| optimized — per-program blocks + `GT4PY_EXNER_AOS=1` | **8.066 ms** | 0.41 | 2.94 |
| **Δ** | **−1.9%** | | |

Two effects, both important:

1. **The kernel gains dilute.** The four block-tuned kernels are only a *slice* of the step; the
   **vertically-implicit solver dominates the step and is unchanged** (256×1 in both configs). So
   −10…−17% on that slice → **~−2% on the whole step**. And the step carries **~36% run-to-run
   variance** (StdDev 2.94 ms on an 8 ms median — the `hipMalloc`-per-program-call jitter noted in
   `model_options.py`), so **−1.9% is barely above the noise**.

2. **The exner-AoS does not apply at all.** The full-model thetarho compile emits a different
   terrain-read expression than the isolated stencil, so the transform's read-redirect matched
   **0 of 18 reads** and failed closed (`[exner_aos] … MISMATCH … reads:0`). No patch → the −14.3%
   contributes **nothing** to the full run.

**Per-kernel: what is genuinely real vs what dilutes**

| kernel | isolated gain | in the full `solve_nonhydro` step |
|---|---|---|
| thetarho | −10% (block) / −14.3% (exner) | block: real but diluted; **exner-AoS did not apply** |
| hvel | −14% (block) | real, diluted |
| #8 hmom | −17% (block) | real, diluted |
| #7 vmom | −12.5% (block) | real, diluted |
| solver (biggest kernel) | 256×1 = −20.8% vs the 32×8 default | **global default — identical in both A/B configs, so not part of the Δ** |

**How to read this for ICON:** the block config is a genuine, free per-kernel win — it just lands on
a small fraction of a step the solver dominates, so the *step-level* number is ~2% and noise-limited.
The exner-AoS is a **demonstrated mechanism that does not survive into production codegen** as
implemented. To show the per-kernel gains *inside* the full run (instead of the noisy step total),
use the per-program GT4Py timer: `benchmark_dycore.sh` emits `dycore_gt4py_program_metrics.json`
(`DYCORE_GT4PY_PROGRAMS_TIMER_FILE`), printed via `amd_scripts/print_gt4py_timers.py`.

**Reproduce:** `amd_scripts/model_ab.sh` (baseline vs optimized full-step A/B; toggles
`ICON4PY_BLOCK2D=256x1` + `GT4PY_EXNER_AOS`).

---

## ❌ Refuted / not worth it (measured — do not re-chase)

| idea | result | why |
|---|---|---|
| exner-AoS **upstream** (producer pre-writes the AoS) | **−20% is a mirage — net-negative** | built + measured: the AoS is stride-3, so the producer's interleaved writes are uncoalesced and cost **+23.6% (+~100 µs)**, exceeding the ~57 µs thetarho saves. The run-once −20% was only a fixed-input benchmark proxy. |
| gradient-AoS (green-gauss transients `gtir_tmp_83/89/95/101`) | ~4% slower | the E2C reads are cache-served on MI300A (Infinity Cache); cold-microbench 3.33× didn't transfer |
| MaterializeNeighborGather pass | ~2.6× slower on #8 | forces redundant per-edge gather recompute |
| fuse-less (disable map fusion) | 2.7–4.8× slower | over-fusion hypothesis wrong; keep fusion ON |
| transient-fusion #2 (solver coeff→scan) | infeasible | blocked by `Koff` vertical-neighbor access the sequential scan can't do |
| transient-fusion #3 (divdamp `gtir_tmp_39`) | entangled | branch-dependent (`concat_where` 2nd/4th-order) with multiple producers/consumers, not a clean transient |
| #7 / #8 AoS | no opportunity | single-field reductions over contiguous coefficients, cache-friendly |
| occupancy / `waves_per_eu` / maxreg | 2.5× slower | MI300A already at max occupancy (few VGPRs); the NVIDIA maxreg trick doesn't apply |
| CSE of redundant global loads | neutral | `-O3` already does it |

Detail on the refuted set: `ATTEMPTED_OPTIMIZATIONS.md` (solver, April), `MATERIALIZE_RESULT.md`,
and the memory notes.

---

## Where we are

The heavy dycore kernels (thetarho, the vertically-implicit solver, hvel, #7/#8) are largely at
their **HBM memory floor** beyond the two landed wins. The solver is ~94% of peak bandwidth.
The June re-analysis of thetarho/#7/#8 (with the fusion lever in hand) confirmed **no missed
clean wins** — the transients that look fusable are structural (vertical-neighbor `Koff`, or
`concat_where` branch assembly). Further gains would need **frontend/SDFG-level** work — chiefly
unblocking `concat_where`-induced K-splits so map-fusion can keep HBM-round-tripped intermediates
in registers. That is a gt4py/DaCe pipeline concern (flagged for that team in
`ATTEMPTED_OPTIMIZATIONS.md`), not a tuning knob.

---

## Enable / reproduce

- **Block config:** automatic on the ROCm branch (icon4py `amd_profiling_rocm723`); no flag.
- **exner-AoS:** gt4py `extend_loopblocking` + `export GT4PY_EXNER_AOS=1` (thetarho, regional grid).
- **Benchmark one kernel:**
  ```bash
  srun -p mi300 -N1 --gres=gpu:1 -t 0:10:00 bash amd_scripts/benchmark_theta.sh
  ```
  or the stencil test directly with `--backend=dace_gpu --grid=icon_benchmark_regional`. The three
  profiled tests (thetarho / solver-predictor / full solve_nonhydro) are in
  `amd_scripts/benchmark_{theta,solver,dycore}.sh`.
- **Setup on a new system:** `amd_scripts/install_icon4py_venv.sh` (uv-based), under the ROCm 7.2.x
  uenv; run on the `mi300` partition (the login node is MI200 — gfx942 binaries segfault there).
