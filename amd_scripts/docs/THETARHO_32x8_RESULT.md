# Thetarho 32×8 thread block — result and why it works

**Date:** 2026-06-19 (beverin, MI300A). **Verdict: setting the 2D thread block to 32×8
for `compute_rho_theta_pgrad_and_update_vn` gives a validated −17.9% — bigger than VLB4,
and it already beats the colleague's current 256×1+VLB4 config.**

Benchmark: stencil test `test_compute_theta_rho_face_values_and_pressure_gradient_and_update_vn`,
variant `is_iau_active[False]-compile_time_domain`, 40 rounds × 10 iters, datatest-gated,
clean sequential A/B on one node (no GPU contention).

## Result matrix

| config | median (ms) | vs baseline |
|---|---|---|
| 256×1, VLB1 (baseline) | 4.7536 | — |
| 256×1, VLB4 (**colleague's current config**) | 3.9678 | −16.5% |
| **32×8, VLB1** | **3.9043** | **−17.9% ← BEST** |
| 32×8, VLB4 | 4.5090 | **−5.1% ← WORST tuned config (the two INTERFERE)** |

Two clear findings:
1. **32×8 alone (−17.9%) beats 256×1+VLB4 (−16.5%)** — the block shape alone out-performs
   the colleague's current setup, without any vertical blocking, and is simpler (no VLB).
2. **32×8 + VLB4 is WORSE than either alone (−5.1%).** They do not stack — they interfere
   destructively. Use ONE, not both. The best single config is **32×8 alone**.

## Why 32×8 beats 256×1 on thetarho

The dominant kernel `_731` (terrain-following branch) has a doubly-indirect gather
`field[E2C[edge,slot], k + ikoffset[edge,slot,k]]` — scattered both horizontally (E2C
neighbor cell) and vertically (ikoffset). It is HBM-bound with only **10.6% L2 hit**
(badly scattered, almost no cache reuse).

Thread-block shapes lay the 64-lane wavefront out differently:
- **256×1**: all 256 threads on the horizontal (edge) dimension, 1 in K. A wavefront =
  **64 consecutive edges at ONE K level**.
- **32×8**: 32 edges × 8 K-levels. A wavefront = **32 edges across 2 adjacent K-levels**.

**The mechanism — K-invariant data sharing.** For a fixed edge, the gather's connectivity
and geometry are **K-invariant**:
- `E2C[edge,slot]` (the neighbor cell index) is identical for all K.
- The geometry factors (`pos_on_tplane_e`, `primal/dual_normal_cell`,
  `inv_dual_edge_length`, `coeff_gradekin`) are per-edge, K-independent.

So adjacent K-levels of the same edge reuse the same cell index and the same geometry
lookups. With **32×8**, a wavefront holds each edge's two K-levels together → the
K-invariant data is fetched once and used twice → the scattered E2C gather's index/geometry
overhead is amortized over 2 K-levels, and those reads cache/coalesce across the K-pair.
With **256×1**, a wavefront holds 64 distinct edges at one K → 64 distinct cells, no
K-pairing → every connectivity/geometry lookup is a fresh scattered access.

(Mechanism is consistent with the data and the kernel source; a direct confirmation would
be a per-kernel L2-hit comparison of `_731` under 256×1 vs 32×8 — expected to rise from
10.6%. Worth a follow-up counter run.)

## Relationship to VLB, and why 256×1 was the global default

- **VLB (vertical loop blocking) exploits the SAME K-invariance** — it blocks the K-loop so
  the K-invariant connectivity/geometry is loaded once per block of K-levels. So 32×8 and
  VLB attack the same inefficiency from two angles (wavefront K-pairing vs. loop blocking).
  **Measured: they INTERFERE destructively** — 32×8+VLB4 (4.5090 ms, −5.1%) is worse than
  32×8 alone (−17.9%) OR VLB4 alone (−16.5%). Because both restructure the K dimension,
  combining them breaks the wavefront K-pairing the block shape relies on (VLB's K-blocking
  leaves the 32×8 block's 8-tall K dimension under-filled / mis-paired, losing the reuse).
  **Conclusion: use exactly ONE K-reuse lever; 32×8 alone is the best.**
- **Why 256×1 is the global ROCm default** (`model_options.py`): it was tuned for the
  vertically-implicit **solver**, whose kernels are Cell-indexed and coalesce best with all
  threads on the contiguous Cell dimension — 256×1 maximizes coalescing there. But
  thetarho's edge-indexed scattered gather does NOT coalesce regardless of block shape, so it
  instead benefits from the K-pairing reuse that 32×8 provides. **Block shape is per-kernel:
  256×1 for the coalesced solver, 32×8 for the scattered thetarho.** (Note: gt4py's own
  default `gpu_block_size` is already (32,8,1); the AMD tuning overrides it to 256×1 globally.)

## Recommendation
- Set `gpu_block_size_2d=(32,8,1)` **for `compute_rho_theta_pgrad_and_update_vn` only** on
  ROCm, and **turn VLB off for it** — 32×8 alone is −17.9%, validated, correctness-gated.
  One-line per-program config; no VLB needed (simpler than the current approach).
- **Do NOT combine 32×8 with VLB** — they interfere (−5.1%, worse than either alone).
- The colleague is at 256×1+VLB4 (−16.5%); switching thetarho to **32×8 alone** is a small
  net win (−17.9%) AND drops VLB, and narrows the 96.87% vs-GH200 gap. Critically, it warns
  against the natural mistake of stacking his VLB4 with 32×8 (which would regress to −5.1%).
## Per-program 32×8 sweep (2026-06-19)

32×8 is a **targeted tool, not a global setting** — it helps scattered K-invariant
neighbor-gather kernels and hurts coalesced ones. Apply per-program. (compile_time_domain,
30 rounds, datatest-gated; sweep script `amd_scripts/sweep_block_32x8.sh`.)

13 programs tested (256×1 vs 32×8), exactly **2 winners**:

| program (GH200 gap) | 256×1 | 32×8 | verdict |
|---|---|---|---|
| **compute_rho_theta_pgrad_and_update_vn** (97%) | 4.75 ms | 3.90 ms | **−17.9% ✅ WIN** |
| **compute_horizontal_velocity_quantities_and_fluxes** (54%) | 1.676 ms | 1.482 ms | **−11.6% ✅ WIN** |
| compute_hydrostatic_correction_term (55%) | 88.4 µs | 87.7 µs | −0.8% (neutral) |
| compute_advection_in_vertical_momentum #7 (80%) | 920 µs | 988 µs | +7.3% ❌ |
| compute_advection_in_horizontal_momentum #8 (73%) | 656 µs | 817 µs | +24.6% ❌ |
| apply_divergence_damping_and_update_vn (60%) | 625 µs | 768 µs | +22.9% ❌ |
| interpolate_rho_theta_v_to_half_levels… (nonhydro_buoy, 57%) | 564 µs | 590 µs | +4.7% ❌ |
| compute_averaged_vn_and_fluxes (42%) | 620 µs | 704 µs | +13.6% ❌ |
| compute_perturbed_quantities_and_interpolation (37%) | 878 µs | 1042 µs | +18.7% ❌ |
| update_mass_flux_weighted (46%) | 200 µs | 247 µs | +23.3% ❌ |
| vertically_implicit_solver_at_corrector_step (35%) | 1.845 ms | 2.036 ms | +10.4% ❌ |
| vertically_implicit_solver_at_predictor_step (21%) | 1.916 ms | 2.129 ms | +11.1% ❌ |

**Apply 32×8 (per-program) to: `compute_rho_theta_pgrad_and_update_vn` (−18%) and
`compute_horizontal_velocity_quantities_and_fluxes` (−12%). Keep 256×1 for everything else
(2 win, 1 neutral, 10 regress).** A global 32×8 flip would slow most of the dycore down —
including the implicit solver, for which 256×1 was originally tuned.

## Production change (model_options.py) — what landed and why

The committed change (`get_dace_options`, ROCm branch) sets, before the default 256×1:

```python
if program_name in (
    "compute_rho_theta_pgrad_and_update_vn",
    "compute_horizontal_velocity_quantities_and_fluxes",
):
    optimization_args.setdefault("gpu_block_size_2d", (32, 8, 1))
optimization_args.setdefault("gpu_block_size_2d", (256, 1, 1))
```

Justification:
- **Per-program, not global:** the sweep shows only 2 of 13 programs benefit; 256×1 stays the
  default because it maximizes Cell-dimension coalescing for the solver and most cell-indexed
  kernels. A global 32×8 would regress the majority (e.g. solver +10–11%, divdamp +23%).
- **`setdefault` order:** the two winners get (32,8,1) first; the subsequent
  `setdefault(256,1,1)` only fills programs that weren't set — so winners keep 32×8, everyone
  else gets 256×1.
- **Replaces the VLB patch (not added to it):** the previous experiment applied vertical loop
  blocking (VLB4) to thetarho. The sweep showed **32×8 alone (−17.9%) beats VLB4 alone
  (−16.5%), and combining them regresses to −5.1%** because both restructure the K dimension
  and interfere. So 32×8 supersedes VLB for thetarho; the VLB patch and its now-unused `dims`
  import were removed. 32×8 must NOT be combined with VLB on these kernels.
