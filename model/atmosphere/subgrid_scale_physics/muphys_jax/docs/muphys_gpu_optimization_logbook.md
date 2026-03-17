# Muphys Graupel GPU Optimization Logbook

> Optimizing JAX graupel microphysics towards DaCe-GPU performance on NVIDIA GH200 and AMD MI300.
> Target: reduce per-iteration time from ~51ms to ~10-13ms (DaCe-GPU baseline).

---

## State: March 2025

### Summary

After extensive exploration across multiple optimization strategies (buffer donation, tiling, Triton, StableHLO injection, XLA/IREE compiler passes, memory layout transposition), the current best result is **29ms per iteration** on GH200 using StableHLO injection + transposed memory layout. Work continues on XLA compiler pass development and IREE backend investigation.

### Current Performance (R2B06, GH200)

| Configuration | Time (ms) | Speedup vs Baseline |
|:---|:---:|:---:|
| JAX baseline (original) | ~51 | 1.0x |
| + transposed layout + StableHLO injection | ~35 | 1.46x |
| + further fusion (fused q_t_update) | ~29 | 1.76x |
| DaCe-GPU target | ~10-13 | ~4-5x |

### Performance Breakdown (nsys, 35ms state)

| Component | Time (ms) | Notes |
|:---|:---:|:---|
| q_t_update kernel | 10.5 | 2 fused kernels (down from ~80) |
| Injected precipitation scan | 11.3 | StableHLO-injected unrolled scan |
| Unexplained overhead | ~13 | Kernel launch overhead, to be investigated |

### Correctness

- Integration tests pass against reference baseline data
- R2B06 test case passes against baseline

---

## Optimization Track 1: JAX-Level Optimizations

### Buffer Donation

- Explored JAX buffer donation to reduce memory allocation overhead
- **Result:** No significant performance improvement observed

### Tiling

- Explored tiling strategies for the graupel computation
- **Result:** No significant performance improvement observed

### Triton-JAX Integration

- Implemented Triton-based kernels called from JAX
- Triton kernel-only execution showed improvement
- However, passing JAX arrays to Triton via DLPack involves copies/conversions, adding ~5-20% overhead for tensor interchange
- **Result:** Overall still slower than baseline due to interchange overhead. Not a viable path forward (also not a desired solution architecturally).

### Memory Layout Transposition

- Transposed from `(ncells, nlev)` to `(nlev, ncells)` layout
- Ensures coalesced GPU memory access for vertical scans
- Pre-transpose data once at load time; zero transposes during computation
- XLA does not seem able to remove the transposes automatically
- **Result:** Significant improvement, especially when combined with StableHLO injection. This was a prerequisite for StableHLO injection benefits to become visible.

### Fused q_t_update

- Rewrote all phase transitions using `lax.select`/`lax.pow` to encourage XLA kernel fusion
- Reduced from ~80 kernels to 2 fused large kernels (confirmed via nsys)
- **Result:** Major kernel count reduction. Plan to apply StableHLO injection here as well.

---

## Optimization Track 2: StableHLO Injection

### Precipitation Scan Injection

- Hand-generated / script-generated unrolled StableHLO for precipitation scans
- Eliminates `while` loops and `dynamic_slice`/`dynamic_update_slice` operations (D2D copies)
- Injected via `mlir.merge_mlir_modules()` during JAX's MLIR lowering
- Benefits were not visible until the transpose issue (Track 1) was fixed
- **Result:** Combined with transpose, brought runtime from ~51ms to ~35ms

### Bottleneck Analysis

- The precipitation scan over 90 vertical levels remains the bottleneck
- JAX/XLA sees 90 independent slice-compute-concat chains and creates ~186 separate GPU kernels
- Each kernel is tiny; most time spent on kernel launch overhead rather than computation

---

## Optimization Track 3: XLA Compiler Pass (LoopifyUnrolledSlices)

### Goal

Detect the unrolled 90-level slice-compute-concat pattern in XLA HLO and re-roll it into a `while` loop. When XLA sees a loop body instead of 186 separate operations, its fusion pass can collapse everything into ~6 kernels per iteration.

### Approach

- Custom XLA compiler pass: `LoopifyUnrolledSlices`
- Built JAX and XLA from source to integrate the pass
- Attempted StableHLO-level re-rolling first, but XLA overrides things during lowering, making it slower

### Status

- Pass compiles and is integrated into JAX (built from source within modified XLA)
- First benchmark (GH200):
  - 29.63ms -- baseline (unrolled, with StableHLO injection)
  - 42.96ms -- with loopify pass (slower, correctness issue)
- **Root cause identified:** pass only loopified 1 out of 11 output concatenates. The precipitation scan produces 11 outputs (temperature, etc.) that share the same 90-level computation. The pass should group all 11 into a single `while` loop with 11 accumulators.
- Remaining 10 unrolled concatenates cause: while loop overhead + unrolled concat overhead
- **Current bug:** wrong temperature update; chain detection/grouping logic does not capture all 11 concatenates
- **Blocked:** debugging chain detection/grouping logic

---

## Optimization Track 4: IREE

### IREE CUDA Backend

- Patched an IREE CUDA bug to run the full JAX graupel pipeline
- Due to limitations JIT-compiling large functions (CUDA memory errors), had to split into separately JIT-compiled functions (precip and temp) rather than a single fused JIT function
- This adds overhead; result is slightly slower than XLA
- **Status:** functional but slower; further investigation needed. CUDA backend receives limited upstream attention.

### IREE HIP/ROCm Backend (AMD MI300)

- Successfully running IREE on Beverin (MI300)
- Runtime: **47ms** for 1 iteration (slower than XLA JAX on GH200 at 29ms)
- AMD backend is more complete than CUDA backend

### IREE Compiler Pass (Codegen Level)

- Writing a custom pass at the linalg level to restructure the computation
- **Current (wrong):** `scf.for(90 levels){ compute(all 327680 cells) }` -- serial at function level
- **Target (correct):** `dispatch(327680 cells){ scf.for(90 levels){ compute(one cell) } }` -- per-cell parallelism with vertical loop inside
- Target structure would be comparable to DaCe's approach (~10ms expected)

#### Pass Details

- Two passes developed: preprocessing level and codegen level (see separate IREE pass notes)
- Preprocessing pass blocked by downstream dispatch creation/tiling issues
- Codegen-level pass (`GPULoopifyUnrolledSliceChain`) runs before tiling in `addGPUTileAndFusePassPipeline()`
- Same chain detection + carry analysis as XLA pass, adapted for MLIR tensor.insert_slice chains
- **Status:** in progress; blocked by reported IREE bug

### MLIR Exploration

- Exploring writing graupel code directly with MLIR core dialects (no stencils involved)
- Writing/exploring MLIR examples as preparation for a potential pure-MLIR implementation path

---

## Summary of Approaches Explored

| Approach | Outcome | Status |
|:---|:---|:---|
| Buffer donation | No improvement | Abandoned |
| Tiling | No improvement | Abandoned |
| Triton-JAX | Kernel improvement, but DLPack overhead negates gains | Abandoned |
| Memory transpose (nlev, ncells) | Significant improvement | **Adopted** |
| StableHLO injection (precip) | 51ms to 35ms | **Adopted** |
| Fused q_t_update | ~80 kernels to 2 | **Adopted** |
| XLA LoopifyUnrolledSlices pass | Slower + correctness bug | In progress |
| IREE CUDA | Functional but slower | Needs investigation |
| IREE HIP (MI300) | 47ms | Baseline established |
| IREE codegen-level pass | Wrong dispatch structure | In progress |
| Pure MLIR | Exploratory | Early stage |

---

## Next Steps

1. **XLA pass:** fix chain detection/grouping to capture all 11 concatenates in a single while loop
2. **q_t_update:** apply StableHLO injection (like precipitation scans) to further reduce kernel count
3. **Investigate 13ms overhead:** profile kernel launch overhead and remaining small kernels
4. **IREE codegen pass:** achieve correct dispatch structure `dispatch(cells){ scf.for(levels){ compute } }`
5. **IREE CUDA:** investigate memory errors when JIT-compiling the full fused function
6. **Transpose elimination:** remove pre-transpose step entirely (currently not measured, but desirable)
