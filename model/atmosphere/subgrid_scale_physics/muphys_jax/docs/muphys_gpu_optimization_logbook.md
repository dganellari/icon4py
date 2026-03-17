# Muphys Graupel GPU Optimization Logbook (JAX → XLA / IREE vs DaCe-GPU)

> Optimizing JAX-compiled graupel microphysics via XLA and IREE compiler backends towards DaCe-GPU performance on NVIDIA GH200 and AMD MI300A.
> Target: reduce per-iteration time from ~51ms to <10ms (DaCe-GPU baseline).
> JAX alone cannot reach the target; the strategy relies on custom XLA and IREE compiler passes to re-roll unrolled loops into efficient GPU kernels.

---

## State: March 2025

### Summary

After extensive exploration across multiple optimization strategies, the current best results are:

- **29ms** on GH200 (Santis) — combined StableHLO injection (q_t_update + precip) + transposed memory layout
- **33ms** on GH200 (Santis) — custom XLA `LoopifyUnrolledSlices` pass (SerialScan mode, single GPU kernel for precipitation scan)
- **47ms** on MI300A (Beverin) — IREE HIP baseline; custom preprocessing pass (`LoopifyInsertSliceChain`) in progress

The core bottleneck is the precipitation scan over 90 vertical levels. JAX/XLA unrolls this into ~186 separate GPU kernels, spending most time on kernel launch overhead. The optimization work focuses on re-rolling these into 1-2 kernels via custom compiler passes or StableHLO injection.

### Current Performance (R2B06)

| Configuration | GPU | Cluster | Time (ms) | Notes |
|:---|:---:|:---:|:---:|:---|
| JAX baseline (original) | GH200 | Santis | ~51 | ~186 kernels for precip scan |
| + transposed layout + StableHLO injection | GH200 | Santis | ~35 | Unrolled but coalesced |
| + combined StableHLO (q_t_update + precip) | GH200 | Santis | ~29 | Single HLO module, best current result on GH200 |
| XLA LoopifyUnrolledSlices (SerialScan) | GH200 | Santis | ~33 | 1 kernel for precip scan (replaces StableHLO injection) |
| IREE HIP baseline (no custom pass) | MI300A | Beverin | ~47 | ~186 dispatches for precip scan |
| IREE HIP + LoopifyInsertSliceChain (WIP) | MI300A | Beverin | ~80 | Correctness bug, not yet optimized |
| DaCe-GPU target | GH200 | Santis | <10 | — |

### Performance Breakdown (nsys profile, GH200, at the 35ms configuration stage)

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
- **Result:** Major kernel count reduction. Now part of the combined StableHLO module (see Track 2).

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

Detect the unrolled 90-level slice-compute-concat pattern in XLA HLO and re-roll it into a loop or single-kernel serial scan. Eliminates ~186 separate GPU kernel launches by fusing the entire precipitation scan into 1-2 kernels.

### Approach

- Custom XLA compiler pass: `LoopifyUnrolledSlices`
- Built JAX and XLA from source to integrate the pass
- Attempted StableHLO-level re-rolling first, but XLA overrides things during lowering, making it slower
- Two modes:
  - `kWhileLoop` — emits an HLO while loop; XLA fuses the body into ~6 kernels
  - `kSerialScan` — emits a custom fusion with a single GPU kernel containing `scf.forall` (parallel over cells) + `scf.for` (serial over levels)

### Algorithm

1. **Chain detection**: Find sequences of `slice → compute → concat` at consecutive offsets along the vertical dimension
2. **Chain grouping**: Group all 11 output concatenates sharing the same 90-level computation into a single loop with 11 accumulators
3. **Slice-dependency propagation**: Forward-propagate from k=0 level slices through iter0 body to identify which ops depend on the level index vs. shared precomputation
4. **Carry detection**: Walk (iter2, iter1) structurally to find inter-iteration dependencies (19 carry pairs including shifted carries for previous-level values)
5. **Body construction**: Use iter1 as the loop body template; iter0 output is used as carry initial values
6. **Scalarization (SerialScan mode)**: Convert tensor-level ops to scalar arithmetic inside the kernel; each workgroup processes one cell, looping over 90 levels

### Status: Working

- Pass compiles and is integrated into JAX (built from source within modified XLA)
- All 11 output concatenates correctly grouped into a single loop
- Correctness verified: all output fields match reference data
- **Benchmark progression (all on NVIDIA GH200, R2B06)**:
  - 29.63ms — baseline (unrolled, with StableHLO injection + fused q_t_update)
  - 42.96ms — first attempt (only 1 of 11 chains loopified)
  - 35ms — after fixing chain grouping (all 11 chains, while-loop mode)
  - **33ms** — SerialScan mode (single GPU kernel)
- SerialScan mode eliminates ~186 kernel launches → 1 kernel for the precipitation scan
- Still ~4ms slower than the StableHLO injection baseline (29ms); the gap is likely due to the scan body being less optimized than XLA's hand-fused kernels
- See [INTEGRATION.md](../xla_passes/INTEGRATION.md) for build/integration instructions

---

## Optimization Track 4: IREE

### IREE CUDA Backend

- Patched an IREE CUDA bug to run the full JAX graupel pipeline
- Due to limitations JIT-compiling large functions (CUDA memory errors), had to split into separately JIT-compiled functions (precip and temp) rather than a single fused JIT function
- This adds overhead; result is slightly slower than XLA
- **Status:** functional but slower; further investigation needed. CUDA backend receives limited upstream attention.

### IREE HIP/ROCm Backend (AMD MI300A, Beverin)

- Successfully running IREE on Beverin cluster (AMD MI300A, gfx942)
- Baseline runtime (no custom pass): **47ms** for 1 iteration
- Not directly comparable to GH200/Santis numbers (different GPU architecture)
- AMD backend is more complete than CUDA backend

### IREE Compiler Pass (Preprocessing Level)

- Custom preprocessing pass `LoopifyInsertSliceChain` at the linalg/tensor level
- Detects unrolled `tensor.insert_slice` chains (from StableHLO concatenate lowering) and converts them to `scf.forall + scf.for` loops inside a `flow.dispatch.region`
- **Target structure:** `flow.dispatch.region { scf.forall(%cell in 0..327680) { scf.for(%k in 0..90) { scalar_compute } } }`
- Uses `WorkgroupMappingAttr` so IREE's codegen distributes forall to GPU workgroups (one thread per cell)

#### Algorithm (mirrors XLA pass)

1. Chain detection: find `tensor.insert_slice` chains at consecutive offsets
2. Chain grouping: group all 11 output chains into one loop with 11 column accumulators
3. Slice-dependency propagation: forward-propagate from k=0 level slices to identify level-dependent vs. shared ops
4. Carry detection: structural (iter2, iter1) walk for 19 carry pairs
5. Body construction: iter1 as template, scalarization of `linalg.generic` ops
6. Pre-computation pull-in: traces backward from level-slice sources to include intermediate ops in the loop body

#### Status: Structurally correct, correctness bug in progress

- Generates correct dispatch structure: `forall(327680) + for(90)` inside `flow.dispatch.region`
- All 11 chains grouped, 19 carries detected, scalarization works
- **Correctness bug:** temperature field has error 5.59e+01 (other 10 fields match reference)
- **Root cause identified:** iter1 boundary construction uses ALL iter0 values as boundaries, but should only use SLICE-DEPENDENT iter0 values (matching XLA's `depends_on_slice` filtering). This makes iter1's body too small (202 ops vs ~784), missing shared precomputed ops.
- **Fix planned:** port XLA's slice-dependency forward propagation to IREE pass
- Runtime on AMD MI300A (Beverin, gfx942): ~80ms (vs 45.7ms IREE baseline without pass); performance optimization deferred until correctness is achieved

#### Earlier attempts

- Initially tried codegen-level pass (`GPULoopifyUnrolledSliceChain`) running before tiling in `addGPUTileAndFusePassPipeline()` — blocked by downstream dispatch creation/tiling issues
- Moved to preprocessing level to avoid codegen pipeline conflicts

---

## Summary of Approaches Explored

| Approach | GPU | Outcome | Status |
|:---|:---:|:---|:---|
| Buffer donation | GH200 | No improvement | Abandoned |
| Tiling | GH200 | No improvement | Abandoned |
| Triton-JAX | GH200 | Kernel improvement, but DLPack overhead negates gains | Abandoned |
| Memory transpose (nlev, ncells) | GH200 | Significant improvement | **Adopted** |
| StableHLO injection (precip only) | GH200 | 51ms to 35ms | **Adopted** |
| Combined StableHLO (q_t_update + precip) | GH200 | 35ms to 29ms | **Adopted** |
| XLA LoopifyUnrolledSlices (WhileLoop) | GH200 | 35ms, correct, ~6 kernels | **Working** |
| XLA LoopifyUnrolledSlices (SerialScan) | GH200 | 33ms, correct, 1 kernel | **Working** |
| IREE CUDA | GH200 | Functional but slower | Low priority |
| IREE HIP baseline | MI300A | 47ms | Baseline established |
| IREE preprocessing pass | MI300A | Correct structure, temperature bug (80ms) | In progress |
| Pure MLIR | — | Exploratory | Future idea |

---

## Next Steps

1. **IREE preprocessing pass:** fix iter1 boundary construction (port `depends_on_slice` from XLA pass) to achieve correctness on AMD MI300A
2. **XLA SerialScan performance:** close the 4ms gap vs StableHLO injection baseline (33ms vs 29ms) — investigate body optimization
3. **Investigate overhead:** profile remaining kernel launch overhead and small kernels
4. **Transpose elimination:** remove pre-transpose step entirely (currently not measured, but desirable)

---

## Future Ideas

- **Pure MLIR rewrite:** write graupel code directly with MLIR core dialects (no stencils), bypassing JAX entirely. Exploratory; not currently active.
