# Muphys Graupel GPU Optimization Logbook

> Optimizing JAX graupel microphysics towards DaCe-GPU performance on NVIDIA GH200 and AMD MI300.
> Target: reduce per-iteration time from ~51ms to ~10-13ms (DaCe-GPU baseline).

---

## State: March 2026

### Summary

After extensive exploration across multiple optimization strategies (buffer donation, tiling, Triton, StableHLO injection, XLA/IREE compiler passes, memory layout transposition, power-op decomposition, XLA flag tuning), the current best result is **11.15ms per iteration** on MI300A (Beverin) using Combined StableHLO injection + fast_pow decomposition + transposed memory layout + optimized XLA flags, with JAX 0.6.0 and XLA ROCm. This is within the DaCe-GPU target range of ~10-13ms. On GH200 (Santis), the best result remains ~29ms without the XLA pass.

### Current Performance (R2B06)

| Configuration | GPU | Cluster | Time (ms) | Notes |
|:---|:---|:---|:---:|:---|
| JAX baseline (original) | GH200 | Santis | ~51 | No optimizations |
| + transposed layout + StableHLO injection | GH200 | Santis | ~35 | First optimization |
| + further fusion (fused q_t_update) | GH200 | Santis | ~29 | Best on GH200 |
| Combined StableHLO (JAX 0.6.0, ROCm) | MI300A | Beverin | 12.97 | Default XLA flags |
| + fast_pow + XLA flag tuning | MI300A | Beverin | **11.15** | **Best overall**, within DaCe target range |
| XLA LoopifyUnrolledSlices (WhileLoop) | GH200 | Santis | 32.99 | Correct results, host-device sync overhead |
| DaCe-GPU target | GH200 | - | ~10-13 | Target |

### Correctness

- Integration tests pass against reference baseline data
- R2B06 test case passes against baseline
- XLA while-loop pass validated: max diff 2.17e-06 vs baseline

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
- Later replaced all `lax.pow(x, c)` with `jnp.exp(c * jnp.log(x))` (`_fast_pow`) — `stablehlo.power` blocks XLA fusion; `exp+multiply+log` fuse into elementwise kernels
- **Result:** Major kernel count reduction. fast_pow saved ~1ms on MI300A (12.27 → 11.88ms → 11.15ms combined with XLA flags).

---

## Optimization Track 2: StableHLO Injection

### Precipitation Scan Injection

- Hand-generated / script-generated unrolled StableHLO for precipitation scans
- Eliminates `while` loops and `dynamic_slice`/`dynamic_update_slice` operations (D2D copies)
- Injected via `mlir.merge_mlir_modules()` during JAX's MLIR lowering
- Benefits were not visible until the transpose issue (Track 1) was fixed
- **Result:** Combined with transpose, brought runtime from ~51ms to ~35ms

### Combined StableHLO (q_t_update + precipitation)

- Combined all optimizations into a single StableHLO injection
- Best result: **12.97ms** on MI300A with JAX 0.6.0
- Significant JAX version sensitivity: JAX 0.9.2 regresses to 23.32ms (+79%)

### Bottleneck Analysis

- The precipitation scan over 90 vertical levels remains the bottleneck
- JAX/XLA sees 90 independent slice-compute-concat chains and creates ~186 separate GPU kernels
- Each kernel is tiny; most time spent on kernel launch overhead rather than computation

### Profiling (rocprofv3, MI300A, Mar 2026)

Per-invocation breakdown (applied to 12.97ms baseline):

| Category | % of total | Est. per run (ms) |
|:---|:---:|:---:|
| Precip scan body computation | 20.2% | 2.6 |
| Concatenation (4 kernels) | 22.8% | 3.0 |
| q_t_update (with power ops) | 8.8% | 1.1 |
| Per-level compute kernels | ~22% | 2.9 |
| Dynamic slicing | 7.1% | 0.9 |
| Transposes | ~11% | 1.4 |
| Copies + broadcasts + other | ~8% | 1.0 |

Key finding: concatenation (23%) + dynamic slicing (7%) = 30% of runtime is pure overhead from the unrolled scan pattern.

### XLA Flag Tuning (MI300A, JAX 0.6.0)

Systematic exploration of XLA flags on the unrolled StableHLO baseline:

| Flags | Time (ms) |
|:---|:---:|
| Default (`autotune=0`) | 12.97 |
| `graph_level=0` | 12.41 |
| `backend_optimization_level=3 + llvm_force_inline_before_split=true` | 12.27 |
| + fast_pow (exp/log replacing stablehlo.power in q_t_update) | 11.88 |
| + fast_pow + `graph_level=0` + LLVM flags | **11.15** |

Best configuration: `--xla_gpu_autotune_level=0 --xla_gpu_graph_level=0 --xla_backend_optimization_level=3 --xla_llvm_force_inline_before_split=true`

Flags that did NOT help: `graph_min_graph_size=2`, `enable_latency_hiding_scheduler`, `disable fusion_merger`, `enable_dynamic_slice_fusion`, `GPU_MAX_HW_QUEUES`, `HSA_FORCE_FINE_GRAIN_PCIE`.

### StableHLO While Loop Attempt (MI300A, Mar 2026)

- Generated `stablehlo.while` version of the precipitation scan to eliminate concat/slice overhead
- Correct results, but 25.25ms — worse than 12.97ms baseline
- WhileThunk host-device sync adds ~12ms (90 iterations x ~130us per sync)
- ROCm command buffer graph capture segfaults on while loops (`--xla_gpu_graph_level=0` required)
- `--xla_gpu_command_buffer_unroll_loops` flag does not exist in JAX 0.6.0's XLA version
- Conclusion: while loops at the StableHLO level do not help

---

## Optimization Track 3: XLA Compiler Pass (LoopifyUnrolledSlices)

### Goal

Detect the unrolled 90-level slice-compute-concat pattern in XLA HLO and replace it with a single GPU kernel. Two modes implemented:
1. **kWhileLoop**: Re-roll into a while loop (XLA fuses into fewer kernels)
2. **kSerialScan**: Create a custom fusion with a serial scan emitter (single kernel)

### Approach

- Custom XLA compiler pass: `LoopifyUnrolledSlices` (located in `xla/service/gpu/transforms/loopify/`)
- Built JAX and XLA from source to integrate the pass
- Pass registered in `fusion_pipeline.cc` inside `HloPassFix<HloPassPipeline>`

### While-Loop Mode (kWhileLoop)

- **Chain detection fixed:** all 10 concatenates now correctly grouped into a single while loop with 10 accumulators
- Carry analysis identifies shifted carries, invariants, sliced inputs, and offset slices
- **Result on GH200:** 32.99ms (correct results validated)
- **Problem:** WhileThunk forces host-device synchronization per iteration (90 round-trips), making it slower than the 29ms baseline
- This overhead is fundamental to XLA's while-loop execution model

### Serial Scan Mode (kSerialScan)

Attempted to generate a single GPU kernel via custom `SerialScanFusion` emitter:

- **Pass side (BuildSerialScanFusion):** Successfully creates a kCustom fusion with `__serial_scan` kind
  - Builds per-level scalar body computation
  - Serializes body ops into metadata string (avoids computation pruning)
  - Inline DCE removes dead unrolled instructions (18420 → 1068)
  - Uses `GpuBackendConfig` → `FusionBackendConfig` wrapper for backend config
- **Emitter side (SerialScanFusion):** Generates MLIR from serialized body ops
  - Supports ~20 HLO opcodes (add, mul, select, compare, convert, cbrt, etc.)
  - **Blocked:** XLA's GPU MLIR lowering pipeline does not support standard `scf.for`/`scf.forall` + `tensor.extract`/`tensor.insert` patterns
  - XLA emitters use custom `EmitXlaLoopOp` → `xla_gpu.loop` ops with special bufferization support
  - `scf.forall` IS used by XLA's LoopFusionKernelEmitter, but only with `EmitXlaLoopOp` inside (not arbitrary scf.for)
  - A sequential scan doesn't fit the per-element `EmitXlaLoopOp` model

### Key Technical Findings

1. XLA's EmitterBase framework expects MLIR using custom `xla_gpu` dialect ops, not standard MLIR SCF/tensor
2. `scf.for` with tensor iter_args cannot be bufferized by XLA's pipeline
3. Body computation pruning solved by serializing HLO ops directly into the fusion's backend config metadata string
4. HloDCE crashes if fusion body has unused parameters — solved with keep-alive custom call chain
5. Significant effort spent on XLA version compatibility (API differences in `Shape::rank()` vs `dimensions_size()`, `FusionBackendConfig` vs `GpuBackendConfig`, `indexing_map` vs `indexing_analysis`, etc.)

### Files

- `loopify_unrolled_slices.h/cc` — Main pass (both modes)
- `serial_scan_emitter.h/cc` — Custom GPU kernel emitter (blocked)
- `BUILD` — Bazel build configuration
- Requires 5 patches to XLA core files (see `XLA_CORE_PATCHES.md`)

---

## Optimization Track 4: IREE

### IREE CUDA Backend

- Patched an IREE CUDA bug to run the full JAX graupel pipeline
- Due to limitations JIT-compiling large functions (CUDA memory errors), had to split into separately JIT-compiled functions (precip and temp) rather than a single fused JIT function
- This adds overhead; result is slightly slower than XLA
- **Status:** functional but slower; further investigation needed. CUDA backend receives limited upstream attention.

### IREE HIP/ROCm Backend (AMD MI300)

- Successfully running IREE on Beverin (MI300)
- Runtime: **45-47ms** for 1 iteration (slower than XLA JAX best of 12.97ms on MI300A)
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
| Combined StableHLO (JAX 0.6.0) | 12.97ms on MI300A (default flags) | **Adopted** |
| + fast_pow + XLA flag tuning | **11.15ms** on MI300A | **Best result** |
| StableHLO while loop | 25.25ms on MI300A (host-device sync overhead) | Not useful |
| XLA LoopifyUnrolledSlices (WhileLoop) | 33ms, correct but slower (host-device sync) | Functional, not useful |
| XLA LoopifyUnrolledSlices (SerialScan) | Blocked by XLA MLIR lowering | Blocked |
| IREE CUDA | Functional but slower | Needs investigation |
| IREE HIP (MI300) | 47ms | Baseline established |
| IREE codegen-level pass | Wrong dispatch structure | In progress |
| Pure MLIR | Exploratory | Early stage |

---

## Next Steps

1. **Push below 11ms:** Profile the 11.15ms result to find remaining overhead; look for further fusion opportunities
2. **JAX version regression:** Investigate why JAX 0.9.2 regresses to 23.32ms (vs 11.15ms on JAX 0.6.0)
3. **XLA SerialScan emitter:** Only path to eliminate concat/slice overhead (~3ms); requires `xla_gpu` dialect work
4. **IREE codegen pass:** Achieve correct dispatch structure `dispatch(cells){ scf.for(levels){ compute } }`
5. **Transpose elimination:** Remove pre-transpose step entirely (~19ms overhead at runtime)
