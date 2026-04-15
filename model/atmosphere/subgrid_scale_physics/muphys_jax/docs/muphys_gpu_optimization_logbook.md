# Muphys Graupel GPU Optimization Logbook (JAX → XLA / IREE)

> Optimizing JAX-compiled graupel microphysics via XLA and IREE compiler backends towards GT4Py DaCe GPU performance on NVIDIA GH200 and AMD MI300A.
> Target: reduce per-iteration time from ~51ms to <10ms (GT4Py DaCe GPU baseline).
> JAX alone cannot reach the target; the strategy combines StableHLO injection with power-op decomposition, XLA flag tuning, and custom XLA/IREE compiler passes to re-roll unrolled loops into efficient GPU kernels.

---

## State: March 2026

### Summary

After extensive exploration across multiple optimization strategies, the current best results are:

- **11.15ms** on MI300A (Beverin) — combined StableHLO injection + fast_pow + optimized XLA flags, JAX 0.6.0 (best overall, within <10ms target range)
- **27.09ms** on GH200 (Santis) — combined StableHLO injection + fast_pow + XLA flags, JAX 0.6.2, XLA CUDA
- **29.89ms** on GH200 (Santis) — custom XLA `LoopifyUnrolledSlices` pass (WhileLoop mode, with fast_pow, correct results, slower due to host-device sync)
- **47ms** on MI300A (Beverin) — IREE HIP baseline; custom preprocessing pass (`LoopifyInsertSliceChain`) in progress

The core bottleneck is the precipitation scan over 90 vertical levels. JAX/XLA unrolls this into ~186 separate GPU kernels, spending most time on kernel launch overhead. The optimization work focuses on re-rolling these into 1-2 kernels via custom compiler passes or StableHLO injection.

### Reference Performance

| Configuration | GPU | Cluster | Time (ms) | Notes |
|:---|:---:|:---:|:---:|:---|
| GT4Py DaCe GPU | GH200 | Santis | <10 | Target |

### Current Performance — XLA (R2B06)

| Configuration | GPU | Cluster | Time (ms) | Date | Notes |
|:---|:---:|:---:|:---:|:---:|:---|
| JAX baseline (pure JAX) | GH200 | Santis | ~51 | Dec 2025 | ~186 kernels for precip scan |
| + transposed layout + StableHLO injection | GH200 | Santis | ~35 | early Feb 2026 | Unrolled but coalesced |
| + combined StableHLO (q_t_update + precip) | GH200 | Santis | ~29 | mid Feb 2026 | Single HLO module |
| Combined StableHLO (XLA CUDA, JAX 0.6.2) | GH200 | Santis | 27.09 | Mar 2026 | fast_pow + XLA flags, best on GH200 |
| XLA LoopifyUnrolledSlices (WhileLoop) | GH200 | Santis | 29.89 | Mar 2026 | Correct, with fast_pow StableHLO (host-device sync) |
| XLA LoopifyUnrolledSlices (SerialScan) | GH200 | Santis | — | mid Mar 2026 | Blocked: XLA MLIR lowering requires custom `xla_gpu.loop` ops |
| Combined StableHLO (XLA ROCm, JAX 0.6.0) | MI300A | Beverin | 12.97 | Mar 2026 | Default XLA flags |
| + fast_pow + XLA flag tuning | MI300A | Beverin | **11.15** | Mar 2026 | **Best overall**, within DaCe target range |
| StableHLO while loop (XLA ROCm) | MI300A | Beverin | 25.25 | Mar 2026 | WhileThunk host-device sync, not useful |

### Current Performance — IREE (R2B06)

| Configuration | GPU | Cluster | Time (ms) | Date | Notes |
|:---|:---:|:---:|:---:|:---:|:---|
| IREE CUDA + LoopifyInsertSliceChain (nsys on H100) | H100 | Santis | ~104 | Mar 2026 | Worse than baseline — pass prevents IREE's own fusion |
| IREE HIP baseline (no custom pass) | MI300A | Beverin | ~47 | Mar 2026 | ~186 dispatches for precip scan |
| IREE HIP + LoopifyInsertSliceChain (WIP) | MI300A | Beverin | ~80 | Mar 2026 | Correctness bug, not yet optimized |

### Correctness

- Integration tests pass against reference baseline data
- R2B06 test case passes against baseline

---

## Optimization Track 1: JAX-Level Optimizations (Dec 2025 – Jan 2026)

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
- **Result:** Major kernel count reduction. Now part of the combined StableHLO module (see Track 2). fast_pow saved ~1ms on MI300A.

---

## Optimization Track 2: StableHLO Injection (Feb 2026)

### Custom Primitive + StableHLO Injection

- Defined a JAX custom primitive (`jax.extend.core.Primitive`) with a custom MLIR lowering (`mlir.register_lowering`)
- The lowering loads pre-generated StableHLO and merges it into the JAX compilation module via `mlir.merge_mlir_modules()`
- This allows replacing arbitrary parts of the JAX-traced computation with hand-optimized or script-generated StableHLO, transparently within JAX's normal JIT pipeline
- First applied to precipitation scans: eliminates `while` loops and `dynamic_slice`/`dynamic_update_slice` operations (D2D copies)
- Benefits were not visible until the transpose issue (Track 1) was fixed
- Later extended to a combined module covering both q_t_update and precipitation scans
- **Result:** Precip-only injection brought runtime from ~51ms to ~35ms; combined module brought it to ~29ms

### Bottleneck Analysis

- The precipitation scan over 90 vertical levels remains the bottleneck
- JAX/XLA sees 90 independent slice-compute-concat chains and creates ~186 separate GPU kernels
- Each kernel is tiny; most time spent on kernel launch overhead rather than computation
- Earlier nsys profiling on GH200 at the 35ms stage showed: q_t_update ~10.5ms, precipitation scan ~11.3ms, overhead ~13ms (later identified as concat/slice + transposes + kernel launch overhead via MI300A rocprofv3, see below)

### Profiling (rocprofv3, MI300A, Mar 2026)

Per-invocation breakdown (applied to 12.97ms baseline, before fast_pow fix):

| Category | % of total | Est. per run (ms) | Status |
|:---|:---:|:---:|:---|
| Precip scan body computation | 20.2% | 2.6 | |
| Concatenation (4 kernels) | 22.8% | 3.0 | |
| q_t_update (power ops) | 8.8% | 1.1 | Fixed: replaced with exp/log (fast_pow) |
| Per-level compute kernels | ~22% | 2.9 | |
| Dynamic slicing | 7.1% | 0.9 | |
| Transposes | ~11% | 1.4 | |
| Copies + broadcasts + other | ~8% | 1.0 | |

Key finding: concatenation (23%) + dynamic slicing (7%) = 30% of runtime is pure overhead from the unrolled scan pattern. The power ops (9%) were fixed by the fast_pow change, contributing to the 12.97ms → 11.15ms improvement.

### XLA Flag Tuning (MI300A, JAX 0.6.0)

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
- Conclusion: while loops at the StableHLO level do not help

---

## Optimization Track 3: XLA Compiler Pass (LoopifyUnrolledSlices) (Mar 2026)

### Goal

Detect the unrolled 90-level slice-compute-concat pattern in XLA HLO and re-roll it into a loop or single-kernel serial scan. Eliminates ~186 separate GPU kernel launches by fusing the entire precipitation scan into 1-2 kernels.

### Approach

- Custom XLA compiler pass: `LoopifyUnrolledSlices`
- Built JAX and XLA from source to integrate the pass
- Attempted StableHLO-level re-rolling first, but XLA overrides things during lowering, making it slower
- Two modes:
  - `kWhileLoop` — emits an HLO while loop; XLA fuses the body into ~6 kernels
  - `kSerialScan` — emits a custom fusion intended to produce a single GPU kernel with `scf.forall` (parallel over cells) + `scf.for` (serial over levels). Currently blocked by XLA's MLIR lowering pipeline (see below).

### Algorithm

1. **Chain detection**: Find sequences of `slice → compute → concat` at consecutive offsets along the vertical dimension
2. **Chain grouping**: Group all 11 output concatenates sharing the same 90-level computation into a single loop with 11 accumulators
3. **Slice-dependency propagation**: Forward-propagate from k=0 level slices through iter0 body to identify which ops depend on the level index vs. shared precomputation
4. **Carry detection**: Walk (iter2, iter1) structurally to find inter-iteration dependencies (19 carry pairs including shifted carries for previous-level values)
5. **Body construction**: Use iter1 as the loop body template; iter0 output is used as carry initial values
6. **Scalarization (SerialScan mode)**: Convert tensor-level ops to scalar arithmetic inside the kernel; each workgroup processes one cell, looping over 90 levels

### Status

**kWhileLoop mode — working, correct, but not useful:**
- Pass compiles and is integrated into JAX (built from source within modified XLA)
- All 10 output concatenates correctly grouped into a single while loop
- Correctness verified: all output fields match reference data (max diff 2.17e-06)
- Result: **29.89ms** on GH200 (with fast_pow StableHLO) — slower than 27ms baseline due to WhileThunk host-device synchronization per iteration (90 round-trips). This overhead is fundamental to XLA's while-loop execution model.
- Benchmark progression (all on NVIDIA GH200, R2B06):
  - 29.63ms — baseline (unrolled, with StableHLO injection + fused q_t_update)
  - 42.96ms — first attempt (only 1 of 11 chains loopified)
  - 32.99ms — after fixing chain grouping (all chains, while-loop mode)
  - 29.89ms — with fast_pow StableHLO
- See [INTEGRATION.md](../xla_passes/INTEGRATION.md) for build/integration instructions

**kSerialScan mode — blocked by XLA MLIR lowering:**
- Pass side works: creates a kCustom fusion with `__serial_scan` kind, inline DCE cleans up dead ops (18420 → 1068 instructions), body ops serialized into metadata string
- Emitter generates MLIR from serialized body ops (~20 HLO opcodes supported)
- **Blocked:** XLA's GPU MLIR lowering pipeline does not support standard `scf.for`/`tensor.extract`/`tensor.insert`. XLA emitters use custom `EmitXlaLoopOp` → `xla_gpu.loop` ops with special bufferization support. A sequential scan (level k depends on level k-1) does not fit the per-element `EmitXlaLoopOp` model.
- Unblocking requires either: (a) writing a custom `xla_gpu` dialect op for sequential scans, or (b) finding a way to express carries within `EmitXlaLoopOp`

---

## Optimization Track 4: IREE (Mar 2026)

### IREE CUDA Backend

- Patched an IREE CUDA bug to run the full JAX graupel pipeline
- Due to limitations JIT-compiling large functions (CUDA memory errors), had to split into separately JIT-compiled functions (precip and temp) rather than a single fused JIT function
- This adds overhead; result is slightly slower than XLA
- **Status:** functional but slower; further investigation needed. CUDA backend receives limited upstream attention.

### IREE HIP/ROCm Backend (AMD MI300A, Beverin)

- Successfully running IREE on Beverin cluster (AMD MI300A, gfx942)
- Baseline runtime (no custom pass): **47ms** for 1 iteration
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
| Combined StableHLO (q_t_update + precip) | GH200 | 35ms → 29ms | **Adopted** |
| Combined StableHLO (XLA ROCm, JAX 0.6.0) | MI300A | 12.97ms (default flags) | **Adopted** |
| + fast_pow + XLA flag tuning | MI300A | **11.15ms**, best overall | **Adopted** |
| StableHLO while loop | MI300A | 25.25ms (host-device sync) | Not useful |
| XLA LoopifyUnrolledSlices (WhileLoop) | GH200 | 29.89ms, correct, host-device sync overhead | Working (not useful — slower than baseline) |
| XLA LoopifyUnrolledSlices (SerialScan) | GH200 | Blocked by XLA MLIR lowering | Blocked |
| IREE CUDA | GH200 | Functional but slower | Low priority |
| IREE HIP baseline | MI300A | 47ms | Baseline established |
| IREE preprocessing pass | MI300A | Correct structure, temperature bug (80ms) | In progress |
| Pure MLIR | — | Exploratory | Future idea |

---

## Next Steps

1. **Push below 11ms:** profile the 11.15ms result to find remaining overhead; look for further fusion opportunities
2. **XLA pass on MI300A (ROCm):** build JAX/XLA from source on Beverin to test LoopifyUnrolledSlices on MI300A — WhileLoop mode may benefit more from MI300A's unified memory and lower host-device sync latency
3. **JAX version regression:** investigate why JAX 0.9.2 regresses to 23.32ms (vs 11.15ms on JAX 0.6.0)
4. **XLA SerialScan emitter:** only path to eliminate concat/slice overhead (~3ms); requires `xla_gpu` dialect work
5. **IREE preprocessing pass:** fix iter1 boundary construction (port `depends_on_slice` from XLA pass) to achieve correctness on AMD MI300A
6. **Transpose elimination:** remove pre-transpose step entirely (~19ms overhead at runtime)

---

## Future Ideas

- **Pure MLIR rewrite:** write graupel code directly with MLIR core dialects (no stencils), bypassing JAX entirely. Exploratory; not currently active.

---

*This logbook was restructured and maintained with the help of Claude (Anthropic).*
