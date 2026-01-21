# Next Steps: MLIR-Level Optimization to Match DaCe Performance

**Current State:** JAX graupel at 51ms vs DaCe at 14.6ms (3.5x gap)

**Bottleneck:** 92.1% of runtime is D2D memory copies (810 copies per iteration)

---

## Why MLIR/Custom Kernel is the Right Path

### What We've Tried (JAX-level):
1. ✅ Carry optimization (5→4 elements): **2.5% improvement**
2. ✅ Scan fusion (180→90 kernels): **2.6% improvement**
3. ✅ XLA compiler flags: **No additional improvement** (already optimized)
4. ⚠️ Tiled scans: Implementation has bugs, needs cleanup

### The Fundamental Problem:
**JAX's lax.scan abstraction creates unavoidable D2D copies:**
- Scan carry state: Copied every iteration (360 copies × 90 levels)
- Dynamic slicing: XLA inserts copies for safety (450 copies × 90 levels)
- Conservative memory model: XLA doesn't know iterations are independent

**DaCe doesn't have this problem** because it:
- Generates specialized code for the exact computation
- Keeps carry state in registers/shared memory
- Single kernel per column (no inter-kernel D2D traffic)

---

## The Path Forward: Custom CUDA Kernel

### Phase 1: Export and Analyze MLIR (1 day)

```bash
# Export current optimized version
cd model/atmosphere/subgrid_scale_physics/muphys_jax
JAX_ENABLE_X64=1 python export_to_mlir.py

# Analyze D2D copy patterns
python mlir_passes/d2d_copy_eliminator.py mlir_output/graupel_fused_stablehlo.mlir
```

**Expected findings:**
- ~1440 dynamic_slice operations (16 per level × 90 levels)
- ~1440 dynamic_update_slice operations
- These are the source of D2D copies

### Phase 2: Prototype Custom Kernel (1-2 weeks)

**Strategy:** Write a CUDA kernel that processes entire vertical column in shared memory.

**Key design:**
```cuda
__global__ void vertical_column_kernel(...) {
    // One thread block = one vertical column (or small batch of columns)
    __shared__ double carry_state[4][4];  // 4 species × 4 elements

    // Load column data
    // Process 90 levels sequentially (carry stays in shared memory!)
    // Write results
}
```

**Benefits over JAX scan:**
- Zero D2D copies for carry state (stays in shared memory)
- Coalesced memory access
- Inlined physics (no function call overhead)
- Single kernel launch vs 90

### Phase 3: Integration with JAX (1 week)

```python
from jax.experimental import jax2cuda

# Register custom kernel
xla_client.register_custom_call_target(
    "vertical_column_cuda",
    kernel_ptr,
    platform="CUDA"
)

@jax.jit
def graupel_optimized(inputs):
    # Phase transitions in JAX (easy to maintain)
    q_updated, t_updated = q_t_update(...)

    # Custom kernel for vertical scans (performance critical)
    results = jax2cuda.custom_call(
        "vertical_column_cuda",
        inputs=[q_updated, t_updated, ...],
        ...
    )

    return results
```

---

## Projected Performance

| Optimization | D2D Copies | Runtime | vs DaCe |
|--------------|------------|---------|---------|
| Current (JAX scan) | 810 | 51ms | 3.5x slower |
| Custom kernel | ~50 | 12-15ms | **1.0-1.1x** ✓ |

**Target achieved:** Within 20% of DaCe performance!

---

## Implementation Guide

### Step 1: Simple Prototype (Start Here)

Create `model/atmosphere/subgrid_scale_physics/muphys_jax/cuda_kernels/vertical_scan.cu`:

```cuda
#include <cuda_runtime.h>

// Simple version: One species, one column per thread
__global__ void precip_scan_kernel(
    const double* __restrict__ q_in,
    const double* __restrict__ rho,
    const double* __restrict__ params,
    double* __restrict__ q_out,
    int ncells,
    int nlev
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncells) return;

    // Carry state (stays in registers!)
    double q_prev = 0.0;
    double flx_prev = 0.0;
    double rhox_prev = 0.0;
    bool activated_prev = false;

    // Scan from top to bottom
    for (int k = 0; k < nlev; k++) {
        int idx = col * nlev + k;
        double q = q_in[idx];
        double rho_k = rho[idx];

        // Physics calculation (inlined)
        bool activated = activated_prev | (q > 1e-12);
        double rho_x = q * rho_k;
        double flx_eff = (rho_x / zeta) + 2.0 * flx_prev;
        // ... rest of physics ...

        // Update carry (stays in registers!)
        q_prev = q_out_val;
        flx_prev = flx_out_val;
        rhox_prev = q_out_val * rho_k;
        activated_prev = activated;

        // Write output
        q_out[idx] = q_out_val;
    }
}
```

### Step 2: Compile and Link

```bash
# Compile CUDA kernel
nvcc -c -O3 --compiler-options '-fPIC' \
    -gencode arch=compute_80,code=sm_80 \
    -o vertical_scan.o vertical_scan.cu

# Create shared library
nvcc -shared -o libvertical_scan.so vertical_scan.o

# Register with JAX
export LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH
```

### Step 3: Python Wrapper

```python
import ctypes
from jax import core
from jax.lib import xla_client

# Load library
lib = ctypes.CDLL("./libvertical_scan.so")

# Register custom call
xla_client.register_custom_call_target(
    b"precip_scan_cuda",
    lib.precip_scan_kernel,
    platform="CUDA"
)

# Wrapper function
def precip_scan_cuda(q, rho, params):
    return core.call_p.bind(
        q, rho, params,
        call_jaxpr=...,
        name="precip_scan_cuda"
    )
```

### Step 4: Validation

```python
# Test against JAX reference
q_jax, flx_jax = precip_scan_batched(...)  # JAX version
q_cuda, flx_cuda = precip_scan_cuda(...)    # CUDA version

# Check correctness
assert jnp.allclose(q_cuda, q_jax, atol=1e-12)
assert jnp.allclose(flx_cuda, flx_jax, atol=1e-12)

# Benchmark
%timeit precip_scan_batched(...)  # ~20ms
%timeit precip_scan_cuda(...)     # Target: ~5ms
```

---

## Resources

### Documentation
- [JAX Custom Calls](https://jax.readthedocs.io/en/latest/Custom_Operation_for_GPUs.html)
- [XLA Custom Call API](https://www.tensorflow.org/xla/custom_call)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

### Reference Implementations
- DaCe graupel: `/path/to/dace/graupel` (14.6ms target)
- JAX custom call examples: `jax/examples/custom_vjp.py`

### Tools
- `nvprof / nsys`: Profile CUDA kernels
- `cuda-memcheck`: Verify memory safety
- `compute-sanitizer`: Debug CUDA code

---

## Timeline Estimate

| Task | Duration | Deliverable |
|------|----------|-------------|
| MLIR analysis | 1 day | D2D copy breakdown |
| Simple CUDA kernel | 3 days | Single-species prototype |
| Multi-species kernel | 1 week | Full 4-species version |
| JAX integration | 3 days | Custom call working |
| Validation & testing | 1 week | Correctness verified |
| Performance tuning | 1 week | Match DaCe (12-15ms) |
| **Total** | **3-4 weeks** | **Production-ready** |

---

## Alternative: If Custom CUDA is Too Much

If writing CUDA is not feasible, consider:

1. **Use DaCe itself**: DaCe can compile Python/NumPy to optimized GPU code
2. **Numba CUDA**: Higher-level than raw CUDA, integrates with JAX
3. **CuPy**: GPU array library with custom kernels
4. **Accept 2-3x overhead**: 51ms is still reasonable for many use cases

---

## Decision Point

**Question:** Do you want to proceed with custom CUDA kernel to match DaCe performance?

**If YES:**
- I'll create detailed CUDA kernel implementation
- Set up build system and JAX integration
- Target: 12-15ms (matching DaCe's 14.6ms)

**If NO:**
- Accept 51ms as JAX's practical limit
- Focus on other optimizations (mixed precision, multi-GPU)
- Use DaCe for production if 3.5x performance matters

**Let me know which path you want to take!**
