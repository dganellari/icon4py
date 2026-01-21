# Graupel MLIR Optimization Strategy

Based on MLIR analysis of the JAX-generated StableHLO code.

## Problem Structure

The graupel microphysics has **2 sequential vertical scans**:

### Scan #1: Precipitation Sedimentation (`precip_scan_batched`)
- **90 iterations** (vertical levels)
- **16 carry state elements**
- **4 species** processed in batches: rain, snow, ice, graupel
- **Outputs**: Updated species concentrations (qr, qs, qi, qg) and precipitation fluxes

### Scan #2: Temperature Update (`temperature_update_scan`)  
- **90 iterations** (vertical levels)
- **17 carry state elements**
- **Depends on**: Scan #1 outputs (qr, qs from `%1092#14`, `%1092#15`)
- **Outputs**: Updated temperature field

**Key Constraint:** Scan #2 depends on Scan #1 → Cannot trivially fuse without restructuring computation

## Optimization Strategies

### 1. Power-to-Multiply Transform ⚡ [IMMEDIATE - LOW EFFORT]

**Target:** 37 power operations with constant exponents

**Implementation:**
```python
# Before (in JAX source)
x_squared = x**2
x_cubed = x**3
sqrt_x = x**0.5

# After
x_squared = x * x
x_cubed = x * x * x
sqrt_x = jnp.sqrt(x)
```

**Expected Impact:**
- Power operations: 2-3x faster
- Overall speedup: ~1.2-1.3x (power ops are ~5-10% of compute)
- **Effort:** 1-2 hours (find and replace in source)
- **Risk:** Low (mathematically equivalent)

**Action Items:**
```bash
# Find all power operations in source
cd implementations
grep -r "\*\*[0-9]" *.py
grep -r "jnp.power" *.py

# Replace systematically
# x**2 → x*x
# x**0.5 → jnp.sqrt(x)
# x**(-1) → 1.0/x
```

---

### 2. Scan Dependency Analysis 🔍 [SHORT TERM - MEDIUM EFFORT]

**Problem:** Temperature scan depends on precipitation outputs

**Current Flow:**
```python
# Scan 1: Precipitation (independent for 4 species)
qr, qs, qi, qg, pr, ps, pi, pg = precip_scan_batched(...)

# Scan 2: Temperature (depends on qr, qs, etc.)
t_new = temperature_update_scan(qr, qs, qi, qg, pr, ps, ...)
```

**Option A: Fused Single Scan** 
Combine both scans into one 90-iteration loop:

```python
def fused_scan_step(carry, inputs):
    # Step 1: Precipitation for this level
    qr, qs, qi, qg, fluxes = precip_step(carry_precip, inputs_precip)
    
    # Step 2: Temperature using fresh precipitation
    t_new = temp_step(carry_temp, qr, qs, qi, qg, fluxes)
    
    return (carry_precip_new, carry_temp_new), (qr, qs, qi, qg, t_new)

# Single scan instead of two
results = lax.scan(fused_scan_step, init, inputs)
```

**Expected Impact:**
- Kernel launches: 180 → 90 (50% reduction)
- Overhead savings: ~1.3-1.5x
- **Effort:** 2-3 days (refactor scan structure)
- **Risk:** Medium (complex state management)

**Option B: Overlap with Compute**
Keep scans separate but pipeline them:

```python
# Start temperature scan as soon as first precip levels are ready
# This requires XLA async execution or manual pipelining
```

**Expected Impact:** 1.1-1.2x (limited by dependencies)

---

### 3. Memory Layout Optimization 💾 [MEDIUM TERM - MEDIUM EFFORT]

**Problem:** Each scan iteration does 8 `dynamic_slice` operations

**Current Pattern:**
```python
# Input shape: (ncells=1024, nlev=90, nspecies=4)
for k in range(90):
    slice_k = input[:, k, :]  # dynamic_slice
    result_k = process(slice_k)
    output[:, k, :] = result_k  # dynamic_update_slice
```

**Optimized Pattern:**
```python
# Transpose to (nlev=90, ncells=1024, nspecies=4)
input_transposed = jnp.transpose(input, (1, 0, 2))

for k in range(90):
    slice_k = input_transposed[k]  # static slice (much faster)
    result_k = process(slice_k)
    output_transposed[k] = result_k

output = jnp.transpose(output_transposed, (1, 0, 2))
```

**Expected Impact:**
- Memory operations: 30-40% faster (static vs dynamic slicing)
- Overall speedup: ~1.2-1.3x
- **Effort:** 1 day (change input layout + transpose overhead)
- **Risk:** Low (verify transpose is hoisted outside scan)

---

### 4. Custom CUDA Kernel 🚀 [LONG TERM - HIGH EFFORT]

**Problem:** JAX `lax.scan` is general-purpose, not optimized for vertical columns

**Approach:** Write specialized GPU kernel

**Kernel Design:**
```cuda
__global__ void vertical_column_kernel(
    float64* inputs,   // (ncells, nlev, nspecies)
    float64* outputs,  // (ncells, nlev, nspecies)
    int ncells,
    int nlev
) {
    // Each thread block processes one vertical column
    int col = blockIdx.x;
    
    // Shared memory for column data
    __shared__ float64 column[90][4];
    
    // Load column into shared memory
    for (int k = threadIdx.x; k < nlev; k += blockDim.x) {
        for (int s = 0; s < 4; s++) {
            column[k][s] = inputs[col * nlev * 4 + k * 4 + s];
        }
    }
    __syncthreads();
    
    // Process column level-by-level
    if (threadIdx.x == 0) {
        for (int k = 0; k < nlev; k++) {
            // Inline the physics computation
            column[k][0] = physics_step(column, k);
        }
    }
    __syncthreads();
    
    // Write back
    for (int k = threadIdx.x; k < nlev; k += blockDim.x) {
        for (int s = 0; s < 4; s++) {
            outputs[col * nlev * 4 + k * 4 + s] = column[k][s];
        }
    }
}
```

**Integration with JAX:**
```python
from jax.experimental import jax2cuda

# Register custom CUDA kernel
@jax2cuda.custom_call
def vertical_column_cuda(inputs):
    # Calls the CUDA kernel above
    return outputs
```

**Expected Impact:**
- Eliminate kernel launch overhead (1 launch vs 180)
- Optimize memory access (shared memory)
- Better instruction-level parallelism
- Overall speedup: **2-3x**
- **Effort:** 2-3 weeks (CUDA + JAX integration + validation)
- **Risk:** High (correctness, portability)

---

### 5. Mixed Precision 🎯 [MEDIUM TERM - MEDIUM EFFORT]

**Problem:** All operations use f64 (double precision)

**Approach:** Use f32 for compute, f64 for accumulation

**Implementation:**
```python
# Input arrays in f32
dz_f32 = dz.astype(jnp.float32)
te_f32 = te.astype(jnp.float32)
q_f32 = q.astype(jnp.float32)

# Compute in f32
result_f32 = graupel_f32(dz_f32, te_f32, q_f32, ...)

# Accumulate/output in f64
result_f64 = result_f32.astype(jnp.float64)
```

**Validation Required:**
- Physics accuracy comparison
- Energy conservation check
- Numerical stability analysis

**Expected Impact:**
- Memory bandwidth: 2x faster
- Compute throughput: ~1.5x faster (on modern GPUs)
- Overall speedup: **1.5-2x**
- **Effort:** 1 week (implement + validate physics)
- **Risk:** Medium (need to verify physics accuracy)

---

### 6. Broadcast Hoisting 📊 [SHORT TERM - LOW EFFORT]

**Problem:** 375 broadcast operations, many inside scans

**Current:**
```python
def scan_step(carry, x):
    const = jnp.broadcast_to(0.5, x.shape)  # Inside loop!
    return carry, x * const
```

**Optimized:**
```python
# Hoist broadcasts outside scan
const_broadcasted = jnp.broadcast_to(0.5, (nlev, ncells))

def scan_step(carry, x):
    return carry, x * carry.const  # Use pre-broadcasted
```

**Expected Impact:**
- Reduce redundant work in scan
- Overall speedup: ~1.1-1.2x
- **Effort:** 1-2 days (refactor broadcasts)
- **Risk:** Low

---

## Implementation Roadmap

### Phase 1: Quick Wins (1 week) - **Target: 1.5x**
1. ✅ Power-to-multiply transforms
2. ✅ Broadcast hoisting  
3. Run benchmarks

### Phase 2: Scan Optimization (2-3 weeks) - **Target: 2x**
4. Fuse precipitation + temperature scans
5. Memory layout optimization
6. Run benchmarks

### Phase 3: Advanced (1-2 months) - **Target: 5-10x**
7. Mixed precision implementation + validation
8. Custom CUDA kernel (if needed)
9. Profile and fine-tune

---

## Benchmark Protocol

Before/after each optimization:

```python
import time
import jax

# Warmup
for _ in range(10):
    _ = graupel_run(dz, te, p, rho, q, dt, qnc)

# Benchmark
times = []
for _ in range(100):
    start = time.perf_counter()
    result = graupel_run(dz, te, p, rho, q, dt, qnc)
    result.block_until_ready()  # Wait for GPU
    times.append(time.perf_counter() - start)

print(f"Mean: {np.mean(times)*1000:.2f} ms")
print(f"Std:  {np.std(times)*1000:.2f} ms")
```

**Validation:**
```python
# Compare against reference
diff = jnp.abs(result_optimized - result_reference)
max_error = jnp.max(diff)
rel_error = max_error / jnp.max(jnp.abs(result_reference))
print(f"Max relative error: {rel_error:.2e}")
assert rel_error < 1e-10  # For f64
```

---

## Risk Mitigation

1. **Correctness:** Always validate against reference implementation
2. **Performance:** Benchmark each change in isolation
3. **Portability:** Test on different GPU architectures
4. **Maintainability:** Document all optimizations clearly

---

## Expected Overall Speedup

| Optimization | Speedup | Cumulative |
|-------------|---------|------------|
| Power-to-multiply | 1.2x | 1.2x |
| Broadcast hoisting | 1.1x | 1.32x |
| Scan fusion | 1.3x | 1.72x |
| Memory layout | 1.2x | 2.06x |
| Mixed precision | 1.5x | 3.09x |
| Custom kernel | 1.5x | **4.6x** |

**Conservative estimate: 3-5x**  
**Optimistic estimate: 5-10x**

---

## Next Steps

1. **Run power analysis:**
   ```bash
   python mlir_passes/power_to_multiply.py mlir_output/graupel_stablehlo.mlir
   ```

2. **Implement Phase 1 optimizations** (power + broadcast)

3. **Benchmark** and validate

4. **Proceed to Phase 2** based on results
