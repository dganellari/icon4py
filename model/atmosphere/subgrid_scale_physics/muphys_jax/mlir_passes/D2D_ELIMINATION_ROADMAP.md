# D2D Copy Elimination Roadmap

**Goal:** Reduce JAX graupel from 51ms to ~15-20ms (approaching DaCe's 14.6ms)

**Current Bottleneck:** 92.1% of runtime is D2D memory copies (810 copies per iteration)

---

## Phase 1: MLIR Analysis & Standard Passes (1 week)

### 1.1 Export and Analyze Current MLIR
```bash
# Export fused graupel to MLIR
python model/atmosphere/subgrid_scale_physics/muphys_jax/export_to_mlir.py

# Analyze D2D copy patterns
python model/atmosphere/subgrid_scale_physics/muphys_jax/mlir_passes/d2d_copy_eliminator.py \
    mlir_output/graupel_fused_stablehlo.mlir
```

**Expected findings:**
- 8 dynamic_slice ops per scan iteration (16 total × 90 levels = 1440 slices)
- 8 dynamic_update_slice ops per scan iteration (1440 updates)
- These are the primary source of D2D copies

### 1.2 Apply Standard XLA/MLIR Optimization Passes

XLA already runs these, but verify they're being applied:

```bash
# Apply standard passes using mlir-opt
mlir-opt graupel_fused_stablehlo.mlir \
    --canonicalize \
    --cse \
    --inline \
    --loop-invariant-code-motion \
    -o graupel_optimized.mlir
```

**Expected impact:** 5-10% improvement (minor)

These passes help but won't eliminate the fundamental scan overhead.

---

## Phase 2: Custom MLIR Transformations (2-3 weeks)

### 2.1 Buffer Forwarding Pass

**Goal:** Eliminate redundant D2D copies from slice-update pairs

**Implementation:**
1. Detect pattern: `dynamic_slice → compute → dynamic_update_slice`
2. Transform to in-place update using buffer aliasing
3. Use XLA buffer donation hints

**MLIR transformation:**
```mlir
// Before (causes 2 D2D copies):
%slice = stablehlo.dynamic_slice %input, %idx
%result = compute(%slice)
%output = stablehlo.dynamic_update_slice %input, %result, %idx

// After (uses buffer aliasing, 0 D2D copies):
%buffer = stablehlo.optimization_barrier %input  // Prevent reordering
%slice_view = "stablehlo.get_slice_view"(%buffer, %idx)  // View, not copy
compute_inplace(%slice_view)
```

**Challenge:** StableHLO doesn't have native in-place operations
**Solution:** Use XLA custom calls with buffer aliasing

**Expected impact:** Eliminate ~200 D2D copies → save ~12ms

### 2.2 Scan Unrolling and Loop Fusion

**Goal:** Help compiler optimize scan loop

**Implementation:**
```mlir
// Before (while loop, 90 iterations):
stablehlo.while(%carry) {
    %cond = ...
    scf.condition %cond, %new_carry
} do {
    // Scan body
}

// After (for loop with unrolling):
scf.for %k = 0 to 90 step 2 iter_args(%carry) {
    // Unroll factor 2
    %carry1 = scan_step(%carry, %inputs[k])
    %carry2 = scan_step(%carry1, %inputs[k+1])
    scf.yield %carry2
}
```

**Benefits:**
- Compiler knows iteration count (90)
- Can apply more aggressive optimizations
- Unrolling exposes ILP

**Expected impact:** 5-8% improvement

### 2.3 Tiled Scan Execution

**Goal:** Process cells in smaller batches that fit in shared memory

**Implementation:**
```python
# Transform at JAX source level:

def tiled_scan(carry, inputs, tile_size=256):
    """Process scan in tiles that fit in shared memory."""
    ncells = carry[0].shape[0]
    n_tiles = ncells // tile_size

    results = []
    for tile_idx in range(n_tiles):
        # Extract tile from carry
        tile_carry = tree_map(
            lambda x: x[tile_idx*tile_size:(tile_idx+1)*tile_size],
            carry
        )

        # Extract tile from inputs
        tile_inputs = tree_map(
            lambda x: x[:, tile_idx*tile_size:(tile_idx+1)*tile_size],
            inputs
        )

        # Run scan on tile (fits in shared memory)
        _, tile_result = lax.scan(scan_step, tile_carry, tile_inputs)

        results.append(tile_result)

    return tree_map(lambda *xs: jnp.concatenate(xs, axis=1), *results)
```

**Benefits:**
- Carry state per tile: 256 cells × 4 elements × 8 bytes = 8KB (fits in L1)
- Reduced D2D traffic (only load/store once per tile)
- Better cache locality

**Expected impact:** 10-15% improvement

---

## Phase 3: Custom CUDA Kernel (4-6 weeks)

### 3.1 Design Custom Vertical Column Kernel

**Architecture:**
- Each thread block processes one tile of columns (e.g., 256 columns)
- Shared memory holds carry state for the tile
- Single kernel launch per graupel call (vs 90 launches for scan)

**Pseudo-code:**
```cuda
__global__ void vertical_column_kernel(
    const double* __restrict__ q_in,      // [ncells][nlev][species]
    const double* __restrict__ rho,
    const double* __restrict__ params,
    double* __restrict__ q_out,
    double* __restrict__ fluxes,
    int ncells, int nlev
) {
    // Tile of columns processed by this thread block
    const int tile_size = 256;
    const int tile_start = blockIdx.x * tile_size;

    // Shared memory for carry state (256 cells × 4 species × 4 elements)
    __shared__ double carry_q[256][4];
    __shared__ double carry_flx[256][4];
    __shared__ double carry_rhox[256][4];
    __shared__ bool carry_activated[256][4];

    // Each thread processes one column
    const int local_col = threadIdx.x;
    const int global_col = tile_start + local_col;

    if (global_col >= ncells) return;

    // Initialize carry state
    for (int s = 0; s < 4; s++) {
        carry_q[local_col][s] = 0.0;
        carry_flx[local_col][s] = 0.0;
        carry_rhox[local_col][s] = 0.0;
        carry_activated[local_col][s] = false;
    }

    // Scan from top to bottom
    for (int k = 0; k < nlev; k++) {
        // Load inputs for this level (coalesced)
        double q[4], vc[4];
        bool mask[4];
        for (int s = 0; s < 4; s++) {
            q[s] = q_in[global_col * nlev * 4 + k * 4 + s];
            vc[s] = ...;  // Load velocity scale
            mask[s] = ...;  // Load activation mask
        }

        double rho_k = rho[global_col * nlev + k];

        // Process all 4 species (inline, no function call overhead)
        for (int s = 0; s < 4; s++) {
            // Precipitation physics (inlined from scan_step_fast)
            bool activated = carry_activated[local_col][s] | mask[s];
            double rho_x = q[s] * rho_k;
            double flx_eff = (rho_x / zeta) + 2.0 * carry_flx[local_col][s];

            // Fall speed calculation
            double fall_speed = prefactor[s] * pow(rho_x + offset[s], exponent[s]);
            double flx_partial = fmin(rho_x * vc[s] * fall_speed, flx_eff);

            // Terminal velocity
            double vt = activated ?
                vc[s] * prefactor[s] * pow(carry_rhox[local_col][s] + offset[s], exponent[s]) :
                0.0;

            double q_activated = (zeta * (flx_eff - flx_partial)) / ((1.0 + zeta * vt) * rho_k);
            double flx_activated = (q_activated * rho_k * vt + flx_partial) * 0.5;

            // Update carry state (stays in shared memory!)
            double q_out_s = activated ? q_activated : q[s];
            double flx_out_s = activated ? flx_activated : 0.0;

            carry_q[local_col][s] = q_out_s;
            carry_flx[local_col][s] = flx_out_s;
            carry_rhox[local_col][s] = q_out_s * rho_k;
            carry_activated[local_col][s] = activated;

            // Write output for this level (coalesced)
            q_out[global_col * nlev * 4 + k * 4 + s] = q_out_s;
            fluxes[global_col * nlev * 4 + k * 4 + s] = flx_out_s;
        }
    }
}
```

**Key optimizations:**
1. **Shared memory carry state:** No D2D copies between levels (stays in shared memory)
2. **Coalesced memory access:** All threads in warp access consecutive memory
3. **Inlined physics:** No function call overhead
4. **Single kernel launch:** vs 90 launches for JAX scan

### 3.2 Integration with JAX via Custom Call

```python
from jax.experimental import jax2cuda
from jax.lib import xla_client

# Register custom call
xla_client.register_custom_call_target(
    "vertical_column_cuda",
    vertical_column_kernel_ptr,  # Function pointer to CUDA kernel
    platform="CUDA"
)

@jax.jit
def optimized_graupel(dz, te, p, rho, q_in, dt, qnc):
    """Graupel with custom CUDA kernel for vertical scans."""

    # Do phase transitions (keep in JAX)
    q_updated, t_updated = q_t_update(te, p, rho, q_in, dt, qnc)

    # Custom call for precipitation + temperature scans
    q_out, t_out, fluxes = jax2cuda.custom_call(
        "vertical_column_cuda",
        result_shape_dtypes=[
            jax.ShapeDtypeStruct(q_updated.r.shape, jnp.float64),  # q_out
            jax.ShapeDtypeStruct(t_updated.shape, jnp.float64),    # t_out
            jax.ShapeDtypeStruct((ncells, nlev, 4), jnp.float64),  # fluxes
        ],
        inputs=[dz, t_updated, p, rho, q_updated, dt, qnc],
    )

    return q_out, t_out, fluxes
```

### 3.3 Expected Performance

**D2D copy reduction:**
- Current: 810 copies × 90 levels = 48.7ms
- Custom kernel: ~50 copies (only input/output) = ~3ms
- **Reduction: 45ms savings**

**Projected performance:**
- Compute time: ~8ms (similar to DaCe)
- Memory transfers: ~3ms (input/output only)
- **Total: ~11-13ms** (within 20% of DaCe's 14.6ms)

---

## Phase 4: Advanced Optimizations (optional, 2-3 weeks)

### 4.1 Mixed Precision (f32 compute, f64 accumulate)

**Benefits:** 2x memory bandwidth, 1.5x compute throughput
**Effort:** Validation required (energy/mass conservation)
**Impact:** Additional 1.5x speedup → **target 7-9ms**

### 4.2 Multi-GPU Scaling

Split cells across GPUs:
```python
# Partition cells across 4 GPUs
cells_per_gpu = ncells // 4

with jax.experimental.maps.Mesh(devices, ["data"]):
    result = jax.pmap(optimized_graupel)(inputs_sharded)
```

**Impact:** Near-linear scaling (4 GPUs → ~3ms per GPU)

---

## Implementation Timeline

| Phase | Duration | Expected Performance | Cumulative Speedup |
|-------|----------|---------------------|-------------------|
| **Baseline** | - | 51ms | 1.0x |
| Phase 1: MLIR Analysis | 1 week | 48ms | 1.06x |
| Phase 2: Custom MLIR Passes | 3 weeks | 38-42ms | 1.3-1.4x |
| Phase 3: Custom CUDA Kernel | 6 weeks | 11-13ms | **4.0-4.6x** |
| Phase 4: Mixed Precision | 3 weeks | 7-9ms | **5.7-7.3x** |

**Final target:** 7-13ms (DaCe is 14.6ms) ✓

---

## Risk Mitigation

### Risk 1: Custom kernel complexity
**Mitigation:**
- Start with simple single-species kernel
- Validate against JAX reference at each step
- Use Nsight Compute to verify D2D reduction

### Risk 2: Physics accuracy with mixed precision
**Mitigation:**
- Thorough validation with long-term climate runs
- May need selective f64 for critical operations
- Compare against established references

### Risk 3: Maintenance burden
**Mitigation:**
- Keep custom kernel minimal (only scan loop)
- Rest of physics stays in JAX (easier to maintain)
- Comprehensive test suite

---

## Success Criteria

1. **Performance:** Achieve 11-15ms per iteration (within 20% of DaCe)
2. **Correctness:** Bit-exact match with JAX reference (f64) or <1e-6 relative error (f32)
3. **D2D Reduction:** Verify <100 D2D copies vs current 810
4. **Maintainability:** Custom kernel <500 lines, well-documented

---

## Next Steps

1. **This week:**
   - Export latest MLIR
   - Run d2d_copy_eliminator.py analysis
   - Profile D2D copies with Nsight

2. **Next 2 weeks:**
   - Implement tiled scan in JAX
   - Benchmark to verify shared memory benefits
   - Start custom kernel prototype

3. **Month 2:**
   - Complete custom CUDA kernel
   - Integration testing
   - Performance validation

4. **Month 3:**
   - Mixed precision exploration
   - Production hardening
   - Documentation

---

## Resources

- CUDA Kernel Guide: `/model/atmosphere/subgrid_scale_physics/muphys_jax/cuda_kernels/README.md` (to be created)
- Profiling Results: `/model/atmosphere/subgrid_scale_physics/muphys_jax/profiling/` (existing)
- MLIR Passes: `/model/atmosphere/subgrid_scale_physics/muphys_jax/mlir_passes/` (existing)
