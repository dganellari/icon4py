# StableHLO Transformation for DaCe-like Performance

## Goal
Transform JAX-generated StableHLO IR to eliminate D2D memory copies and achieve DaCe-like performance (~14.6ms vs current 51ms).

## Problem Analysis

### Current JAX Implementation
- **Runtime**: 51ms
- **Bottleneck**: 92.1% time in D2D memory copies
- **Root cause**: `lax.scan` generates `stablehlo.while` loops with memory-based iteration

### StableHLO IR Pattern (from JAX)
```mlir
%5:6 = stablehlo.while(%iterArg = %arg2, %iterArg_0 = %4, ...) {
  cond {
    %6 = stablehlo.constant dense<90> : tensor<i32>
    %7 = stablehlo.compare LT, %iterArg_0, %6
    stablehlo.return %7
  }
  do {
    // Dynamic slice - D2D READ!
    %13 = stablehlo.dynamic_slice %iterArg, %11, %12, sizes = [1, 1000]

    // Computation
    %18 = stablehlo.add %iterArg_1, %17
    %25 = stablehlo.add %21, %24

    // Dynamic update - D2D WRITE!
    %34 = stablehlo.dynamic_update_slice %iterArg_3, %26, %32, %33
    %43 = stablehlo.dynamic_update_slice %iterArg_4, %35, %41, %42

    stablehlo.return ...
  }
}
```

**Key issues:**
1. `stablehlo.while` with dynamic iteration counter
2. `dynamic_slice` reads from input tensor (D2D copy from HBM)
3. `dynamic_update_slice` writes to output tensors (D2D copy to HBM)
4. Carry state (`iterArg_1`, `iterArg_2`) carried through memory

### Target: DaCe-like Performance
- **Runtime**: 14.6ms (3.5× faster)
- **Strategy**: Keep carry state in registers, single kernel launch
- **Implementation**: Static unrolling with SSA values

## Transformation Strategy

### Phase 1: Unroll While Loop
Replace dynamic while loop with static sequence:

**Before:**
```mlir
stablehlo.while (i=0; i<90; i++) {
  x = dynamic_slice(input, i)
  carry = compute(carry, x)
}
```

**After:**
```mlir
x0 = slice input[0:1]
carry1 = compute(carry0, x0)
x1 = slice input[1:2]
carry2 = compute(carry1, x1)
...
x89 = slice input[89:90]
carry90 = compute(carry89, x89)
```

### Phase 2: Static Slicing
Replace `dynamic_slice` with compile-time `stablehlo.slice`:

**Before:**
```mlir
%13 = stablehlo.dynamic_slice %input, %i, %0, sizes = [1, 1000]
```

**After (iteration k=5):**
```mlir
%slice_5 = stablehlo.slice %input [5:6, 0:1000]
```

### Phase 3: SSA Carry State
Keep carry values as SSA values (registers), not tensor iterArgs (memory):

**Before:**
```mlir
stablehlo.while(%iterArg_1 = %init_a, %iterArg_2 = %init_b) {
  // iterArg_1, iterArg_2 live in memory (D2D copy)
}
```

**After:**
```mlir
%a_0 = %init_a  // Register
%b_0 = %init_b  // Register
// Iteration 0
%a_1 = stablehlo.add %a_0, %x_0
%b_1 = stablehlo.add %b_0, %x_0
// Iteration 1
%a_2 = stablehlo.add %a_1, %x_1
%b_2 = stablehlo.add %b_1, %x_1
...
```

### Phase 4: Delayed Output Writes
Eliminate `dynamic_update_slice` inside loop. Build output tensor once at end:

**Before:**
```mlir
stablehlo.while(..., %iterArg_out = %empty) {
  %new_out = dynamic_update_slice %iterArg_out, %result, %i  // D2D write!
}
```

**After:**
```mlir
// Accumulate results
%result_0 = compute(...)
%result_1 = compute(...)
...
%result_89 = compute(...)

// Single output construction
%out = stablehlo.concatenate [%result_0, %result_1, ..., %result_89], dim=0
```

## Implementation Tools

### 1. `export_stablehlo.py`
Exports StableHLO IR from simple scan test case.

**Usage:**
```bash
python tools/export_stablehlo.py
# Generates: stablehlo_scan_baseline.mlir
```

### 2. `export_graupel_stablehlo.py`
Exports StableHLO IR from full graupel physics.

**Usage:**
```bash
python tools/export_graupel_stablehlo.py
# Generates: stablehlo_graupel_full.mlir
```

### 3. `transform_stablehlo.py`
Transforms StableHLO IR to eliminate D2D copies.

**Usage:**
```bash
python tools/transform_stablehlo.py stablehlo_scan_baseline.mlir stablehlo_unrolled.mlir
```

**Current status:**
- ✅ Parse while loop structure
- ✅ Extract loop bounds
- ✅ Generate unrolled sequence
- ✅ Replace dynamic_slice with static slice
- ✅ Keep carry as SSA values
- ⚠️ TODO: Build output tensors from results
- ⚠️ TODO: Handle multiple scans (90 scans in graupel!)

## Workflow

### Step 1: Export Baseline
```bash
cd model/atmosphere/subgrid_scale_physics/muphys_jax
python tools/export_graupel_stablehlo.py
```

**Expected output:**
- `stablehlo_graupel_full.mlir` (~500KB)
- Analysis showing ~180 while loops (90 scans × 2 fused/unfused)

### Step 2: Transform
```bash
python tools/transform_stablehlo.py stablehlo_graupel_full.mlir stablehlo_graupel_optimized.mlir
```

**Transformations applied:**
1. Unroll all while loops
2. Static slicing
3. SSA carry state
4. Delayed output construction

### Step 3: MLIR Optimization Passes
```bash
mlir-opt stablehlo_graupel_optimized.mlir \
  --canonicalize \
  --cse \
  --symbol-dce \
  -o stablehlo_graupel_opt.mlir
```

### Step 4: Lower to GPU
```bash
mlir-opt stablehlo_graupel_opt.mlir \
  --convert-stablehlo-to-linalg \
  --linalg-fuse-elementwise-ops \
  --convert-linalg-to-gpu \
  --gpu-kernel-outlining \
  -o stablehlo_graupel_gpu.mlir
```

### Step 5: Compile and Benchmark
```bash
# Compile to executable
mlir-opt stablehlo_graupel_gpu.mlir \
  --convert-gpu-to-nvvm \
  --gpu-to-llvm \
  -o stablehlo_graupel_nvvm.mlir

# Compare performance
python -m muphys_jax.driver.run_graupel_jax input.nc 100 30.0 100.0  # Baseline
# Run compiled MLIR version
# Target: <17ms (DaCe-like performance)
```

## Expected Performance Impact

### Before Transformation (JAX baseline)
- Total runtime: 51ms
- D2D copies: 47ms (92.1%)
- Compute: 4ms (7.9%)
- Kernel launches: 180

### After Transformation (Target)
- Total runtime: ~15ms (3.4× speedup)
- D2D copies: ~0ms (eliminated)
- Compute: ~15ms
- Kernel launches: 1 (single fused kernel)

## Alternative Approaches Considered

### 1. MLIR Direct (muphys_mlir)
- **Status**: Implemented
- **Pros**: Full control, custom dialects
- **Cons**: Reimplementing all physics in MLIR

### 2. Triton
- **Status**: Implemented
- **Pros**: High-level GPU kernel language
- **Cons**: Still requires manual kernel writing

### 3. Pallas
- **Status**: Tested
- **Pros**: JAX-native
- **Cons**: Limited to JAX ecosystem

### 4. StableHLO Transformation (THIS APPROACH)
- **Status**: In progress
- **Pros**: Reuses JAX frontend, transforms IR automatically
- **Cons**: Complex IR transformation, must handle all scan patterns
- **Advantage**: Can benefit entire JAX codebase, not just graupel!

## Success Criteria

1. ✅ Extract StableHLO IR from JAX
2. ✅ Parse while loop structure
3. ⚠️ Transform all 180 scan loops
4. ⚠️ Compile to GPU
5. ⚠️ Measure performance: target <17ms
6. ⚠️ Verify correctness: match JAX output

## Future Work

### Generalize Transformation
- Handle arbitrary scan patterns
- Support reverse scans
- Support nested scans

### Integration with JAX
- Package as JAX transformation pass
- Integrate with XLA pipeline
- Upstream to JAX/IREE?

### Extend to Other Operations
- Reduce/scan fusion
- Cross-scan optimization
- Memory layout optimization

## References

- JAX StableHLO: https://github.com/openxla/stablehlo
- MLIR Dialects: https://mlir.llvm.org/docs/Dialects/
- IREE Compiler: https://iree.dev/
- DaCe Performance Analysis: (internal benchmark data)
