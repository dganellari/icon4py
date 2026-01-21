# MLIR Transformation Passes

Optimization passes for JAX-generated StableHLO MLIR.

## Overview

These scripts analyze and optimize MLIR generated from JAX code. They identify common performance issues and suggest/apply transformations.

## Usage

### 1. Export to MLIR
```bash
cd model/atmosphere/subgrid_scale_physics/muphys_jax
python export_to_mlir.py
```

### 2. Analyze MLIR Structure
```bash
python mlir_passes/analyze_mlir.py mlir_output/graupel_stablehlo.mlir
```

### 3. Analyze Power Operations
```bash
python mlir_passes/power_to_multiply.py mlir_output/graupel_stablehlo.mlir
```

### 4. Analyze Scan Patterns
```bash
python mlir_passes/analyze_scans.py mlir_output/graupel_stablehlo.mlir
```

## Passes

### `analyze_mlir.py`
General MLIR analysis:
- Operation counts
- Tensor types (precision)
- Function definitions
- Memory operations
- Identifies optimization opportunities

### `power_to_multiply.py`
Power operation optimization:
- Identifies `x^2`, `x^3`, `x^0.5`, etc.
- Suggests replacements with multiply/sqrt
- Estimates performance impact
- 2-3x speedup for power operations

### `analyze_scans.py`
Vertical scan analysis:
- Identifies scan structure (90-level loops)
- Analyzes carry state complexity
- Checks fusion opportunities
- Memory access patterns
- Suggests custom kernel optimizations

## Optimization Strategies

### 1. Power → Multiply
```mlir
// Before
%y = stablehlo.power %x, %c2 : tensor<1024x90xf64>

// After
%y = stablehlo.multiply %x, %x : tensor<1024x90xf64>
```
**Impact:** 2-3x speedup for power ops (37 occurrences)

### 2. Scan Fusion
```python
# Before: 2 separate scans (180 kernel launches)
scan1 = jax.lax.scan(step1, init1, xs)
scan2 = jax.lax.scan(step2, init2, ys)

# After: 1 fused scan (90 kernel launches)
scan_fused = jax.lax.scan(step_combined, (init1, init2), (xs, ys))
```
**Impact:** 50% reduction in kernel launches

### 3. Custom Column Kernel
Replace JAX scan with hand-written GPU kernel:
- Process entire vertical column in one thread block
- Shared memory for intermediate values
- Reduced memory traffic

**Impact:** 2-3x overall speedup

### 4. Mixed Precision
```python
# Before: all f64
x = jnp.float64(...)

# After: f32 compute, f64 accumulate
x = jnp.float32(...)
result = jnp.float64(result)
```
**Impact:** 2x memory bandwidth, 1.5x compute

## Key Findings

From `graupel_stablehlo.mlir`:

| Metric | Value | Optimization |
|--------|-------|--------------|
| Total lines | ~12,000 | - |
| While loops (scans) | 2 | Fuse into 1 |
| Power operations | 37 | Replace 20-30 with multiply |
| Broadcasts | 375 | Hoist out of loops |
| Precision | f64 | Consider mixed precision |

## Next Steps

1. **Immediate (JAX source level):**
   - Replace `x**2` with `x*x` in Python
   - Fuse the two scans in graupel
   - Experiment with f32

2. **Medium term (MLIR level):**
   - Implement automatic power-to-multiply pass
   - Implement broadcast hoisting pass
   - Test with MLIR optimization pipeline

3. **Long term (Custom kernels):**
   - Write CUDA kernel for vertical column
   - Integrate with JAX via custom call
   - Profile and optimize memory access

## Performance Targets

| Optimization | Estimated Speedup | Effort |
|--------------|-------------------|--------|
| Power → Multiply | 1.2-1.3x | Low (source change) |
| Scan fusion | 1.3-1.5x | Medium (refactor) |
| Mixed precision | 1.5-2x | Medium (validation) |
| Custom kernel | 2-3x | High (CUDA) |
| **Combined** | **5-10x** | - |

## References

- [StableHLO Spec](https://github.com/openxla/stablehlo/blob/main/docs/spec.md)
- [JAX MLIR Lowering](https://jax.readthedocs.io/en/latest/jep/9407-jax-mlir.html)
- [XLA Optimizations](https://www.tensorflow.org/xla/operation_semantics)
