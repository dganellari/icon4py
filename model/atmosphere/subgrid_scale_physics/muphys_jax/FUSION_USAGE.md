# How to Use Scan Fusion

## Overview

Scan fusion combines the two sequential scans (precipitation + temperature) into a single scan, reducing kernel launches from **180 to 90** (~1.3-1.4x expected speedup).

## Quick Start

### 1. Test Correctness & Benchmark

```bash
cd /Users/gandanie/scratch/icon/icon4py/model/atmosphere/subgrid_scale_physics/muphys_jax
python test_scan_fusion.py
```

This will:
- Run both unfused (baseline) and fused versions
- Compare outputs for correctness
- Benchmark performance of both versions

### 2. Use in Your Code

The fusion is controlled by the `use_fused_scans` parameter:

```python
from muphys_jax.implementations.graupel import graupel_run
from muphys_jax.core.definitions import Q

# ... create your inputs ...

# Without fusion (baseline, 180 kernel launches)
result = graupel_run(dz, te, p, rho, q_in, dt, qnc, use_fused_scans=False)

# With fusion (optimized, 90 kernel launches)
result = graupel_run(dz, te, p, rho, q_in, dt, qnc, use_fused_scans=True)
```

### 3. Export MLIR to Verify

```bash
# Export unfused version
python export_to_mlir.py

# Edit export_to_mlir.py to add use_fused_scans=True to graupel_run call
# Then export fused version
python export_to_mlir.py
```

Check the MLIR files in `mlir_output/` - you should see:
- **Unfused**: 2 `stablehlo.while` operations (2 scans)
- **Fused**: 1 `stablehlo.while` operation (single scan)

### 4. Modify Existing Scripts

If you have existing code using `graupel_run()`, just add the parameter:

**Before:**
```python
result = graupel_run(dz, te, p, rho, q_in, dt, qnc)
```

**After (with fusion):**
```python
result = graupel_run(dz, te, p, rho, q_in, dt, qnc, use_fused_scans=True)
```

## Performance Expectations

| Configuration | Kernel Launches | Expected Runtime |
|--------------|----------------|------------------|
| **Unfused (baseline)** | 180 (2×90 levels) | ~53-54ms |
| **Fused (optimized)** | 90 (1×90 levels) | ~38-40ms (~1.3-1.4x faster) |

## Implementation Details

- **File**: [implementations/graupel.py](implementations/graupel.py)
- **Function**: `precipitation_effects()`
  - When `fused=False`: Uses 2 separate scans (original)
  - When `fused=True`: Uses `precipitation_effects_fused()` (single scan)
- **Strategy**: Processes precipitation for all 4 species first, then immediately uses results for temperature update within same scan iteration

## Troubleshooting

### Import errors
Make sure you're running from the correct directory and Python path is set:
```bash
cd /path/to/muphys_jax
PYTHONPATH=..:$PYTHONPATH python test_scan_fusion.py
```

### Numerical differences
Small numerical differences (< 1e-12) are expected due to floating point rounding. The correctness test checks for close equality with appropriate tolerances.

### Performance not improving
- Make sure GPU is being used: `jax.devices()` should show CudaDevice
- Check that JIT compilation is working: First run should be slower (compilation)
- Verify kernel count reduction: Export MLIR and count `stablehlo.while` ops
