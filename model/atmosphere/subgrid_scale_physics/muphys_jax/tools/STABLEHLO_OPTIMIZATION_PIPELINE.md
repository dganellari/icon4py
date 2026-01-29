# StableHLO Optimization Pipeline for Graupel Precipitation Effects

This document describes the complete pipeline for optimizing the graupel precipitation effects kernel using StableHLO loop unrolling.

## Overview

The optimization eliminates overhead from JAX's `lax.scan` by unrolling the vertical level loop at the IR level. This removes:
- While loop control overhead
- `dynamic_slice` / `dynamic_update_slice` operations
- Tuple packing/unpacking between iterations

**Measured Speedup**: ~1.60x faster on real data (327680 cells x 90 levels)

## Prerequisites

- JAX with GPU support
- NetCDF4 for reading input data
- Real atmospheric data in NetCDF format

## Pipeline Steps

### Step 1: Export Baseline StableHLO

Lower the baseline precipitation effects function to StableHLO:

```bash
cd model/atmosphere/subgrid_scale_physics/muphys_jax

python tools/export_stablehlo.py \
    --input /path/to/data.nc \
    --output shlo/precip_effect_x64_lowered.stablehlo
```

This generates the baseline IR with `lax.scan` compiled to a while loop.

### Step 2: Generate Unrolled StableHLO

Generate the unrolled version with the loop explicitly expanded:

```bash
python tools/generate_unrolled_stablehlo.py \
    --input /path/to/data.nc
```

Output filename is auto-generated based on dimensions, e.g.:
- `shlo/precip_effect_x64_unrolled_327680x90.stablehlo`

### Step 3: Benchmark Comparison

Compare execution times between baseline and unrolled versions:

```bash
python tools/benchmark_stablehlo.py \
    shlo/precip_effect_x64_unrolled_327680x90.stablehlo \
    --compare shlo/precip_effect_x64_lowered.stablehlo \
    --input /path/to/data.nc \
    --num-runs 10
```

Expected output:
```
COMPARISON SUMMARY
======================================================================
precip_effect_x64_unrolled_327680x90       19.24 ms  (baseline)
precip_effect_x64_lowered                  24.09 ms  1.25x slower
```

### Step 4: Run Full Graupel with Optimized HLO

Benchmark the full graupel implementation with HLO injection:

```bash
python tools/run_graupel_optimized.py \
    --compare-hlo \
        shlo/precip_effect_x64_unrolled_327680x90.stablehlo \
        shlo/precip_effect_x64_lowered.stablehlo \
    --input /path/to/data.nc \
    --num-runs 10
```

### Step 5: Validate Correctness

Run the test suite to ensure the optimized pipeline produces correct results:

```bash
cd model/atmosphere/subgrid_scale_physics/muphys_jax

# Run optimization tests
pytest muphys_jax/tests/test_graupel_only.py -v -k "optimized"

# Run full test suite
pytest muphys_jax/tests/test_graupel_only.py -v
```

## File Structure

```
muphys_jax/
├── tools/
│   ├── export_stablehlo.py          # Step 1: Lower to baseline StableHLO
│   ├── generate_unrolled_stablehlo.py  # Step 2: Generate unrolled version
│   ├── benchmark_stablehlo.py       # Step 3: Compare execution times
│   ├── run_graupel_optimized.py     # Step 4: Full graupel benchmark
│   └── STABLEHLO_OPTIMIZATION_PIPELINE.md  # This documentation
├── core/
│   └── scans_stablehlo.py           # StableHLO-based scan implementation
└── shlo/                            # Generated StableHLO files
    ├── precip_effect_x64_lowered.stablehlo
    └── precip_effect_x64_unrolled_327680x90.stablehlo
```

## Key Technical Details

### Dimension Baking

StableHLO files have dimensions (ncells, nlev) baked into the IR. You must regenerate the StableHLO when changing grid sizes:

```bash
# For different grid sizes, regenerate:
python tools/generate_unrolled_stablehlo.py --input /path/to/different_grid.nc
```

### Data Types

The pipeline uses `float64` precision. The scripts enable x64 mode automatically:
```python
jax.config.update("jax_enable_x64", True)
```

### Input Format

All tools require real NetCDF input data with these variables:
- `hus` (specific humidity)
- `clw` (cloud liquid water)
- `qr`, `qs`, `cli`, `qg` (hydrometeor mixing ratios)
- `ta` (temperature)
- `rho` (density)
- `zg` (geometric height, for computing dz)

Dimensions: `cell` (or `ncells`) and `height`

## Troubleshooting

### Shape Mismatch Error

```
Executable expected shape f64[20480,90] but got incompatible shape f64[327680,90]
```

**Cause**: StableHLO was generated for a different grid size.
**Fix**: Regenerate with `--input` pointing to your data file.

### Dtype Mismatch Error

```
Executable expected shape f64[...] but got incompatible shape f32[...]
```

**Cause**: x64 mode not enabled.
**Fix**: The scripts should enable this automatically. If not, set:
```bash
export JAX_ENABLE_X64=1
```

### Performance Regression

If unrolled version is slower than baseline:
- Check compilation time (unrolled takes longer to compile)
- Ensure warmup runs complete before benchmarking
- Use `--num-runs 20` for more stable measurements

## Results Summary

| Version | Execution Time | Speedup |
|---------|---------------|---------|
| Baseline (lax.scan) | 425.10 ms | - |
| Unrolled (loop unrolled) | 265.10 ms | 1.60x |

Grid: 327680 cells x 90 levels

Note: Results measured with direct HLO execution (no JAX overhead) using `run_graupel_optimized.py --compare-hlo`.
