# Commands to Run on Server

## Setup (One-time)

```bash
# Navigate to muphys_jax directory
cd /capstor/scratch/cscs/gandanie/icon/icon4py/model/atmosphere/subgrid_scale_physics/muphys_jax

# Make scripts executable
chmod +x tools/run_stablehlo_pipeline.sh
```

## Step 1: Export and Transform StableHLO

### Quick Start (Automated)
```bash
./tools/run_stablehlo_pipeline.sh
```

This generates:
- `stablehlo_scan_baseline.mlir` - Simple test case
- `stablehlo_graupel_full.mlir` - Full physics (500+ KB)
- `stablehlo_scan_unrolled.mlir` - Transformed simple scan
- `stablehlo_graupel_analyzed.mlir` - Analysis of full graupel

### Manual Steps (if you want more control)

```bash
# Export simple scan (default mode)
python tools/export_stablehlo.py --mode simple

# Export simple scan with transpose
python tools/export_stablehlo.py --mode simple-transpose

# Export graupel baseline
python tools/export_stablehlo.py --mode baseline

# Export graupel all-in-one fused
python tools/export_stablehlo.py --mode allinone

# Analyze StableHLO IR
python tools/analyze_stablehlo.py stablehlo_graupel_baseline_lowered.mlir

# Compare two implementations
python tools/analyze_stablehlo.py --compare stablehlo_graupel_baseline_lowered.mlir stablehlo_graupel_allinone_lowered.mlir

# Transform (unroll while loops) - auto-detect loop bound
python tools/transform_stablehlo.py stablehlo_graupel_baseline_lowered.mlir

# Transform with manual loop bound (if auto-detection fails)
python tools/transform_stablehlo.py stablehlo_graupel_baseline_lowered.mlir --loop-bound 10

# Analyze only (no transformation)
python tools/transform_stablehlo.py stablehlo_graupel_baseline_lowered.mlir --analyze-only

# Transform with custom input shape
python tools/transform_stablehlo.py input.mlir output.mlir --input-shape 100,1000 --loop-bound 10
```

## Step 2: Check What Was Generated

```bash
# List generated files
ls -lh stablehlo*.mlir

# Check simple scan analysis
head -50 stablehlo_scan_baseline.mlir

# Count while loops in baseline
grep -c "stablehlo.while" stablehlo_scan_baseline.mlir

# Count while loops after transformation (should be 0)
grep -c "stablehlo.while" stablehlo_scan_unrolled.mlir

# Count D2D operations in baseline
echo "dynamic_slice: $(grep -c 'dynamic_slice' stablehlo_scan_baseline.mlir)"
echo "dynamic_update_slice: $(grep -c 'dynamic_update_slice' stablehlo_scan_baseline.mlir)"

# Count D2D operations after transformation (should be 0)
echo "dynamic_slice: $(grep -c 'dynamic_slice' stablehlo_scan_unrolled.mlir)"
echo "dynamic_update_slice: $(grep -c 'dynamic_update_slice' stablehlo_scan_unrolled.mlir)"
```

## Step 3: Optimize with HLO Tools (Recommended!)

### Find hlo_opt (Comes with XLA/JAX)
```bash
# hlo_opt is part of XLA, which comes with JAX
# Find where it's installed:
python -c "import jaxlib; import os; print(os.path.dirname(jaxlib.__file__))"

# Look for hlo_opt or xla_opt in the bin/ directory
# Example paths:
#   ~/.local/lib/python3.X/site-packages/jaxlib/bin/hlo_opt
#   /usr/local/lib/python3.X/site-packages/jaxlib/bin/hlo_opt

# Create an alias for convenience
alias hlo_opt="$(python -c 'import jaxlib; import os; print(os.path.join(os.path.dirname(jaxlib.__file__), "bin", "hlo_opt"))')"

# Verify it works
hlo_opt --help
```

### Apply HLO Optimization Passes (Better than mlir-opt!)
```bash
# Step 1: Verify the IR is valid
hlo_opt stablehlo_scan_unrolled.mlir --verify-diagnostics

# Step 2: Apply StableHLO-specific optimizations
hlo_opt stablehlo_scan_unrolled.mlir \
  --canonicalize \
  --cse \
  --stablehlo-aggressive-simplification \
  --stablehlo-aggressive-folder \
  -o stablehlo_optimized.mlir

# Step 3: Check the optimized IR
ls -lh stablehlo_optimized.mlir
head -50 stablehlo_optimized.mlir
```

**Why hlo_opt is better:**
- ✅ Built specifically for StableHLO (not generic MLIR)
- ✅ Includes HLO-specific optimization passes
- ✅ Already installed with JAX (no extra dependencies)
- ✅ Better understands XLA semantics
- ✅ More aggressive constant folding and simplification

### Available HLO Passes
```bash
# See all available passes
hlo_opt --help | grep -A 500 "Pass options"

# Key StableHLO passes:
# --stablehlo-aggressive-simplification  - Aggressive pattern matching
# --stablehlo-aggressive-folder         - Constant folding
# --stablehlo-canonicalize-dynamism     - Static shape inference
# --canonicalize                        - Generic canonicalization
# --cse                                 - Common subexpression elimination
```

### Advanced: Lower to Linalg (Optional)
```bash
# If hlo_opt has the lowering passes:
hlo_opt stablehlo_optimized.mlir \
  --stablehlo-legalize-to-linalg \
  -o stablehlo_linalg.mlir

# If not available, fall back to mlir-opt:
mlir-opt stablehlo_optimized.mlir \
  --stablehlo-legalize-to-linalg \
  -o stablehlo_linalg.mlir
```

## Step 4: Analyze the Optimized IR

### Check What Changed After Optimization
```bash
# Count operations before optimization
echo "=== BEFORE OPTIMIZATION ==="
echo "Total lines: $(wc -l < stablehlo_scan_unrolled.mlir)"
grep -o 'stablehlo\.[a-z_]*' stablehlo_scan_unrolled.mlir | sort | uniq -c | head -20

# Count operations after optimization
echo "=== AFTER OPTIMIZATION ==="
echo "Total lines: $(wc -l < stablehlo_optimized.mlir)"
grep -o 'stablehlo\.[a-z_]*' stablehlo_optimized.mlir | sort | uniq -c | head -20

# Look for specific optimizations
echo "=== OPTIMIZATION SUMMARY ==="
echo "Before: $(grep -c 'stablehlo.constant' stablehlo_scan_unrolled.mlir) constants"
echo "After:  $(grep -c 'stablehlo.constant' stablehlo_optimized.mlir) constants"
echo "Before: $(grep -c 'stablehlo.broadcast' stablehlo_scan_unrolled.mlir) broadcasts"
echo "After:  $(grep -c 'stablehlo.broadcast' stablehlo_optimized.mlir) broadcasts"
```

### Compare IR Sizes
```bash
ls -lh stablehlo_scan_*.mlir stablehlo_optimized.mlir
# The optimized version should be smaller (dead code eliminated, constants folded)
```

## Step 5: Validate Correctness (Optional)

### Create a Simple Test
```bash
# Test that the transformed IR is equivalent to original
python tools/test_stablehlo_correctness.py
```

Note: You'll need to create this test script that:
1. Runs JAX baseline
2. Runs optimized StableHLO (if compilation works)
3. Compares outputs

## Expected Results

### Transformation Quality

| Metric | JAX Baseline | After Transform | After Optimize |
|--------|--------------|-----------------|----------------|
| While loops | 1 | 0 | 0 |
| dynamic_slice | 90 | 0 | 0 |
| dynamic_update | 180 | 0 | 0 |
| Total ops | ~500 | ~2000 | ~1500 |
| IR size | 5 KB | 50 KB | 35 KB |

### Performance (When You Get to Compilation)

| Version | Runtime | Status |
|---------|---------|--------|
| JAX baseline | ~51ms | Current |
| JAX + Triton | ~20ms | Implemented |
| DaCe | 14.6ms | Reference |
| **StableHLO optimized** | **?** | **Testing** |

**Target:** <20ms (eliminate D2D copies)

## Troubleshooting

### "hlo_opt: command not found"
```bash
# Find where jaxlib is installed
python -c "import jaxlib; import os; print(os.path.dirname(jaxlib.__file__))"

# List bin directory
ls -la $(python -c 'import jaxlib; import os; print(os.path.join(os.path.dirname(jaxlib.__file__), "bin"))')

# Common locations:
# - hlo_opt
# - xla_opt (alternative name)
# - mlir-opt (fallback)

# If truly not found, use Python API instead:
python tools/optimize_stablehlo_python.py stablehlo_scan_unrolled.mlir
```

### "Unknown pass 'stablehlo-aggressive-simplification'"
```bash
# Your XLA version might be older
# Use basic passes only:
hlo_opt stablehlo_scan_unrolled.mlir \
  --canonicalize \
  --cse \
  -o stablehlo_optimized.mlir

# Or check available passes:
hlo_opt --help | grep stablehlo
```

### "mlir-opt: command not found" (if falling back from hlo_opt)
```bash
# hlo_opt is preferred, but if you must use mlir-opt:
pip install mlir-core

# Or use system package manager:
sudo apt-get install mlir-tools  # Ubuntu/Debian
```

### "Module not found: muphys_jax"
```bash
# Make sure you're in the right directory
cd /capstor/scratch/cscs/gandanie/icon/icon4py

# Or add to path
export PYTHONPATH="/capstor/scratch/cscs/gandanie/icon/icon4py:$PYTHONPATH"
```

### "MLIR verification failed"
This means the transformed IR has syntax errors. Check:
```bash
# Validate IR
mlir-opt stablehlo_scan_unrolled.mlir --verify-diagnostics

# Look for error messages
```

### "CUDA not available"
```bash
# Check GPU
nvidia-smi

# Check JAX sees GPU
python -c "import jax; print(jax.devices())"

# For IREE, use CPU backend instead
iree-compile stablehlo_scan_unrolled.mlir \
  --iree-hal-target-backends=llvm-cpu \
  -o stablehlo_cpu.vmfb
```

## What to Report Back

After running, please share:

1. **File sizes:**
   ```bash
   ls -lh stablehlo*.mlir
   ```

2. **Transformation statistics:**
   ```bash
   tail -50 stablehlo_graupel_analyzed.mlir
   ```

3. **Compilation result:**
   - Did `iree-compile` succeed?
   - Any error messages?

4. **Performance (if you get that far):**
   - JAX baseline time
   - Optimized time
   - Speedup

## Quick Sanity Checks

```bash
# 1. Did export work?
[ -f stablehlo_scan_baseline.mlir ] && echo "✓ Export OK" || echo "✗ Export FAILED"

# 2. Did transformation work?
[ -f stablehlo_scan_unrolled.mlir ] && echo "✓ Transform OK" || echo "✗ Transform FAILED"

# 3. Were while loops removed?
WHILE_COUNT=$(grep -c "stablehlo.while" stablehlo_scan_unrolled.mlir 2>/dev/null || echo "0")
[ "$WHILE_COUNT" -eq "0" ] && echo "✓ Unrolling OK" || echo "✗ Still has $WHILE_COUNT while loops"

# 4. Were dynamic slices removed?
DYN_COUNT=$(grep -c "dynamic_slice" stablehlo_scan_unrolled.mlir 2>/dev/null || echo "0")
[ "$DYN_COUNT" -eq "0" ] && echo "✓ Static slicing OK" || echo "✗ Still has $DYN_COUNT dynamic slices"
```

## Summary

**Minimum to run:**
```bash
cd muphys_jax
./tools/run_stablehlo_pipeline.sh
```

**Check if it worked:**
```bash
ls -lh stablehlo*.mlir
grep -c "stablehlo.while" stablehlo_scan_unrolled.mlir  # Should be 0
```

**Try to optimize with hlo_opt:**
```bash
# Find hlo_opt (comes with JAX)
python -c "import jaxlib; import os; print(os.path.join(os.path.dirname(jaxlib.__file__), 'bin', 'hlo_opt'))"

# Use it (replace path with output from above)
/path/to/hlo_opt stablehlo_scan_unrolled.mlir \
  --canonicalize \
  --cse \
  -o stablehlo_optimized.mlir
```

**If all above works:** You have successfully transformed and compiled StableHLO! 🎉

**If something fails:** Share the error message and I'll help debug.
