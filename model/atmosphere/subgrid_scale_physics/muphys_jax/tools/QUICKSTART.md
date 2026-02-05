# StableHLO Optimization Quick Start

## Goal
Transform JAX graupel to eliminate 92.1% D2D memory copy overhead and achieve DaCe-like performance (51ms → ~15ms).

## Quick Run


# Step 1: Generate q_t_update StableHLO
python generate_qt_update_stablehlo.py -o stablehlo/qt_update.stablehlo

# Step 2: Generate precip StableHLO (already exists, or regenerate)
python generate_unrolled_transposed.py -o stablehlo/precip_transposed.stablehlo

# Step 3: Combine them
python generate_combined_graupel.py \
  --qt-update stablehlo/qt_update.stablehlo \
  --precip stablehlo/precip_transposed.stablehlo \
  -o stablehlo/graupel_combined.stablehlo

# Step 4: Test with combined HLO
python test_graupel_native_transposed.py \
  --input /path/to/graupel_input.nc \
  --graupel-hlo stablehlo/graupel_combined.stablehlo \
  --num-runs 60

# Or test precip-only HLO (existing path, still works)
python test_graupel_native_transposed.py \
  --input /path/to/graupel_input.nc \
  --optimized-hlo stablehlo/precip_transposed.stablehlo \
  --num-runs 60

Now you can benchmark the combined graupel directly:


# Benchmark combined graupel vs precip-only
python benchmark_stablehlo.py \
  stablehlo/graupel_combined.stablehlo \
  --compare stablehlo/precip_transposed.stablehlo \
  --input /path/to/input.nc --transposed --num-runs 100

# Or just the combined graupel alone
python benchmark_stablehlo.py \
  stablehlo/graupel_combined.stablehlo \
  --input /path/to/input.nc --transposed --num-runs 100

The benchmark auto-detects the input count from the @main signature in each StableHLO file. When comparing files with different signatures (14 inputs vs 13), it loads the right inputs for each file separately. This gives you clean, isolated XLA-only execution timings without any JAX dispatch overhead.

## Understanding the Output

### Before Transformation
```mlir
%5:6 = stablehlo.while(%iterArg = %arg2, ...) {
  cond { ... }
  do {
    %13 = stablehlo.dynamic_slice %iterArg, %11, %12  // D2D READ
    %34 = stablehlo.dynamic_update_slice ...          // D2D WRITE
  }
}
```

**Problem:**
- Dynamic slicing forces memory traffic
- Carry state lives in memory
- 92.1% of runtime is D2D copies!

### After Transformation
```mlir
// Iteration 0
%x_0 = stablehlo.slice %arg2 [0:1, 0:1000]  // Static slice
%a_1 = stablehlo.add %a_0, %x_0             // Register ops
%b_1 = stablehlo.add %b_0, %x_0

// Iteration 1
%x_1 = stablehlo.slice %arg2 [1:2, 0:1000]
%a_2 = stablehlo.add %a_1, %x_1
%b_2 = stablehlo.add %b_1, %x_1
...
```

**Benefits:**
- No dynamic indexing → compiler can optimize
- Carry in SSA values → stays in registers
- Single kernel launch possible

## Current Status

### ✅ Completed
1. StableHLO export from JAX (simple + full graupel)
2. While loop structure analysis
3. Basic unrolling transformation (simple scan)
4. Multi-loop analysis (v2 transformer)
5. Documentation and pipeline automation

### 🚧 In Progress
1. Full body operation parsing
2. SSA value renaming/tracking
3. Output tensor reconstruction from unrolled results

### ⏳ TODO
1. Handle all 180 graupel scans
2. GPU lowering passes
3. Compilation to executable
4. Performance benchmarking
5. Correctness verification

## Expected Performance

| Version | Runtime | Speedup | Method |
|---------|---------|---------|--------|
| JAX Baseline | 51ms | 1.0× | lax.scan → while loops |
| JAX Triton | ~20ms | 2.5× | Custom CUDA kernel |
| DaCe | 14.6ms | 3.5× | Static schedule, fused |
| **StableHLO (Target)** | **~15ms** | **~3.4×** | **Unrolled + optimized** |

## Debugging

### Check Export Worked
```bash
# Should see module declaration
head -5 stablehlo_scan_baseline.mlir

# Should see while loop
grep -n "stablehlo.while" stablehlo_scan_baseline.mlir
```

### Count D2D Operations
```bash
# Baseline
echo "dynamic_slice: $(grep -c 'dynamic_slice' stablehlo_graupel_full.mlir)"
echo "dynamic_update_slice: $(grep -c 'dynamic_update_slice' stablehlo_graupel_full.mlir)"

# After transformation (should be 0)
echo "dynamic_slice: $(grep -c 'dynamic_slice' stablehlo_unrolled.mlir)"
```

### Verify JAX Version
```bash
python -c "import jax; print(f'JAX {jax.__version__}')"
# Need JAX >= 0.4.0 for StableHLO export
```

## Troubleshooting

### "jax.experimental.export not available"
- This is OK! We use `jax.jit().lower().compiler_ir()` instead
- Works on JAX 0.4.x - 0.6.x

### "No while loops found"
- Check if scans were already optimized away
- Try with `use_fused_scans=False` in export script

### "mlir-opt not found"
- Install with: `pip install mlir`
- Or use IREE compiler: `iree-compile`
- Or skip optimization passes for now

### Large IR files (>1GB)
- This is normal for full graupel (180 scans × 90 iters = 16,200 operations)
- Consider transforming one scan at a time first
- Use `unroll_threshold` parameter to limit unrolling

## How to Run the Optimized StableHLO

### Method 1: Using IREE Compiler (Recommended)

IREE compiles StableHLO directly to GPU executable:

```bash
# Install IREE
pip install iree-compiler iree-runtime

# Compile transformed StableHLO to GPU
iree-compile stablehlo_unrolled.mlir \
  --iree-hal-target-backends=cuda \
  -o stablehlo_gpu.vmfb

# Run with benchmark script
python tools/run_optimized_stablehlo.py stablehlo_unrolled.mlir \
  --compiler iree \
  --benchmark
```

### Method 2: Using MLIR Tools

Lower through MLIR dialect hierarchy:

```bash
# Install MLIR tools
pip install mlir

# Step 1: Optimize StableHLO
mlir-opt stablehlo_unrolled.mlir \
  --canonicalize \
  --cse \
  --symbol-dce \
  -o stablehlo_opt.mlir

# Step 2: Lower to Linalg dialect
mlir-opt stablehlo_opt.mlir \
  --stablehlo-legalize-to-linalg \
  -o stablehlo_linalg.mlir

# Step 3: Lower to GPU dialect
mlir-opt stablehlo_linalg.mlir \
  --convert-linalg-to-parallel-loops \
  --gpu-map-parallel-loops \
  --convert-parallel-loops-to-gpu \
  -o stablehlo_gpu.mlir

# Step 4: Compile to CUDA
mlir-opt stablehlo_gpu.mlir \
  --gpu-kernel-outlining \
  --convert-gpu-to-nvvm \
  --gpu-to-llvm \
  -o stablehlo_nvvm.mlir
```

### Method 3: Quick Test (Without Full Compilation)

Just validate the IR syntax:

```bash
# Check if transformed IR is valid
mlir-opt stablehlo_unrolled.mlir --verify-diagnostics

# If successful, you'll see no errors
# If errors, the IR needs fixing
```

## Benchmark Commands

### JAX Baseline
```bash
# Run JAX version (baseline)
python -m muphys_jax.driver.run_graupel_jax \
  -o /dev/null \
  input.nc 100 30.0 100.0

# Expected: ~51ms per iteration
```

### Optimized StableHLO
```bash
# Run optimized version (after IREE compilation)
python tools/run_optimized_stablehlo.py \
  stablehlo_unrolled.mlir \
  --compiler iree \
  --benchmark

# Target: <20ms per iteration
```

## Next Steps

After running the pipeline:

1. **Inspect the analysis output**
   ```bash
   less stablehlo_graupel_analyzed.mlir
   # Look for loop count, D2D operations
   ```

2. **Test compilation**
   ```bash
   # Try IREE compilation on simple scan first
   iree-compile stablehlo_scan_unrolled.mlir \
     --iree-hal-target-backends=cuda \
     -o test.vmfb

   # If successful, you have a working pipeline!
   ```

3. **Implement full transformation**
   - Edit `transform_stablehlo_v2.py`
   - Parse body operations (see `analyze_scan_pattern`)
   - Generate correct SSA renaming

4. **Scale to full graupel**
   - Apply to all 180 scans
   - Handle different scan patterns
   - Benchmark against JAX baseline

## Questions?

See [README_STABLEHLO_OPTIMIZATION.md](README_STABLEHLO_OPTIMIZATION.md) for:
- Detailed transformation strategy
- MLIR optimization passes
- GPU lowering pipeline
- Performance analysis

---

**Remember:** Sync these files to server before running!
