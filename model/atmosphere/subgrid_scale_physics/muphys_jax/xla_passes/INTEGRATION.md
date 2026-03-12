# Integrating LoopifyUnrolledSlices into XLA

## Overview

This custom XLA pass converts unrolled `slice→compute→concat` patterns into
`while` loops, enabling XLA GPU to fuse N per-level kernels into 1-2 kernels
with an internal loop.

**Target**: graupel microphysics precipitation scan — 90 vertical levels
unrolled into 186 separate GPU kernels → 1-2 kernels.

## Quick Start

### 1. Clone JAX and XLA

```bash
# Clone JAX and check out the desired version
git clone https://github.com/jax-ml/jax.git
cd jax
git checkout jax-v0.6.2  # note: tag is jax-v0.6.2, NOT jaxlib-v0.6.2

# Clone XLA separately (any recent version works — JAX builds it from source)
cd ..
git clone https://github.com/openxla/xla.git
```

### 2. Copy the pass files

```bash
# Create directory for the pass
mkdir -p xla/xla/service/gpu/transforms/loopify

# Copy files (from icon4py repo)
PASS_SRC=/path/to/icon4py/model/atmosphere/subgrid_scale_physics/muphys_jax/xla_passes
XLA_DST=xla/xla/service/gpu/transforms/loopify

cp $PASS_SRC/loopify_unrolled_slices.h   $XLA_DST/
cp $PASS_SRC/loopify_unrolled_slices.cc  $XLA_DST/
cp $PASS_SRC/BUILD                       $XLA_DST/
cp $PASS_SRC/loopify_unrolled_slices_test.cc $XLA_DST/
```

The include path in the `.cc` file should already be:
```cpp
#include "xla/service/gpu/transforms/loopify/loopify_unrolled_slices.h"
```

### 3. Register the pass in the fusion pipeline

Edit `xla/xla/service/gpu/fusion_pipeline.cc`:

```cpp
// Add include at top:
#include "xla/service/gpu/transforms/loopify/loopify_unrolled_slices.h"

// In FusionPipeline(), add BEFORE PriorityFusion:
HloPassPipeline FusionPipeline(...) {
  HloPassFix<HloPassPipeline> fusion("fusion");
  fusion.AddPass<VariadicOpSplitter>();
  // ... verifier ...
  fusion.AddPass<SortIotaFusion>();

  // === ADD THIS ===
  fusion.AddPass<LoopifyUnrolledSlices>(/*min_iterations=*/4);
  // ================

  fusion.AddPass<PriorityFusion>(...);
  // ... rest unchanged ...
}
```

Also add `"//xla/service/gpu/transforms/loopify:loopify_unrolled_slices"` to
the `deps` in `xla/xla/service/gpu/BUILD` for the `fusion_pipeline` target.

### 4. Build JAX with modified XLA

JAX's `build.py` compiles XLA from source, so no separate XLA build is needed.
The `--local_xla_path` flag tells JAX to use your modified XLA.

**Full build (first time — builds jaxlib + CUDA plugin + PJRT):**

```bash
cd jax

python build/build.py build \
  --wheels=jaxlib,jax-cuda-plugin,cuda-pjrt \
  --local_xla_path=/path/to/xla \
  --use_clang=false \
  --gcc_path=$(which gcc) \
  --cuda_version=12.6.3 \
  --cudnn_version=9.8.0 \
  --cuda_compute_capabilities="9.0" \
  --bazel_startup_options="--output_base=/path/to/bazel_output"
```

**Incremental rebuild (after modifying only pass .cc/.h files):**

Only the CUDA plugin and PJRT wheels need rebuilding — jaxlib doesn't include
the fusion pipeline code. Copy the updated files first:

```bash
# Copy updated pass files to XLA
cp $PASS_SRC/loopify_unrolled_slices.cc  $XLA_DST/
cp $PASS_SRC/loopify_unrolled_slices.h   $XLA_DST/

# Rebuild only the CUDA wheels (much faster — bazel caches unchanged targets)
python build/build.py build \
  --wheels=jax-cuda-plugin,cuda-pjrt \
  --local_xla_path=/path/to/xla \
  --use_clang=false \
  --gcc_path=$(which gcc) \
  --cuda_version=12.6.3 \
  --cudnn_version=9.8.0 \
  --cuda_compute_capabilities="9.0" \
  --bazel_startup_options="--output_base=/path/to/bazel_output"
```

**Notes on build flags:**
- `--use_clang=false`: Use GCC instead of clang (avoids needing system clang).
  Bazel's toolchain wraps GCC behind a "clang" script anyway.
- `--cuda_version`: Must be an exact supported version (e.g., `12.6.3`),
  not just `12`. Check error output for the list of supported versions.
- `--cudnn_version`: Same — use exact version like `9.8.0`.
- `--cuda_compute_capabilities`: Restrict to your GPU arch to speed up build.
  Use `"9.0"` for GH200/H100. Omitting this may fail if nvcc doesn't support
  newer architectures (e.g., `sm_120` requires CUDA 12.8+).
- `--bazel_startup_options`: Use a persistent output base to cache builds
  across invocations. This makes incremental rebuilds much faster.

### 5. Install

```bash
# Install the base JAX package (pure Python, not built from source)
pip install --no-deps jax==0.6.2

# Install runtime dependencies
pip install numpy scipy ml_dtypes opt_einsum

# Install NVIDIA runtime libraries (cuDNN, cuPTI, etc.)
# These are needed because custom-built wheels don't bundle them
# (unlike PyPI's jax[cuda12] which bundles everything).
pip install nvidia-cudnn-cu12

# Install the three compiled wheels (use --no-deps to avoid PyPI conflicts)
pip install --force-reinstall --no-deps \
  dist/jaxlib-*.whl \
  dist/jax_cuda12_plugin-*.whl \
  dist/jax_cuda12_pjrt-*.whl
```

**Why `--no-deps`?** The dev wheels (e.g., `0.6.2.dev20260217`) have dependency
pins that don't exist on PyPI. Using `--no-deps` skips dependency resolution.

**Reinstalling after incremental rebuild:** Only reinstall the wheels you rebuilt:
```bash
pip install --force-reinstall --no-deps \
  dist/jax_cuda12_plugin-*.whl \
  dist/jax_cuda12_pjrt-*.whl
```

### 6. Runtime environment (CSCS Alps / Santis)

The custom-built wheels link against system CUDA libraries at runtime.
You need to set `LD_LIBRARY_PATH` and `XLA_FLAGS` for cuPTI and libdevice:

```bash
# Add cuPTI to library path (needed for CUDA plugin initialization)
export LD_LIBRARY_PATH=/user-environment/linux-sles15-neoverse_v2/gcc-13.2.0/cuda-12.6.0-cqxe545cshmxocfoqzdwolerb4i447t5/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# Point XLA to the CUDA SDK (needed for libdevice.10.bc used by PTX compilation)
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/user-environment/linux-sles15-neoverse_v2/gcc-13.2.0/cuda-12.6.0-cqxe545cshmxocfoqzdwolerb4i447t5"
```

Put these in your `.bashrc` or a setup script to avoid repeating them.

### 7. Run the benchmark

```bash
cd /path/to/icon4py

# Generate unrolled StableHLO (if not already)
cd model/atmosphere/subgrid_scale_physics/muphys_jax
python tools/generate_unrolled_transposed.py
python tools/generate_qt_update_stablehlo.py
python tools/generate_combined_graupel.py
cd /path/to/icon4py

# Run on compute node (login node GPU has limited memory)
CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 srun -n1 \
  python model/atmosphere/subgrid_scale_physics/muphys_jax/tests/test_graupel_native_transposed.py \
  --input /capstor/store/cscs/userlab/d126/muphys_grids/inputs/atm_R2B06.nc \
  --graupel-hlo stablehlo/graupel_combined.stablehlo \
  --num-runs 10
```

### 8. Verify the pass fired (XLA dump)

Use a shared filesystem path for the dump (not `/tmp` — compute nodes have
local `/tmp` that isn't visible from login nodes):

```bash
CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 \
XLA_FLAGS="$XLA_FLAGS --xla_dump_to=/capstor/scratch/cscs/$USER/xla_dump --xla_dump_hlo_as_text" \
srun -n1 python model/atmosphere/subgrid_scale_physics/muphys_jax/tests/test_graupel_native_transposed.py \
  --input /capstor/store/cscs/userlab/d126/muphys_grids/inputs/atm_R2B06.nc \
  --graupel-hlo stablehlo/graupel_combined.stablehlo \
  --num-runs 1

# Verify while loop exists in optimized HLO
grep -l "while" /capstor/scratch/cscs/$USER/xla_dump/*.txt

# Count fusions (should be ~6 in while body, ~200+ in ENTRY for qt_update)
grep -c "fusion" /capstor/scratch/cscs/$USER/xla_dump/module_*.gpu_after_optimizations.txt
```

## How It Works

### Pattern Detection

The pass scans for `concatenate` instructions whose operands trace back to
consecutive `slice` operations from the same source tensor:

```
slice(input, [0:1, :]) → compute_0 → ┐
slice(input, [1:2, :]) → compute_1 → ├→ concatenate → output
...                                   │
slice(input, [89:90,:]) → compute_89 →┘
```

The backward trace from each concat operand `i` only records slices at
position `i` (matching `starts[cat_dim] == i`). This prevents carry
dependencies from polluting the slice detection.

Multiple concatenates that share the same computation (e.g., 11 output
fields from a precipitation scan) are grouped into a single while loop
with one accumulator per output.

### Transformation

Replaces with:

```
while (k < 90) {
  slice_k = dynamic-slice(input, k, 0)
  result_k = compute(slice_k, carry_state)
  output = dynamic-update-slice(output, result_k, k, 0)
  carry_state = updated_carry
  k = k + 1
}
```

### Why This Helps

XLA GPU fuses the while-loop body into ~6 kernels (one per major
computation phase). The unrolled version creates ~186 separate kernels
because XLA's PriorityFusion treats each slice→compute chain as independent.

With the while loop:
- **~6 kernels** instead of ~186
- **Better register/L2 utilization** — carry state stays in cache
- **No kernel launch overhead** for 90 iterations
- **Dynamic slicing** is nearly free on GPU

## Alternative: XLA Flag Control

If you want to conditionally enable the pass:

```cpp
// In fusion_pipeline.cc:
if (debug_options.xla_gpu_enable_loopify_slices()) {
  fusion.AddPass<LoopifyUnrolledSlices>();
}
```

Add the flag to `xla/xla.proto`:
```protobuf
bool xla_gpu_enable_loopify_slices = 999 [default = false];
```

Then enable via:
```python
os.environ['XLA_FLAGS'] = '--xla_gpu_enable_loopify_slices=true'
```

## Troubleshooting

### No performance improvement / incorrect results after pass fires

**Symptom**: The while loop appears in the thunk sequence but the ~180 unrolled
kernels are ALSO still present, and outputs are wrong.

**Cause**: After `ReplaceAllUsesWith`, the unrolled concat operands have zero
users but are not removed.  `PriorityFusion` runs next and fuses these dead
instructions into real GPU kernels.  The buffer allocator then reuses the
while-loop output buffers for dead-fusion outputs, which overwrites correct
results at runtime.

**Fix** (in the pass): The pass now calls
`RemoveInstructionAndUnusedOperands(concat)` for every replaced concat after
`ReplaceAllUsesWith`, cascading the removal through all 90×N dead per-level ops.
This is included in the current version of `loopify_unrolled_slices.cc`.

**Alternative** (pipeline-level): Add `HloDCE` after the pass in
`fusion_pipeline.cc`:
```cpp
fusion.AddPass<LoopifyUnrolledSlices>(/*min_iterations=*/4);
fusion.AddPass<HloDCE>();  // eliminate dead unrolled ops before PriorityFusion
fusion.AddPass<PriorityFusion>(...);
```
(Requires `#include "xla/service/hlo_dce.h"` and adding `//xla/service:hlo_dce`
to the deps.)

### `RunImpl` vs `Run` override error
Different XLA versions use different method names on `HloModulePass`.
Check your XLA's `xla/hlo/pass/hlo_pass_interface.h`:
```bash
grep "virtual.*StatusOr.*bool.*Run" xla/hlo/pass/hlo_pass_interface.h
```
Update the header and .cc to match (`Run` or `RunImpl`).

### `std::pair` hash error with `absl::flat_hash_map`
`absl::flat_hash_map` does not support `std::pair` keys by default.
Use a string key (e.g., `absl::StrCat(num_iters, "_", slice_dim)`) instead.

### CUDA compute capability errors
If nvcc reports `Unsupported gpu architecture 'compute_120'`, restrict
capabilities with `--cuda_compute_capabilities="9.0"` (or your GPU's arch).

### `jax-cuda12-pjrt` not found on PyPI
The dev wheels have version pins that don't exist on PyPI. Build and install
all three wheels together with `--no-deps`.

### cuDNN / cuPTI not found at runtime
Custom-built wheels don't bundle NVIDIA libraries like PyPI wheels do.
Install `nvidia-cudnn-cu12` via pip, and add cuPTI to `LD_LIBRARY_PATH`
(see Step 6 above).

### libdevice not found
XLA needs `libdevice.10.bc` for PTX compilation. Set:
```bash
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/path/to/cuda"
```
where `/path/to/cuda` contains `nvvm/libdevice/libdevice.10.bc`.

### XLA dump is empty
When running with `srun`, `/tmp` on compute nodes is local. Use a shared
filesystem path for `--xla_dump_to` (e.g., your scratch directory).

### GPU out of memory on login node
Login node GPUs have limited free memory. Use `srun -n1` to get a dedicated
compute node allocation.
