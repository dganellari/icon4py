# Integrating LoopifyUnrolledSlices into XLA

## Overview

Custom XLA pass that converts unrolled `slice->compute->concat` patterns into
either while loops or single-kernel serial scan fusions.

**Target**: graupel microphysics precipitation scan — 90 vertical levels
unrolled into ~186 separate GPU kernels -> 1-2 kernels.

**Two modes**:
- `kWhileLoop` — emits an HLO while loop; XLA fuses the body into ~6 kernels
- `kSerialScan` — emits a custom fusion with a single GPU kernel containing
  `scf.forall` (parallel over cells) + `scf.for` (serial over levels)

**Performance (R2B06, GH200)**:
- Baseline (StableHLO injection + fused q_t_update): ~29ms
- WhileLoop mode: ~35ms (correct, ~6 kernels for precip scan)
- SerialScan mode: ~33ms (correct, 1 kernel for precip scan)

> **Note**: `kSerialScan` is the current focus. It produces a single kernel but
> is still ~4ms slower than the hand-optimized StableHLO injection baseline.

## Files

```
xla_passes/
  BUILD                           # Bazel build rules
  loopify_unrolled_slices.h       # Pass header (Mode enum, constructor)
  loopify_unrolled_slices.cc      # Pass implementation (pattern detection + transformation)
  serial_scan_emitter.h           # SerialScanFusion emitter header
  serial_scan_emitter.cc          # MLIR codegen for serial scan kernel
  loopify_unrolled_slices_test.cc # Unit tests
```

## Step 1. Clone JAX and XLA

```bash
git clone https://github.com/jax-ml/jax.git
cd jax
git checkout jax-v0.6.2  # note: tag is jax-v0.6.2, NOT jaxlib-v0.6.2

cd ..
git clone https://github.com/openxla/xla.git
```

## Step 2. Copy the pass files

```bash
mkdir -p xla/xla/service/gpu/transforms/loopify

PASS_SRC=icon4py/model/atmosphere/subgrid_scale_physics/muphys_jax/xla_passes
XLA_DST=xla/xla/service/gpu/transforms/loopify

cp $PASS_SRC/BUILD                       $XLA_DST/
cp $PASS_SRC/loopify_unrolled_slices.h   $XLA_DST/
cp $PASS_SRC/loopify_unrolled_slices.cc  $XLA_DST/
cp $PASS_SRC/serial_scan_emitter.h       $XLA_DST/
cp $PASS_SRC/serial_scan_emitter.cc      $XLA_DST/
cp $PASS_SRC/loopify_unrolled_slices_test.cc $XLA_DST/
```

Or use rsync:

```bash
rsync -avz $PASS_SRC/ $XLA_DST/
```

## Step 3. Apply XLA core patches

Six small edits across 5 files. All paths are relative to the XLA root.

### 3a. `xla/service/gpu/ir_emission_utils.h` — Add kind constant

After the existing `kUncompilableFusion` constant, add:

```cpp
inline constexpr absl::string_view kSerialScanFusionKind = "__serial_scan";
```

### 3b. `xla/service/gpu/hlo_fusion_analysis.h` — Add enum value

In the `EmitterFusionKind` enum, add `kSerialScan` at the end (before the
closing brace):

```cpp
    kDynamicMemcpy,
    kSerialScan,
  };
```

> **Note**: The enum values differ across XLA versions. Check your version's
> enum and add `kSerialScan` after the last existing entry.

### 3c. `xla/service/gpu/hlo_fusion_analysis.cc` — Detect kind

In `GetEmitterFusionKind()`, add after the last `FusionKind` check (e.g. after
the `kDynamicMemcpyFusionKind` block):

```cpp
  if (fusion_backend_config_.kind() == kSerialScanFusionKind) {
    return EmitterFusionKind::kSerialScan;
  }
```

> **Note**: The variable name for the backend config may differ across XLA
> versions. Check what the surrounding code uses — it could be
> `fusion_backend_config_`, `fusion_backend_config()`, or a local variable.
> Match the existing style.

### 3d. `xla/backends/gpu/codegen/fusions.cc` — Register emitter

Add include at top:

```cpp
#include "xla/service/gpu/transforms/loopify/serial_scan_emitter.h"
```

In `GetFusionEmitter()`, add a case in the switch statement:

```cpp
    case HloFusionAnalysis::EmitterFusionKind::kSerialScan:
      return std::make_unique<SerialScanFusion>(analysis);
```

Also add to the `deps` in `xla/backends/gpu/codegen/BUILD` for the `fusions`
target:

```
"//xla/service/gpu/transforms/loopify:serial_scan_emitter",
```

### 3e. `xla/service/gpu/fusion_pipeline.cc` — Register the pass

Add include at top:

```cpp
#include "xla/service/gpu/transforms/loopify/loopify_unrolled_slices.h"
```

In `FusionPipeline()`, add **before** `PriorityFusion`:

```cpp
  fusion.AddPass<LoopifyUnrolledSlices>(
      /*min_iterations=*/4, /*unroll_factor=*/1,
      LoopifyUnrolledSlices::Mode::kSerialScan);
```

Also add to the `deps` in `xla/service/gpu/BUILD` for the `fusion_pipeline`
target:

```
"//xla/service/gpu/transforms/loopify:loopify_unrolled_slices",
```

> For while-loop mode only, use `Mode::kWhileLoop` (default) and skip
> patches 3a-3d.

## Step 4. Build JAX with modified XLA

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

**Incremental rebuild** (after modifying only pass .cc/.h files):

```bash
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

**Build flag notes**:
- `--use_clang=false`: Use GCC instead of clang
- `--cuda_compute_capabilities="9.0"`: GH200/H100. Restricting to your arch
  speeds up the build and avoids unsupported `sm_120` errors
- `--bazel_startup_options`: Persistent output base for incremental builds

## Step 5. Install

```bash
pip install --no-deps jax==0.6.2
pip install numpy scipy ml_dtypes opt_einsum
pip install nvidia-cudnn-cu12

pip install --force-reinstall --no-deps \
  dist/jaxlib-*.whl \
  dist/jax_cuda12_plugin-*.whl \
  dist/jax_cuda12_pjrt-*.whl
```

**Reinstalling after incremental rebuild:**

```bash
pip install --force-reinstall --no-deps \
  dist/jax_cuda12_plugin-*.whl \
  dist/jax_cuda12_pjrt-*.whl
```

Use `--no-deps` because dev wheels have dependency pins that don't exist on PyPI.

## Step 6. Runtime environment (CSCS Alps / Santis)

```bash
export LD_LIBRARY_PATH=/user-environment/linux-sles15-neoverse_v2/gcc-13.2.0/cuda-12.6.0-cqxe545cshmxocfoqzdwolerb4i447t5/extras/CUPTI/lib64:$LD_LIBRARY_PATH
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/user-environment/linux-sles15-neoverse_v2/gcc-13.2.0/cuda-12.6.0-cqxe545cshmxocfoqzdwolerb4i447t5"
```

## Step 7. Run the benchmark

```bash
cd /path/to/icon4py

CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 srun -n1 \
  python model/atmosphere/subgrid_scale_physics/muphys_jax/tests/test_graupel_native_transposed.py \
  --input /capstor/store/cscs/userlab/d126/muphys_grids/inputs/atm_R2B06.nc \
  --graupel-hlo stablehlo/graupel_combined.stablehlo \
  --num-runs 10
```

## Step 8. Verify the pass fired

```bash
CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 \
  XLA_FLAGS="$XLA_FLAGS --xla_dump_to=/capstor/scratch/cscs/$USER/xla_dump --xla_dump_hlo_as_text" \
  srun -n1 python model/atmosphere/subgrid_scale_physics/muphys_jax/tests/test_graupel_native_transposed.py \
  --input /capstor/store/cscs/userlab/d126/muphys_grids/inputs/atm_R2B06.nc \
  --graupel-hlo stablehlo/graupel_combined.stablehlo \
  --num-runs 1

# For while-loop mode: check for while loops
grep -l "while" /capstor/scratch/cscs/$USER/xla_dump/*.txt

# For serial-scan mode: check for __serial_scan fusions
grep -l "serial_scan" /capstor/scratch/cscs/$USER/xla_dump/*.txt
```

## Troubleshooting

### BUILD target names differ across XLA versions

The BUILD dependency names vary between XLA versions. Common differences:

| What you need | Possible target names |
|---|---|
| Backend config protos | `backend_configs_cc`, `backend_configs_cc_proto` |
| Indexing map | `indexing_analysis`, `indexing_map` |
| Emitter base | `emitter_base`, `fusion_emitter` |

If you get `no such target` errors, run:
```bash
grep "name =" xla/<path>/BUILD | head -20
```
to find the correct target name.

### `ComputeThreadIdToInputIndexing` override error

The base class signature for this method varies across XLA versions.
Check your version:
```bash
grep "ComputeThreadIdToInputIndexing" xla/backends/gpu/codegen/emitters/emitter_base.h
```
Update `serial_scan_emitter.h` to match (number of `int64_t` parameters and
return type may differ).

### `FusionBackendConfig` not found

The protobuf type is in `xla/service/gpu/backend_configs.pb.h` (generated at
build time). Add to your .cc file:
```cpp
#include "xla/service/gpu/backend_configs.pb.h"
```
When setting the config, wrap in `GpuBackendConfig`:
```cpp
GpuBackendConfig gpu_config;
FusionBackendConfig* fc = gpu_config.mutable_fusion_backend_config();
fc->set_kind("__serial_scan");
fc->mutable_custom_fusion_config()->set_name(metadata);
TF_RETURN_IF_ERROR(fusion_inst->set_backend_config(gpu_config));
```

### `RunImpl` vs `Run` override error

Check your XLA's `xla/hlo/pass/hlo_pass_interface.h`:
```bash
grep "virtual.*StatusOr.*bool.*Run" xla/hlo/pass/hlo_pass_interface.h
```
Update the header and .cc to match (`Run` or `RunImpl`).

### XLA dump is empty

When running with `srun`, `/tmp` on compute nodes is local. Use a shared
filesystem path for `--xla_dump_to`.

### GPU out of memory on login node

Use `srun -n1` to get a dedicated compute node.

## Related: IREE Preprocessing Pass

A parallel effort ports the same algorithm to IREE as a preprocessing pass
(`LoopifyInsertSliceChain`) targeting AMD MI300A via IREE's HIP/ROCm backend.

- Same chain detection, carry analysis, and iter1-as-template approach
- Generates `flow.dispatch.region { scf.forall + scf.for }` with `WorkgroupMappingAttr`
- Key difference: operates on MLIR `tensor.insert_slice` chains (from StableHLO
  concatenate lowering) rather than HLO `slice → concat` patterns
- The XLA pass's `depends_on_slice` forward propagation is critical for
  correctness and is being ported to the IREE version
- Source: `iree-jax/compiler/src/iree/compiler/Preprocessing/Common/LoopifyInsertSliceChain.cpp`
