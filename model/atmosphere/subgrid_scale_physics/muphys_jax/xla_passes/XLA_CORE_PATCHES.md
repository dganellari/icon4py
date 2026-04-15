# XLA Core Patches for Serial Scan Fusion

These patches must be applied to the XLA source tree to register the
SerialScanFusion emitter. Apply them after copying the `loopify/` directory.

## 1. `xla/service/gpu/ir_emission_utils.h` — Add kind constant

After `kUncompilableFusion`:

```cpp
// Fusions that emit a single kernel with a serial scan loop.
inline constexpr absl::string_view kSerialScanFusionKind = "__serial_scan";
```

## 2. `xla/service/gpu/hlo_fusion_analysis.h` — Add enum value

In `EmitterFusionKind` enum, add after `kSort`:

```cpp
    kSort,
    kSerialScan,
```

## 3. `xla/service/gpu/hlo_fusion_analysis.cc` — Detect kind

In `GetEmitterFusionKind()`, add after the `kDynamicMemcpyFusionKind` check
(around line 131):

```cpp
  if (fusion_backend_config.kind() == kSerialScanFusionKind) {
    return HloFusionAnalysis::EmitterFusionKind::kSerialScan;
  }
```

## 4. `xla/backends/gpu/codegen/fusions.cc` — Register emitter

Add include at top:

```cpp
#include "xla/service/gpu/transforms/loopify/serial_scan_emitter.h"
```

In `GetFusionEmitter()`, add a case before the closing brace of the switch:

```cpp
    case HloFusionAnalysis::EmitterFusionKind::kSerialScan:
      return std::make_unique<SerialScanFusion>(analysis);
```

Also add to the `deps` in the BUILD for `fusions`:

```
"//xla/service/gpu/transforms/loopify:serial_scan_emitter",
```

## 5. `xla/service/gpu/fusion_pipeline.cc` — Register pass (serial scan mode)

Change the pass registration from:

```cpp
fusion.AddPass<LoopifyUnrolledSlices>(/*min_iterations=*/4);
```

To:

```cpp
fusion.AddPass<LoopifyUnrolledSlices>(
    /*min_iterations=*/4, /*unroll_factor=*/1,
    LoopifyUnrolledSlices::Mode::kSerialScan);
```

## Summary of files to modify

| File | Change |
|------|--------|
| `xla/service/gpu/ir_emission_utils.h` | Add `kSerialScanFusionKind` constant |
| `xla/service/gpu/hlo_fusion_analysis.h` | Add `kSerialScan` to enum |
| `xla/service/gpu/hlo_fusion_analysis.cc` | Add kind detection |
| `xla/backends/gpu/codegen/fusions.cc` | Add emitter creation + include |
| `xla/backends/gpu/codegen/BUILD` (fusions target) | Add dep on serial_scan_emitter |
| `xla/service/gpu/fusion_pipeline.cc` | Switch to kSerialScan mode |
