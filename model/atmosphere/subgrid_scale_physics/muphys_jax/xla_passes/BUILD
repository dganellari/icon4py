# Build rules for custom XLA pass: LoopifyUnrolledSlices + SerialScanFusion
#
# This pass converts unrolled slice→compute→concat patterns into while loops
# or serial scan fusions (single GPU kernels with internal loops).
#
# To integrate into XLA:
#   1. Copy this directory into openxla/xla/xla/service/gpu/transforms/loopify/
#   2. Add the pass to fusion_pipeline.cc (before PriorityFusion)
#   3. Apply XLA core patches (see INTEGRATION.md)
#   4. Rebuild JAX with the modified XLA

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla:xla.default.bzl", "xla_cc_test")

package(
    default_visibility = [
        "//xla:friends",
        "//xla/backends/gpu/codegen:__subpackages__",
        "//xla/service/gpu:__subpackages__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "serial_scan_emitter",
    srcs = ["serial_scan_emitter.cc"],
    hdrs = ["serial_scan_emitter.h"],
    deps = [
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:util",
        "//xla/backends/gpu/codegen/emitters:emitter_base",
        "//xla/codegen/emitters:computation_partitioner",
        "//xla/hlo/analysis:indexing_analysis",
        "//xla/hlo/ir:hlo",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/service/gpu:hlo_fusion_analysis",
        "//xla/service/gpu:launch_dimensions",
        "//xla/tsl/platform:errors",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:TensorDialect",
    ],
)

cc_library(
    name = "loopify_unrolled_slices",
    srcs = ["loopify_unrolled_slices.cc"],
    hdrs = ["loopify_unrolled_slices.h"],
    deps = [
        "//xla:comparison_util",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "loopify_unrolled_slices_test",
    srcs = ["loopify_unrolled_slices_test.cc"],
    deps = [
        ":loopify_unrolled_slices",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_main",
    ],
)
