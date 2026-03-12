# Build rules for custom XLA pass: LoopifyUnrolledSlices
#
# This pass converts unrolled slice→compute→concat patterns into while loops,
# allowing XLA GPU to fuse per-level kernels into a single kernel.
#
# To integrate into XLA:
#   1. Copy this directory into openxla/xla/xla/service/gpu/transforms/
#   2. Add the pass to fusion_pipeline.cc (before PriorityFusion)
#   3. Rebuild JAX with the modified XLA

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla:xla.default.bzl", "xla_cc_test")

package(
    default_visibility = [
        "//xla:friends",
        "//xla/service/gpu:__subpackages__",
    ],
    licenses = ["notice"],
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
