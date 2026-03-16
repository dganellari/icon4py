/* Copyright 2024 ETH Zurich and MeteoSwiss.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef XLA_PASSES_LOOPIFY_UNROLLED_SLICES_H_
#define XLA_PASSES_LOOPIFY_UNROLLED_SLICES_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {
namespace gpu {

// Rewrites unrolled slice→compute→concat chains into while loops.
//
// Matches:
//   slice(input,[k:k+1]) → f(slice_k, carry) → concat(r0,r1,...,r89)
// Emits:
//   while(k < N) { dynamic-slice → f → dynamic-update-slice }
//
// Runs before PriorityFusion so the loop body gets fused into few kernels
// (~6 instead of ~186 for 90-level precipitation scans).
class LoopifyUnrolledSlices : public HloModulePass {
 public:
  // min_iterations: minimum number of consecutive slices to trigger
  //                 transformation (default 4, set to 2 for testing)
  // unroll_factor:  number of levels processed per while iteration
  //                 (default 1; use e.g. 10 to reduce loop overhead 10×)
  explicit LoopifyUnrolledSlices(int min_iterations = 4,
                                  int unroll_factor = 10)
      : min_iterations_(min_iterations),
        unroll_factor_(unroll_factor) {}

  absl::string_view name() const override {
    return "loopify-unrolled-slices";
  }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads)
      override;

 private:
  int min_iterations_;
  int unroll_factor_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_PASSES_LOOPIFY_UNROLLED_SLICES_H_
