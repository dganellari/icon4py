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

#ifndef XLA_PASSES_SERIAL_SCAN_EMITTER_H_
#define XLA_PASSES_SERIAL_SCAN_EMITTER_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "xla/backends/gpu/codegen/emitters/emitter_base.h"
#include "xla/codegen/emitters/computation_partitioner.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/launch_dimensions.h"

namespace xla {
namespace gpu {

// A deserialized instruction from the serialized body computation.
struct SerializedInst {
  std::string opcode;        // "add", "P0", "K", "T", "CMP", "CVT", "pass", etc.
  std::vector<int> operands; // indices into instruction list
  PrimitiveType type;        // element type
  double constant_value;     // for constants ("K")
  std::string extra;         // comparison direction for "CMP"
};

// Metadata parsed from the fusion's backend config custom_fusion_config.name.
//
// Format: "serial_scan;nlev=90;sd=1;nc=2;no=4;ns=8;noff=2;ni=3;offvals=1;
//          body_ops=<serialized>"
//
// Fusion parameter layout (in order):
//   [0, nc)                          : carry_inits      [C, 1]
//   [nc, nc+no)                      : iter0_level_outs  [C, 1]
//   [nc+no, nc+no+ns)                : sliced_inputs     [C, L]
//   [nc+no+ns, nc+no+ns+noff)        : offset_sources    [C, L]
//   [nc+no+ns+noff, nc+no+ns+noff+ni): invariants        various
//
// Fusion output: no tensors of shape [C, L] (the filled accumulators).
struct SerialScanConfig {
  int64_t num_levels;    // Number of scan levels (e.g. 90)
  int64_t slice_dim;     // Dimension along which levels are indexed (0 or 1)
  int64_t cell_dim;      // The parallel dimension (the other one)
  int num_carries;       // nc
  int num_outputs;       // no  (= number of accumulator outputs)
  int num_sliced;        // ns
  int num_offset;        // noff
  int num_invariants;    // ni
  std::vector<int64_t> offset_values;   // one per offset source
  std::vector<SerializedInst> body_ops; // deserialized body computation

  // Parameter index helpers.
  int carry_start() const { return 0; }
  int iter0_start() const { return num_carries; }
  int sliced_start() const { return num_carries + num_outputs; }
  int offset_start() const { return num_carries + num_outputs + num_sliced; }
  int invariant_start() const {
    return num_carries + num_outputs + num_sliced + num_offset;
  }
  int total_params() const {
    return num_carries + num_outputs + num_sliced + num_offset + num_invariants;
  }
};

// Emitter for serial scan fusions.
//
// Generates a single GPU kernel with:
//   scf.forall  (parallel over cells, one thread per cell)
//     scf.for   (serial over levels, carries as iter_args)
//       per-level body computation (translated HLO -> scalar MLIR ops)
//       tensor.insert to write level outputs into accumulator tensors
//
// This replaces the WhileThunk approach (one kernel launch per iteration)
// with a single kernel that loops internally.
class SerialScanFusion final : public EmitterBase {
 public:
  explicit SerialScanFusion(const HloFusionAnalysis& analysis)
      : analysis_(analysis) {}

  LaunchDimensions launch_dimensions() const override;

  std::optional<IndexingMap> ComputeThreadIdToOutputIndexing(
      int64_t root_index, mlir::MLIRContext* mlir_context) const override;

  std::optional<IndexingMap> ComputeThreadIdToInputIndexing(
      int64_t root_index, int64_t hero_operand_index,
      mlir::MLIRContext* mlir_context) const override;

 protected:
  absl::Status EmitEntryFunction(
      const emitters::PartitionedComputations& computations,
      const emitters::CallTargetProvider& call_targets,
      mlir::func::FuncOp entry_function,
      const HloFusionInstruction& fusion) const override;

 private:
  static absl::StatusOr<SerialScanConfig> ParseConfig(
      const HloFusionInstruction& fusion);

  // Translate a single scalar HLO instruction to an MLIR op.
  static absl::StatusOr<mlir::Value> EmitScalarOp(
      mlir::ImplicitLocOpBuilder& builder,
      const HloInstruction* inst,
      const absl::flat_hash_map<const HloInstruction*, mlir::Value>& value_map);

  // Translate a serialized instruction to an MLIR op.
  static absl::StatusOr<mlir::Value> EmitSerializedOp(
      mlir::ImplicitLocOpBuilder& builder,
      const SerializedInst& inst,
      const std::vector<mlir::Value>& values);

  const HloFusionAnalysis& analysis_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_PASSES_SERIAL_SCAN_EMITTER_H_
