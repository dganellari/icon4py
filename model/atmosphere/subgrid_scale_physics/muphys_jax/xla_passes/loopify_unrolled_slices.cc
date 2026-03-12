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

#include "xla/service/gpu/transforms/loopify/loopify_unrolled_slices.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {

namespace {

// Consecutive slices from the same source, joined by a concatenate.
struct SliceChain {
  HloInstruction* concat;
  // source → ordered slices at positions 0..N-1
  absl::flat_hash_map<HloInstruction*, std::vector<HloInstruction*>>
      source_slices;
  int64_t num_iterations;
  int64_t slice_dim;
};

// Combined iteration body for a group of related concatenates transformed
// into a single while loop.  Uses iteration 1 as the body template because
// XLA's algebraic simplifier may eliminate iter-0 ops (add(0,x)->x, etc.),
// making it structurally incomplete.  Iteration 0 is peeled out.
struct IterationBody {
  std::vector<HloInstruction*> instructions;  // iter-1, topo-sorted
  std::vector<HloInstruction*> sliced_inputs;  // source tensors, deduped
  absl::flat_hash_map<HloInstruction*, HloInstruction*> template_slice_to_source;
  std::vector<HloInstruction*> carry_inputs;   // iter-0 → iter-1 boundary
  std::vector<HloInstruction*> carry_outputs;  // iter-1 → iter-2 boundary
  std::vector<HloInstruction*> level_outputs;  // per-concat iter-1 results
  std::vector<HloInstruction*> iter0_level_outputs;  // for pre-filling accum
  std::vector<HloInstruction*> concats;        // replaced concatenates
  std::vector<HloInstruction*> invariant_inputs;  // constants, params, etc.
  // k=1 slice → level_output for intra-group concat dependencies
  absl::flat_hash_map<HloInstruction*, HloInstruction*>
      intra_loop_slice_to_level_output;
};

// ─────────────────────────────────────────────────────────────────────────────
// Step 1: Find concatenate instructions that join consecutive slices
// ─────────────────────────────────────────────────────────────────────────────

std::optional<SliceChain> AnalyzeConcatenate(HloInstruction* concat,
                                              int min_iterations) {
  if (concat->opcode() != HloOpcode::kConcatenate) return std::nullopt;

  int64_t cat_dim = concat->concatenate_dimension();
  int64_t num_ops = concat->operand_count();

  if (num_ops < min_iterations) return std::nullopt;

  VLOG(3) << "Analyzing concatenate " << concat->name()
          << " with " << num_ops << " operands along dim " << cat_dim;

  absl::flat_hash_map<HloInstruction*, std::vector<HloInstruction*>>
      source_slices;

  // Stop backward trace at carry deps (other iterations' operands)
  absl::flat_hash_set<HloInstruction*> concat_operands;
  for (int64_t i = 0; i < num_ops; ++i) {
    concat_operands.insert(concat->mutable_operand(i));
  }

  for (int64_t i = 0; i < num_ops; ++i) {
    HloInstruction* operand = concat->mutable_operand(i);

    std::vector<HloInstruction*> worklist = {operand};
    absl::flat_hash_set<HloInstruction*> visited;

    while (!worklist.empty()) {
      HloInstruction* inst = worklist.back();
      worklist.pop_back();
      if (!visited.insert(inst).second) continue;

      if (inst->opcode() == HloOpcode::kSlice) {
        auto& starts = inst->slice_starts();
        auto& limits = inst->slice_limits();
        // Only record slices at the expected position for this operand
        if (limits[cat_dim] - starts[cat_dim] == 1 &&
            starts[cat_dim] == i) {
          HloInstruction* source = inst->mutable_operand(0);
          source_slices[source].push_back(inst);
        }
        // Slices are leaf inputs — don't trace further
      } else {
        for (HloInstruction* op : inst->operands()) {
          // Stop at other iterations' operands (carry deps)
          if (op != operand && concat_operands.contains(op)) continue;
          worklist.push_back(op);
        }
      }
    }
  }

  // Need at least one source with slices at all positions
  bool found_complete_source = false;
  for (auto& [source, slices] : source_slices) {
    if (static_cast<int64_t>(slices.size()) >= num_ops) {
      found_complete_source = true;

      std::sort(slices.begin(), slices.end(),
                [cat_dim](HloInstruction* a, HloInstruction* b) {
                  return a->slice_starts()[cat_dim] <
                         b->slice_starts()[cat_dim];
                });

      for (int64_t i = 0; i < num_ops; ++i) {
        if (slices[i]->slice_starts()[cat_dim] != i) {
          found_complete_source = false;
          break;
        }
      }
      if (found_complete_source) break;
    }
  }

  if (!found_complete_source) {
    VLOG(3) << "No complete slice source for " << concat->name();
    return std::nullopt;
  }

  SliceChain chain;
  chain.concat = concat;
  chain.source_slices = std::move(source_slices);
  chain.num_iterations = num_ops;
  chain.slice_dim = cat_dim;

  VLOG(2) << "Found slice chain: " << concat->name()
          << " with " << num_ops << " iterations along dim " << cat_dim
          << " from " << chain.source_slices.size() << " source tensors";

  return chain;
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 2: Extract combined iteration body from a group of chains.
// ─────────────────────────────────────────────────────────────────────────────
// Iter-0 is peeled; iter-1 is the body template (iter-0 may be simplified
// by algebraic passes, making it unsuitable as template).

std::optional<IterationBody> ExtractGroupedIterationBody(
    const std::vector<SliceChain>& chains, HloComputation* computation) {

  if (chains.empty()) return std::nullopt;
  int64_t num_iters = chains[0].num_iterations;
  int64_t slice_dim = chains[0].slice_dim;

  // Need ≥3 iters: 0 (peeled), 1 (template), 2 (carry detection)
  if (num_iters < 3) return std::nullopt;

  // Collect k=0/k=1/k=2 slices across all chains
  absl::flat_hash_set<HloInstruction*> all_k0_slices;
  absl::flat_hash_set<HloInstruction*> all_k1_slices;
  absl::flat_hash_set<HloInstruction*> all_k2_slices;
  absl::flat_hash_set<HloInstruction*> all_concat_operands;

  for (const auto& chain : chains) {
    for (int64_t i = 0; i < num_iters; ++i) {
      all_concat_operands.insert(chain.concat->mutable_operand(i));
    }
    for (auto& [source, slices] : chain.source_slices) {
      if (!slices.empty() && slices[0]->slice_starts()[slice_dim] == 0) {
        all_k0_slices.insert(slices[0]);
      }
      if (slices.size() > 1 && slices[1]->slice_starts()[slice_dim] == 1) {
        all_k1_slices.insert(slices[1]);
      }
      if (slices.size() > 2 && slices[2]->slice_starts()[slice_dim] == 2) {
        all_k2_slices.insert(slices[2]);
      }
    }
  }

  if (all_k0_slices.empty() || all_k1_slices.empty()) return std::nullopt;

  // ── Phase A: Build iteration_0_set for carry boundary detection ──
  absl::flat_hash_set<HloInstruction*> iteration_0_set;
  {
    std::vector<HloInstruction*> worklist;
    for (const auto& chain : chains) {
      worklist.push_back(chain.concat->mutable_operand(0));
    }
    absl::flat_hash_set<HloInstruction*> visited;
    while (!worklist.empty()) {
      HloInstruction* inst = worklist.back();
      worklist.pop_back();
      if (!visited.insert(inst).second) continue;
      iteration_0_set.insert(inst);
      for (HloInstruction* op : inst->operands()) {
        if (all_k0_slices.contains(op)) {
          iteration_0_set.insert(op);
          continue;
        }
        if (op->opcode() == HloOpcode::kParameter ||
            op->opcode() == HloOpcode::kConstant) {
          continue;
        }
        // Stop at concat operands from other iterations
        if (all_concat_operands.contains(op) &&
            !iteration_0_set.contains(op)) {
          bool is_k0_output = false;
          for (const auto& chain : chains) {
            if (chain.concat->mutable_operand(0) == op) {
              is_k0_output = true;
              break;
            }
          }
          if (!is_k0_output) continue;
        }
        worklist.push_back(op);
      }
    }
  }

  // Forward-propagate k=0 slice dependency
  absl::flat_hash_set<HloInstruction*> depends_on_k0_slice;
  {
    std::vector<HloInstruction*> worklist(all_k0_slices.begin(),
                                           all_k0_slices.end());
    while (!worklist.empty()) {
      HloInstruction* inst = worklist.back();
      worklist.pop_back();
      if (!depends_on_k0_slice.insert(inst).second) continue;
      for (HloInstruction* user : inst->users()) {
        if (iteration_0_set.contains(user)) {
          worklist.push_back(user);
        }
      }
    }
  }

  // Prune to slice-dependent only
  {
    absl::flat_hash_set<HloInstruction*> pruned;
    for (HloInstruction* inst : iteration_0_set) {
      if (depends_on_k0_slice.contains(inst)) {
        pruned.insert(inst);
      }
    }
    iteration_0_set = std::move(pruned);
  }

  // ── Phase B: Find iter-0 carry outputs ──
  // Backward trace from concat operand[1] into iter-0.
  std::vector<HloInstruction*> iter0_carry_outputs;
  absl::flat_hash_set<HloInstruction*> iter0_carry_output_set;
  {
    std::vector<HloInstruction*> worklist;
    for (const auto& chain : chains) {
      worklist.push_back(chain.concat->mutable_operand(1));
    }
    absl::flat_hash_set<HloInstruction*> visited;
    while (!worklist.empty()) {
      HloInstruction* inst = worklist.back();
      worklist.pop_back();
      if (!visited.insert(inst).second) continue;
      if (all_k1_slices.contains(inst)) continue;

      if (iteration_0_set.contains(inst) &&
          depends_on_k0_slice.contains(inst)) {
        // Distinguish carry outputs (e.g., or(False, mask_k0)) from
        // shifted inputs (rho[k-1]).  Only k=0 slices feeding an OR
        // outside iter-0 are carries.
        if (all_k0_slices.contains(inst)) {
          bool is_activated_carry = false;
          for (HloInstruction* user : inst->users()) {
            if (!iteration_0_set.contains(user) &&
                user->opcode() == HloOpcode::kOr) {
              is_activated_carry = true;
              break;
            }
          }
          if (!is_activated_carry) {
            continue;  // Shifted input (rho/vc at k-1), not a carry
          }
        }
        if (iter0_carry_output_set.insert(inst).second) {
          iter0_carry_outputs.push_back(inst);
        }
        continue;  // Don't trace into iter-0
      }

      for (HloInstruction* op : inst->operands()) {
        worklist.push_back(op);
      }
    }
  }

  VLOG(2) << "Phase A: iteration_0_set=" << iteration_0_set.size()
          << " iter0_carry_outputs=" << iter0_carry_outputs.size();

  // ── Phase C: Build iteration_1_set (body template) ──
  // Backward from operand[1], stop at k=1 slices / carries / constants.
  absl::flat_hash_set<HloInstruction*> iteration_1_set;
  {
    std::vector<HloInstruction*> worklist;
    for (const auto& chain : chains) {
      worklist.push_back(chain.concat->mutable_operand(1));
    }
    absl::flat_hash_set<HloInstruction*> visited;
    while (!worklist.empty()) {
      HloInstruction* inst = worklist.back();
      worklist.pop_back();
      if (!visited.insert(inst).second) continue;
      iteration_1_set.insert(inst);
      for (HloInstruction* op : inst->operands()) {
        if (all_k1_slices.contains(op)) {
          iteration_1_set.insert(op);
          continue;  // Stop at k=1 slices
        }
        if (op->opcode() == HloOpcode::kParameter ||
            op->opcode() == HloOpcode::kConstant) {
          continue;
        }
        // Stop at iter-0 carry outputs (these become carry_inputs)
        if (iter0_carry_output_set.contains(op)) {
          continue;
        }
        // Stop at concat operands from other iterations
        if (all_concat_operands.contains(op) &&
            !iteration_1_set.contains(op)) {
          bool is_k1_output = false;
          for (const auto& chain : chains) {
            if (chain.concat->mutable_operand(1) == op) {
              is_k1_output = true;
              break;
            }
          }
          if (!is_k1_output) continue;
        }
        worklist.push_back(op);
      }
    }
  }

  // Compute which iter-1 instructions depend on k=1 slices
  absl::flat_hash_set<HloInstruction*> depends_on_k1_slice;
  {
    std::vector<HloInstruction*> worklist(all_k1_slices.begin(),
                                           all_k1_slices.end());
    while (!worklist.empty()) {
      HloInstruction* inst = worklist.back();
      worklist.pop_back();
      if (!depends_on_k1_slice.insert(inst).second) continue;
      for (HloInstruction* user : inst->users()) {
        if (iteration_1_set.contains(user)) {
          worklist.push_back(user);
        }
      }
    }
  }

  // Also mark instructions that depend on carry inputs as slice-dependent
  // (carries transitively depend on earlier slices through iter-0)
  {
    std::vector<HloInstruction*> worklist;
    for (HloInstruction* ci : iter0_carry_outputs) {
      for (HloInstruction* user : ci->users()) {
        if (iteration_1_set.contains(user)) {
          worklist.push_back(user);
        }
      }
    }
    while (!worklist.empty()) {
      HloInstruction* inst = worklist.back();
      worklist.pop_back();
      if (!depends_on_k1_slice.insert(inst).second) continue;
      for (HloInstruction* user : inst->users()) {
        if (iteration_1_set.contains(user)) {
          worklist.push_back(user);
        }
      }
    }
  }

  // Also propagate from shifted inputs: k=0 slices from sliced sources
  // that are not carries.  These read position k-1 each iteration, so
  // dependent computations must stay in the loop body.
  {
    absl::flat_hash_set<HloInstruction*> source_set;
    for (const auto& chain : chains) {
      for (auto& [source, slices] : chain.source_slices) {
        if (static_cast<int64_t>(slices.size()) >= num_iters) {
          source_set.insert(source);
        }
      }
    }

    std::vector<HloInstruction*> worklist;
    for (HloInstruction* inst : iteration_1_set) {
      for (HloInstruction* op : inst->operands()) {
        if (op->opcode() == HloOpcode::kSlice &&
            op->slice_starts()[slice_dim] == 0 &&
            (op->slice_limits()[slice_dim] -
             op->slice_starts()[slice_dim]) == 1 &&
            source_set.contains(op->mutable_operand(0)) &&
            !iter0_carry_output_set.contains(op)) {
          worklist.push_back(inst);
          break;
        }
      }
    }

    while (!worklist.empty()) {
      HloInstruction* inst = worklist.back();
      worklist.pop_back();
      if (!depends_on_k1_slice.insert(inst).second) continue;
      for (HloInstruction* user : inst->users()) {
        if (iteration_1_set.contains(user)) {
          worklist.push_back(user);
        }
      }
    }
  }

  // Prune iteration_1_set to slice/carry/shifted-input-dependent instructions
  {
    absl::flat_hash_set<HloInstruction*> pruned;
    for (HloInstruction* inst : iteration_1_set) {
      if (depends_on_k1_slice.contains(inst)) {
        pruned.insert(inst);
      }
    }
    iteration_1_set = std::move(pruned);
  }

  // ── Phase D: Find carry outputs of iter-1 (backward from operand[2]) ──
  std::vector<HloInstruction*> iter1_carry_outputs;
  absl::flat_hash_set<HloInstruction*> iter1_carry_output_set;
  {
    std::vector<HloInstruction*> worklist;
    for (const auto& chain : chains) {
      if (num_iters >= 3) {
        worklist.push_back(chain.concat->mutable_operand(2));
      }
    }
    absl::flat_hash_set<HloInstruction*> visited;
    while (!worklist.empty()) {
      HloInstruction* inst = worklist.back();
      worklist.pop_back();
      if (!visited.insert(inst).second) continue;
      if (all_k2_slices.contains(inst)) continue;

      if (iteration_1_set.contains(inst) && !all_k1_slices.contains(inst) &&
          depends_on_k1_slice.contains(inst)) {
        if (iter1_carry_output_set.insert(inst).second) {
          iter1_carry_outputs.push_back(inst);
        }
        continue;
      }

      for (HloInstruction* op : inst->operands()) {
        worklist.push_back(op);
      }
    }
  }

  VLOG(2) << "Phase C: iteration_1_set=" << iteration_1_set.size()
          << " Phase D: iter1_carry_outputs=" << iter1_carry_outputs.size();

  // ── Phase E: Pair carry inputs with carry outputs (forward BFS) ──
  IterationBody body;
  absl::flat_hash_set<HloInstruction*> iter1_carry_output_set_local(
      iter1_carry_outputs.begin(), iter1_carry_outputs.end());
  absl::flat_hash_set<HloInstruction*> claimed_carry_outputs;

  for (HloInstruction* ci : iter0_carry_outputs) {
    std::queue<HloInstruction*> q;
    for (HloInstruction* user : ci->users()) {
      if (iteration_1_set.contains(user)) {
        q.push(user);
      }
    }
    absl::flat_hash_set<HloInstruction*> seen;
    HloInstruction* matched_co = nullptr;
    while (!q.empty() && matched_co == nullptr) {
      HloInstruction* inst = q.front();
      q.pop();
      if (!seen.insert(inst).second) continue;
      if (iter1_carry_output_set_local.contains(inst) &&
          !claimed_carry_outputs.contains(inst)) {
        matched_co = inst;
        break;
      }
      for (HloInstruction* user : inst->users()) {
        if (iteration_1_set.contains(user)) {
          q.push(user);
        }
      }
    }
    if (matched_co != nullptr) {
      claimed_carry_outputs.insert(matched_co);
      body.carry_inputs.push_back(ci);
      body.carry_outputs.push_back(matched_co);
      VLOG(2) << "carry pair: " << ci->name() << " -> " << matched_co->name();
    } else {
      VLOG(2) << "carry_in=" << ci->name() << " has no matching carry output";
    }
  }

  // Topo-sort iteration-1 instructions
  for (HloInstruction* inst : computation->MakeInstructionPostOrder()) {
    if (iteration_1_set.contains(inst)) {
      body.instructions.push_back(inst);
    }
  }

  absl::flat_hash_set<HloInstruction*> concat_set;
  absl::flat_hash_map<HloInstruction*, HloInstruction*> concat_to_level_output;
  for (const auto& chain : chains) {
    concat_set.insert(chain.concat);
    concat_to_level_output[chain.concat] = chain.concat->mutable_operand(1);
  }

  absl::flat_hash_set<HloInstruction*> seen_sources;
  for (const auto& chain : chains) {
    for (auto& [source, slices] : chain.source_slices) {
      if (concat_set.contains(source)) {
        // Intra-loop dep: use level_output instead of slicing the concat
        if (slices.size() > 1 && slices[1]->slice_starts()[slice_dim] == 1) {
          auto level_it = concat_to_level_output.find(source);
          if (level_it != concat_to_level_output.end()) {
            body.intra_loop_slice_to_level_output[slices[1]] =
                level_it->second;
            VLOG(2) << "Intra-loop slice: " << slices[1]->name()
                    << " -> level_output " << level_it->second->name()
                    << " (source concat: " << source->name() << ")";
          }
        }
        continue;
      }
      if (seen_sources.insert(source).second) {
        body.sliced_inputs.push_back(source);
      }
      if (slices.size() > 1 && slices[1]->slice_starts()[slice_dim] == 1) {
        body.template_slice_to_source[slices[1]] = source;
      }
    }
  }

  for (const auto& chain : chains) {
    body.level_outputs.push_back(chain.concat->mutable_operand(1));
    body.iter0_level_outputs.push_back(chain.concat->mutable_operand(0));
    body.concats.push_back(chain.concat);
  }

  // Find invariant inputs (outside iter_1_set, not carries/sources/concats).
  // Exclude k=0 slices from sliced sources — those are shifted inputs
  // (e.g., rho[k-1]) handled via dynamic_slice(source, counter-1).
  absl::flat_hash_set<HloInstruction*> carry_input_set(
      body.carry_inputs.begin(), body.carry_inputs.end());
  absl::flat_hash_set<HloInstruction*> sliced_input_set(
      body.sliced_inputs.begin(), body.sliced_inputs.end());
  absl::flat_hash_set<HloInstruction*> invariant_set;
  for (HloInstruction* inst : body.instructions) {
    for (HloInstruction* op : inst->operands()) {
      if (!iteration_1_set.contains(op) && !carry_input_set.contains(op) &&
          !concat_set.contains(op) && !sliced_input_set.contains(op)) {
        // Shifted inputs (k=0 slices from sliced sources) are handled
        // separately in BuildWhileLoop as dynamic_slice(source, counter-1).
        if (op->opcode() == HloOpcode::kSlice &&
            op->slice_starts()[slice_dim] == 0 &&
            (op->slice_limits()[slice_dim] -
             op->slice_starts()[slice_dim]) == 1 &&
            sliced_input_set.contains(op->mutable_operand(0))) {
          continue;
        }
        if (invariant_set.insert(op).second) {
          body.invariant_inputs.push_back(op);
        }
      }
    }
  }

  VLOG(1) << "Body: " << body.instructions.size() << " insts, "
          << body.sliced_inputs.size() << " sliced, "
          << body.carry_inputs.size() << " carry, "
          << body.level_outputs.size() << " outputs, "
          << body.invariant_inputs.size() << " invariant";

  return body;
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 3: Build the while loop for a group of concatenates
// ─────────────────────────────────────────────────────────────────────────────

absl::StatusOr<bool> BuildWhileLoop(
    HloComputation* computation,
    const std::vector<SliceChain>& chains,
    const IterationBody& body) {

  int64_t num_iters = chains[0].num_iterations;
  int64_t slice_dim = chains[0].slice_dim;

  VLOG(1) << "Building while loop for " << chains.size() << " concatenates"
          << " (" << num_iters << " iterations, peeling iter-0)";

  int num_carry = body.carry_outputs.size();
  int num_accum = body.level_outputs.size();
  int num_invariant = body.sliced_inputs.size() + body.invariant_inputs.size();
  int tuple_size = 1 + num_carry + num_accum + num_invariant;

  VLOG(2) << "While tuple: 1 counter + " << num_carry << " carry + "
          << num_accum << " accumulators + " << num_invariant << " invariant = "
          << tuple_size;

  std::vector<Shape> tuple_shapes;
  tuple_shapes.push_back(ShapeUtil::MakeShape(S64, {}));  // counter
  for (HloInstruction* carry : body.carry_outputs) {
    tuple_shapes.push_back(carry->shape());
  }
  for (int i = 0; i < num_accum; ++i) {
    tuple_shapes.push_back(body.concats[i]->shape());
  }
  for (HloInstruction* src : body.sliced_inputs) {
    tuple_shapes.push_back(src->shape());
  }
  for (HloInstruction* inv : body.invariant_inputs) {
    tuple_shapes.push_back(inv->shape());
  }

  Shape tuple_shape = ShapeUtil::MakeTupleShape(tuple_shapes);

  // Counter starts at 1 (iter-0 is peeled)
  HloInstruction* one_init = computation->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int64_t>(1)));

  std::vector<HloInstruction*> init_values;
  init_values.push_back(one_init);

  // Carry initial values = peeled iter-0 carry outputs
  for (HloInstruction* carry_in : body.carry_inputs) {
    init_values.push_back(carry_in);
  }

  // Accumulators: zeros + iter-0 result at position 0
  HloInstruction* zero_idx = computation->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int64_t>(0)));

  for (int i = 0; i < num_accum; ++i) {
    const Shape& accum_shape = body.concats[i]->shape();
    PrimitiveType elem_type = accum_shape.element_type();

    HloInstruction* zero_const = computation->AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::Zero(elem_type)));
    HloInstruction* zeros_accum = computation->AddInstruction(
        HloInstruction::CreateBroadcast(accum_shape, zero_const, {}));

    HloInstruction* iter0_out = body.iter0_level_outputs[i];

    int64_t rank = accum_shape.dimensions_size();
    std::vector<HloInstruction*> dus_indices;
    for (int64_t d = 0; d < rank; ++d) {
      dus_indices.push_back(zero_idx);  // All start at 0 for iter-0
    }

    HloInstruction* accum_with_iter0 = computation->AddInstruction(
        HloInstruction::CreateDynamicUpdateSlice(
            accum_shape, zeros_accum, iter0_out, dus_indices));

    init_values.push_back(accum_with_iter0);
  }

  for (HloInstruction* src : body.sliced_inputs) {
    init_values.push_back(src);
  }
  for (HloInstruction* inv : body.invariant_inputs) {
    init_values.push_back(inv);
  }

  HloInstruction* init_tuple = computation->AddInstruction(
      HloInstruction::CreateTuple(init_values));

  // Condition: counter < num_iters
  HloComputation::Builder cond_builder("while_cond");
  HloInstruction* cond_param = cond_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "param"));
  HloInstruction* cond_counter = cond_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(cond_param, 0));
  HloInstruction* cond_limit = cond_builder.AddInstruction(
      HloInstruction::CreateConstant(
          LiteralUtil::CreateR0<int64_t>(num_iters)));
  cond_builder.AddInstruction(HloInstruction::CreateCompare(
      ShapeUtil::MakeShape(PRED, {}), cond_counter, cond_limit,
      ComparisonDirection::kLt));
  HloComputation* cond_comp =
      computation->parent()->AddEmbeddedComputation(cond_builder.Build());

  // Body computation
  HloComputation::Builder body_builder("while_body");
  HloInstruction* body_param = body_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "param"));

  HloInstruction* counter = body_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(body_param, 0));

  std::vector<HloInstruction*> carry_values;
  for (int i = 0; i < num_carry; ++i) {
    carry_values.push_back(body_builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(body_param, 1 + i)));
  }

  std::vector<HloInstruction*> accumulators;
  for (int i = 0; i < num_accum; ++i) {
    accumulators.push_back(body_builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(body_param,
                                               1 + num_carry + i)));
  }

  int inv_offset = 1 + num_carry + num_accum;
  std::vector<HloInstruction*> sliced_inputs_in_body;
  for (int i = 0; i < static_cast<int>(body.sliced_inputs.size()); ++i) {
    sliced_inputs_in_body.push_back(body_builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(body_param, inv_offset + i)));
  }

  std::vector<HloInstruction*> invariant_in_body;
  int other_inv_offset = inv_offset + body.sliced_inputs.size();
  for (int i = 0; i < static_cast<int>(body.invariant_inputs.size()); ++i) {
    invariant_in_body.push_back(body_builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(body_param,
                                               other_inv_offset + i)));
  }

  HloInstruction* zero_body = body_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int64_t>(0)));

  absl::flat_hash_map<HloInstruction*, int> source_to_idx;
  for (int i = 0; i < static_cast<int>(body.sliced_inputs.size()); ++i) {
    source_to_idx[body.sliced_inputs[i]] = i;
  }

  // Replace k=1 template slices with dynamic_slice(source, counter)
  absl::flat_hash_map<HloInstruction*, HloInstruction*> slice_replacement;
  for (const auto& chain : chains) {
    for (auto& [source, slices] : chain.source_slices) {
      if (slices.size() < 2) continue;
      HloInstruction* k1_slice = slices[1];
      if (slice_replacement.contains(k1_slice)) continue;

      if (body.intra_loop_slice_to_level_output.contains(k1_slice)) continue;

      auto src_it = source_to_idx.find(source);
      if (src_it == source_to_idx.end()) continue;
      HloInstruction* source_in_body = sliced_inputs_in_body[src_it->second];

      int64_t rank = source->shape().dimensions_size();
      std::vector<HloInstruction*> start_indices;
      for (int64_t d = 0; d < rank; ++d) {
        if (d == slice_dim) {
          start_indices.push_back(counter);
        } else {
          start_indices.push_back(zero_body);
        }
      }

      std::vector<int64_t> slice_sizes;
      for (int64_t d = 0; d < rank; ++d) {
        slice_sizes.push_back(
            k1_slice->slice_limits()[d] - k1_slice->slice_starts()[d]);
      }

      HloInstruction* dyn_slice = body_builder.AddInstruction(
          HloInstruction::CreateDynamicSlice(
              k1_slice->shape(), source_in_body, start_indices, slice_sizes));

      slice_replacement[k1_slice] = dyn_slice;
    }
  }

  // Clone the iteration body
  absl::flat_hash_map<HloInstruction*, HloInstruction*> cloned;
  for (auto& [orig, replacement] : slice_replacement) {
    cloned[orig] = replacement;
  }

  for (int i = 0; i < static_cast<int>(body.invariant_inputs.size()); ++i) {
    cloned[body.invariant_inputs[i]] = invariant_in_body[i];
  }

  for (int i = 0; i < num_carry; ++i) {
    cloned[body.carry_inputs[i]] = carry_values[i];
  }

  // Shifted inputs: k=0 slices → dynamic_slice(source, counter-1)
  HloInstruction* one_shift = body_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int64_t>(1)));
  HloInstruction* counter_minus_1 = body_builder.AddInstruction(
      HloInstruction::CreateBinary(
          ShapeUtil::MakeShape(S64, {}), HloOpcode::kSubtract,
          counter, one_shift));

  for (HloInstruction* inst : body.instructions) {
    for (int oi = 0; oi < inst->operand_count(); ++oi) {
      HloInstruction* op = inst->mutable_operand(oi);
      if (cloned.contains(op)) continue;
      if (op->opcode() != HloOpcode::kSlice) continue;

      if (op->slice_starts()[slice_dim] != 0) continue;
      if (op->slice_limits()[slice_dim] - op->slice_starts()[slice_dim] != 1)
        continue;

      HloInstruction* source = op->mutable_operand(0);
      auto src_it = source_to_idx.find(source);
      if (src_it == source_to_idx.end()) continue;

      HloInstruction* source_in_body = sliced_inputs_in_body[src_it->second];
      int64_t rank = source->shape().dimensions_size();
      std::vector<HloInstruction*> start_indices;
      for (int64_t d = 0; d < rank; ++d) {
        start_indices.push_back(d == slice_dim ? counter_minus_1 : zero_body);
      }
      std::vector<int64_t> ss;
      for (int64_t d = 0; d < rank; ++d) {
        ss.push_back(op->slice_limits()[d] - op->slice_starts()[d]);
      }

      HloInstruction* dyn_slice = body_builder.AddInstruction(
          HloInstruction::CreateDynamicSlice(
              op->shape(), source_in_body, start_indices, ss));
      cloned[op] = dyn_slice;
      VLOG(2) << "shifted input: " << op->name() << " from " << source->name();
    }
  }

  // Intra-loop slices: map level_output → template slices that read it
  absl::flat_hash_map<HloInstruction*, std::vector<HloInstruction*>>
      level_output_to_template_slices;
  for (auto& [template_slice, level_out] :
       body.intra_loop_slice_to_level_output) {
    level_output_to_template_slices[level_out].push_back(template_slice);
  }

  for (HloInstruction* inst : body.instructions) {
    if (cloned.contains(inst)) continue;

    std::vector<HloInstruction*> new_operands;
    for (int oi = 0; oi < inst->operand_count(); ++oi) {
      HloInstruction* op = inst->mutable_operand(oi);
      auto it = cloned.find(op);
      if (it != cloned.end()) {
        new_operands.push_back(it->second);
      } else if (op->opcode() == HloOpcode::kConstant ||
                 (op->opcode() == HloOpcode::kBroadcast &&
                  op->operand_count() > 0 &&
                  op->mutable_operand(0)->opcode() == HloOpcode::kConstant)) {
        HloInstruction* cloned_const =
            body_builder.AddInstruction(op->Clone());
        cloned[op] = cloned_const;
        new_operands.push_back(cloned_const);
      } else {
        LOG(ERROR) << "BuildWhileLoop: unmapped operand " << op->name()
                   << " (" << HloOpcodeString(op->opcode()) << ")"
                   << " for " << inst->name();
        return false;
      }
    }

    cloned[inst] = body_builder.AddInstruction(
        inst->CloneWithNewOperands(inst->shape(), new_operands));

    auto lo_it = level_output_to_template_slices.find(inst);
    if (lo_it != level_output_to_template_slices.end()) {
      for (HloInstruction* template_slice : lo_it->second) {
        cloned[template_slice] = cloned[inst];
        VLOG(2) << "Mapped intra-loop template_slice " << template_slice->name()
                << " -> cloned level_output " << cloned[inst]->name();
      }
    }
  }

  // Write results into accumulators
  std::vector<HloInstruction*> new_accumulators;
  for (int i = 0; i < num_accum; ++i) {
    HloInstruction* level_out = cloned[body.level_outputs[i]];
    HloInstruction* accum = accumulators[i];

    int64_t rank = accum->shape().dimensions_size();
    std::vector<HloInstruction*> update_indices;
    for (int64_t d = 0; d < rank; ++d) {
      if (d == slice_dim) {
        update_indices.push_back(counter);
      } else {
        update_indices.push_back(zero_body);
      }
    }

    HloInstruction* updated = body_builder.AddInstruction(
        HloInstruction::CreateDynamicUpdateSlice(
            accum->shape(), accum, level_out, update_indices));
    new_accumulators.push_back(updated);
  }

  // k++
  HloInstruction* one = body_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int64_t>(1)));
  HloInstruction* next_counter = body_builder.AddInstruction(
      HloInstruction::CreateBinary(
          ShapeUtil::MakeShape(S64, {}), HloOpcode::kAdd, counter, one));

  // Output tuple
  std::vector<HloInstruction*> output_values;
  output_values.push_back(next_counter);

  for (int i = 0; i < num_carry; ++i) {
    output_values.push_back(cloned[body.carry_outputs[i]]);
  }
  for (auto* accum : new_accumulators) {
    output_values.push_back(accum);
  }
  for (auto* src : sliced_inputs_in_body) {
    output_values.push_back(src);
  }
  for (auto* inv : invariant_in_body) {
    output_values.push_back(inv);
  }

  body_builder.AddInstruction(HloInstruction::CreateTuple(output_values));

  HloComputation* body_comp =
      computation->parent()->AddEmbeddedComputation(body_builder.Build());

  // Create while and extract results
  HloInstruction* while_inst = computation->AddInstruction(
      HloInstruction::CreateWhile(tuple_shape, cond_comp, body_comp,
                                   init_tuple));

  // Replace concatenates with GTE from while loop
  for (int i = 0; i < num_accum; ++i) {
    int accum_idx = 1 + num_carry + i;
    HloInstruction* result = computation->AddInstruction(
        HloInstruction::CreateGetTupleElement(while_inst, accum_idx));

    TF_RETURN_IF_ERROR(body.concats[i]->ReplaceAllUsesWith(result));
    VLOG(2) << "Replaced " << body.concats[i]->name()
            << " with GTE(" << accum_idx << ") from while loop";
  }

  // Remove dead concat trees
  for (HloInstruction* concat : body.concats) {
    TF_RETURN_IF_ERROR(
        computation->RemoveInstructionAndUnusedOperands(concat));
  }

  VLOG(1) << "Successfully created while loop replacing "
          << chains.size() << " concatenates (iter-0 peeled, dead ops removed)";
  return true;
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// Main pass entry point
// ─────────────────────────────────────────────────────────────────────────────

absl::StatusOr<bool> LoopifyUnrolledSlices::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {

  bool changed = false;

  for (HloComputation* computation : module->computations(execution_threads)) {
    // Only transform the entry computation.
    if (!computation->IsEntryComputation()) {
      continue;
    }

    // Step 1: Find all matching slice chains
    std::vector<SliceChain> all_chains;
    for (HloInstruction* inst : computation->instructions()) {
      if (inst->opcode() == HloOpcode::kConcatenate) {
        auto chain = AnalyzeConcatenate(inst, min_iterations_);
        if (chain.has_value()) {
          all_chains.push_back(std::move(*chain));
        }
      }
    }

    if (all_chains.empty()) continue;

    VLOG(1) << "Found " << all_chains.size() << " slice chains in "
            << computation->name();

    // Step 2: Group chains by (num_iterations, slice_dim)
    absl::flat_hash_map<std::string, std::vector<int>> groups;
    for (int i = 0; i < static_cast<int>(all_chains.size()); ++i) {
      std::string key = absl::StrCat(all_chains[i].num_iterations, "_",
                                      all_chains[i].slice_dim);
      groups[key].push_back(i);
    }

    for (auto& [key, indices] : groups) {
      std::vector<SliceChain> group_chains;
      for (int idx : indices) {
        group_chains.push_back(std::move(all_chains[idx]));
      }

      VLOG(1) << "Processing group: " << group_chains.size()
              << " chains (" << key << ")";

      auto body = ExtractGroupedIterationBody(group_chains, computation);
      if (!body.has_value()) {
        VLOG(2) << "Could not extract grouped iteration body";
        continue;
      }

      TF_ASSIGN_OR_RETURN(bool transformed,
                           BuildWhileLoop(computation, group_chains, *body));
      if (transformed) {
        changed = true;
      }
    }
  }

  return changed;
}

}  // namespace gpu
}  // namespace xla
