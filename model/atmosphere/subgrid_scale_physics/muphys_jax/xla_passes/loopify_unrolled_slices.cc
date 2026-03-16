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

// A "SliceChain" is a sequence of concatenated slices from the same source
// tensor, where each slice operates on a consecutive position along dim 0.
struct SliceChain {
  HloInstruction* concat;
  absl::flat_hash_map<HloInstruction*, std::vector<HloInstruction*>>
      source_slices;
  int64_t num_iterations;
  int64_t slice_dim;
};

// Combined iteration body for a GROUP of related concatenates.
// Uses iteration-1 as the body template (not iter-0, which may be
// constant-folded by XLA and have fewer instructions).
struct IterationBody {
  // Iter-1 template instructions, topologically sorted
  std::vector<HloInstruction*> instructions;

  // Source tensors that get sliced (loop-invariant)
  std::vector<HloInstruction*> sliced_inputs;

  // Map from each k=1 slice to its source tensor
  absl::flat_hash_map<HloInstruction*, HloInstruction*> k1_slice_to_source;

  // Structurally matched carry pairs.
  // carry_inputs[i]  = iter-0 value or k=0 slice (initial carry value)
  // carry_outputs[i] = iter-1 value or k=1 slice (updated carry value)
  // carry_is_shifted[i] = true if this is a k=0/k=1 slice pair (shifted carry)
  std::vector<HloInstruction*> carry_inputs;
  std::vector<HloInstruction*> carry_outputs;
  std::vector<bool> carry_is_shifted;

  // Per-chain level outputs
  std::vector<HloInstruction*> iter0_level_outputs;  // for accumulator prefill
  std::vector<HloInstruction*> iter1_level_outputs;  // body template outputs
  std::vector<HloInstruction*> concats;

  // Invariant inputs (constants, broadcasts, etc.)
  std::vector<HloInstruction*> invariant_inputs;

  // Intra-loop: k1_slice -> level_output for cross-chain dependencies
  absl::flat_hash_map<HloInstruction*, HloInstruction*>
      intra_loop_slice_to_level_output;

  // Offset slices: slices from known sources at position k+offset instead of k.
  // Maps iter-1 offset slice instruction -> (source tensor, offset).
  // Example: t_kp1[k] = slice(t, [k+1:k+2]) has offset = 1.
  absl::flat_hash_map<HloInstruction*, std::pair<HloInstruction*, int64_t>>
      offset_slices;
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
        if (limits[cat_dim] - starts[cat_dim] == 1 &&
            starts[cat_dim] == i) {
          HloInstruction* source = inst->mutable_operand(0);
          source_slices[source].push_back(inst);
        }
      } else {
        for (HloInstruction* op : inst->operands()) {
          if (op != operand && concat_operands.contains(op)) continue;
          bool is_other_concat_operand = false;
          for (HloInstruction* user : op->users()) {
            if (user->opcode() == HloOpcode::kConcatenate && user != concat) {
              is_other_concat_operand = true;
              break;
            }
          }
          if (is_other_concat_operand) continue;
          worklist.push_back(op);
        }
      }
    }
  }

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
    VLOG(3) << "  No complete source found for concatenate " << concat->name();
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
// Step 2: Extract combined iteration body using iter-1 as template
//
// Uses STRUCTURAL MATCHING between iter-2 and iter-1 operand trees to
// discover carry pairs (both regular and shifted), avoiding unreliable
// DFS discovery-order pairing.
// ─────────────────────────────────────────────────────────────────────────────

std::optional<IterationBody> ExtractGroupedIterationBody(
    const std::vector<SliceChain>& chains, HloComputation* computation) {

  if (chains.empty()) return std::nullopt;
  int64_t num_iters = chains[0].num_iterations;
  int64_t slice_dim = chains[0].slice_dim;

  // Need at least 3 iterations for k=0, k=1, k=2 template detection
  if (num_iters < 3) return std::nullopt;

  // ── Collect k=0, k=1, k=2 slices across all chains ──
  absl::flat_hash_set<HloInstruction*> all_k0_slices;
  absl::flat_hash_set<HloInstruction*> all_k1_slices;
  absl::flat_hash_set<HloInstruction*> all_k2_slices;
  absl::flat_hash_set<HloInstruction*> all_concat_operands;

  for (const auto& chain : chains) {
    for (int64_t i = 0; i < num_iters; ++i) {
      all_concat_operands.insert(chain.concat->mutable_operand(i));
    }
    for (auto& [source, slices] : chain.source_slices) {
      for (auto* s : slices) {
        int64_t k = s->slice_starts()[slice_dim];
        if (k == 0) all_k0_slices.insert(s);
        if (k == 1) all_k1_slices.insert(s);
        if (k == 2) all_k2_slices.insert(s);
      }
    }
  }

  if (all_k0_slices.empty() || all_k1_slices.empty()) return std::nullopt;

  // ── Phase A: Build iteration_0_set ──
  // Backward trace from k=0 concat operands.
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

  // Forward-propagate slice dependency within iter-0
  absl::flat_hash_set<HloInstruction*> depends_on_slice;
  {
    std::vector<HloInstruction*> worklist(all_k0_slices.begin(),
                                           all_k0_slices.end());
    while (!worklist.empty()) {
      HloInstruction* inst = worklist.back();
      worklist.pop_back();
      if (!depends_on_slice.insert(inst).second) continue;
      for (HloInstruction* user : inst->users()) {
        if (iteration_0_set.contains(user)) {
          worklist.push_back(user);
        }
      }
    }
  }

  // Prune iteration_0_set to only slice-dependent instructions
  {
    absl::flat_hash_set<HloInstruction*> pruned;
    for (HloInstruction* inst : iteration_0_set) {
      if (depends_on_slice.contains(inst)) {
        pruned.insert(inst);
      }
    }
    iteration_0_set = std::move(pruned);
  }

  VLOG(2) << "Phase A: iteration_0_set=" << iteration_0_set.size()
          << " slice-dependent instructions";

  // ── Collect known source tensors (for offset slice detection) ──
  absl::flat_hash_set<HloInstruction*> known_sources;
  for (const auto& chain : chains) {
    for (auto& [source, slices] : chain.source_slices) {
      known_sources.insert(source);
    }
  }

  // ── Phase B: Build iteration_1_set ──
  // Backward trace from k=1 concat operands. Stops at:
  //   - k=1 slices (regular slice inputs)
  //   - k=0 slices along slice_dim (potential shifted carries)
  //   - offset slices from known sources (position != 1, e.g. t_kp1[k]=t[k+1])
  //   - iter-0 instructions (regular carry boundary)
  //   - parameters/constants
  absl::flat_hash_set<HloInstruction*> iteration_1_set;
  absl::flat_hash_map<HloInstruction*, std::pair<HloInstruction*, int64_t>>
      offset_slices;
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

      // k=1 slice: regular sliced input leaf
      if (all_k1_slices.contains(inst)) continue;

      // Any size-1 slice at position 0 along slice_dim: shifted carry leaf
      if (inst->opcode() == HloOpcode::kSlice) {
        int64_t start = inst->slice_starts()[slice_dim];
        int64_t size = inst->slice_limits()[slice_dim] - start;
        if (start == 0 && size == 1) {
          continue;
        }
        // Offset slice from a known source: leaf with offset tracking.
        // e.g. t_kp1[1] = slice(t, [2:3]) when processing iter-1 template.
        if (size == 1 && start > 1) {
          HloInstruction* source = inst->mutable_operand(0);
          if (known_sources.contains(source)) {
            int64_t offset = start - 1;  // offset relative to counter
            offset_slices[inst] = {source, offset};
            VLOG(2) << "Offset slice: " << inst->name()
                    << " from " << source->name()
                    << " at position " << start
                    << " (offset=" << offset << ")";
            continue;
          }
        }
      }

      // iter-0 instruction: regular carry boundary
      if (iteration_0_set.contains(inst) &&
          depends_on_slice.contains(inst)) continue;

      if (inst->opcode() == HloOpcode::kParameter ||
          inst->opcode() == HloOpcode::kConstant) continue;

      iteration_1_set.insert(inst);
      for (HloInstruction* op : inst->operands()) {
        worklist.push_back(op);
      }
    }
  }

  // Prune iteration_1_set: only keep instructions depending on iter-1 inputs.
  // Collect boundary inputs (k=0 slices + iter-0 instructions) for this.
  absl::flat_hash_set<HloInstruction*> iter1_boundary;
  for (HloInstruction* inst : computation->instructions()) {
    // k=0 slices along slice_dim
    if (inst->opcode() == HloOpcode::kSlice) {
      int64_t start = inst->slice_starts()[slice_dim];
      int64_t size = inst->slice_limits()[slice_dim] - start;
      if (start == 0 && size == 1) {
        // Check if any iter-1 instruction uses this
        for (HloInstruction* user : inst->users()) {
          if (iteration_1_set.contains(user)) {
            iter1_boundary.insert(inst);
            break;
          }
        }
      }
    }
    // iter-0 instructions used by iter-1
    if (iteration_0_set.contains(inst) && depends_on_slice.contains(inst)) {
      for (HloInstruction* user : inst->users()) {
        if (iteration_1_set.contains(user)) {
          iter1_boundary.insert(inst);
          break;
        }
      }
    }
  }

  {
    absl::flat_hash_set<HloInstruction*> depends_on_iter1_input;
    std::vector<HloInstruction*> worklist;
    for (auto* s : all_k1_slices) worklist.push_back(s);
    for (auto* bi : iter1_boundary) worklist.push_back(bi);
    // Offset slices are also iter-1 inputs
    for (auto& [inst, src_off] : offset_slices) worklist.push_back(inst);
    while (!worklist.empty()) {
      HloInstruction* inst = worklist.back();
      worklist.pop_back();
      if (!depends_on_iter1_input.insert(inst).second) continue;
      for (HloInstruction* user : inst->users()) {
        if (iteration_1_set.contains(user)) {
          worklist.push_back(user);
        }
      }
    }
    absl::flat_hash_set<HloInstruction*> pruned;
    for (HloInstruction* inst : iteration_1_set) {
      if (depends_on_iter1_input.contains(inst)) pruned.insert(inst);
    }
    iteration_1_set = std::move(pruned);
  }

  VLOG(2) << "Phase B: iteration_1_set=" << iteration_1_set.size()
          << " instructions (body template)";

  // ── Phase AC: Structural carry matching ──
  // Walk backward from (k=2_concat_operand, k=1_concat_operand) pairs per
  // chain. At each step, match operands by position. At carry boundaries
  // (iter-1 value ↔ iter-0 value, or k=1 slice ↔ k=0 slice), record pairs.
  //
  // This replaces the old Phase A2 + Phase C + Phase D, fixing:
  //   Bug 1: DFS discovery-order pairing was unreliable
  //   Bug 2: Shifted k=0 slices not in source_slices were missed

  struct CarryPairInfo {
    HloInstruction* input;   // iter-0 value or k=0 slice
    HloInstruction* output;  // iter-1 value or k=1 slice
    bool is_shifted;
  };

  std::vector<CarryPairInfo> carry_pairs;
  absl::flat_hash_set<HloInstruction*> carry_input_set;
  absl::flat_hash_set<HloInstruction*> carry_output_set;
  bool match_failed = false;

  // Visited set for (i2, i1) pairs to avoid infinite recursion
  absl::flat_hash_map<HloInstruction*, absl::flat_hash_set<HloInstruction*>>
      match_visited;

  // Recursive structural matcher
  std::function<void(HloInstruction*, HloInstruction*)> match_carries;
  match_carries = [&](HloInstruction* i2, HloInstruction* i1) {
    if (match_failed) return;

    // Same instruction → invariant (used identically by both iterations)
    if (i2 == i1) return;

    // Already visited this pair
    if (match_visited[i2].contains(i1)) return;
    match_visited[i2].insert(i1);

    // Both are slices from the same source along slice_dim?
    if (i2->opcode() == HloOpcode::kSlice &&
        i1->opcode() == HloOpcode::kSlice &&
        i2->mutable_operand(0) == i1->mutable_operand(0)) {
      int64_t p2 = i2->slice_starts()[slice_dim];
      int64_t p1 = i1->slice_starts()[slice_dim];
      int64_t sz2 = i2->slice_limits()[slice_dim] - p2;
      int64_t sz1 = i1->slice_limits()[slice_dim] - p1;
      if (sz2 == 1 && sz1 == 1) {
        if (p2 == 2 && p1 == 1) {
          // Regular slice input: current-level for iter-2 and iter-1
          return;
        }
        if (p2 == 1 && p1 == 0) {
          // Shifted carry: previous-level for iter-2 and iter-1
          if (!carry_input_set.contains(i1)) {
            carry_pairs.push_back({i1, i2, /*is_shifted=*/true});
            carry_input_set.insert(i1);
            carry_output_set.insert(i2);
            VLOG(2) << "Structural match: SHIFTED carry "
                    << i1->name() << " -> " << i2->name()
                    << " (source: " << i1->mutable_operand(0)->name() << ")";
          }
          return;
        }
        if (p2 - p1 == 1 && p1 > 1) {
          // Offset slice input: e.g. t_kp1[k] = slice(t, [k+1:k+2])
          // Both iter-2 and iter-1 use the same offset from their
          // respective levels. Not a carry — just an input.
          VLOG(3) << "Structural match: offset slice pair "
                  << i1->name() << " (pos=" << p1 << ") / "
                  << i2->name() << " (pos=" << p2 << ")";
          return;
        }
      }
    }

    // Check carry boundary: i2 from iter-1, i1 from iter-0
    bool i2_from_iter1 = iteration_1_set.contains(i2) ||
                          all_k1_slices.contains(i2);
    bool i1_from_iter0 = (iteration_0_set.contains(i1) &&
                           depends_on_slice.contains(i1));

    // Also check if i1 is a k=0 slice not in iteration_0_set
    // (shifted carry from a source not tracked by AnalyzeConcatenate)
    if (!i1_from_iter0 && i1->opcode() == HloOpcode::kSlice) {
      int64_t start = i1->slice_starts()[slice_dim];
      int64_t size = i1->slice_limits()[slice_dim] - start;
      if (start == 0 && size == 1) {
        i1_from_iter0 = true;
      }
    }

    if (i2_from_iter1 && i1_from_iter0) {
      // Regular carry boundary
      if (!carry_input_set.contains(i1)) {
        bool shifted = (i1->opcode() == HloOpcode::kSlice &&
                        i1->slice_starts()[slice_dim] == 0 &&
                        i2->opcode() == HloOpcode::kSlice &&
                        i2->slice_starts()[slice_dim] == 1 &&
                        i1->mutable_operand(0) == i2->mutable_operand(0));
        carry_pairs.push_back({i1, i2, shifted});
        carry_input_set.insert(i1);
        carry_output_set.insert(i2);
        VLOG(2) << "Structural match: " << (shifted ? "SHIFTED" : "REGULAR")
                << " carry " << i1->name() << " -> " << i2->name();
      }
      return;
    }

    // If one side is a boundary but not the other, possible mismatch
    if (i2_from_iter1 || i1_from_iter0) {
      VLOG(1) << "WARNING: asymmetric carry boundary: i2=" << i2->name()
              << " (from_iter1=" << i2_from_iter1 << ") i1=" << i1->name()
              << " (from_iter0=" << i1_from_iter0 << ")";
      // Treat as carry pair anyway (best effort)
      if (!carry_input_set.contains(i1)) {
        carry_pairs.push_back({i1, i2, false});
        carry_input_set.insert(i1);
        carry_output_set.insert(i2);
      }
      return;
    }

    // Both constants → invariant
    if ((i2->opcode() == HloOpcode::kConstant ||
         i2->opcode() == HloOpcode::kParameter) &&
        (i1->opcode() == HloOpcode::kConstant ||
         i1->opcode() == HloOpcode::kParameter)) {
      return;
    }

    // Both are body instructions of their respective iterations → recurse
    if (i2->opcode() != i1->opcode() ||
        i2->operand_count() != i1->operand_count()) {
      VLOG(1) << "Structural mismatch: i2=" << i2->name()
              << " (" << HloOpcodeString(i2->opcode()) << "/"
              << i2->operand_count() << ") vs i1=" << i1->name()
              << " (" << HloOpcodeString(i1->opcode()) << "/"
              << i1->operand_count() << ")";
      match_failed = true;
      return;
    }

    for (int64_t j = 0; j < i2->operand_count(); ++j) {
      match_carries(i2->mutable_operand(j), i1->mutable_operand(j));
    }
  };

  // Run structural matching from each chain's (k=2 operand, k=1 operand)
  for (const auto& chain : chains) {
    match_carries(chain.concat->mutable_operand(2),
                  chain.concat->mutable_operand(1));
  }

  if (match_failed) {
    VLOG(1) << "Structural carry matching failed — aborting";
    return std::nullopt;
  }

  VLOG(2) << "Phase AC: " << carry_pairs.size()
          << " structurally matched carry pairs";
  for (const auto& cp : carry_pairs) {
    VLOG(2) << "  carry: " << cp.input->name() << " -> " << cp.output->name()
            << (cp.is_shifted ? " [SHIFTED]" : " [REGULAR]");
  }

  // ── Build IterationBody ──
  IterationBody body;

  for (const auto& cp : carry_pairs) {
    body.carry_inputs.push_back(cp.input);
    body.carry_outputs.push_back(cp.output);
    body.carry_is_shifted.push_back(cp.is_shifted);
  }

  // Transfer offset slices
  body.offset_slices = std::move(offset_slices);

  // Topologically sort iter-1 instructions
  for (HloInstruction* inst : computation->MakeInstructionPostOrder()) {
    if (iteration_1_set.contains(inst)) {
      body.instructions.push_back(inst);
    }
  }

  // Concats, level outputs, and intra-loop dependencies
  absl::flat_hash_set<HloInstruction*> concat_set;
  absl::flat_hash_map<HloInstruction*, HloInstruction*> concat_to_k1_output;
  for (const auto& chain : chains) {
    concat_set.insert(chain.concat);
    concat_to_k1_output[chain.concat] = chain.concat->mutable_operand(1);
  }

  // Collect sliced inputs from k=1 slices in chain.source_slices
  absl::flat_hash_set<HloInstruction*> seen_sources;
  for (const auto& chain : chains) {
    for (auto& [source, slices] : chain.source_slices) {
      // Intra-loop: source is a concat of another chain
      if (concat_set.contains(source)) {
        for (auto* s : slices) {
          if (s->slice_starts()[slice_dim] == 1) {
            auto level_it = concat_to_k1_output.find(source);
            if (level_it != concat_to_k1_output.end()) {
              body.intra_loop_slice_to_level_output[s] = level_it->second;
              VLOG(2) << "Intra-loop k1_slice: " << s->name()
                      << " -> level_output " << level_it->second->name();
            }
          }
        }
        continue;
      }
      if (seen_sources.insert(source).second) {
        body.sliced_inputs.push_back(source);
      }
      // Record k=1 slice → source
      for (auto* s : slices) {
        if (s->slice_starts()[slice_dim] == 1) {
          body.k1_slice_to_source[s] = source;
        }
      }
    }
  }

  // Add offset slice sources to sliced_inputs if not already present
  for (auto& [inst, src_off] : body.offset_slices) {
    auto [source, offset] = src_off;
    if (!concat_set.contains(source) &&
        seen_sources.insert(source).second) {
      body.sliced_inputs.push_back(source);
      VLOG(2) << "Added offset slice source: " << source->name();
    }
  }

  // Add shifted carry sources to sliced_inputs (they may not be in
  // chain.source_slices because AnalyzeConcatenate only tracks slices
  // whose position matches the concat operand index)
  for (size_t i = 0; i < body.carry_inputs.size(); ++i) {
    if (body.carry_is_shifted[i]) {
      HloInstruction* k0_slice = body.carry_inputs[i];
      HloInstruction* k1_slice = body.carry_outputs[i];
      HloInstruction* source = k0_slice->mutable_operand(0);
      if (!concat_set.contains(source) &&
          seen_sources.insert(source).second) {
        body.sliced_inputs.push_back(source);
        VLOG(2) << "Added shifted carry source: " << source->name();
      }
      // Also record k=1 slice → source for the shifted carry
      body.k1_slice_to_source[k1_slice] = k1_slice->mutable_operand(0);
    }
  }

  // Level outputs
  for (const auto& chain : chains) {
    body.iter0_level_outputs.push_back(chain.concat->mutable_operand(0));
    body.iter1_level_outputs.push_back(chain.concat->mutable_operand(1));
    body.concats.push_back(chain.concat);
  }

  // ── Find derived concatenates ──
  // Some concatenates have operands that are computed inside the iteration body
  // but were not found by AnalyzeConcatenate (because their operands are computed
  // values, not slices from a source tensor).  Example: pflx_tot = concat(
  //   ps[0]+pi[0]+pg[0], ..., ps[89]+pi[89]+pg[89]).
  // If their k=1 operand is in the iteration body, we add them as accumulators
  // so the original per-level computations can be removed.
  for (HloInstruction* inst : computation->MakeInstructionPostOrder()) {
    if (inst->opcode() != HloOpcode::kConcatenate) continue;
    if (concat_set.contains(inst)) continue;
    if (inst->operand_count() != num_iters) continue;
    if (inst->concatenate_dimension() != slice_dim) continue;

    HloInstruction* k1_op = inst->mutable_operand(1);
    if (!iteration_1_set.contains(k1_op)) continue;

    HloInstruction* k0_op = inst->mutable_operand(0);
    if (k0_op->shape() != k1_op->shape()) continue;

    body.iter0_level_outputs.push_back(k0_op);
    body.iter1_level_outputs.push_back(k1_op);
    body.concats.push_back(inst);
    concat_set.insert(inst);

    VLOG(1) << "Found derived concatenate: " << inst->name()
            << " (k1_op=" << k1_op->name() << ")";
  }

  // Invariant inputs: operands of iter-1 instructions that are not
  // in iter-1 set, not carry inputs, not k=1 slices, not sliced sources,
  // not offset slices.
  absl::flat_hash_set<HloInstruction*> sliced_set(
      body.sliced_inputs.begin(), body.sliced_inputs.end());
  absl::flat_hash_set<HloInstruction*> offset_slice_set;
  for (auto& [inst, src_off] : body.offset_slices) {
    offset_slice_set.insert(inst);
  }
  absl::flat_hash_set<HloInstruction*> invariant_set;
  for (HloInstruction* inst : body.instructions) {
    for (HloInstruction* op : inst->operands()) {
      if (!iteration_1_set.contains(op) && !carry_input_set.contains(op) &&
          !all_k1_slices.contains(op) && !concat_set.contains(op) &&
          !sliced_set.contains(op) && !offset_slice_set.contains(op)) {
        if (invariant_set.insert(op).second) {
          body.invariant_inputs.push_back(op);
        }
      }
    }
  }

  VLOG(2) << "Combined body (iter-1 template): " << body.instructions.size()
          << " instructions, " << body.sliced_inputs.size() << " sliced inputs, "
          << body.carry_inputs.size() << " carry pairs ("
          << [&]() { int n = 0; for (auto s : body.carry_is_shifted) if (s) n++; return n; }()
          << " shifted), "
          << body.offset_slices.size() << " offset slices, "
          << body.iter1_level_outputs.size() << " level outputs, "
          << body.invariant_inputs.size() << " invariant inputs";

  return body;
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 3: Build the while loop (iter-0 peeled, iter-1 as body template)
// ─────────────────────────────────────────────────────────────────────────────

absl::StatusOr<bool> BuildWhileLoop(
    HloComputation* computation,
    const std::vector<SliceChain>& chains,
    const IterationBody& body,
    int unroll_factor) {

  int64_t num_iters = chains[0].num_iterations;
  int64_t slice_dim = chains[0].slice_dim;

  VLOG(1) << "Building while loop for " << chains.size() << " concatenates"
          << " (" << num_iters << " iterations, iter-0 peeled)";

  int num_carry = body.carry_outputs.size();
  int num_accum = body.iter1_level_outputs.size();
  int num_sliced = body.sliced_inputs.size();
  int num_invariant = body.invariant_inputs.size();

  // ── Classify carries: regular vs shifted ──
  // Shifted carries are just input[k-1]; compute via dynamic_slice(source,
  // counter-1) instead of carrying through the tuple.
  std::vector<int> regular_carry_indices;
  std::vector<int> shifted_carry_indices;
  for (int i = 0; i < num_carry; ++i) {
    if (body.carry_is_shifted[i]) {
      shifted_carry_indices.push_back(i);
    } else {
      regular_carry_indices.push_back(i);
    }
  }
  int num_regular_carry = regular_carry_indices.size();

  // ── Classify invariants: constants (recreated in body) vs tuple ──
  std::vector<HloInstruction*> const_invariants;
  std::vector<HloInstruction*> tuple_invariants;
  for (auto* inv : body.invariant_inputs) {
    if (inv->opcode() == HloOpcode::kConstant ||
        (inv->opcode() == HloOpcode::kBroadcast &&
         inv->operand(0)->opcode() == HloOpcode::kConstant)) {
      const_invariants.push_back(inv);
    } else {
      tuple_invariants.push_back(inv);
    }
  }
  int num_tuple_inv = tuple_invariants.size();

  VLOG(1) << "Carries: " << num_regular_carry << " regular + "
          << shifted_carry_indices.size() << " shifted (eliminated from tuple)";
  VLOG(1) << "Invariants: " << const_invariants.size()
          << " constant (recreated in body) + " << num_tuple_inv << " in tuple";

  // ── Check if accumulators can be stacked into a single tensor ──
  // Stacking reduces N dynamic-update-slice ops per iteration to 1.
  bool use_stacked = (num_accum > 1);
  int64_t stack_dim = -1;
  std::vector<int64_t> accum_widths;
  std::vector<int64_t> accum_cum_offsets;
  int64_t total_stacked_width = 0;

  if (use_stacked) {
    PrimitiveType first_type = body.concats[0]->shape().element_type();
    int64_t first_rank = body.concats[0]->shape().dimensions_size();
    if (first_rank == 2) {
      stack_dim = (slice_dim == 0) ? 1 : 0;
      for (int i = 0; i < num_accum; ++i) {
        const Shape& s = body.concats[i]->shape();
        if (s.element_type() != first_type ||
            s.dimensions_size() != first_rank ||
            s.dimensions(slice_dim) !=
                body.concats[0]->shape().dimensions(slice_dim)) {
          use_stacked = false;
          break;
        }
        accum_cum_offsets.push_back(total_stacked_width);
        accum_widths.push_back(s.dimensions(stack_dim));
        total_stacked_width += s.dimensions(stack_dim);
      }
    } else {
      use_stacked = false;
    }
  }

  int num_accum_slots = use_stacked ? 1 : num_accum;

  VLOG(1) << "Stacked accumulators: " << (use_stacked ? "YES" : "NO")
          << " (" << num_accum << " accums, " << num_accum_slots << " slots)";

  // Tuple layout:
  // [0]                                  : counter (starts at 1)
  // [1 .. 1+num_regular_carry)           : regular carries only
  // [+num_accum_slots)                   : accumulators (1 if stacked)
  // [+num_sliced .. )                    : sliced inputs
  // [+num_tuple_inv .. )                 : non-constant invariant inputs
  int carry_off = 1;
  int accum_off = carry_off + num_regular_carry;
  int sliced_off = accum_off + num_accum_slots;
  int inv_off = sliced_off + num_sliced;
  int tuple_size = inv_off + num_tuple_inv;

  VLOG(2) << "Tuple: 1 counter + " << num_regular_carry << " carry + "
          << num_accum_slots << " accum_slots + " << num_sliced << " sliced + "
          << num_tuple_inv << " invariant = " << tuple_size;

  // ── Tuple shapes ──
  std::vector<Shape> tuple_shapes;
  tuple_shapes.push_back(ShapeUtil::MakeShape(S64, {}));  // counter
  for (int idx : regular_carry_indices) {
    tuple_shapes.push_back(body.carry_outputs[idx]->shape());
  }
  if (use_stacked) {
    PrimitiveType accum_type = body.concats[0]->shape().element_type();
    std::vector<int64_t> stacked_dims(2);
    stacked_dims[slice_dim] = num_iters;
    stacked_dims[stack_dim] = total_stacked_width;
    tuple_shapes.push_back(ShapeUtil::MakeShape(accum_type, stacked_dims));
  } else {
    for (auto* concat : body.concats) {
      tuple_shapes.push_back(concat->shape());
    }
  }
  for (auto* src : body.sliced_inputs) {
    tuple_shapes.push_back(src->shape());
  }
  for (auto* inv : tuple_invariants) {
    tuple_shapes.push_back(inv->shape());
  }
  Shape tuple_shape = ShapeUtil::MakeTupleShape(tuple_shapes);

  // ── Initial values ──
  // Counter starts at 1 (iter-0 is peeled)
  HloInstruction* one_init = computation->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int64_t>(1)));

  std::vector<HloInstruction*> init_values;
  init_values.push_back(one_init);

  // Carry initial values = iter-0 outputs (only regular carries in tuple)
  for (int idx : regular_carry_indices) {
    init_values.push_back(body.carry_inputs[idx]);
    VLOG(2) << "carry init: " << body.carry_inputs[idx]->name()
            << " (opcode=" << HloOpcodeString(body.carry_inputs[idx]->opcode()) << ")";
  }

  // Accumulators: zero-filled, then prefill position 0 with iter-0 outputs
  HloInstruction* zero_idx = computation->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int64_t>(0)));
  if (use_stacked) {
    const Shape& stacked_shape_ref = tuple_shapes[accum_off];
    PrimitiveType elem_type = stacked_shape_ref.element_type();
    HloInstruction* zero_elem = computation->AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::Zero(elem_type)));
    HloInstruction* stacked_accum = computation->AddInstruction(
        HloInstruction::CreateBroadcast(stacked_shape_ref, zero_elem, {}));
    for (int i = 0; i < num_accum; ++i) {
      HloInstruction* iter0_out = body.iter0_level_outputs[i];
      HloInstruction* offset_val = computation->AddInstruction(
          HloInstruction::CreateConstant(
              LiteralUtil::CreateR0<int64_t>(accum_cum_offsets[i])));
      std::vector<HloInstruction*> update_indices(2);
      update_indices[slice_dim] = zero_idx;
      update_indices[stack_dim] = offset_val;
      stacked_accum = computation->AddInstruction(
          HloInstruction::CreateDynamicUpdateSlice(
              stacked_shape_ref, stacked_accum, iter0_out, update_indices));
    }
    init_values.push_back(stacked_accum);
  } else {
    for (int i = 0; i < num_accum; ++i) {
      PrimitiveType elem_type = body.concats[i]->shape().element_type();
      HloInstruction* zero_elem = computation->AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::Zero(elem_type)));
      HloInstruction* zero_accum = computation->AddInstruction(
          HloInstruction::CreateBroadcast(body.concats[i]->shape(),
                                           zero_elem, {}));
      HloInstruction* iter0_out = body.iter0_level_outputs[i];
      int64_t rank = zero_accum->shape().dimensions_size();
      std::vector<HloInstruction*> update_indices;
      for (int64_t d = 0; d < rank; ++d) {
        update_indices.push_back(zero_idx);
      }
      HloInstruction* prefilled = computation->AddInstruction(
          HloInstruction::CreateDynamicUpdateSlice(
              zero_accum->shape(), zero_accum, iter0_out, update_indices));
      init_values.push_back(prefilled);
    }
  }

  // Sliced inputs and non-constant invariants (pass-through)
  for (auto* src : body.sliced_inputs) init_values.push_back(src);
  for (auto* inv : tuple_invariants) init_values.push_back(inv);

  HloInstruction* init_tuple = computation->AddInstruction(
      HloInstruction::CreateTuple(init_values));

  // ── Condition computation: counter < num_iters ──
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

  // ── Body computation ──
  HloComputation::Builder body_builder("while_body");
  HloInstruction* body_param = body_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "param"));

  // Extract tuple elements
  HloInstruction* counter = body_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(body_param, 0));

  std::vector<HloInstruction*> carry_values;  // indexed by regular carry index
  for (int i = 0; i < num_regular_carry; ++i) {
    carry_values.push_back(body_builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(body_param, carry_off + i)));
  }

  std::vector<HloInstruction*> accumulators;
  for (int i = 0; i < num_accum_slots; ++i) {
    accumulators.push_back(body_builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(body_param, accum_off + i)));
  }

  std::vector<HloInstruction*> sliced_inputs_in_body;
  for (int i = 0; i < num_sliced; ++i) {
    sliced_inputs_in_body.push_back(body_builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(body_param, sliced_off + i)));
  }

  std::vector<HloInstruction*> tuple_inv_in_body;
  for (int i = 0; i < num_tuple_inv; ++i) {
    tuple_inv_in_body.push_back(body_builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(body_param, inv_off + i)));
  }

  // ── Shared setup for unrolled sub-iterations ──
  HloInstruction* zero_body = body_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int64_t>(0)));

  absl::flat_hash_map<HloInstruction*, int> source_to_idx;
  for (int i = 0; i < num_sliced; ++i) {
    source_to_idx[body.sliced_inputs[i]] = i;
  }

  // Intra-loop: level_output → list of k1_slices to map after cloning
  absl::flat_hash_map<HloInstruction*, std::vector<HloInstruction*>>
      level_output_to_k1_slices;
  for (auto& [k1_slice, level_out] :
       body.intra_loop_slice_to_level_output) {
    level_output_to_k1_slices[level_out].push_back(k1_slice);
  }

  // Shared invariant/constant clone map (same for all sub-iterations)
  absl::flat_hash_map<HloInstruction*, HloInstruction*> shared_cloned;
  for (int i = 0; i < num_tuple_inv; ++i) {
    shared_cloned[tuple_invariants[i]] = tuple_inv_in_body[i];
  }
  for (auto* inv : const_invariants) {
    if (inv->opcode() == HloOpcode::kConstant) {
      shared_cloned[inv] = body_builder.AddInstruction(inv->Clone());
    } else {
      HloInstruction* const_clone = body_builder.AddInstruction(
          inv->operand(0)->Clone());
      shared_cloned[inv] = body_builder.AddInstruction(
          inv->CloneWithNewOperands(inv->shape(), {const_clone}));
    }
  }

  // Tail handling: when (num_iters-1) % unroll_factor != 0, the last
  // while iteration has sub-iterations past num_iters that must be gated.
  int num_body_iters = num_iters - 1;  // iter-0 is peeled
  bool needs_tail = (num_body_iters % unroll_factor != 0);
  HloInstruction* num_iters_body = nullptr;
  if (needs_tail) {
    num_iters_body = body_builder.AddInstruction(
        HloInstruction::CreateConstant(
            LiteralUtil::CreateR0<int64_t>(num_iters)));
  }

  LOG(INFO) << "loopify: unroll_factor=" << unroll_factor
            << ", num_body_iters=" << num_body_iters
            << (needs_tail ? ", needs tail handling" : "");

  // Current carries and accumulators for chaining between sub-iterations
  std::vector<HloInstruction*> current_carries = carry_values;
  HloInstruction* current_stacked_accum =
      use_stacked ? accumulators[0] : nullptr;
  std::vector<HloInstruction*> current_accums;
  if (!use_stacked) {
    current_accums.assign(accumulators.begin(),
                           accumulators.begin() + num_accum_slots);
  }

  // ── Unrolled sub-iterations ──
  for (int u = 0; u < unroll_factor; ++u) {
    // sub_counter = counter + u
    HloInstruction* sub_counter;
    if (u == 0) {
      sub_counter = counter;
    } else {
      HloInstruction* u_const = body_builder.AddInstruction(
          HloInstruction::CreateConstant(
              LiteralUtil::CreateR0<int64_t>(u)));
      sub_counter = body_builder.AddInstruction(
          HloInstruction::CreateBinary(
              ShapeUtil::MakeShape(S64, {}), HloOpcode::kAdd,
              counter, u_const));
    }

    // Fresh maps for this sub-iteration
    absl::flat_hash_map<HloInstruction*, HloInstruction*> slice_replacement;
    absl::flat_hash_map<HloInstruction*, HloInstruction*> source_to_dyn_slice;

    // ── Dynamic-slice k=1 slices at sub_counter ──
    for (const auto& chain : chains) {
      for (auto& [source, slices] : chain.source_slices) {
        HloInstruction* k1_slice = nullptr;
        for (auto* s : slices) {
          if (s->slice_starts()[slice_dim] == 1) {
            k1_slice = s;
            break;
          }
        }
        if (!k1_slice) continue;
        if (slice_replacement.contains(k1_slice)) continue;
        if (body.intra_loop_slice_to_level_output.contains(k1_slice)) continue;

        auto src_it = source_to_idx.find(source);
        if (src_it == source_to_idx.end()) continue;
        HloInstruction* source_in_body = sliced_inputs_in_body[src_it->second];

        int64_t rank = source->shape().dimensions_size();
        std::vector<HloInstruction*> start_indices;
        for (int64_t d = 0; d < rank; ++d) {
          start_indices.push_back(d == slice_dim ? sub_counter : zero_body);
        }
        std::vector<int64_t> slice_sizes;
        for (int64_t d = 0; d < rank; ++d) {
          slice_sizes.push_back(
              k1_slice->slice_limits()[d] - k1_slice->slice_starts()[d]);
        }

        HloInstruction* dyn_slice = body_builder.AddInstruction(
            HloInstruction::CreateDynamicSlice(
                k1_slice->shape(), source_in_body, start_indices,
                slice_sizes));
        slice_replacement[k1_slice] = dyn_slice;
        source_to_dyn_slice[source] = dyn_slice;
      }
    }

    // ── Dynamic-slice offset slices at sub_counter + offset ──
    for (auto& [k1_slice, src_off] : body.offset_slices) {
      auto [source, offset] = src_off;
      if (slice_replacement.contains(k1_slice)) continue;

      auto src_it = source_to_idx.find(source);
      if (src_it == source_to_idx.end()) {
        LOG(WARNING) << "loopify: offset slice source " << source->name()
                     << " not in sliced_inputs (sub-iter " << u << ")";
        return false;
      }
      HloInstruction* source_in_body = sliced_inputs_in_body[src_it->second];

      HloInstruction* offset_val = body_builder.AddInstruction(
          HloInstruction::CreateConstant(
              LiteralUtil::CreateR0<int64_t>(offset)));
      HloInstruction* sc_plus_offset = body_builder.AddInstruction(
          HloInstruction::CreateBinary(
              ShapeUtil::MakeShape(S64, {}), HloOpcode::kAdd,
              sub_counter, offset_val));

      int64_t rank = source->shape().dimensions_size();
      std::vector<HloInstruction*> start_indices;
      for (int64_t d = 0; d < rank; ++d) {
        start_indices.push_back(d == slice_dim ? sc_plus_offset : zero_body);
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

    // ── Dynamic-slice shifted carry outputs at sub_counter ──
    for (int i = 0; i < num_carry; ++i) {
      if (!body.carry_is_shifted[i]) continue;
      HloInstruction* k1_slice = body.carry_outputs[i];
      if (slice_replacement.contains(k1_slice)) continue;

      HloInstruction* source = k1_slice->mutable_operand(0);
      auto ds_it = source_to_dyn_slice.find(source);
      if (ds_it != source_to_dyn_slice.end()) {
        slice_replacement[k1_slice] = ds_it->second;
        continue;
      }

      auto src_it = source_to_idx.find(source);
      if (src_it == source_to_idx.end()) {
        LOG(WARNING) << "loopify: shifted carry source " << source->name()
                     << " not in sliced_inputs (sub-iter " << u << ")";
        return false;
      }
      HloInstruction* source_in_body = sliced_inputs_in_body[src_it->second];

      int64_t rank = source->shape().dimensions_size();
      std::vector<HloInstruction*> start_indices;
      for (int64_t d = 0; d < rank; ++d) {
        start_indices.push_back(d == slice_dim ? sub_counter : zero_body);
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
      source_to_dyn_slice[source] = dyn_slice;
    }

    // ── Shifted carry inputs: dynamic_slice(source, sub_counter - 1) ──
    HloInstruction* sc_minus_1 = nullptr;
    if (!shifted_carry_indices.empty()) {
      HloInstruction* one_sc = body_builder.AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::CreateR0<int64_t>(1)));
      sc_minus_1 = body_builder.AddInstruction(
          HloInstruction::CreateBinary(
              ShapeUtil::MakeShape(S64, {}), HloOpcode::kSubtract,
              sub_counter, one_sc));
    }

    for (int idx : shifted_carry_indices) {
      HloInstruction* carry_in = body.carry_inputs[idx];
      HloInstruction* source = carry_in->mutable_operand(0);

      auto src_it = source_to_idx.find(source);
      if (src_it == source_to_idx.end()) {
        LOG(WARNING) << "loopify: shifted carry input source " << source->name()
                     << " not in sliced_inputs (sub-iter " << u << ")";
        return false;
      }
      HloInstruction* source_in_body = sliced_inputs_in_body[src_it->second];

      int64_t rank = source->shape().dimensions_size();
      std::vector<HloInstruction*> start_indices;
      for (int64_t d = 0; d < rank; ++d) {
        start_indices.push_back(d == slice_dim ? sc_minus_1 : zero_body);
      }
      std::vector<int64_t> slice_sizes;
      for (int64_t d = 0; d < rank; ++d) {
        slice_sizes.push_back(carry_in->shape().dimensions(d));
      }

      HloInstruction* prev_level = body_builder.AddInstruction(
          HloInstruction::CreateDynamicSlice(
              carry_in->shape(), source_in_body, start_indices, slice_sizes));
      slice_replacement[carry_in] = prev_level;
    }

    // ── Build clone map for this sub-iteration ──
    absl::flat_hash_map<HloInstruction*, HloInstruction*> cloned =
        shared_cloned;

    for (auto& [orig, repl] : slice_replacement) {
      cloned[orig] = repl;
    }
    for (int i = 0; i < num_regular_carry; ++i) {
      int idx = regular_carry_indices[i];
      cloned[body.carry_inputs[idx]] = current_carries[i];
    }

    // ── Clone body instructions ──
    for (HloInstruction* inst : body.instructions) {
      if (cloned.contains(inst)) continue;

      std::vector<HloInstruction*> new_operands;
      for (HloInstruction* op : inst->operands()) {
        auto it = cloned.find(op);
        if (it != cloned.end()) {
          new_operands.push_back(it->second);
        } else {
          LOG(WARNING) << "loopify: unmapped operand " << op->name()
                       << " (opcode=" << HloOpcodeString(op->opcode()) << ")"
                       << " for instruction " << inst->name()
                       << " (opcode=" << HloOpcodeString(inst->opcode()) << ")"
                       << " (sub-iter " << u << ")";
          return false;
        }
      }

      cloned[inst] = body_builder.AddInstruction(
          inst->CloneWithNewOperands(inst->shape(), new_operands));

      auto lo_it = level_output_to_k1_slices.find(inst);
      if (lo_it != level_output_to_k1_slices.end()) {
        for (HloInstruction* k1_slice : lo_it->second) {
          cloned[k1_slice] = cloned[inst];
        }
      }
    }

    // ── DUS level outputs into accumulator(s) ──
    // Tail handling: gate the VALUE being written, not the whole accumulator.
    // This avoids a full-tensor select (O(90*N)) per sub-iteration.
    // Instead: gated_val = select(valid, new_val, old_slice); DUS(accum, gated_val)
    HloInstruction* tail_valid = nullptr;
    if (needs_tail && u > 0) {
      tail_valid = body_builder.AddInstruction(
          HloInstruction::CreateCompare(
              ShapeUtil::MakeShape(PRED, {}), sub_counter, num_iters_body,
              ComparisonDirection::kLt));
    }

    if (use_stacked) {
      std::vector<HloInstruction*> level_outs;
      for (int i = 0; i < num_accum; ++i) {
        level_outs.push_back(cloned[body.iter1_level_outputs[i]]);
      }
      PrimitiveType elem_type = body.concats[0]->shape().element_type();
      std::vector<int64_t> cat_dims(2);
      cat_dims[slice_dim] = 1;
      cat_dims[stack_dim] = total_stacked_width;
      Shape cat_shape = ShapeUtil::MakeShape(elem_type, cat_dims);
      HloInstruction* cat_level = body_builder.AddInstruction(
          HloInstruction::CreateConcatenate(cat_shape, level_outs, stack_dim));

      if (tail_valid) {
        // Read old slice, select between new and old at slice granularity
        std::vector<HloInstruction*> read_indices(2);
        read_indices[slice_dim] = sub_counter;
        read_indices[stack_dim] = zero_body;
        HloInstruction* old_slice = body_builder.AddInstruction(
            HloInstruction::CreateDynamicSlice(
                cat_shape, current_stacked_accum, read_indices,
                cat_shape.dimensions()));
        HloInstruction* valid_bcast = body_builder.AddInstruction(
            HloInstruction::CreateBroadcast(
                ShapeUtil::MakeShape(PRED, cat_shape.dimensions()),
                tail_valid, {}));
        cat_level = body_builder.AddInstruction(
            HloInstruction::CreateTernary(
                cat_shape, HloOpcode::kSelect,
                valid_bcast, cat_level, old_slice));
      }

      std::vector<HloInstruction*> update_indices(2);
      update_indices[slice_dim] = sub_counter;
      update_indices[stack_dim] = zero_body;
      current_stacked_accum = body_builder.AddInstruction(
          HloInstruction::CreateDynamicUpdateSlice(
              current_stacked_accum->shape(), current_stacked_accum,
              cat_level, update_indices));
    } else {
      for (int i = 0; i < num_accum; ++i) {
        HloInstruction* level_out = cloned[body.iter1_level_outputs[i]];
        int64_t rank = current_accums[i]->shape().dimensions_size();
        std::vector<HloInstruction*> update_indices;
        for (int64_t d = 0; d < rank; ++d) {
          update_indices.push_back(d == slice_dim ? sub_counter : zero_body);
        }

        if (tail_valid) {
          // Read old slice, select at slice granularity
          std::vector<int64_t> slice_sizes;
          for (int64_t d = 0; d < rank; ++d) {
            slice_sizes.push_back(level_out->shape().dimensions(d));
          }
          HloInstruction* old_slice = body_builder.AddInstruction(
              HloInstruction::CreateDynamicSlice(
                  level_out->shape(), current_accums[i],
                  update_indices, slice_sizes));
          HloInstruction* valid_bcast = body_builder.AddInstruction(
              HloInstruction::CreateBroadcast(
                  ShapeUtil::MakeShape(PRED, level_out->shape().dimensions()),
                  tail_valid, {}));
          level_out = body_builder.AddInstruction(
              HloInstruction::CreateTernary(
                  level_out->shape(), HloOpcode::kSelect,
                  valid_bcast, level_out, old_slice));
        }

        current_accums[i] = body_builder.AddInstruction(
            HloInstruction::CreateDynamicUpdateSlice(
                current_accums[i]->shape(), current_accums[i],
                level_out, update_indices));
      }
    }

    // ── Extract carry outputs for chaining to next sub-iteration ──
    std::vector<HloInstruction*> new_carries;
    for (int i = 0; i < num_regular_carry; ++i) {
      int idx = regular_carry_indices[i];
      auto it = cloned.find(body.carry_outputs[idx]);
      if (it != cloned.end()) {
        new_carries.push_back(it->second);
      } else {
        LOG(WARNING) << "loopify: carry output " << body.carry_outputs[idx]->name()
                     << " not in cloned map (sub-iter " << u << ")";
        return false;
      }
    }
    current_carries = std::move(new_carries);

    VLOG(2) << "Sub-iteration " << u << " complete";
  }  // end unroll loop

  // ── Increment counter by unroll_factor ──
  HloInstruction* step = body_builder.AddInstruction(
      HloInstruction::CreateConstant(
          LiteralUtil::CreateR0<int64_t>(unroll_factor)));
  HloInstruction* next_counter = body_builder.AddInstruction(
      HloInstruction::CreateBinary(
          ShapeUtil::MakeShape(S64, {}), HloOpcode::kAdd, counter, step));

  // ── Output tuple ──
  std::vector<HloInstruction*> output_values;
  output_values.push_back(next_counter);

  // Final carries from last sub-iteration
  for (auto* carry : current_carries) {
    output_values.push_back(carry);
  }

  if (use_stacked) {
    output_values.push_back(current_stacked_accum);
  } else {
    for (auto* accum : current_accums) {
      output_values.push_back(accum);
    }
  }
  for (auto* src : sliced_inputs_in_body) {
    output_values.push_back(src);
  }
  for (auto* inv : tuple_inv_in_body) {
    output_values.push_back(inv);
  }

  body_builder.AddInstruction(HloInstruction::CreateTuple(output_values));

  HloComputation* body_comp =
      computation->parent()->AddEmbeddedComputation(body_builder.Build());

  // ── Create the while instruction ──
  HloInstruction* while_inst = computation->AddInstruction(
      HloInstruction::CreateWhile(tuple_shape, cond_comp, body_comp,
                                   init_tuple));

  // ── Extract accumulators and replace concatenates ──
  if (use_stacked) {
    HloInstruction* stacked_result = computation->AddInstruction(
        HloInstruction::CreateGetTupleElement(while_inst, accum_off));
    for (int i = 0; i < num_accum; ++i) {
      std::vector<int64_t> starts(2, 0);
      starts[stack_dim] = accum_cum_offsets[i];
      std::vector<int64_t> limits(2);
      limits[slice_dim] = num_iters;
      limits[stack_dim] = accum_cum_offsets[i] + accum_widths[i];
      std::vector<int64_t> strides(2, 1);
      HloInstruction* result = computation->AddInstruction(
          HloInstruction::CreateSlice(body.concats[i]->shape(),
                                       stacked_result, starts, limits, strides));
      TF_RETURN_IF_ERROR(body.concats[i]->ReplaceAllUsesWith(result));
      VLOG(2) << "Replaced " << body.concats[i]->name()
              << " with stacked slice [" << starts[stack_dim]
              << ":" << limits[stack_dim] << "]";
    }
  } else {
    for (int i = 0; i < num_accum; ++i) {
      int idx = accum_off + i;
      HloInstruction* result = computation->AddInstruction(
          HloInstruction::CreateGetTupleElement(while_inst, idx));
      TF_RETURN_IF_ERROR(body.concats[i]->ReplaceAllUsesWith(result));
      VLOG(2) << "Replaced " << body.concats[i]->name()
              << " with GTE(" << idx << ") from while loop";
    }
  }

  // ── Remove dead concatenates and cascade ──
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

    VLOG(1) << "Found " << all_chains.size() << " slice chains";

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
                           BuildWhileLoop(computation, group_chains, *body,
                                          unroll_factor_));
      if (!transformed && unroll_factor_ > 1) {
        VLOG(1) << "Unroll factor " << unroll_factor_
                << " failed, retrying with unroll_factor=1";
        // Re-extract body since BuildWhileLoop may have partially modified state
        auto body2 = ExtractGroupedIterationBody(group_chains, computation);
        if (body2.has_value()) {
          TF_ASSIGN_OR_RETURN(transformed,
                               BuildWhileLoop(computation, group_chains, *body2,
                                              1));
        }
      }
      if (transformed) {
        changed = true;
      }
    }
  }

  return changed;
}

}  // namespace gpu
}  // namespace xla
