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

#include "xla/service/gpu/transforms/loopify/serial_scan_emitter.h"

#include <cctype>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/tsl/platform/errors.h"
#include "xla/util.h"

namespace xla {
namespace gpu {

// ─────────────────────────────────────────────────────────────────────────────
// Body ops deserialization
// ─────────────────────────────────────────────────────────────────────────────

static PrimitiveType ParseTypeName(absl::string_view name) {
  if (name == "F16") return F16;
  if (name == "F32") return F32;
  if (name == "F64") return F64;
  if (name == "BF16") return BF16;
  if (name == "PRED") return PRED;
  if (name == "S8") return S8;
  if (name == "S16") return S16;
  if (name == "S32") return S32;
  if (name == "S64") return S64;
  if (name == "U8") return U8;
  if (name == "U16") return U16;
  if (name == "U32") return U32;
  if (name == "U64") return U64;
  return PRIMITIVE_TYPE_INVALID;
}

static std::vector<int> ParseIntList(absl::string_view s) {
  std::vector<int> result;
  for (absl::string_view part : absl::StrSplit(s, ',')) {
    int v;
    if (absl::SimpleAtoi(part, &v)) {
      result.push_back(v);
    }
  }
  return result;
}

static std::vector<SerializedInst> ParseBodyOps(absl::string_view serialized) {
  std::vector<SerializedInst> ops;
  for (absl::string_view token : absl::StrSplit(serialized, '|')) {
    if (token.empty()) continue;
    SerializedInst inst;
    inst.type = PRIMITIVE_TYPE_INVALID;
    inst.constant_value = 0.0;

    std::vector<absl::string_view> parts = absl::StrSplit(token, '/');
    if (parts.empty()) continue;

    absl::string_view first = parts[0];

    // Parameter: P<index>/<type>
    if (first.size() >= 2 && first[0] == 'P' && std::isdigit(first[1])) {
      inst.opcode = std::string(first);  // e.g. "P0", "P1"
      if (parts.size() >= 2) {
        inst.type = ParseTypeName(parts[1]);
      }
      ops.push_back(std::move(inst));
      continue;
    }

    // Constant: K/<type>/<value>
    if (first == "K") {
      inst.opcode = "K";
      if (parts.size() >= 3) {
        inst.type = ParseTypeName(parts[1]);
        double v;
        if (absl::SimpleAtod(parts[2], &v)) {
          inst.constant_value = v;
        }
      }
      ops.push_back(std::move(inst));
      continue;
    }

    // Tuple: T/<op0>,<op1>,...
    if (first == "T") {
      inst.opcode = "T";
      if (parts.size() >= 2) {
        inst.operands = ParseIntList(parts[1]);
      }
      ops.push_back(std::move(inst));
      continue;
    }

    // Compare: CMP/<direction>/<op0>,<op1>/<type>
    if (first == "CMP") {
      inst.opcode = "CMP";
      if (parts.size() >= 4) {
        inst.extra = std::string(parts[1]);  // direction
        inst.operands = ParseIntList(parts[2]);
        inst.type = ParseTypeName(parts[3]);
      }
      ops.push_back(std::move(inst));
      continue;
    }

    // Convert: CVT/<op0>/<to_type>
    if (first == "CVT") {
      inst.opcode = "CVT";
      if (parts.size() >= 3) {
        int v;
        if (absl::SimpleAtoi(parts[1], &v)) {
          inst.operands.push_back(v);
        }
        inst.type = ParseTypeName(parts[2]);
      }
      ops.push_back(std::move(inst));
      continue;
    }

    // Passthrough: pass/<op0>/<type>
    if (first == "pass") {
      inst.opcode = "pass";
      if (parts.size() >= 3) {
        int v;
        if (absl::SimpleAtoi(parts[1], &v)) {
          inst.operands.push_back(v);
        }
        inst.type = ParseTypeName(parts[2]);
      }
      ops.push_back(std::move(inst));
      continue;
    }

    // Generic: <opcode>/<operands>/<type>
    // Handles unary: <opcode>/<op0>/<type>
    //         binary: <opcode>/<op0>,<op1>/<type>
    //         ternary: <opcode>/<op0>,<op1>,<op2>/<type>
    inst.opcode = std::string(first);
    if (parts.size() >= 3) {
      inst.operands = ParseIntList(parts[1]);
      inst.type = ParseTypeName(parts[2]);
    } else if (parts.size() == 2) {
      inst.operands = ParseIntList(parts[1]);
    }
    ops.push_back(std::move(inst));
  }
  return ops;
}

// ─────────────────────────────────────────────────────────────────────────────
// Config parsing
// ─────────────────────────────────────────────────────────────────────────────

absl::StatusOr<SerialScanConfig> SerialScanFusion::ParseConfig(
    const HloFusionInstruction& fusion) {
  auto gpu_config = fusion.backend_config<GpuBackendConfig>();
  if (!gpu_config.ok()) {
    return absl::InternalError("Failed to get GpuBackendConfig");
  }
  const FusionBackendConfig& backend_config =
      gpu_config->fusion_backend_config();
  const std::string& name = backend_config.custom_fusion_config().name();

  SerialScanConfig config;
  config.num_levels = 0;
  config.slice_dim = 0;
  config.cell_dim = 1;
  config.num_carries = 0;
  config.num_outputs = 0;
  config.num_sliced = 0;
  config.num_offset = 0;
  config.num_invariants = 0;

  // Parse "serial_scan;key=val;key=val;..."
  for (absl::string_view token : absl::StrSplit(name, ';')) {
    if (token == "serial_scan") continue;

    std::vector<absl::string_view> kv = absl::StrSplit(token, '=');
    if (kv.size() != 2) continue;

    absl::string_view key = kv[0];
    absl::string_view val = kv[1];

    if (key == "nlev") {
      if (!absl::SimpleAtoi(val, &config.num_levels))
        return absl::InvalidArgumentError("Bad nlev");
    } else if (key == "sd") {
      if (!absl::SimpleAtoi(val, &config.slice_dim))
        return absl::InvalidArgumentError("Bad sd");
    } else if (key == "nc") {
      if (!absl::SimpleAtoi(val, &config.num_carries))
        return absl::InvalidArgumentError("Bad nc");
    } else if (key == "no") {
      if (!absl::SimpleAtoi(val, &config.num_outputs))
        return absl::InvalidArgumentError("Bad no");
    } else if (key == "ns") {
      if (!absl::SimpleAtoi(val, &config.num_sliced))
        return absl::InvalidArgumentError("Bad ns");
    } else if (key == "noff") {
      if (!absl::SimpleAtoi(val, &config.num_offset))
        return absl::InvalidArgumentError("Bad noff");
    } else if (key == "ni") {
      if (!absl::SimpleAtoi(val, &config.num_invariants))
        return absl::InvalidArgumentError("Bad ni");
    } else if (key == "offvals") {
      for (absl::string_view v : absl::StrSplit(val, ',')) {
        int64_t ov;
        if (!absl::SimpleAtoi(v, &ov))
          return absl::InvalidArgumentError("Bad offvals");
        config.offset_values.push_back(ov);
      }
    } else if (key == "body_ops") {
      // Parse serialized body computation.
      config.body_ops = ParseBodyOps(val);
    }
  }

  // Derive cell_dim from slice_dim (the other dimension of a 2D tensor).
  config.cell_dim = (config.slice_dim == 0) ? 1 : 0;

  if (config.num_levels == 0)
    return absl::InvalidArgumentError("serial_scan: nlev not set");
  if (config.body_ops.empty())
    return absl::InvalidArgumentError("serial_scan: body_ops not set");

  return config;
}

// ─────────────────────────────────────────────────────────────────────────────
// Launch dimensions: one thread per cell
// ─────────────────────────────────────────────────────────────────────────────

LaunchDimensions SerialScanFusion::launch_dimensions() const {
  // Single thread for now — the entry function processes all cells
  // sequentially via scf.for. TODO: parallelize with thread indexing.
  return LaunchDimensions(1, 1);
}

// ─────────────────────────────────────────────────────────────────────────────
// Indexing maps (best-effort for analysis; the real indexing is in MLIR)
// ─────────────────────────────────────────────────────────────────────────────

std::optional<IndexingMap>
SerialScanFusion::ComputeThreadIdToOutputIndexing(
    int64_t root_index, mlir::MLIRContext* mlir_context) const {
  // Each thread writes one full row/column of the output.
  // For a rigorous IndexingMap we'd need to express:
  //   output[cell, k] where cell = blockIdx.x * 256 + threadIdx.x,
  //                         k = symbol_0 in [0, nlev)
  // For now, return nullopt; the MLIR emission handles indexing directly.
  return std::nullopt;
}

std::optional<IndexingMap>
SerialScanFusion::ComputeThreadIdToInputIndexing(
    int64_t root_index, int64_t hero_operand_index,
    mlir::MLIRContext* mlir_context) const {
  return std::nullopt;
}

// ─────────────────────────────────────────────────────────────────────────────
// Scalar HLO → MLIR translation
// ─────────────────────────────────────────────────────────────────────────────

static mlir::arith::CmpFPredicate GetCmpFPredicate(
    ComparisonDirection direction) {
  switch (direction) {
    case ComparisonDirection::kEq: return mlir::arith::CmpFPredicate::OEQ;
    case ComparisonDirection::kNe: return mlir::arith::CmpFPredicate::ONE;
    case ComparisonDirection::kLt: return mlir::arith::CmpFPredicate::OLT;
    case ComparisonDirection::kLe: return mlir::arith::CmpFPredicate::OLE;
    case ComparisonDirection::kGt: return mlir::arith::CmpFPredicate::OGT;
    case ComparisonDirection::kGe: return mlir::arith::CmpFPredicate::OGE;
  }
}

static mlir::arith::CmpIPredicate GetCmpIPredicate(
    ComparisonDirection direction) {
  switch (direction) {
    case ComparisonDirection::kEq: return mlir::arith::CmpIPredicate::eq;
    case ComparisonDirection::kNe: return mlir::arith::CmpIPredicate::ne;
    case ComparisonDirection::kLt: return mlir::arith::CmpIPredicate::slt;
    case ComparisonDirection::kLe: return mlir::arith::CmpIPredicate::sle;
    case ComparisonDirection::kGt: return mlir::arith::CmpIPredicate::sgt;
    case ComparisonDirection::kGe: return mlir::arith::CmpIPredicate::sge;
  }
}

static bool IsFloatType(PrimitiveType type) {
  return type == F16 || type == F32 || type == F64 || type == BF16;
}

absl::StatusOr<mlir::Value> SerialScanFusion::EmitScalarOp(
    mlir::ImplicitLocOpBuilder& b,
    const HloInstruction* inst,
    const absl::flat_hash_map<const HloInstruction*, mlir::Value>& vm) {

  auto op = [&](int i) -> mlir::Value { return vm.at(inst->operand(i)); };
  PrimitiveType elem_type = inst->shape().element_type();

  switch (inst->opcode()) {
    // ── Binary arithmetic ──
    case HloOpcode::kAdd:
      if (IsFloatType(elem_type))
        return b.create<mlir::arith::AddFOp>(op(0), op(1)).getResult();
      return b.create<mlir::arith::AddIOp>(op(0), op(1)).getResult();

    case HloOpcode::kSubtract:
      if (IsFloatType(elem_type))
        return b.create<mlir::arith::SubFOp>(op(0), op(1)).getResult();
      return b.create<mlir::arith::SubIOp>(op(0), op(1)).getResult();

    case HloOpcode::kMultiply:
      if (IsFloatType(elem_type))
        return b.create<mlir::arith::MulFOp>(op(0), op(1)).getResult();
      return b.create<mlir::arith::MulIOp>(op(0), op(1)).getResult();

    case HloOpcode::kDivide:
      if (IsFloatType(elem_type))
        return b.create<mlir::arith::DivFOp>(op(0), op(1)).getResult();
      return b.create<mlir::arith::DivSIOp>(op(0), op(1)).getResult();

    case HloOpcode::kRemainder:
      if (IsFloatType(elem_type))
        return b.create<mlir::arith::RemFOp>(op(0), op(1)).getResult();
      return b.create<mlir::arith::RemSIOp>(op(0), op(1)).getResult();

    // ── Unary arithmetic ──
    case HloOpcode::kNegate:
      if (IsFloatType(elem_type))
        return b.create<mlir::arith::NegFOp>(op(0)).getResult();
      // Integer negate: 0 - x
      return b.create<mlir::arith::SubIOp>(
          b.create<mlir::arith::ConstantIntOp>(0, op(0).getType()),
          op(0)).getResult();

    case HloOpcode::kAbs:
      return b.create<mlir::math::AbsFOp>(op(0)).getResult();

    case HloOpcode::kSqrt:
      return b.create<mlir::math::SqrtOp>(op(0)).getResult();

    case HloOpcode::kRsqrt:
      return b.create<mlir::math::RsqrtOp>(op(0)).getResult();

    case HloOpcode::kExp:
      return b.create<mlir::math::ExpOp>(op(0)).getResult();

    case HloOpcode::kLog:
      return b.create<mlir::math::LogOp>(op(0)).getResult();

    case HloOpcode::kCeil:
      return b.create<mlir::math::CeilOp>(op(0)).getResult();

    case HloOpcode::kFloor:
      return b.create<mlir::math::FloorOp>(op(0)).getResult();

    case HloOpcode::kTanh:
      return b.create<mlir::math::TanhOp>(op(0)).getResult();

    // ── Min / Max ──
    case HloOpcode::kMaximum:
      if (IsFloatType(elem_type))
        return b.create<mlir::arith::MaximumFOp>(op(0), op(1)).getResult();
      return b.create<mlir::arith::MaxSIOp>(op(0), op(1)).getResult();

    case HloOpcode::kMinimum:
      if (IsFloatType(elem_type))
        return b.create<mlir::arith::MinimumFOp>(op(0), op(1)).getResult();
      return b.create<mlir::arith::MinSIOp>(op(0), op(1)).getResult();

    // ── Compare ──
    case HloOpcode::kCompare:
      if (IsFloatType(inst->operand(0)->shape().element_type()))
        return b.create<mlir::arith::CmpFOp>(
            GetCmpFPredicate(inst->comparison_direction()),
            op(0), op(1)).getResult();
      return b.create<mlir::arith::CmpIOp>(
          GetCmpIPredicate(inst->comparison_direction()),
          op(0), op(1)).getResult();

    // ── Select ──
    case HloOpcode::kSelect:
      return b.create<mlir::arith::SelectOp>(
          op(0), op(1), op(2)).getResult();

    // ── Clamp: max(lo, min(x, hi)) ──
    case HloOpcode::kClamp: {
      auto clamped_lo = b.create<mlir::arith::MaximumFOp>(op(0), op(1));
      return b.create<mlir::arith::MinimumFOp>(
          clamped_lo, op(2)).getResult();
    }

    // ── Power ──
    case HloOpcode::kPower:
      return b.create<mlir::math::PowFOp>(op(0), op(1)).getResult();

    // ── Constants ──
    case HloOpcode::kConstant: {
      const auto& literal = inst->literal();
      if (elem_type == F64) {
        double val = literal.GetFirstElement<double>();
        return b.create<mlir::arith::ConstantOp>(
            b.getFloatAttr(b.getF64Type(), val)).getResult();
      } else if (elem_type == F32) {
        float val = literal.GetFirstElement<float>();
        return b.create<mlir::arith::ConstantOp>(
            b.getFloatAttr(b.getF32Type(), val)).getResult();
      } else if (elem_type == PRED) {
        bool val = literal.GetFirstElement<bool>();
        return b.create<mlir::arith::ConstantOp>(
            b.getIntegerAttr(b.getI1Type(), val ? 1 : 0)).getResult();
      } else if (elem_type == S8) {
        int8_t val = literal.GetFirstElement<int8_t>();
        return b.create<mlir::arith::ConstantOp>(
            b.getIntegerAttr(b.getIntegerType(8), val)).getResult();
      } else if (elem_type == S32) {
        int32_t val = literal.GetFirstElement<int32_t>();
        return b.create<mlir::arith::ConstantOp>(
            b.getIntegerAttr(b.getI32Type(), val)).getResult();
      } else if (elem_type == S64) {
        int64_t val = literal.GetFirstElement<int64_t>();
        return b.create<mlir::arith::ConstantOp>(
            b.getIntegerAttr(b.getI64Type(), val)).getResult();
      }
      return absl::UnimplementedError(
          absl::StrCat("Unsupported constant type: ",
                       PrimitiveType_Name(elem_type)));
    }

    // ── Type conversions ──
    case HloOpcode::kConvert: {
      PrimitiveType src = inst->operand(0)->shape().element_type();
      PrimitiveType dst = elem_type;
      mlir::Value operand = op(0);

      // Float widening/narrowing.
      if (IsFloatType(src) && IsFloatType(dst)) {
        int src_bits = primitive_util::BitWidth(src);
        int dst_bits = primitive_util::BitWidth(dst);
        if (dst_bits > src_bits)
          return b.create<mlir::arith::ExtFOp>(
              b.getF64Type(), operand).getResult();
        return b.create<mlir::arith::TruncFOp>(
            b.getF32Type(), operand).getResult();
      }
      // Int → Float.
      if (!IsFloatType(src) && IsFloatType(dst)) {
        mlir::Type ft = dst == F64 ? (mlir::Type)b.getF64Type()
                                   : (mlir::Type)b.getF32Type();
        if (src == PRED) {
          // bool → float: uitofp
          return b.create<mlir::arith::UIToFPOp>(ft, operand).getResult();
        }
        return b.create<mlir::arith::SIToFPOp>(ft, operand).getResult();
      }
      // Float → Int.
      if (IsFloatType(src) && !IsFloatType(dst)) {
        mlir::Type it = b.getIntegerType(primitive_util::BitWidth(dst));
        return b.create<mlir::arith::FPToSIOp>(it, operand).getResult();
      }
      // Int → Int (sign extend or truncate).
      {
        int src_bits = primitive_util::BitWidth(src);
        int dst_bits = primitive_util::BitWidth(dst);
        mlir::Type it = b.getIntegerType(dst_bits);
        if (dst_bits > src_bits)
          return b.create<mlir::arith::ExtSIOp>(it, operand).getResult();
        return b.create<mlir::arith::TruncIOp>(it, operand).getResult();
      }
    }

    // ── Structural no-ops at scalar level ──
    case HloOpcode::kBroadcast:
    case HloOpcode::kReshape:
    case HloOpcode::kBitcast:
    case HloOpcode::kCopy:
      return op(0);

    // ── Boolean ops ──
    case HloOpcode::kAnd:
      return b.create<mlir::arith::AndIOp>(op(0), op(1)).getResult();
    case HloOpcode::kOr:
      return b.create<mlir::arith::OrIOp>(op(0), op(1)).getResult();
    case HloOpcode::kXor:
      return b.create<mlir::arith::XOrIOp>(op(0), op(1)).getResult();
    case HloOpcode::kNot: {
      // Bitwise NOT: xor with all-ones.
      auto ones = b.create<mlir::arith::ConstantOp>(
          b.getIntegerAttr(op(0).getType(), -1));
      return b.create<mlir::arith::XOrIOp>(op(0), ones).getResult();
    }

    // ── Tuple: should not appear in the scalar body ──
    case HloOpcode::kTuple:
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kParameter:
      return absl::InternalError(
          absl::StrCat("Structural op should not reach EmitScalarOp: ",
                       HloOpcodeString(inst->opcode())));

    default:
      return absl::UnimplementedError(
          absl::StrCat("Unsupported HLO opcode in serial scan body: ",
                       HloOpcodeString(inst->opcode())));
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Serialized instruction → MLIR translation
// ─────────────────────────────────────────────────────────────────────────────

static mlir::Type GetMlirFloatType(mlir::ImplicitLocOpBuilder& b,
                                    PrimitiveType type) {
  switch (type) {
    case F16: return b.getF16Type();
    case F32: return b.getF32Type();
    case F64: return b.getF64Type();
    case BF16: return b.getBF16Type();
    default: return b.getF32Type();
  }
}

absl::StatusOr<mlir::Value> SerialScanFusion::EmitSerializedOp(
    mlir::ImplicitLocOpBuilder& b,
    const SerializedInst& inst,
    const std::vector<mlir::Value>& values) {

  auto get = [&](int idx) -> mlir::Value { return values[inst.operands[idx]]; };

  // ── Constants ──
  if (inst.opcode == "K") {
    if (inst.type == PRED) {
      return b.create<mlir::arith::ConstantOp>(
          b.getIntegerAttr(b.getI1Type(), inst.constant_value != 0.0 ? 1 : 0))
          .getResult();
    } else if (inst.type == F32) {
      return b.create<mlir::arith::ConstantOp>(
          b.getFloatAttr(b.getF32Type(), inst.constant_value)).getResult();
    } else if (inst.type == F64) {
      return b.create<mlir::arith::ConstantOp>(
          b.getFloatAttr(b.getF64Type(), inst.constant_value)).getResult();
    } else if (inst.type == F16) {
      return b.create<mlir::arith::ConstantOp>(
          b.getFloatAttr(b.getF16Type(), inst.constant_value)).getResult();
    } else if (inst.type == BF16) {
      return b.create<mlir::arith::ConstantOp>(
          b.getFloatAttr(b.getBF16Type(), inst.constant_value)).getResult();
    } else if (inst.type == S8) {
      return b.create<mlir::arith::ConstantOp>(
          b.getIntegerAttr(b.getIntegerType(8),
                           static_cast<int64_t>(inst.constant_value)))
          .getResult();
    } else if (inst.type == S16) {
      return b.create<mlir::arith::ConstantOp>(
          b.getIntegerAttr(b.getIntegerType(16),
                           static_cast<int64_t>(inst.constant_value)))
          .getResult();
    } else if (inst.type == S32) {
      return b.create<mlir::arith::ConstantOp>(
          b.getIntegerAttr(b.getI32Type(),
                           static_cast<int64_t>(inst.constant_value)))
          .getResult();
    } else if (inst.type == S64) {
      return b.create<mlir::arith::ConstantOp>(
          b.getIntegerAttr(b.getI64Type(),
                           static_cast<int64_t>(inst.constant_value)))
          .getResult();
    } else if (inst.type == U8) {
      return b.create<mlir::arith::ConstantOp>(
          b.getIntegerAttr(b.getIntegerType(8),
                           static_cast<int64_t>(inst.constant_value)))
          .getResult();
    } else if (inst.type == U16) {
      return b.create<mlir::arith::ConstantOp>(
          b.getIntegerAttr(b.getIntegerType(16),
                           static_cast<int64_t>(inst.constant_value)))
          .getResult();
    } else if (inst.type == U32) {
      return b.create<mlir::arith::ConstantOp>(
          b.getIntegerAttr(b.getI32Type(),
                           static_cast<int64_t>(inst.constant_value)))
          .getResult();
    } else if (inst.type == U64) {
      return b.create<mlir::arith::ConstantOp>(
          b.getIntegerAttr(b.getI64Type(),
                           static_cast<int64_t>(inst.constant_value)))
          .getResult();
    }
    return absl::UnimplementedError(
        absl::StrCat("Unsupported constant type: ",
                     PrimitiveType_Name(inst.type)));
  }

  // ── Passthrough (broadcast/reshape/bitcast/copy) ──
  if (inst.opcode == "pass") {
    return get(0);
  }

  // ── Compare ──
  if (inst.opcode == "CMP") {
    // Determine if operand is float by checking its MLIR type.
    mlir::Value lhs = get(0);
    bool is_float = mlir::isa<mlir::FloatType>(lhs.getType());
    if (is_float) {
      mlir::arith::CmpFPredicate pred;
      if (inst.extra == "EQ") pred = mlir::arith::CmpFPredicate::OEQ;
      else if (inst.extra == "NE") pred = mlir::arith::CmpFPredicate::ONE;
      else if (inst.extra == "LT") pred = mlir::arith::CmpFPredicate::OLT;
      else if (inst.extra == "LE") pred = mlir::arith::CmpFPredicate::OLE;
      else if (inst.extra == "GT") pred = mlir::arith::CmpFPredicate::OGT;
      else if (inst.extra == "GE") pred = mlir::arith::CmpFPredicate::OGE;
      else return absl::InvalidArgumentError(
          absl::StrCat("Unknown CMP direction: ", inst.extra));
      return b.create<mlir::arith::CmpFOp>(pred, get(0), get(1)).getResult();
    } else {
      mlir::arith::CmpIPredicate pred;
      if (inst.extra == "EQ") pred = mlir::arith::CmpIPredicate::eq;
      else if (inst.extra == "NE") pred = mlir::arith::CmpIPredicate::ne;
      else if (inst.extra == "LT") pred = mlir::arith::CmpIPredicate::slt;
      else if (inst.extra == "LE") pred = mlir::arith::CmpIPredicate::sle;
      else if (inst.extra == "GT") pred = mlir::arith::CmpIPredicate::sgt;
      else if (inst.extra == "GE") pred = mlir::arith::CmpIPredicate::sge;
      else return absl::InvalidArgumentError(
          absl::StrCat("Unknown CMP direction: ", inst.extra));
      return b.create<mlir::arith::CmpIOp>(pred, get(0), get(1)).getResult();
    }
  }

  // ── Convert ──
  if (inst.opcode == "CVT") {
    mlir::Value operand = get(0);
    bool src_float = mlir::isa<mlir::FloatType>(operand.getType());
    bool dst_float = IsFloatType(inst.type);

    if (src_float && dst_float) {
      mlir::Type dst_mlir = GetMlirFloatType(b, inst.type);
      unsigned src_bits = mlir::cast<mlir::FloatType>(operand.getType()).getWidth();
      unsigned dst_bits = mlir::cast<mlir::FloatType>(dst_mlir).getWidth();
      if (dst_bits > src_bits)
        return b.create<mlir::arith::ExtFOp>(dst_mlir, operand).getResult();
      return b.create<mlir::arith::TruncFOp>(dst_mlir, operand).getResult();
    }
    if (!src_float && dst_float) {
      mlir::Type ft = GetMlirFloatType(b, inst.type);
      // Check if source is i1 (PRED)
      if (auto itype = mlir::dyn_cast<mlir::IntegerType>(operand.getType());
          itype && itype.getWidth() == 1) {
        return b.create<mlir::arith::UIToFPOp>(ft, operand).getResult();
      }
      return b.create<mlir::arith::SIToFPOp>(ft, operand).getResult();
    }
    if (src_float && !dst_float) {
      int dst_bits = primitive_util::BitWidth(inst.type);
      mlir::Type it = b.getIntegerType(dst_bits);
      return b.create<mlir::arith::FPToSIOp>(it, operand).getResult();
    }
    // Int → Int
    {
      int dst_bits = primitive_util::BitWidth(inst.type);
      mlir::Type it = b.getIntegerType(dst_bits);
      unsigned src_bits =
          mlir::cast<mlir::IntegerType>(operand.getType()).getWidth();
      if (static_cast<unsigned>(dst_bits) > src_bits)
        return b.create<mlir::arith::ExtSIOp>(it, operand).getResult();
      return b.create<mlir::arith::TruncIOp>(it, operand).getResult();
    }
  }

  // ── Binary arithmetic ──
  if (inst.opcode == "add") {
    if (IsFloatType(inst.type))
      return b.create<mlir::arith::AddFOp>(get(0), get(1)).getResult();
    return b.create<mlir::arith::AddIOp>(get(0), get(1)).getResult();
  }
  if (inst.opcode == "subtract") {
    if (IsFloatType(inst.type))
      return b.create<mlir::arith::SubFOp>(get(0), get(1)).getResult();
    return b.create<mlir::arith::SubIOp>(get(0), get(1)).getResult();
  }
  if (inst.opcode == "multiply") {
    if (IsFloatType(inst.type))
      return b.create<mlir::arith::MulFOp>(get(0), get(1)).getResult();
    return b.create<mlir::arith::MulIOp>(get(0), get(1)).getResult();
  }
  if (inst.opcode == "divide") {
    if (IsFloatType(inst.type))
      return b.create<mlir::arith::DivFOp>(get(0), get(1)).getResult();
    return b.create<mlir::arith::DivSIOp>(get(0), get(1)).getResult();
  }
  if (inst.opcode == "remainder") {
    if (IsFloatType(inst.type))
      return b.create<mlir::arith::RemFOp>(get(0), get(1)).getResult();
    return b.create<mlir::arith::RemSIOp>(get(0), get(1)).getResult();
  }

  // ── Unary arithmetic ──
  if (inst.opcode == "negate") {
    if (IsFloatType(inst.type))
      return b.create<mlir::arith::NegFOp>(get(0)).getResult();
    return b.create<mlir::arith::SubIOp>(
        b.create<mlir::arith::ConstantIntOp>(0, get(0).getType()),
        get(0)).getResult();
  }
  if (inst.opcode == "abs") {
    return b.create<mlir::math::AbsFOp>(get(0)).getResult();
  }
  if (inst.opcode == "sqrt") {
    return b.create<mlir::math::SqrtOp>(get(0)).getResult();
  }
  if (inst.opcode == "cbrt") {
    return b.create<mlir::math::CbrtOp>(get(0)).getResult();
  }
  if (inst.opcode == "rsqrt") {
    return b.create<mlir::math::RsqrtOp>(get(0)).getResult();
  }
  if (inst.opcode == "exp") {
    return b.create<mlir::math::ExpOp>(get(0)).getResult();
  }
  if (inst.opcode == "log") {
    return b.create<mlir::math::LogOp>(get(0)).getResult();
  }
  if (inst.opcode == "ceil") {
    return b.create<mlir::math::CeilOp>(get(0)).getResult();
  }
  if (inst.opcode == "floor") {
    return b.create<mlir::math::FloorOp>(get(0)).getResult();
  }
  if (inst.opcode == "tanh") {
    return b.create<mlir::math::TanhOp>(get(0)).getResult();
  }

  // ── Min / Max ──
  if (inst.opcode == "maximum") {
    if (IsFloatType(inst.type))
      return b.create<mlir::arith::MaximumFOp>(get(0), get(1)).getResult();
    return b.create<mlir::arith::MaxSIOp>(get(0), get(1)).getResult();
  }
  if (inst.opcode == "minimum") {
    if (IsFloatType(inst.type))
      return b.create<mlir::arith::MinimumFOp>(get(0), get(1)).getResult();
    return b.create<mlir::arith::MinSIOp>(get(0), get(1)).getResult();
  }

  // ── Power ──
  if (inst.opcode == "power") {
    return b.create<mlir::math::PowFOp>(get(0), get(1)).getResult();
  }

  // ── Select ──
  if (inst.opcode == "select") {
    return b.create<mlir::arith::SelectOp>(
        get(0), get(1), get(2)).getResult();
  }

  // ── Clamp: max(lo, min(x, hi)) ──
  if (inst.opcode == "clamp") {
    auto clamped_lo = b.create<mlir::arith::MaximumFOp>(get(0), get(1));
    return b.create<mlir::arith::MinimumFOp>(clamped_lo, get(2)).getResult();
  }

  // ── Boolean ops ──
  if (inst.opcode == "and") {
    return b.create<mlir::arith::AndIOp>(get(0), get(1)).getResult();
  }
  if (inst.opcode == "or") {
    return b.create<mlir::arith::OrIOp>(get(0), get(1)).getResult();
  }
  if (inst.opcode == "xor") {
    return b.create<mlir::arith::XOrIOp>(get(0), get(1)).getResult();
  }
  if (inst.opcode == "not") {
    auto ones = b.create<mlir::arith::ConstantOp>(
        b.getIntegerAttr(get(0).getType(), -1));
    return b.create<mlir::arith::XOrIOp>(get(0), ones).getResult();
  }

  return absl::UnimplementedError(
      absl::StrCat("Unsupported serialized opcode: ", inst.opcode));
}

// ─────────────────────────────────────────────────────────────────────────────
// Entry function emission
// ─────────────────────────────────────────────────────────────────────────────

absl::Status SerialScanFusion::EmitEntryFunction(
    const emitters::PartitionedComputations& computations,
    const emitters::CallTargetProvider& call_targets,
    mlir::func::FuncOp entry_function,
    const HloFusionInstruction& fusion) const {

  TF_ASSIGN_OR_RETURN(auto config, ParseConfig(fusion));

  VLOG(1) << "SerialScanFusion: emitting entry function"
          << " nlev=" << config.num_levels
          << " nc=" << config.num_carries
          << " no=" << config.num_outputs
          << " ns=" << config.num_sliced
          << " noff=" << config.num_offset
          << " ni=" << config.num_invariants
          << " body_ops=" << config.body_ops.size() << " instructions";

  mlir::ImplicitLocOpBuilder builder(entry_function.getLoc(), entry_function);
  builder.setInsertionPointToStart(entry_function.addEntryBlock());

  // Separate function args into inputs and outputs.
  int num_fusion_params =
      fusion.fused_instructions_computation()->num_parameters();
  auto input_args = entry_function.getArguments().take_front(num_fusion_params);
  auto output_args = entry_function.getArguments().drop_front(num_fusion_params);

  // Sanity checks.
  if (static_cast<int>(output_args.size()) != config.num_outputs) {
    return absl::InternalError(absl::StrCat(
        "Output count mismatch: ", output_args.size(),
        " vs config.num_outputs=", config.num_outputs));
  }

  // Get ncells from the first sliced input parameter shape.
  int64_t ncells = fusion.fused_instructions_computation()
      ->parameter_instruction(config.sliced_start())
      ->shape().dimensions(config.cell_dim);

  // ── Constants ──
  mlir::Value c0 = builder.create<mlir::arith::ConstantIndexOp>(0);
  mlir::Value c1 = builder.create<mlir::arith::ConstantIndexOp>(1);
  mlir::Value c_nlev =
      builder.create<mlir::arith::ConstantIndexOp>(config.num_levels);
  mlir::Value c_ncells =
      builder.create<mlir::arith::ConstantIndexOp>(ncells);

  // ── Sequential for over cells (single-thread for now) ──
  // Output tensors are iter_args of the outer cell loop.
  llvm::SmallVector<mlir::Value> output_tensor_args(
      output_args.begin(), output_args.end());

  auto cell_loop = builder.create<mlir::scf::ForOp>(
      c0, c_ncells, c1, output_tensor_args,
      [&](mlir::OpBuilder& cb, mlir::Location cloc, mlir::Value cell,
          mlir::ValueRange cell_iter_args) {
    mlir::ImplicitLocOpBuilder nb(cloc, cb);

    // cell_iter_args are the output tensors being updated.
    llvm::SmallVector<mlir::Value> cur_outs(cell_iter_args.begin(),
                                             cell_iter_args.end());

    // ── Extract carry initial values: carry_init[cell, 0] ──
    llvm::SmallVector<mlir::Value> carry_inits;
    for (int i = 0; i < config.num_carries; ++i) {
      mlir::Value tensor = input_args[config.carry_start() + i];
      llvm::SmallVector<mlir::Value> idx;
      if (config.cell_dim == 0) {
        idx = {cell, c0};
      } else {
        idx = {c0, cell};
      }
      carry_inits.push_back(
          nb.create<mlir::tensor::ExtractOp>(tensor, idx));
    }

    // ── Extract invariant values (done once per cell, before the loop) ──
    llvm::SmallVector<mlir::Value> invariant_vals;
    for (int i = 0; i < config.num_invariants; ++i) {
      mlir::Value tensor = input_args[config.invariant_start() + i];
      auto tensor_type =
          mlir::cast<mlir::RankedTensorType>(tensor.getType());
      llvm::SmallVector<mlir::Value> idx;
      if (tensor_type.getRank() == 2) {
        if (config.cell_dim == 0) {
          idx = {cell, c0};
        } else {
          idx = {c0, cell};
        }
      } else if (tensor_type.getRank() == 1) {
        idx = {cell};
      } else {
        idx = {};
      }
      invariant_vals.push_back(
          nb.create<mlir::tensor::ExtractOp>(tensor, idx));
    }

    // ── Fill k=0 from iter0_level_outputs ──
    for (int i = 0; i < config.num_outputs; ++i) {
      mlir::Value iter0_tensor = input_args[config.iter0_start() + i];
      llvm::SmallVector<mlir::Value> src_idx;
      if (config.cell_dim == 0) {
        src_idx = {cell, c0};
      } else {
        src_idx = {c0, cell};
      }
      mlir::Value iter0_val =
          nb.create<mlir::tensor::ExtractOp>(iter0_tensor, src_idx);

      llvm::SmallVector<mlir::Value> dst_idx;
      if (config.cell_dim == 0) {
        dst_idx = {cell, c0};
      } else {
        dst_idx = {c0, cell};
      }
      cur_outs[i] = nb.create<mlir::tensor::InsertOp>(
          iter0_val, cur_outs[i], dst_idx);
    }

    // ── scf.for over levels: carries (scalars) + output accums (tensors) ──
    llvm::SmallVector<mlir::Value> for_init_args;
    for (auto& c : carry_inits) for_init_args.push_back(c);
    for (auto& o : cur_outs) for_init_args.push_back(o);

    auto for_op = nb.create<mlir::scf::ForOp>(
        c1, c_nlev, c1, for_init_args,
        [&](mlir::OpBuilder& fb2, mlir::Location loc2, mlir::Value k,
            mlir::ValueRange iter_args) {
          mlir::ImplicitLocOpBuilder nb2(loc2, fb2);

          auto carry_args = iter_args.take_front(config.num_carries);
          auto accum_args = iter_args.drop_front(config.num_carries);

          // ── Extract sliced inputs at [cell, k] ──
          llvm::SmallVector<mlir::Value> sliced_vals;
          for (int i = 0; i < config.num_sliced; ++i) {
            mlir::Value tensor = input_args[config.sliced_start() + i];
            llvm::SmallVector<mlir::Value> idx;
            if (config.cell_dim == 0) {
              idx = {cell, k};
            } else {
              idx = {k, cell};
            }
            sliced_vals.push_back(
                nb2.create<mlir::tensor::ExtractOp>(tensor, idx));
          }

          // ── Extract offset inputs at [cell, k + offset] ──
          llvm::SmallVector<mlir::Value> offset_vals;
          for (int i = 0; i < config.num_offset; ++i) {
            mlir::Value tensor = input_args[config.offset_start() + i];
            mlir::Value off = nb2.create<mlir::arith::ConstantIndexOp>(
                config.offset_values[i]);
            mlir::Value k_off = nb2.create<mlir::arith::AddIOp>(k, off);
            mlir::Value max_idx = nb2.create<mlir::arith::ConstantIndexOp>(
                config.num_levels - 1);
            mlir::Value k_clamped_lo =
                nb2.create<mlir::arith::MaxSIOp>(k_off, c0);
            mlir::Value k_clamped =
                nb2.create<mlir::arith::MinSIOp>(k_clamped_lo, max_idx);
            llvm::SmallVector<mlir::Value> idx;
            if (config.cell_dim == 0) {
              idx = {cell, k_clamped};
            } else {
              idx = {k_clamped, cell};
            }
            offset_vals.push_back(
                nb2.create<mlir::tensor::ExtractOp>(tensor, idx));
          }

          // ── Map body parameters to MLIR values ──
          std::vector<mlir::Value> inst_values(config.body_ops.size());
          int bp = 0;
          for (int i = 0; i < static_cast<int>(config.body_ops.size()); ++i) {
            const auto& op = config.body_ops[i];
            if (op.opcode.size() >= 2 && op.opcode[0] == 'P' &&
                std::isdigit(op.opcode[1])) {
              if (bp < config.num_carries) {
                inst_values[i] = carry_args[bp];
              } else if (bp < config.num_carries + config.num_sliced) {
                inst_values[i] = sliced_vals[bp - config.num_carries];
              } else if (bp < config.num_carries + config.num_sliced +
                             config.num_offset) {
                inst_values[i] =
                    offset_vals[bp - config.num_carries - config.num_sliced];
              } else {
                inst_values[i] =
                    invariant_vals[bp - config.num_carries -
                                   config.num_sliced - config.num_offset];
              }
              bp++;
            }
          }

          // ── Translate body instructions to MLIR ──
          for (int i = 0; i < static_cast<int>(config.body_ops.size()); ++i) {
            const auto& op = config.body_ops[i];
            if (op.opcode.size() >= 2 && op.opcode[0] == 'P' &&
                std::isdigit(op.opcode[1]))
              continue;
            if (op.opcode == "T") continue;

            auto result_or = EmitSerializedOp(nb2, op, inst_values);
            if (!result_or.ok()) {
              LOG(FATAL) << "EmitSerializedOp failed for opcode="
                         << op.opcode << " at index " << i << ": "
                         << result_or.status().message();
            }
            inst_values[i] = *result_or;
          }

          // ── Extract outputs from the root tuple ──
          const auto& root_op = config.body_ops.back();

          llvm::SmallVector<mlir::Value> new_carries;
          for (int i = 0; i < config.num_carries; ++i) {
            new_carries.push_back(inst_values[root_op.operands[i]]);
          }

          llvm::SmallVector<mlir::Value> level_outputs;
          for (int i = 0; i < config.num_outputs; ++i) {
            level_outputs.push_back(
                inst_values[root_op.operands[config.num_carries + i]]);
          }

          // ── Insert level outputs into accumulator tensors ──
          llvm::SmallVector<mlir::Value> new_accums;
          for (int i = 0; i < config.num_outputs; ++i) {
            llvm::SmallVector<mlir::Value> idx;
            if (config.cell_dim == 0) {
              idx = {cell, k};
            } else {
              idx = {k, cell};
            }
            new_accums.push_back(
                nb2.create<mlir::tensor::InsertOp>(
                    level_outputs[i], accum_args[i], idx));
          }

          // ── Yield: new carries + new accums ──
          llvm::SmallVector<mlir::Value> yield_vals;
          for (auto& v : new_carries) yield_vals.push_back(v);
          for (auto& v : new_accums) yield_vals.push_back(v);
          nb2.create<mlir::scf::YieldOp>(yield_vals);
        });

    // ── Yield updated output tensors from the cell loop ──
    auto for_results = for_op.getResults();
    auto accum_results = for_results.drop_front(config.num_carries);
    llvm::SmallVector<mlir::Value> cell_yield;
    for (int i = 0; i < config.num_outputs; ++i) {
      cell_yield.push_back(accum_results[i]);
    }
    nb.create<mlir::scf::YieldOp>(cell_yield);
  });

  builder.create<mlir::func::ReturnOp>(cell_loop.getResults());

  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla
