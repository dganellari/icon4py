/* Copyright 2024 ETH Zurich and MeteoSwiss.
 *
 * Licensed under the Apache License, Version 2.0.
 */

#include "xla/service/gpu/transforms/loopify/loopify_unrolled_slices.h"


#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_matchers.h"

namespace xla {
namespace gpu {
namespace {

namespace op = ::xla::testing::opcode_matchers;

class LoopifyUnrolledSlicesTest : public HloHardwareIndependentTestBase {};

TEST_F(LoopifyUnrolledSlicesTest, SimpleSliceConcatBecomesWhile) {
  const char* hlo_string = R"(
    HloModule test

    ENTRY main {
      %p0_ = f64[4,100] parameter(0)
      %s0_ = f64[1,100] slice(%p0_), slice={[0:1], [0:100]}
      %s1_ = f64[1,100] slice(%p0_), slice={[1:2], [0:100]}
      %s2_ = f64[1,100] slice(%p0_), slice={[2:3], [0:100]}
      %s3_ = f64[1,100] slice(%p0_), slice={[3:4], [0:100]}
      %c1_ = f64[] constant(2.0)
      %b1_ = f64[1,100] broadcast(%c1_), dimensions={}
      %r0_ = f64[1,100] multiply(%s0_, %b1_)
      %r1_ = f64[1,100] multiply(%s1_, %b1_)
      %r2_ = f64[1,100] multiply(%s2_, %b1_)
      %r3_ = f64[1,100] multiply(%s3_, %b1_)
      ROOT %result_ = f64[4,100] concatenate(%r0_, %r1_, %r2_, %r3_), dimensions={0}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                           ParseAndReturnVerifiedModule(hlo_string));

  LoopifyUnrolledSlices pass(/*min_iterations=*/4);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get(), {}));

  EXPECT_TRUE(changed);

  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::GetTupleElement(op::While()));
}

TEST_F(LoopifyUnrolledSlicesTest, WithCarryState) {
  const char* hlo_string = R"(
    HloModule test

    ENTRY main {
      %p0_ = f64[4,100] parameter(0)
      %s0_ = f64[1,100] slice(%p0_), slice={[0:1], [0:100]}
      %s1_ = f64[1,100] slice(%p0_), slice={[1:2], [0:100]}
      %s2_ = f64[1,100] slice(%p0_), slice={[2:3], [0:100]}
      %s3_ = f64[1,100] slice(%p0_), slice={[3:4], [0:100]}
      %zero_ = f64[] constant(0.0)
      %init_ = f64[1,100] broadcast(%zero_), dimensions={}
      %r0_ = f64[1,100] add(%s0_, %init_)
      %r1_ = f64[1,100] add(%s1_, %r0_)
      %r2_ = f64[1,100] add(%s2_, %r1_)
      %r3_ = f64[1,100] add(%s3_, %r2_)
      ROOT %result_ = f64[4,100] concatenate(%r0_, %r1_, %r2_, %r3_), dimensions={0}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                           ParseAndReturnVerifiedModule(hlo_string));

  LoopifyUnrolledSlices pass(/*min_iterations=*/4);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get(), {}));

  EXPECT_TRUE(changed);

  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::GetTupleElement(op::While()));
}

TEST_F(LoopifyUnrolledSlicesTest, MultiConcatSharedComputation) {
  // Two concatenates share slices from the same source and carry state.
  // Both should be replaced by a SINGLE while loop with 2 accumulators.
  const char* hlo_string = R"(
    HloModule test

    ENTRY main {
      %p0_ = f64[4,100] parameter(0)
      %s0_ = f64[1,100] slice(%p0_), slice={[0:1], [0:100]}
      %s1_ = f64[1,100] slice(%p0_), slice={[1:2], [0:100]}
      %s2_ = f64[1,100] slice(%p0_), slice={[2:3], [0:100]}
      %s3_ = f64[1,100] slice(%p0_), slice={[3:4], [0:100]}
      %c2_ = f64[] constant(2.0)
      %b2_ = f64[1,100] broadcast(%c2_), dimensions={}
      %a0_ = f64[1,100] multiply(%s0_, %b2_)
      %a1_ = f64[1,100] multiply(%s1_, %b2_)
      %a2_ = f64[1,100] multiply(%s2_, %b2_)
      %a3_ = f64[1,100] multiply(%s3_, %b2_)
      %c3_ = f64[] constant(3.0)
      %b3_ = f64[1,100] broadcast(%c3_), dimensions={}
      %d0_ = f64[1,100] add(%a0_, %b3_)
      %d1_ = f64[1,100] add(%a1_, %b3_)
      %d2_ = f64[1,100] add(%a2_, %b3_)
      %d3_ = f64[1,100] add(%a3_, %b3_)
      %out_a_ = f64[4,100] concatenate(%a0_, %a1_, %a2_, %a3_), dimensions={0}
      ROOT %out_d_ = f64[4,100] concatenate(%d0_, %d1_, %d2_, %d3_), dimensions={0}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                           ParseAndReturnVerifiedModule(hlo_string));

  LoopifyUnrolledSlices pass(/*min_iterations=*/4);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get(), {}));

  EXPECT_TRUE(changed);

  // Both concats should be replaced. The root (out_d_) should be a GTE
  // from a while loop, and out_a_ should also be a GTE from the SAME while.
  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::GetTupleElement(op::While()));
}

TEST_F(LoopifyUnrolledSlicesTest, TooFewIterationsNoTransform) {
  const char* hlo_string = R"(
    HloModule test

    ENTRY main {
      %p0_ = f64[2,100] parameter(0)
      %s0_ = f64[1,100] slice(%p0_), slice={[0:1], [0:100]}
      %s1_ = f64[1,100] slice(%p0_), slice={[1:2], [0:100]}
      ROOT %result_ = f64[2,100] concatenate(%s0_, %s1_), dimensions={0}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                           ParseAndReturnVerifiedModule(hlo_string));

  LoopifyUnrolledSlices pass(/*min_iterations=*/4);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get(), {}));

  EXPECT_FALSE(changed);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
