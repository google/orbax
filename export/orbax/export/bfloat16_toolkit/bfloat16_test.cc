/* Copyright 2026 The Orbax Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <string>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "third_party/absl/container/linked_hash_map.h"
#include "third_party/absl/strings/string_view.h"
#include "orbax/export/bfloat16_toolkit/optim_utils.h"
#include "tensorflow/compiler/xla/tsl/platform/env.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_matcher.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/framework/types.proto.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"

namespace tensorflow::orbax {
namespace {

using ::tensorflow::test::TensorEq;
using ::testing::ElementsAre;
using ::testing::Pair;

constexpr absl::string_view kWeightKey = "w/.ATTRIBUTES/VARIABLE_VALUE";

TEST(BFloat16Test, ReadVariablesAsTensorsReassemblesSlicedEntries) {
  const std::string prefix = io::JoinPath(testing::TempDir(), "sliced_ckpt");
  const TensorShape full_shape({4, 3});
  Tensor expected(DT_FLOAT, full_shape);
  BundleWriter writer(tsl::Env::Default(), prefix);
  for (int shard = 0; shard < 2; ++shard) {
    Tensor slice_tensor(DT_FLOAT, TensorShape({2, 3}));
    for (int index = 0; index < 6; ++index) {
      const float value = shard * 6.0f + index;
      slice_tensor.flat<float>()(index) = value;
      expected.flat<float>()(shard * 6 + index) = value;
    }
    ASSERT_OK(writer.AddSlice(kWeightKey, full_shape,
                              TensorSlice({{shard * 2, 2}, {0, 3}}),
                              slice_tensor));
  }
  ASSERT_OK(writer.Finish());

  absl::linked_hash_map<std::string, Tensor> tensors;
  ASSERT_OK(ReadVariablesAsTensors(prefix, &tensors));

  EXPECT_THAT(tensors, ElementsAre(Pair(kWeightKey, TensorEq(expected))));
}

TEST(BFloat16Test, WriteTensorsToVariablesPersistsTensorsInNestedDirectory) {
  const std::string prefix =
      io::JoinPath(testing::TempDir(), "nested_dir", "variables", "variables");
  Tensor expected(DT_FLOAT, TensorShape({2, 2}));
  expected.flat<float>().setConstant(2.5f);

  ASSERT_OK(
      WriteTensorsToVariables(prefix, {{std::string(kWeightKey), expected}}));

  absl::linked_hash_map<std::string, Tensor> tensors;
  ASSERT_OK(ReadVariablesAsTensors(prefix, &tensors));
  EXPECT_THAT(tensors, ElementsAre(Pair(kWeightKey, TensorEq(expected))));
}

}  // namespace
}  // namespace tensorflow::orbax
