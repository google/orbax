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

#include "orbax/export/bfloat16_toolkit/bfloat16.h"

#include <cstdint>
#include <initializer_list>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "third_party/absl/container/linked_hash_map.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/status/status_matchers.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/absl/strings/match.h"
#include "third_party/absl/strings/string_view.h"
#include "orbax/export/bfloat16_toolkit/converter_options_v2.proto.h"
#include "orbax/export/bfloat16_toolkit/function_tree.h"
#include "orbax/export/bfloat16_toolkit/optim_utils.h"
#include "tensorflow/compiler/xla/tsl/platform/env.h"
#include "tensorflow/compiler/xla/tsl/platform/errors.h"
#include "tensorflow/compiler/xla/tsl/platform/statusor.h"
#include "tensorflow/compiler/xla/tsl/util/proto/parse_text_proto.h"
#include "tensorflow/compiler/xla/tsl/util/proto/proto_matchers.h"
#include "tensorflow/core/framework/attr_value.proto.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.proto.h"
#include "tensorflow/core/framework/graph.proto.h"
#include "tensorflow/core/framework/node_def.proto.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_matcher.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.proto.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/protobuf/meta_graph.proto.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"

namespace tensorflow::orbax {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::tensorflow::test::TensorEq;
using ::testing::ElementsAre;
using ::testing::NotNull;
using ::testing::Pair;
using ::testing::Pointee;
using ::testing::Property;
using ::tsl::proto_testing::EqualsProto;
using ::tsl::proto_testing::ParseTextProtoOrDie;

constexpr absl::string_view kInferFunction = "infer_function";
constexpr absl::string_view kCallInferName = "call_infer";
constexpr absl::string_view kAllScopeOptions = R"pb(scope: ALL)pb";
constexpr absl::string_view kServingKey = "serving_default";
constexpr absl::string_view kFloatSignatureProto = R"pb(
  inputs {
    key: "x"
    value { name: "x:0" dtype: DT_FLOAT }
  }
  outputs {
    key: "y"
    value { name: "call_infer:0" dtype: DT_FLOAT }
  }
)pb";
constexpr absl::string_view kInputName = "x";
constexpr absl::string_view kOutputName = "y";
constexpr absl::string_view kVariableName = "weight_variable";
constexpr absl::string_view kSharedInitName = "shared_init";
constexpr absl::string_view kVariableArgDef = "weight_handle: resource";
constexpr absl::string_view kVariableArg = "weight_handle";
constexpr absl::string_view kReadWeightName = "read_weight";
constexpr absl::string_view kReadWeightPort = "read_weight:value:0";
constexpr absl::string_view kWeightKey = "w/.ATTRIBUTES/VARIABLE_VALUE";
constexpr absl::string_view kTypeAttr = "T";
constexpr absl::string_view kDtypeAttr = "dtype";
constexpr absl::string_view kRetvalOp = "_Retval";
constexpr absl::string_view kPlaceholderOp = "Placeholder";
constexpr absl::string_view kVarHandleOp = "VarHandleOp";
constexpr absl::string_view kCastOp = "Cast";
constexpr absl::string_view kIdentityOp = "Identity";
constexpr absl::string_view kReadVariableOp = "ReadVariableOp";
constexpr absl::string_view kAssignVariableOp = "AssignVariableOp";
constexpr absl::string_view kTruncatedNormalOp = "TruncatedNormal";
constexpr absl::string_view kMatMulOp = "MatMul";

template <typename T>
FunctionDefHelper::Node MakeFunctionNode(
    absl::string_view name, absl::string_view op_type,
    std::initializer_list<absl::string_view> inputs,
    absl::string_view attr_name, const T& value) {
  return {{std::string(name)},
          std::string(op_type),
          {inputs.begin(), inputs.end()},
          {{std::string(attr_name), value}}};
}

absl::StatusOr<const Node*> FindOp(const Graph& graph,
                                   absl::string_view op_type) {
  for (const Node* node : graph.nodes()) {
    if (node->type_string() == op_type) return node;
  }
  return absl::NotFoundError(op_type);
}

absl::StatusOr<const Node*> GetOpInput(const Graph& graph,
                                       absl::string_view op_type, int index) {
  TF_ASSIGN_OR_RETURN(const Node* op_node, FindOp(graph, op_type));
  const Node* source_node = nullptr;
  TF_RETURN_IF_ERROR(op_node->input_node(index, &source_node));
  if (source_node == nullptr) return absl::NotFoundError(op_type);
  return source_node;
}

absl::StatusOr<DataType> GetOpType(const Graph& graph,
                                   absl::string_view op_type,
                                   absl::string_view attr_name = kTypeAttr) {
  TF_ASSIGN_OR_RETURN(const Node* op_node, FindOp(graph, op_type));
  DataType dtype;
  if (GetNodeAttr(op_node->attrs(), attr_name, &dtype).ok() ||
      GetNodeAttr(op_node->attrs(), kDtypeAttr, &dtype).ok()) {
    return dtype;
  }
  return absl::NotFoundError(op_type);
}

absl::StatusOr<const Graph*> FindChildGraph(FunctionInfo& root,
                                            absl::string_view name_substring) {
  std::vector<FunctionInfo*> children;
  TF_RETURN_IF_ERROR(root.get_children_ptrs(&children));
  for (FunctionInfo* child : children) {
    if (absl::StrContains(child->name(), name_substring)) {
      return child->get_graph();
    }
  }
  return absl::NotFoundError(name_substring);
}

std::map<std::string, SignatureDef> MakeSignatureMap() {
  return {{std::string(kServingKey),
           ParseTextProtoOrDie<SignatureDef>(kFloatSignatureProto)}};
}

NodeDef* AddOp(absl::string_view name, absl::string_view op_type,
               std::initializer_list<absl::string_view> inputs,
               absl::string_view attr_name, DataType dtype,
               GraphDef* graph_def) {
  NodeDef* node = graph_def->add_node();
  node->set_name(std::string(name));
  node->set_op(std::string(op_type));
  for (absl::string_view input : inputs) node->add_input(std::string(input));
  AddNodeAttr(attr_name, dtype, node);
  return node;
}

NodeDef* AddCall(absl::string_view name, absl::string_view function_name,
                 std::initializer_list<absl::string_view> inputs,
                 DataTypeSlice input_types, DataTypeSlice output_types,
                 GraphDef* graph_def) {
  NodeDef* node = graph_def->add_node();
  node->set_name(std::string(name));
  node->set_op("StatefulPartitionedCall");
  for (absl::string_view input : inputs) node->add_input(std::string(input));
  (*node->mutable_attr())["f"].mutable_func()->set_name(
      std::string(function_name));
  AddNodeAttr("Tin", input_types, node);
  AddNodeAttr("Tout", output_types, node);
  return node;
}

// Builds `y = MatMul(x, AddV2(ReadVariableOp(weight_handle), TruncatedNormal))`
// called from root via `StatefulPartitionedCall`, optionally with a float value
// shared by a variable assignment and a float consumer.
GraphDef BuildTestGraphDef(bool with_shared_init = false) {
  GraphDef graph_def;
  *graph_def.mutable_library()->add_function() = FunctionDefHelper::Create(
      std::string(kInferFunction), {"x: float", std::string(kVariableArgDef)},
      {"y: float"}, {},
      {MakeFunctionNode(kReadWeightName, kReadVariableOp, {kVariableArg},
                        kDtypeAttr, DT_FLOAT),
       FunctionDefHelper::Const<int32_t>("noise_shape", {2, 2}),
       {{"noise"},
        std::string(kTruncatedNormalOp),
        {"noise_shape:output:0"},
        {{std::string(kTypeAttr), DT_INT32},
         {std::string(kDtypeAttr), DT_FLOAT}}},
       MakeFunctionNode("noisy_weight", "AddV2",
                        {kReadWeightPort, "noise:output:0"}, kTypeAttr,
                        DT_FLOAT),
       MakeFunctionNode("matmul", kMatMulOp, {kInputName, "noisy_weight:z:0"},
                        kTypeAttr, DT_FLOAT)},
      {{std::string(kOutputName), "matmul:product:0"}});
  AddOp(kInputName, kPlaceholderOp, {}, kDtypeAttr, DT_FLOAT, &graph_def);
  AddNodeAttr(
      "shape", TensorShape({2, 2}),
      AddOp(kVariableName, kVarHandleOp, {}, kDtypeAttr, DT_FLOAT, &graph_def));
  AddCall(kCallInferName, kInferFunction, {kInputName, kVariableName},
          {DT_FLOAT, DT_RESOURCE}, {DT_FLOAT}, &graph_def);

  if (with_shared_init) {
    AddOp(kSharedInitName, kPlaceholderOp, {}, kDtypeAttr, DT_FLOAT,
          &graph_def);
    AddOp("assign_weight_init", kAssignVariableOp,
          {kVariableName, kSharedInitName}, kDtypeAttr, DT_FLOAT, &graph_def);
    AddOp("float_consumer", kIdentityOp, {kSharedInitName}, kTypeAttr, DT_FLOAT,
          &graph_def);
  }
  return graph_def;
}

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

TEST(BFloat16Test, ApplyBfloat16OptimizationV2ConvertsAllScopeModel) {
  std::map<std::string, SignatureDef> signature_map = MakeSignatureMap();
  GraphDef graph_def = BuildTestGraphDef();
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<FunctionInfo> root,
                       GetFunctionInfoFromGraphDef(graph_def));
  ASSERT_THAT(root, NotNull());
  Graph* root_graph = root->get_graph();
  ASSERT_THAT(root_graph, NotNull());

  ASSERT_OK(ApplyBfloat16OptimizationV2(ParseTextProtoOrDie(kAllScopeOptions),
                                        root.get(), &signature_map));

  EXPECT_THAT(
      signature_map,
      ElementsAre(Pair(kServingKey, EqualsProto(R"pb(
                         inputs {
                           key: "x"
                           value { name: "x:0" dtype: DT_BFLOAT16 }
                         }
                         outputs {
                           key: "y"
                           value { name: "call_infer:0" dtype: DT_BFLOAT16 }
                         }
                       )pb"))));
  EXPECT_THAT(GetOpType(*root_graph, kPlaceholderOp),
              IsOkAndHolds(DT_BFLOAT16));
  EXPECT_THAT(GetOpType(*root_graph, kVarHandleOp), IsOkAndHolds(DT_BFLOAT16));

  ASSERT_OK_AND_ASSIGN(const Graph* infer_graph,
                       FindChildGraph(*root, kInferFunction));
  ASSERT_THAT(infer_graph, NotNull());
  EXPECT_THAT(GetOpType(*infer_graph, kReadVariableOp),
              IsOkAndHolds(DT_BFLOAT16));
  EXPECT_THAT(GetOpType(*infer_graph, kTruncatedNormalOp, kDtypeAttr),
              IsOkAndHolds(DT_FLOAT));
  EXPECT_THAT(GetOpType(*infer_graph, kMatMulOp), IsOkAndHolds(DT_BFLOAT16));
  EXPECT_THAT(GetOpType(*infer_graph, kRetvalOp), IsOkAndHolds(DT_BFLOAT16));
}

TEST(BFloat16Test,
     ApplyBfloat16OptimizationV2IsolatesCastFromSharedFloatAssignSource) {
  GraphDef graph_def = BuildTestGraphDef(/*with_shared_init=*/true);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<FunctionInfo> root,
                       GetFunctionInfoFromGraphDef(graph_def));
  ASSERT_THAT(root, NotNull());

  ASSERT_OK(ApplyBfloat16OptimizationV2(ParseTextProtoOrDie(R"pb(
                                          scope: ALL
                                          preserve_signature: true
                                          skip_safety_checks: true
                                        )pb"),
                                        root.get(),
                                        /*signature_def=*/nullptr));

  Graph* root_graph = root->get_graph();
  ASSERT_THAT(root_graph, NotNull());
  EXPECT_THAT(GetOpType(*root_graph, kAssignVariableOp),
              IsOkAndHolds(DT_BFLOAT16));
  EXPECT_THAT(GetOpInput(*root_graph, kAssignVariableOp, 1),
              IsOkAndHolds(Pointee(Property(&Node::type_string, kCastOp))));
  EXPECT_THAT(GetOpType(*root_graph, kIdentityOp), IsOkAndHolds(DT_FLOAT));
  EXPECT_THAT(GetOpInput(*root_graph, kIdentityOp, 0),
              IsOkAndHolds(Pointee(Property(&Node::name, kSharedInitName))));
}

}  // namespace
}  // namespace tensorflow::orbax
