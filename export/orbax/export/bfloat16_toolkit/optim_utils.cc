/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "orbax/export/bfloat16_toolkit/optim_utils.h"

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "base/logging.h"
#include "third_party/absl/algorithm/container.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/strings/str_cat.h"
#include "third_party/absl/strings/str_join.h"
#include "third_party/absl/strings/string_view.h"
#include "orbax/export/bfloat16_toolkit/function_tree.h"
#include "orbax/export/bfloat16_toolkit/utils.h"
#include "tensorflow/compiler/jit/shape_inference.h"
#include "tensorflow/compiler/xla/tsl/platform/errors.h"
#include "tensorflow/core/common_runtime/graph_runner.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"
#include "util/gtl/linked_hash_map.h"

namespace tensorflow {
namespace orbax {

// Read variables and name - variable mappings.
absl::Status ReadVariablesAsTensors(
    const string& input_prefix,
    ::gtl::linked_hash_map<string, Tensor>* var_tensors) {
  VLOG(1) << "Reading Variables from " << input_prefix;
  BundleReader reader(Env::Default(), input_prefix);
  reader.Seek(tensorflow::kHeaderEntryKey);
  reader.Next();
  while (reader.Valid()) {
    string tensor_name(reader.key());
    Tensor tensor_value;
    TF_RETURN_IF_ERROR(reader.ReadCurrent(&tensor_value));
    (*var_tensors)[tensor_name] = tensor_value;
    VLOG(3) << "Variable: " << tensor_name;
    reader.Next();
  }
  return absl::OkStatus();
}

// Persist variables and name - variable mappings.
absl::Status WriteTensorsToVariables(
    const string& output_prefix,
    const ::gtl::linked_hash_map<string, Tensor>& var_tensors) {
  VLOG(1) << "Writing Variables to " << output_prefix;
  BundleWriter writer(Env::Default(), output_prefix);
  for (const auto& it : var_tensors) {
    VLOG(3) << "Variable: " << it.first;
    TF_RETURN_IF_ERROR(writer.Add(it.first, it.second));
  }
  TF_RETURN_IF_ERROR(writer.Finish());
  return absl::OkStatus();
}

bool IsStandardFunctionNode(Node* node) {
  return (node->type_string() == "TPUPartitionedCall" ||
          node->type_string() == "BatchFunction" ||
          node->type_string() == "StatefulPartitionedCall" ||
          node->type_string() == "PartitionedCall");
}

absl::Status InferShapesRecursively(FunctionInfo* func,
                                    GraphShapeInfo* shape_info) {
  // Infer parent's shape before self.
  GraphShapeInfo parent_shape_info;
  if (!func->IsRoot())
    TF_RETURN_IF_ERROR(
        InferShapesRecursively(func->parent(), &parent_shape_info));

  // If function is not root, get its args' shapes.
  std::map<int, InferredShape> arg_shapes = std::map<int, InferredShape>();
  if (!func->IsRoot()) {
    for (const Edge* edge : GetInEdges(func->node_in_parent_graph())) {
      if (!parent_shape_info.count(edge->src()->name())) continue;
      // If Arg's shape has been inferred.
      int arg_index = edge->dst_input();
      if (edge->dst()->type_string() == "If" ||
          edge->dst()->type_string() == "StatelessIf") {
        // The first input of "If"/"StatelessIf" is Tcond, not counted in args.
        arg_index = edge->dst_input() - 1;
      }
      auto inferred_shape_vec = parent_shape_info.at(edge->src()->name());
      if (!inferred_shape_vec.empty() && edge->src_output() >= 0) {
        arg_shapes.emplace(arg_index,
                           inferred_shape_vec.at(edge->src_output()));
      }
    }
  }

  // Run shape inference on the current graph.
  TF_RETURN_IF_ERROR(InferShapes(func->get_graph(), arg_shapes,
                                 &func->GetRoot()->get_graph()->flib_def(),
                                 shape_info));

  return absl::OkStatus();
}

// Rewrites a TPUPartitionedCall to a StatefulPartitionedCall with same name.
absl::Status RewriteTPUPartitionedCall(FunctionInfo* tpu_func) {
  VLOG(3) << "Rewriting TPUPartitionedCall for " << tpu_func->name();
  Graph* graph = tpu_func->parent()->get_graph();
  Node* call_node = tpu_func->node_in_parent_graph();
  string orig_name = call_node->name();
  ::gtl::linked_hash_map<int, const Edge*> input_edges, output_edges;

  // Collect input and output edges.
  for (const Edge* edge : GetInEdges(call_node)) {
    input_edges[edge->dst_input()] = edge;
  }
  for (const Edge* edge : GetOutEdges(call_node)) {
    output_edges[edge->src_output()] = edge;
  }

  // Create an input list.
  std::vector<NodeBuilder::NodeOut> input_list;
  for (int i = 0; i < input_edges.size(); i++) {
    // This works because TPUOrdinalSelector is always the last argument.
    if (input_edges.at(i)->src()->type_string() != "TPUOrdinalSelector") {
      input_list.push_back(NodeBuilder::NodeOut(
          input_edges.at(i)->src(), input_edges.at(i)->src_output()));
    }
  }

  // Build StatefulPartitionedCall node.
  Node* stateful_partitioned_call_node = nullptr;
  TF_RETURN_IF_ERROR(
      NodeBuilder(absl::StrCat(call_node->name(), "/stateful_partitioned_call"),
                  "StatefulPartitionedCall")
          .Attr("Tin", *(call_node->attrs().Find("Tin")))
          .Attr("Tout", *(call_node->attrs().Find("Tout")))
          .Attr("f", *(call_node->attrs().Find("f")))
          .Input(input_list)
          .Finalize(graph, &stateful_partitioned_call_node));

  // Replace TPUPartitionedCall.
  for (const auto& it : input_edges) {
    const Edge* edge = it.second;
    graph->RemoveEdge(edge);
    if (edge->src()->type_string() == "TPUOrdinalSelector")
      graph->RemoveNode(edge->src());
  }
  for (const auto& it : output_edges) {
    int output_index = it.first;
    const Edge* edge = it.second;
    graph->RemoveEdge(edge);
    graph->AddEdge(stateful_partitioned_call_node, output_index, edge->dst(),
                   edge->dst_input());
  }
  graph->RemoveNode(call_node);

  // Set new node's name to be the same as original one.
  stateful_partitioned_call_node->set_name(orig_name);

  // Update Node in parent graph.
  tpu_func->set_node_in_parent_graph(stateful_partitioned_call_node);

  return absl::OkStatus();
}

bool IsNodeBFloat16(Node* node) {
  for (absl::string_view name : kDataTypeAttrs) {
    const AttrValue* attr = node->attrs().Find(name);
    if (attr == nullptr)
      continue;

    if (attr->has_type() && attr->type() == DT_BFLOAT16)
      return true;

    if (attr->has_list() && attr->list().type_size() > 0) {
      if (absl::c_linear_search(attr->list().type(), DT_BFLOAT16))
        return true;
    }
  }

  return false;
}

// This function checks if there are bfloat16 ops in the graph. In general,
// models that have been converted to bfloat16 should not pass through the
// converter again. Otherwise, it may lead to unintended behaviors.
absl::Status ValidateGraphHasNoBFloat16Ops(Graph* graph) {
  ::gtl::linked_hash_map<string, int> func_overhead;
  auto func = std::make_unique<FunctionInfo>("root");
  func->set_graph(graph);
  TF_RETURN_IF_ERROR(func->Build(graph->flib_def(), &func_overhead));

  std::vector<FunctionInfo*> funcs;
  std::vector<string> found_bfloat16_ops;
  TF_RETURN_IF_ERROR(func->Find(
      [&](FunctionInfo* func) {
        for (Node* node : func->get_graph()->nodes()) {
          if (IsNodeBFloat16(node))
            found_bfloat16_ops.push_back(node->name());
        }
        return false;
      },
      &funcs));

  if (!found_bfloat16_ops.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Found bfloat16 ops in the model. The model may have been "
                     "converted before. It should not be converted again. \n",
                     absl::StrJoin(found_bfloat16_ops, ", \n")));
  }

  return absl::OkStatus();
}

}  // namespace orbax
}  // namespace tensorflow
