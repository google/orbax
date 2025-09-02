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

#include "orbax/export/bfloat16_toolkit/utils.h"

#include <memory>
#include <string>

#include "third_party/absl/status/status.h"
#include "third_party/absl/strings/match.h"
#include "tensorflow/compiler/xla/tsl/platform/errors.h"
#include "tensorflow/core/common_runtime/function_body.h"
#include "tensorflow/core/common_runtime/function_def_utils.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/framework/attr_value.proto.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.proto.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/types.h"
#include "util/gtl/linked_hash_map.h"
#include "util/gtl/linked_hash_set.h"

namespace tensorflow {
namespace orbax {

absl::Status ExtractGraphFromFunction(
    const FunctionLibraryDefinition& flib_def, const std::string& func_name,
    Graph* func_graph, ::gtl::linked_hash_set<std::string>* control_ret_nodes) {
  // Get TPU Graph from function library.
  if (!flib_def.Contains(func_name)) {
    return errors::InvalidArgument("Function ", func_name, " not found!");
  }

  std::unique_ptr<FunctionBody> func_body(nullptr);
  const FunctionDef* func_def = flib_def.Find(func_name);
  if (func_def == nullptr) {
    return errors::InvalidArgument("FunctionDef of ", func_name, " not found!");
  }
  AttrValueMap attr_val_map = AttrValueMap();
  attr_val_map.insert(func_def->attr().begin(), func_def->attr().end());
  AttrSlice attrs = AttrSlice(&attr_val_map);
  TF_RETURN_IF_ERROR(
      FunctionDefToBodyHelper(*func_def, attrs, &flib_def, &func_body));

  CopyGraph(*func_body->graph, func_graph);

  if (control_ret_nodes != nullptr) {
    for (const auto& it : func_def->control_ret()) {
      control_ret_nodes->insert(it.second);
    }
  }
  return absl::OkStatus();
}

bool IsInferenceFunctionCallNode(Node* node,
                                 const FunctionLibraryDefinition& flib_def) {
  if (node->type_string() == "SymbolicGradient") {  // Not for inference.
    return false;
  } else if (absl::StrContains(node->type_string(), "PartitionedCall") ||
             node->type_string() == "BatchFunction" ||
             node->type_string() == "While" || node->type_string() == "If" ||
             node->type_string() == "StatelessWhile" ||
             node->type_string() == "StatelessIf" ||
             flib_def.Contains(node->type_string())  // Customized functions.
  ) {
    return true;
  } else {
    return false;
  }
}

int GetNodeOverhead(const Node* n,
                    const ::gtl::linked_hash_map<string, int>& func_overhead) {
  // TODO(lzr,ylc): figure out a better cost model.
  if (absl::StartsWith(n->type_string(), "Conv") ||
      absl::StartsWith(n->type_string(), "FusedConv2DBiasActivation") ||
      absl::StartsWith(n->type_string(), "_FusedConv2D") ||
      absl::StartsWith(n->type_string(), "FusedConv2DBiasActivation") ||
      absl::StartsWith(n->type_string(), "XlaConv")) {
    return 5000;
  } else if (absl::StartsWith(n->type_string(), "MatMul") ||
             absl::StartsWith(n->type_string(), "BatchMatMul") ||
             absl::StartsWith(n->type_string(), "XlaDot")) {
    return 1000;
  } else if (absl::StartsWith(n->type_string(), "Einsum") ||
             absl::StartsWith(n->type_string(), "XlaEinsum")) {
    return 1001;
  } else if (!func_overhead.empty()) {
    if ((n->type_string() == "PartitionedCall" ||
         n->type_string() == "StatefulPartitionedCall") &&
        func_overhead.contains(n->attrs().Find("f")->func().name())) {
      return func_overhead.at(n->attrs().Find("f")->func().name());
    } else if ((n->type_string() == "While" ||
                n->type_string() == "StatelessWhile") &&
               func_overhead.contains(n->attrs().Find("body")->func().name()) &&
               func_overhead.contains(n->attrs().Find("cond")->func().name())) {
      return (func_overhead.at(n->attrs().Find("body")->func().name()) +
              func_overhead.at(n->attrs().Find("cond")->func().name()));
    } else if ((n->type_string() == "If" ||
                n->type_string() == "StatelessIf") &&
               func_overhead.contains(
                   n->attrs().Find("then_branch")->func().name()) &&
               func_overhead.contains(
                   n->attrs().Find("else_branch")->func().name())) {
      return (func_overhead.at(n->attrs().Find("then_branch")->func().name()) +
              func_overhead.at(n->attrs().Find("else_branch")->func().name()));
    } else if (func_overhead.contains(n->name())) {  // Defun.
      return func_overhead.at(n->name());
    }
  }
  return 1;
}

bool GetBooleanAttribute(const Node* node, const std::string& attribute_name,
                         bool default_value) {
  const AttrValue* attr = node->attrs().FindByString(attribute_name);
  if (attr == nullptr) {
    return default_value;
  }
  return attr->b();
}

}  // namespace orbax
}  // namespace tensorflow
