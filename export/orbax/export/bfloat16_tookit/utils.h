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

#ifndef THIRD_PARTY_PY_ORBAX_EXPORT_BFLOAT16_TOOKIT_UTILS_H_
#define THIRD_PARTY_PY_ORBAX_EXPORT_BFLOAT16_TOOKIT_UTILS_H_

#include <algorithm>
#include <string>
#include <vector>

#include "third_party/absl/status/status.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/types.h"
#include "util/gtl/linked_hash_map.h"
#include "util/gtl/linked_hash_set.h"

namespace tensorflow {
namespace orbax {

// Compares two edges based on their ids.
struct EdgeComparatorID {
  bool operator()(const Edge* e1, const Edge* e2) const {
    return e1->id() < e2->id();
  }
};

// Returns a node's in edges in a deterministic order. The edges it returns may
// include control edges.
// This function should be called whenever getting a node's in_edges in the
// inference converter because Node::in_edges() is not deterministic but the
// inference converter is deterministic.
inline std::vector<const Edge*> GetInEdges(Node* node) {
  std::vector<const Edge*> in_edges;
  for (const Edge* edge : node->in_edges()) {
    in_edges.push_back(edge);
  }
  std::sort(in_edges.begin(), in_edges.end(), EdgeComparatorID());
  return in_edges;
}

// Returns a node's out edges in a deterministic order. The edges it returns may
// include control edges.
// This function should be called whenever getting a node's out_edges in the
// inference converter because Node::out_edges() is not deterministic but the
// inference converter is deterministic.
inline std::vector<const Edge*> GetOutEdges(Node* node) {
  std::vector<const Edge*> out_edges;
  for (const Edge* edge : node->out_edges()) {
    out_edges.push_back(edge);
  }
  std::sort(out_edges.begin(), out_edges.end(), EdgeComparatorID());
  return out_edges;
}

// Extracts Graph from FunctionDef with name `func_name`.
absl::Status ExtractGraphFromFunction(
    const FunctionLibraryDefinition& flib_def, const std::string& func_name,
    Graph* func_graph,
    ::gtl::linked_hash_set<std::string>* control_ret_nodes = nullptr);

// Returns true if the node is a call node runnable during inference.
bool IsInferenceFunctionCallNode(Node* node,
                                 const FunctionLibraryDefinition& flib_def);

// Looks up overhead of a node under cost model, optionally w/ recursive
// function calls taken into consideration.
int GetNodeOverhead(
    const Node* n,
    const ::gtl::linked_hash_map<string, int>& func_overhead = {});

// Returns the boolean value of the attribute with the given name in the node.
// Returns the default value in the parameter if the attribute is not present in
// the node.
bool GetBooleanAttribute(const Node* node, const std::string& attribute_name,
                         bool default_value = false);

}  // namespace orbax
}  // namespace tensorflow

#endif  // THIRD_PARTY_PY_ORBAX_EXPORT_BFLOAT16_TOOKIT_UTILS_H_
