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

#include "orbax/export/bfloat16_toolkit/graph_analysis.h"

#include <algorithm>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <stack>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "base/logging.h"
#include "third_party/absl/algorithm/container.h"
#include "third_party/absl/container/btree_map.h"
#include "third_party/absl/functional/function_ref.h"
#include "third_party/absl/log/log.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/absl/strings/match.h"
#include "third_party/absl/strings/numbers.h"
#include "third_party/absl/strings/str_cat.h"
#include "third_party/absl/strings/str_join.h"
#include "third_party/absl/strings/str_split.h"
#include "third_party/absl/strings/string_view.h"
#include "orbax/export/bfloat16_toolkit/function_tree.h"
#include "orbax/export/bfloat16_toolkit/utils.h"
#include "tensorflow/compiler/jit/shape_inference.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/tsl/platform/errors.h"
#include "tensorflow/compiler/xla/tsl/platform/status.h"
#include "tensorflow/compiler/xla/tsl/platform/statusor.h"
#include "tensorflow/core/framework/attr_value.proto.h"
#include "tensorflow/core/framework/device_factory.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.proto.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session_options.h"
#include "util/gtl/linked_hash_map.h"
#include "util/gtl/linked_hash_set.h"
#include "util/regexp/re2/re2.h"

namespace tensorflow {
namespace orbax {

struct ConstNodeInfo {
  NodeInfo node_info;
  // Three possible kinds of nodes can be contained in `node`:
  // 1. Const op
  // 2. _Arg node
  // 3. End consumers
  absl::btree_map<const Edge*, std::unique_ptr<ConstNodeInfo>, EdgeComparatorID>
      consumers;
};

template <typename NodeInfoContainer>
using ProcessConsumerCallable =
    absl::FunctionRef<void(NodeInfoContainer& node_info, FunctionInfo* func,
                           Node* node, const Edge* edge, bool is_end_consumer)>;

absl::StatusOr<OutputTensor> ParseOutputTensorFromString(const Graph& g,
                                                         const string& tensor) {
  std::vector<string> parts = absl::StrSplit(tensor, ':');
  if (parts.size() > 2) {
    return errors::InvalidArgument("Invalid tensor name: ", tensor);
  }
  auto node_name_index = g.BuildNodeNameIndex();
  if (node_name_index.find(parts[0]) == node_name_index.end()) {
    return errors::InvalidArgument("Cannot find node for tensor: ", tensor);
  }

  Node* node = g.FindNodeId(node_name_index.at(parts[0])->id());
  OutputTensor result(node, 0);
  if (parts.size() == 2) {
    if (!absl::SimpleAtoi(parts[1], &result.index)) {
      return errors::InvalidArgument("Invalid node output \"", parts[1],
                                     "\" in tensor name \"", tensor, "\"");
    }
  }

  if (result.index < 0) {
    return errors::InvalidArgument("Output index cannot be negative: ", tensor);
  } else if (result.index >= node->num_outputs()) {
    return errors::InvalidArgument(
        "Invalid output index : ", result.index,
        ". Number of node outputs: ", node->num_outputs());
  }

  return result;
}

bool IsCompileTimeConstant(
    Node* node, ::gtl::linked_hash_map<Node*, bool>* compile_time_constant,
    const ::gtl::linked_hash_map<Node*, bool>& compatibility) {
  if (compile_time_constant->contains(node))
    return compile_time_constant->at(node);

  if (node->IsConstant() ||
      // For Inference, Variable ops are assumed constant.
      node->IsVariable() || node->type_string() == "ReadVariableOp" ||
      // These ops(Rank, Size, Shape) are deduced from the shape of inputs, and
      // are considered constant.
      (XlaOpRegistry::IsMetadataOp(node->type_string()) &&
       compatibility.at(node)) ||
      node->type_string() == "TensorArraySizeV3") {
    (*compile_time_constant)[node] = true;
  } else if (node->type_string() == "Placeholder" ||
             // `Placeholder` is dynamic since the value are feeded by caller.
             // `Where` is also dynamic since it has two branches.
             node->type_string() == "Where") {
    // TODO(ylc,lzr): Add more ops which are known dynamic.
    (*compile_time_constant)[node] = false;
  } else if (node->IsArg()) {
    // If the arg node is for variable input, it is compile-time constant.
    // Such arg nodes have type DT_RESOURCE.
    (*compile_time_constant)[node] =
        (node->attrs().Find("T")->type() == DT_RESOURCE);
  } else {
    // Recursively check predecessors.
    bool is_predecessors_constant = true;
    for (const Edge* e : GetInEdges(node)) {
      if (e->IsControlEdge()) continue;
      if (!IsCompileTimeConstant(e->src(), compile_time_constant,
                                 compatibility)) {
        is_predecessors_constant = false;
        break;
      }
    }
    (*compile_time_constant)[node] = is_predecessors_constant;
  }
  VLOG(4) << node->name() << " is constant? "
          << compile_time_constant->at(node);
  return compile_time_constant->at(node);
}

void EnforceCompileTimeConstantConstraints(
    Node* node, ::gtl::linked_hash_map<Node*, bool>* compile_time_constant,
    ::gtl::linked_hash_map<Node*, bool>* compatibility) {
  std::vector<int> const_input_idxs;
  TF_CHECK_OK(XlaOpRegistry::CompileTimeConstantInputs(
      node->def(), node->op_def(), &const_input_idxs));

  for (const int i : const_input_idxs) {
    Node* input;
    TF_CHECK_OK(node->input_node(i, &input));
    if (!IsCompileTimeConstant(input, compile_time_constant, *compatibility)) {
      VLOG(3) << node->name()
              << " violates the compile-time constant constraint for input "
              << i << ": " << input->name();
      (*compatibility)[node] = false;
    }
  }
  VLOG(3) << node->name() << " meets compile-time constant constraint.";
}

bool IsCallNodeTPUCompatible(Node* node, FunctionInfo* func_info, int depth,
                             absl::string_view xla_jit_device) {
  // This RegExp matches the names of DT_STRING Const nodes introduced by an
  // Assert op. Examples are `Assert/Const`, `Assert/Const_1`,
  // `Assert/Assert/data_2`, `Assert/AssertGuard/Assert/data_3`
  static const LazyRE2 assert_const_name = {"Assert/(data_\\d|Const(_\\d)?)$"};

  // `func_info` is the function in which `node` lies.
  std::vector<FunctionInfo*> children_ptrs;
  TF_CHECK_OK(func_info->get_children_ptrs(&children_ptrs));
  for (FunctionInfo* child : children_ptrs) {
    if (child->node_in_parent_graph() != node) continue;
    for (Node* node_in_child_fn : child->get_graph()->nodes()) {
      // When trying to determine whether a function call node is compilable,
      // the _SOURCE and _SINK generated from instantiating the function
      // should not be taken into account.
      if (node_in_child_fn->IsSource() || node_in_child_fn->IsSink()) continue;
      if (node_in_child_fn->type_string() == "Const") {
        // Edge case: Const with type DT_STRING not supported in functions
        // of control flows, but supported in TPUPartitionedCall. However if
        // it's introduced by an Assert op, we ignore it because Assert is NoOp
        // on the TPU. Otherwise any Assert will make a function
        // TPU-incompatible, which is not a desired behavior (b/197176454).
        if (node_in_child_fn->attrs().Find("dtype")->type() == DT_STRING &&
           !RE2::PartialMatch(node_in_child_fn->name(), *assert_const_name)) {
          VLOG(4) << node->name() << " incompatible because of "
                  << node_in_child_fn->name() << " in " << child->name();
          return false;
        }
      }
      if (!IsTPUCompatible(node_in_child_fn, child, depth, xla_jit_device)) {
        VLOG(4) << node->name() << " incompatible because of "
                << node_in_child_fn->name() << " in " << child->name();
        return false;
      }
    }
  }
  return true;
}

// Check if a node is TPU compatible.
// `node`: The node to be examined.
// `func_info`: The function that contains the `node`.
// `depth`: The depth of `func_info` w.r.t. the partition candidate function.
// When calling this method for a node in partition candidate function (the
// function at root), the depth is 0.
bool IsTPUCompatible(Node* node, FunctionInfo* func_info, int depth,
                     absl::string_view xla_jit_device) {
  // Function Node cases:
  if (IsInferenceFunctionCallNode(node, func_info->get_graph()->flib_def())) {
    return IsCallNodeTPUCompatible(node, func_info, depth + 1, xla_jit_device);
  }
  // _Arg and _Retval in partition candidate function (depth == 0) should be
  // considered incompatible to avoid being absorbed into TPU cluster.
  // For other functions (depth > 0), _Arg / _Retval should be considered
  // compatible and have no effect on compatibility of the function itself.
  if (node->type_string() == "_Arg" || node->type_string() == "_Retval") {
    return (depth > 0);
  }

  if (node->IsSource() || node->IsSink()) return false;
  if (node->type_string() == "Placeholder" ||
      node->type_string() == "LegacyFedInput")
    return false;
  if (node->attrs().Find("_scoped_allocator") ||
      node->attrs().Find("_forward_from"))
    return false;
  if (node->type_string() == "NoOp" &&
      (absl::StrContains(node->name(), "restore_all") ||
       absl::StrContains(node->name(), "restore_shard")))
    return false;

  auto is_variant = [](DataType dtype) { return dtype == DT_VARIANT; };
  if (absl::c_any_of(node->input_types(), is_variant) ||
      absl::c_any_of(node->output_types(), is_variant))
    return false;

  // Nodes that whose inputs are a identity string cannot be placed on the TPU
  // because the identity string node can neither be placed on the TPU nor be
  // an input to the TPU.
  for (const Edge* e : GetInEdges(node)) {
    Node* src = e->src();
    if ((src->IsIdentity() && src->attrs().Find("T")->type() == DT_STRING) ||
        (src->type_string() == "Placeholder" &&
         src->attrs().Find("dtype")->type() == DT_STRING)) {
      return false;
    }
  }

  // FindKernelDef read nodes on variables. Need to explicitly pull them on TPU.
  if (node->IsIdentity()) {
    for (const Edge* e : GetInEdges(node))
      if (!e->IsControlEdge() &&
          IsRefType(e->src()->output_type(e->src_output())))
        return true;
  }

  if (node->type_string() == "SymbolicGradient") return false;

  if (node->IsVariable() || node->IsConstant() || node->IsControlFlow())
    return true;

  absl::Status s =
      FindKernelDef(DeviceType(xla_jit_device), node->def(), nullptr, nullptr);
  if (!s.ok()) {
    VLOG(4) << node->name() << " not compilable: " << s;
    return false;
  }

  return true;
}

// `func_info` is the function in which `nodes` lies.
absl::Status GetTpuCompatibility(
    const FunctionLibraryDefinition& flib_def, const std::vector<Node*>& nodes,
    const ::gtl::linked_hash_set<string>& disallowed_nodes,
    FunctionInfo* func_info, ::gtl::linked_hash_map<Node*, bool>* compatibility,
    absl::string_view xla_jit_device) {
  // Env preparation.
  SessionOptions options;
  auto* device_count = options.config.mutable_device_count();
  device_count->insert({"CPU", 1});
  std::vector<std::unique_ptr<Device>> devices;
  TF_CHECK_OK(DeviceFactory::AddDevices(
      options, "/job:localhost/replica:0/task:0", &devices));
  ::gtl::linked_hash_map<Node*, bool> compile_time_constant;

  // XLA ops don't register themselves automatically. In order to make them
  // visible to FindKernelDef, we need to call the following function to
  // properly register them. Normally, this is done during the initialization of
  // TF. But if the converter runs before TF is loaded (e.g. converter invoked
  // via the C++ API), we need to do it ourselves to make sure those ops are
  // registered.
  XlaOpRegistry::RegisterCompilationKernels();

  // First pass: Compilability check.
  for (Node* node : nodes) {
    // Partition candidate function is at root (depth == 0).
    (*compatibility)[node] =
        IsTPUCompatible(node, func_info, /*depth=*/0, xla_jit_device);
    VLOG(4) << node->name() << " of type " << node->type_string()
            << " is compatible? " << compatibility->at(node);
  }

  // Exclude disallowed nodes.
  for (auto& it : *compatibility) {
    if (disallowed_nodes.contains(it.first->name())) {
      it.second = false;
      VLOG(4) << "Set TPU incompatible for node: " << it.first->name();
    }
  }

  // Second pass: Compile-time constant constraint check.
  for (Node* node : nodes) {
    if (!compatibility->at(node)) continue;
    EnforceCompileTimeConstantConstraints(node, &compile_time_constant,
                                          compatibility);
  }
  return absl::OkStatus();
}

void GetPredecessors(const Graph* g,
                     const std::vector<Node*>& reverse_post_order,
                     Predecessors* predecessors) {
  // predecessors: { node -> { predecessor -> distance } }
  // predecessors map is O(N^2) in space. This allows for O(1) time
  // lookup of node-predecessor relationship. Since GetTpuDependency() is
  // O(N^2) w.r.t. lookup, this time-space tradeoff is necessary.
  for (Node* node : reverse_post_order) {
    // Initialize empty set.
    predecessors->insert(
        std::make_pair(node, ::gtl::linked_hash_map<Node*, int>()));
  }
  // A node will be visited only if all its predecessors have been visited.
  for (Node* node : reverse_post_order) {
    for (const Edge* e : GetInEdges(node)) {
      // Add input nodes to set.
      predecessors->at(node).insert(std::make_pair(e->src(), 1));
      // Add predecessors of input nodes to set.
      if (!predecessors->contains(e->src())) continue;
      for (auto it : predecessors->at(e->src())) {
        Node* n = it.first;
        int dist = it.second;
        predecessors->at(node).insert(std::make_pair(n, dist + 1));
      }
    }
  }
}

void GetTpuDependency(
    const Graph* g, const std::vector<Node*>& reverse_post_order,
    const Predecessors& predecessors,
    const ::gtl::linked_hash_map<Node*, bool>& compatibility,
    ::gtl::linked_hash_set<std::pair<Node*, Node*>>* completely_depend_on) {
  // A node will be visited only if all its predecessors have been visited.
  for (Node* to_node : reverse_post_order) {
    if (!compatibility.at(to_node)) continue;
    // If a node is tpu-compatible, it completely depends on itself on TPU.
    completely_depend_on->insert(std::make_pair(to_node, to_node));
    // Inspect all predecessors of this node.
    // Avoid indexing non-existent node.
    if (!predecessors.contains(to_node)) continue;
    for (auto it : predecessors.at(to_node)) {
      Node* from_node = it.first;
      bool complete_dependency = true;
      for (const Edge* edge_in : GetInEdges(to_node)) {
        Node* child_node = edge_in->src();
        if (child_node != from_node &&
            predecessors.at(child_node).contains(from_node) &&
            !completely_depend_on->contains(
                std::make_pair(child_node, from_node))) {
          // If a child_node is on the path from from_node to to_node
          // and the child_node does not completely depend on to_node,
          // complete dependency between from_node and to_node does not hold.
          complete_dependency = false;
          break;
        }
      }
      if (complete_dependency)
        completely_depend_on->insert(std::make_pair(to_node, from_node));
    }
  }
}

bool IsDescendantOf(Node* node, const ::gtl::linked_hash_set<Node*>& node_set,
                    const Predecessors& predecessors) {
  for (Node* n : node_set)
    if (predecessors.contains(node) && predecessors.at(node).contains(n))
      return true;
  return false;
}

bool IsPredecessorOf(Node* node, const ::gtl::linked_hash_set<Node*>& node_set,
                     const Predecessors& predecessors) {
  for (Node* n : node_set)
    if (predecessors.contains(n) && predecessors.at(n).contains(node))
      return true;
  return false;
}

bool HasDescendant(Node* node, Graph* g,
                   const std::function<bool(Node*)>& condition_fn,
                   const Predecessors& predecessors) {
  for (Node* n : g->nodes()) {
    if (predecessors.contains(n) && predecessors.at(n).contains(node) &&
        condition_fn(n)) {
      return true;
    }
  }
  return false;
}

bool HasPredecessor(Node* node, const std::function<bool(Node*)>& condition_fn,
                    const Predecessors& predecessors) {
  for (auto it : predecessors.at(node)) {
    if (condition_fn(it.first)) return true;
  }
  return false;
}

bool IsValidInputEdge(const Edge* e) {
  // Source node cannot be input.
  if (!e->src()->IsSource() &&
      // Variables cannot be input.
      (e->src_output() < 0 ||
       !IsRefType(e->src()->output_type(e->src_output()))) &&
      // --- Regarding VarHandleOp ---
      // We no longer blacklist VarHandleOp, but instead capture VarHandleOp as
      // input explicitly, and then treat their corresponding TPUReplicatedInput
      // as captured nodes by checking "T=DT_RESOURCE".
      // One of the reason is because in TF2, ReadVariableOps are wrapped in
      // StatefulPartitionedCall and cannot be identified as captured nodes.
      // Another reason is because some resource ops with two inputs,
      // e.g. ResourceGather.
      // ------
      // When partitioning a function's graph, the edge _Arg --> ReadVariableOp
      // will be captured twice if not skipped here.
      // (Later in RewriteVarToResourceVar().)
      e->dst()->type_string() != "ReadVariableOp") {
    // If all inputs to a node are variables,
    // this node is used for reading variables.
    bool has_non_variable_input = false;
    for (const Edge* e_in : GetInEdges(e->src())) {
      if (!(e_in->src()->type_string() == kVariable) &&
          !(e_in->src()->type_string() == kVariableV2)) {
        has_non_variable_input = true;
        break;
      }
    }
    if (has_non_variable_input) return true;
  }
  return false;
}

bool IsValidOutputEdge(const Edge* e) {
  // Source node cannot be output.
  if (!e->src()->IsSource() &&
      // Variables cannot be output.
      (e->IsControlEdge() ||
       !IsRefType(e->src()->output_type(
           e->src_output()))) &&  // VarHandleOp cannot be output.
      e->src()->type_string() != "VarHandleOp") {
    // If all inputs to a node are variables,
    // this node is used for reading variables.
    bool has_non_variable_input = false;
    for (const Edge* e_in : GetInEdges(e->src())) {
      if (e_in->IsControlEdge()) continue;
      if (!(e_in->src()->type_string() == kVariable) &&
          !(e_in->src()->type_string() == kVariableV2) &&
          !(e_in->src()->type_string() == "VarHandleOp")) {
        has_non_variable_input = true;
        break;
      }
    }
    if (has_non_variable_input) return true;
  }
  return false;
}

// Identifies input_tensors, output_tensors and compute overhead.
absl::Status AnalyseConnectedComponent(
    const Graph* g, ::gtl::linked_hash_set<int>* node_ids,
    ::gtl::linked_hash_set<const Edge*>* input_edges,
    ::gtl::linked_hash_set<const Edge*>* output_edges,
    ::gtl::linked_hash_set<const Edge*>* nonsplit_output_edges, int* overhead,
    const ::gtl::linked_hash_map<string, int>& func_overhead) {
  for (int node_id : *node_ids) {
    Node* n = g->FindNodeId(node_id);
    *overhead += GetNodeOverhead(n, func_overhead);
    for (const Edge* e : GetInEdges(n)) {
      if (!node_ids->count(e->src()->id()) && IsValidInputEdge(e)) {
        input_edges->insert(e);
      }
    }
    ::gtl::linked_hash_set<const Edge*> candidate_output_edges =
        ::gtl::linked_hash_set<const Edge*>();
    for (const Edge* e : GetOutEdges(n)) {
      if (!node_ids->count(e->dst()->id()) && IsValidOutputEdge(e)) {
        candidate_output_edges.insert(e);
      }
    }
    if (!candidate_output_edges.empty()) {
      if (candidate_output_edges.size() == n->out_edges().size()) {
        for (const Edge* edge : candidate_output_edges) {
          output_edges->insert(edge);
        }
      } else {
        // If this node has outputs to inside and outside the component,
        // it should be nonsplit_output_edges.
        for (const Edge* edge : candidate_output_edges) {
          nonsplit_output_edges->insert(edge);
        }
      }
    }
  }
  return absl::OkStatus();
}

// This function, starting from a given node in the graph, iteratively adds
// TPU-compatible nodes that are connected to the current node, iff the node
// to be added does not introduce partial dependency.
// If the given node is not TPU-compatible, this function returns false.
bool ExpandTpuCoverage(Graph* g, Node* node,
                       const ::gtl::linked_hash_map<Node*, bool>& compatibility,
                       const Predecessors& predecessors,
                       const ::gtl::linked_hash_set<std::pair<Node*, Node*>>&
                           tpu_complete_dependency,
                       const ::gtl::linked_hash_set<Node*>& input_nodes_spec,
                       const ::gtl::linked_hash_set<Node*>& output_nodes_spec,
                       ::gtl::linked_hash_set<int>* node_ids,
                       std::vector<bool>* discovered,
                       bool enforce_subgraph_convexity) {
  discovered->at(node->id()) = true;
  // Cannot start expansion from an incompatible node.
  if (!compatibility.at(node)) return false;
  std::queue<Node*> queue;
  VLOG(3) << "Expand TPU converage from node " << node->name();
  queue.push(node);
  while (!queue.empty()) {
    Node* n = queue.front();
    queue.pop();

    bool is_convex_subgraph = true;
    if (enforce_subgraph_convexity) {
      // It checks that the node completely depends on all its ancestors in the
      // cluster and all its descendants in the cluster completely depends on
      // this node. Checking this relationship in only one way is not enough to
      // guarantee that the formed cluster is convex. See b/273546525.
      //
      // There could be some duplication with the code below, but we choose this
      // implementation because we want to avoid modifying the V1 converters
      // graph partitioning logic as much as possible as because multiple users
      // depend on it and it's in a maintenance mode at the moment.
      for (int node_id : *node_ids) {
        if (predecessors.at(n).contains(g->FindNodeId(node_id)) &&
            !tpu_complete_dependency.contains(
                std::make_pair(n, g->FindNodeId(node_id)))) {
          is_convex_subgraph = false;
          break;
        }
        if (predecessors.at(g->FindNodeId(node_id)).contains(n) &&
            !tpu_complete_dependency.contains(
                std::make_pair(g->FindNodeId(node_id), n))) {
          is_convex_subgraph = false;
          break;
        }
      }
    }
    if (is_convex_subgraph) {
      node_ids->insert(n->id());
    }

    // First consider output nodes then input nodes.
    // Because nodes are accessed by ExpandTpuCoverage in post order,
    // and if a node can either be attached to a upper or a lower cluster,
    // the upper is preferred because the lower one usually contains
    // ops for preprocessing stage.

    // Expand TPU coverage to output nodes.
    for (const Edge* e : GetOutEdges(n)) {
      Node* node_out = e->dst();
      if (discovered->at(node_out->id())) continue;
      if (!compatibility.at(node_out)) continue;

      bool maintains_complete_dependency = true;
      // Check if expanding to node_out introduces incomplete dependency
      // for any of the current component.
      for (int node_id : *node_ids) {
        // node_out <-- node_id
        // if node node_id is a predecessor of node_out
        // but node_out does not completely depend on node_id on TPU
        // expanding to node_out will introduce partial dependency
        if (predecessors.at(node_out).contains(g->FindNodeId(node_id)) &&
            !tpu_complete_dependency.contains(
                std::make_pair(node_out, g->FindNodeId(node_id)))) {
          maintains_complete_dependency = false;
          break;
        }
      }
      if (!maintains_complete_dependency) continue;

      // Don't go beyond specified output node's output.
      if (!output_nodes_spec.empty() &&
          IsDescendantOf(node_out, output_nodes_spec, predecessors)) {
        discovered->at(node_out->id()) = true;
        continue;
      }
      VLOG(3) << "Expand TPU converage to output node " << node_out->name();
      queue.push(node_out);
      discovered->at(node_out->id()) = true;
    }

    // Expand TPU coverage to input nodes.
    // TODO(lzr): Take into consideration compile-time constant.
    for (const Edge* e : GetInEdges(n)) {
      Node* node_in = e->src();
      if (discovered->at(node_in->id())) continue;
      if (!compatibility.at(node_in)) continue;

      bool maintains_complete_dependency = true;
      // Check if expanding to node_in introduces incomplete dependency
      // for any of the current component.
      for (int node_id : *node_ids) {
        // node_id <-- node_in
        // if node_in is a predecessor of node node_id
        // but node node_id does not completely depend on node_in on TPU
        // expanding to node_in will introduce partial dependency
        if (predecessors.at(g->FindNodeId(node_id)).contains(node_in) &&
            !tpu_complete_dependency.contains(
                std::make_pair(g->FindNodeId(node_id), node_in))) {
          maintains_complete_dependency = false;
          break;
        }
      }

      if (!maintains_complete_dependency) continue;

      bool parent_dt_resource_maintains_complete_dependency = true;
      // If a parent of node node_id is of type DT_RESOURCE,
      // it should also maintain complete dependency,
      // so that DT_RESOURCE will always be included as TPU node and
      // never be an input.
      // This check is added because DT_RESOURCE is not allowed as Arg type.
      for (const Edge* edge : GetInEdges(node_in)) {
        if (edge->IsControlEdge() ||
            edge->src()->output_type(edge->src_output()) != DT_RESOURCE)
          continue;
        Node* parent_node = edge->src();
        for (int node_id : *node_ids) {
          // node_id <-- parent_node
          // if parent_node is a predecessor of node node_id
          // but node node_id does not completely depend on node_in on TPU
          // expanding to node_in will introduce partial dependency
          if (predecessors.at(g->FindNodeId(node_id)).contains(parent_node) &&
              !tpu_complete_dependency.contains(
                  std::make_pair(g->FindNodeId(node_id), parent_node))) {
            parent_dt_resource_maintains_complete_dependency = false;
            break;
          }
        }
        if (!parent_dt_resource_maintains_complete_dependency) break;
      }
      if (!parent_dt_resource_maintains_complete_dependency) continue;

      // Don't go beyond specified input node.
      if (!input_nodes_spec.empty() &&
          (input_nodes_spec.contains(node_in) ||
           IsPredecessorOf(node_in, input_nodes_spec, predecessors))) {
        discovered->at(node_in->id()) = true;
        continue;
      }
      VLOG(3) << "Expand TPU converage to input node " << node_in->name();
      queue.push(node_in);
      discovered->at(node_in->id()) = true;
    }
  }
  return true;
}

absl::Status AnalyseGraph(
    Graph* g, const FunctionLibraryDefinition& flib_def,
    FunctionInfo* func_info, std::vector<Node*>* reverse_post_order,
    ::gtl::linked_hash_map<Node*, bool>* compatibility,
    Predecessors* predecessors,
    ::gtl::linked_hash_set<std::pair<Node*, Node*>>* tpu_complete_dependency,
    const ::gtl::linked_hash_set<string>& disallowed_nodes,
    absl::string_view xla_jit_device) {
  if (func_info != nullptr && g != func_info->get_graph())
    return errors::Internal("Input graph is not the same graph in ",
                            func_info->name());
  GetReversePostOrder(*g, reverse_post_order, NodeComparatorID());
  TF_RETURN_IF_ERROR(GetTpuCompatibility(flib_def, *reverse_post_order,
                                         disallowed_nodes, func_info,
                                         compatibility, xla_jit_device));
  GetPredecessors(g, *reverse_post_order, predecessors);
  GetTpuDependency(g, *reverse_post_order, *predecessors, *compatibility,
                   tpu_complete_dependency);
  return absl::OkStatus();
}

absl::Status GetConnectedComponentsOnTpu(
    Graph* g, const std::vector<OutputTensor>& input_tensors,
    const std::vector<OutputTensor>& output_tensors,
    const ::gtl::linked_hash_map<Node*, bool>& compatibility,
    const Predecessors& predecessors,
    const ::gtl::linked_hash_set<std::pair<Node*, Node*>>&
        tpu_complete_dependency,
    std::vector<ConnectedComponent>* connected_components,
    const ::gtl::linked_hash_map<string, int>& func_overhead,
    bool enforce_subgraph_convexity) {
  connected_components->clear();  // Reuse. Save memory.

  // Perform expansion on graph nodes.
  std::vector<bool> discovered(g->num_node_ids(), false);
  std::vector<Node*> expansion_entries;
  ::gtl::linked_hash_set<Node*> input_nodes,
      output_nodes = ::gtl::linked_hash_set<Node*>();

  // Expand TPU subgraph from each node between the input and output tensors.
  GetPostOrder(*g, &expansion_entries, NodeComparatorID());

  if (!input_tensors.empty()) {
    for (const OutputTensor& tensor : input_tensors) {
      input_nodes.insert(tensor.node);
    }
  }
  if (!output_tensors.empty()) {
    for (const OutputTensor& tensor : output_tensors) {
      output_nodes.insert(tensor.node);
    }
  }
  for (Node* n : expansion_entries) {
    ::gtl::linked_hash_set<int> node_ids;
    ::gtl::linked_hash_set<const Edge*> input_edges;
    ::gtl::linked_hash_set<const Edge*> output_edges;
    ::gtl::linked_hash_set<const Edge*> nonsplit_output_edges;
    int overhead = 0;
    if (!discovered.at(n->id()) &&
        !IsPredecessorOf(n, input_nodes, predecessors) &&
        !IsDescendantOf(n, output_nodes, predecessors) &&
        ExpandTpuCoverage(g, n, compatibility, predecessors,
                          tpu_complete_dependency, input_nodes, output_nodes,
                          &node_ids, &discovered, enforce_subgraph_convexity)) {
      // Get `input_edges`, `output_edges` and `overhead` for this component.
      TF_RETURN_IF_ERROR(AnalyseConnectedComponent(
          g, &node_ids, &input_edges, &output_edges, &nonsplit_output_edges,
          &overhead, func_overhead));

      if (!output_tensors.empty()) {
        // Remove unused output.
        std::vector<const Edge*> output_edges_to_remove;
        for (const Edge* edge : output_edges) {
          // If an identified output edge points to SINK and is not a specified
          // output node, SINK must be its only outgoing edge. (Otherwise it
          // would not have been connected to SINK in the very first place.)
          // If that is the case, this identified output edge is not needed by
          // users.
          // Since sending it back to CPU is not costless, we remove it.
          if (edge->dst()->IsSink() && !output_nodes.contains(edge->src()))
            output_edges_to_remove.push_back(edge);
        }
        for (const Edge* edge : output_edges_to_remove) {
          output_edges.erase(edge);
        }
        std::vector<const Edge*> nonsplit_output_edges_to_remove;
        for (const Edge* edge : nonsplit_output_edges) {
          if (edge->dst()->IsSink() && !output_nodes.contains(edge->src()))
            nonsplit_output_edges_to_remove.push_back(edge);
        }
        for (const Edge* edge : nonsplit_output_edges_to_remove) {
          nonsplit_output_edges.erase(edge);
        }
      }

      // Construct ConnectedComponent.
      ConnectedComponent cc(&overhead, &input_edges, &output_edges,
                            &nonsplit_output_edges, &node_ids);
      if (VLOG_IS_ON(4)) {
        LOG(INFO) << "Connected component on TPU detected. "
                  << "(Overhead=" << cc.overhead << ")\nNodes:";
        for (int nid : cc.node_ids) LOG(INFO) << g->FindNodeId(nid)->name();
        VLOG(4) << "Input edges:";
        for (const Edge* e : cc.input_edges) LOG(INFO) << e->DebugString();
        VLOG(4) << "Output edges:";
        for (const Edge* e : cc.output_edges) LOG(INFO) << e->DebugString();
        VLOG(4) << "Non-split output edges:";
        for (const Edge* e : cc.nonsplit_output_edges)
          LOG(INFO) << e->DebugString();
      }
      connected_components->push_back(cc);
    }
  }
  return absl::OkStatus();
}

absl::Status GetEdgeOverhead(
    Graph* g, const FunctionLibraryDefinition& flib_def, int unknown_dim_weight,
    int edge_penalty, ::gtl::linked_hash_map<const Edge*, int>* edge_overhead) {
  GraphShapeInfo shape_info;
  TF_RETURN_IF_ERROR(
      InferShapes(g, std::map<int, InferredShape>(), &flib_def, &shape_info));
  for (const Edge* edge : g->edges()) {
    if (shape_info.count(edge->src()->name()) && edge->src_output() >= 0) {
      auto inferred_shape =
          shape_info.at(edge->src()->name()).at(edge->src_output());
      int overhead = 1;
      for (int d = 0; d < inferred_shape.shape.dims(); ++d) {
        overhead *= (inferred_shape.shape.dim_size(d) < 0
                         ? unknown_dim_weight
                         : inferred_shape.shape.dim_size(d));
      }
      (*edge_overhead)[edge] = overhead + edge_penalty;
    }
  }
  return absl::OkStatus();
}

// This function computes cost of merging operation as
// op_cost + edge_cost
// We consider ops to be cheap on TPU, and expensive on TPU, so
// op_cost is the sum of overhead of nodes being excluded from TPU.
// edge_cost consists of two parts:
// removal of input edges reduces edge_cost (negative), and
// inclusion of input edges set by the merge_node increases edge_cost.
int ComputeMergeCost(
    const Graph* g, Node* merge_node,
    const ::gtl::linked_hash_set<Node*>& descs,
    const ::gtl::linked_hash_map<const Edge*, int>& edge_overhead,
    const int unknown_dim_weight,
    const ::gtl::linked_hash_map<string, int>& func_overhead) {
  ::gtl::linked_hash_set<const Edge*> edges_charged =
      ::gtl::linked_hash_set<const Edge*>();
  // Collect ops cost on path to descs.
  ::gtl::linked_hash_set<Node*> visited;
  int op_cost = 0;
  std::stack<Node*> stack;
  stack.push(merge_node);
  while (!stack.empty()) {
    Node* n = stack.top();
    stack.pop();
    op_cost += GetNodeOverhead(n, func_overhead);
    for (const Edge* e : GetInEdges(n)) {
      if (e->src()->IsSource()) continue;
      if (!visited.contains(e->src()) && !descs.contains(e->src())) {
        stack.push(e->src());
        visited.insert(e->src());
      } else {
        // Either leaf node or input node.
        edges_charged.insert(e);
      }
    }
  }
  // Subtract cost of descendant nodes input edges
  // since they no longer charge any cost for transfer.
  int edge_cost = 0;
  for (const Edge* edge : edges_charged) {
    if (edge_overhead.contains(edge)) {
      edge_cost -= edge_overhead.at(edge);
    } else {
      // If shape is not inferred for this edge.
      edge_cost -= unknown_dim_weight;
    }
  }
  // Add cost of merge node output edges
  // since it is crossing device border.
  for (const Edge* edge : GetOutEdges(merge_node)) {
    if (edge_overhead.contains(edge)) {
      edge_cost += edge_overhead.at(edge);
    }
  }
  return op_cost + edge_cost;
}

void GetMergeCandidates(
    const Graph* g, const ::gtl::linked_hash_set<int>& tpu_node_ids,
    const Predecessors& predecessors,
    const ::gtl::linked_hash_set<Node*>& original_input_nodes,
    ::gtl::linked_hash_map<Node*, ::gtl::linked_hash_set<Node*>>*
        merge_candidates) {
  // common_descendants: { { node_1, node_2 } -> { descendant, distance } }
  ::gtl::linked_hash_map<std::pair<Node*, Node*>, std::pair<Node*, int>>
      common_descendants;
  // Search for common descendants.
  for (int node_id : tpu_node_ids) {
    Node* node = g->FindNodeId(node_id);
    for (auto it_1 : predecessors.at(node)) {
      Node* node_1 = it_1.first;
      int distance_1 = it_1.second;
      if (!original_input_nodes.contains(node_1)) continue;
      for (auto it_2 : predecessors.at(node)) {
        Node* node_2 = it_2.first;
        int distance_2 = it_2.second;
        if (!original_input_nodes.contains(node_2)) continue;
        if (!common_descendants.contains(std::make_pair(node_1, node_2)) ||
            distance_1 + distance_2 <
                common_descendants.at(std::make_pair(node_1, node_2)).second) {
          common_descendants[std::make_pair(node_1, node_2)] =
              std::make_pair(node, distance_1 + distance_2);
        }
      }
    }
  }

  // Find merge candidates from common descendants.
  for (auto it : common_descendants) {
    Node* desc_node = it.second.first;
    if (!merge_candidates->contains(desc_node)) {
      merge_candidates->insert(
          std::make_pair(desc_node, ::gtl::linked_hash_set<Node*>()));
    }
    if (original_input_nodes.contains(it.first.first))
      merge_candidates->at(desc_node).insert(it.first.first);
    if (original_input_nodes.contains(it.first.second))
      merge_candidates->at(desc_node).insert(it.first.second);
  }
}

void GetMergeCandidatesByType(
    const Graph* g, const ::gtl::linked_hash_set<string>& merge_node_types,
    const ::gtl::linked_hash_set<int>& tpu_node_ids,
    const Predecessors& predecessors,
    const ::gtl::linked_hash_set<Node*>& original_input_nodes,
    ::gtl::linked_hash_map<Node*, ::gtl::linked_hash_set<Node*>>*
        merge_candidates) {
  for (int node_id : tpu_node_ids) {
    Node* merge_node = g->FindNodeId(node_id);
    if (!merge_node_types.contains(merge_node->type_string())) continue;
    (*merge_candidates)[merge_node] = ::gtl::linked_hash_set<Node*>();
    for (Node* input_node : original_input_nodes)
      if (predecessors.at(merge_node).contains(input_node))
        merge_candidates->at(merge_node).insert(input_node);
  }
}

absl::Status MaybeFindBetterCut(
    Graph* g, const FunctionLibraryDefinition& flib_def,
    ConnectedComponent* candidate, const std::vector<Node*>& reverse_post_order,
    const ::gtl::linked_hash_map<Node*, bool>& compatibility,
    const Predecessors& predecessors,
    const ::gtl::linked_hash_set<std::pair<Node*, Node*>>&
        tpu_complete_dependency,
    std::vector<ConnectedComponent>* connected_components,
    int edge_unknown_dim_weight, int edge_penalty, bool get_merge_node_by_type,
    bool enforce_subgraph_convexity,
    const ::gtl::linked_hash_map<string, int>& func_overhead) {
  // Get edge overhead.
  ::gtl::linked_hash_map<const Edge*, int> edge_overhead;
  TF_RETURN_IF_ERROR(GetEdgeOverhead(g, flib_def, edge_unknown_dim_weight,
                                     edge_penalty, &edge_overhead));
  // Original input nodes set.
  ::gtl::linked_hash_set<Node*> input_nodes = ::gtl::linked_hash_set<Node*>();
  for (const Edge* edge : candidate->input_edges) {
    input_nodes.insert(edge->src());
  }
  // Set of allowed merge node types.
  ::gtl::linked_hash_set<string> merge_node_types({"Concat", "ConcatV2"});
  // Get merge candidates.
  ::gtl::linked_hash_map<Node*, ::gtl::linked_hash_set<Node*>>
      merge_candidates =
          ::gtl::linked_hash_map<Node*, ::gtl::linked_hash_set<Node*>>();
  if (get_merge_node_by_type)
    GetMergeCandidatesByType(g, merge_node_types, candidate->node_ids,
                             predecessors, input_nodes, &merge_candidates);
  else
    GetMergeCandidates(g, candidate->node_ids, predecessors, input_nodes,
                       &merge_candidates);

  // Search for cheaper candidates.
  input_nodes.clear();
  for (const auto& it : merge_candidates) {
    int cost = ComputeMergeCost(g, it.first, it.second, edge_overhead,
                                edge_unknown_dim_weight, func_overhead);
    if (cost < 0) input_nodes.insert(it.first);
  }
  // If cheapest candidate has gain
  if (!input_nodes.empty()) {
    // Build input boundary.
    std::vector<OutputTensor> input_tensors = std::vector<OutputTensor>();
    for (Node* node : input_nodes) {
      OutputTensor tensor_in(node, 0);
      input_tensors.push_back(tensor_in);
      VLOG(4) << "Input cut: " << tensor_in.node->name();
    }
    // Build output boundary.
    std::vector<OutputTensor> output_tensors = std::vector<OutputTensor>();
    for (const Edge* e : candidate->output_edges) {
      OutputTensor tensor_out(e->src(), e->src_output());
      output_tensors.push_back(tensor_out);
      VLOG(4) << "Output cut: " << tensor_out.node->name();
    }

    TF_RETURN_IF_ERROR(GetConnectedComponentsOnTpu(
        g, input_tensors, output_tensors, compatibility, predecessors,
        tpu_complete_dependency, connected_components, func_overhead,
        enforce_subgraph_convexity));
    *candidate =
        *absl::c_max_element(*connected_components, CompareByOverhead());
  }

  return absl::OkStatus();
}

absl::Status GetFuncArgsList(FunctionInfo* func, FuncArgsList* func_args_list) {
  if (!func->IsRoot()) {
    ::gtl::linked_hash_map<int, Node*> args_list;
    for (Node* arg_node : func->get_graph()->nodes()) {
      if (arg_node->type_string() != "_Arg") continue;
      args_list[arg_node->attrs().Find("index")->i()] = arg_node;
    }
    if (!func_args_list->contains(func->node_in_parent_graph())) {
      (*func_args_list)[func->node_in_parent_graph()] =
          ::gtl::linked_hash_map<FunctionInfo*,
                                 ::gtl::linked_hash_map<int, Node*>>();
    }
    func_args_list->at(func->node_in_parent_graph())[func] = args_list;
    VLOG(3) << "ArgsList for " << func->name() << ", size " << args_list.size();
  }
  std::vector<FunctionInfo*> children_ptrs;
  TF_RETURN_IF_ERROR(func->get_children_ptrs(&children_ptrs));
  for (FunctionInfo* child : children_ptrs) {
    TF_RETURN_IF_ERROR(GetFuncArgsList(child, func_args_list));
  }
  return absl::OkStatus();
}

template <typename NodeInfoContainer>
void TrackConsumption(
    const FuncArgsList& func_args_list, NodeInfoContainer& node_info_container,
    ProcessConsumerCallable<NodeInfoContainer> process_consumer) {
  const NodeInfo& node_info = node_info_container.node_info;
  for (const Edge* edge : GetOutEdges(node_info.node)) {
    if (edge->IsControlEdge()) continue;
    Node* dst_node = edge->dst();
    if (dst_node->type_string() == "TPUPartitionedCall" ||
        dst_node->type_string() == "BatchFunction" ||
        dst_node->type_string() == "StatefulPartitionedCall" ||
        dst_node->type_string() == "PartitionedCall" ||
        dst_node->type_string() == "While" || dst_node->type_string() == "If" ||
        dst_node->type_string() == "StatelessWhile" ||
        dst_node->type_string() == "StatelessIf" ||
        node_info.func->get_graph()->flib_def().Contains(
            dst_node->type_string())) {
      auto func_args = func_args_list.at(dst_node);
      for (auto it : func_args) {
        // _Arg Node in function referred to by dst_node.
        int arg_index = edge->dst_input();

        // If/StatelessIf is a special case for functions. The 0th arg is
        // a tensor that corresponds to Tcond and is not an input to the
        // then and else branch functions.
        if (dst_node->type_string() == "If" ||
            dst_node->type_string() == "StatelessIf") {
          // If the node is in the 0th arg position, treat it like any other
          // destination consumer because it is not an input to the functions.
          if (arg_index == 0) {
            process_consumer(node_info_container, node_info.func, dst_node,
                             edge, true);
            continue;
          }
          // If the node is not in the 0th arg position, it is an input to the
          // then or else branch functions. In that case, we need to update the
          // arg_index to match the expect index of the function. We can do this
          // by subtracting 1 from the arg_index. This will ensure the Nth
          // function arg will now correspond to arg_index N.
          arg_index = edge->dst_input() - 1;
        }
        process_consumer(node_info_container, it.first, it.second.at(arg_index),
                         edge, false);
      }
    } else if (dst_node->type_string() == "Switch") {
      process_consumer(node_info_container, node_info.func, dst_node, edge,
                       false);
    } else {
      // ReadVariableOp / ResourceGather / AssignVariableOp /
      // copybara:strip_begin
      // copybara:strip_end
      process_consumer(node_info_container, node_info.func, dst_node, edge,
                       true);
    }
  }
}

// Get the consumers for a particular variable op.
void TrackResourceConsumption(const FuncArgsList& func_args_list,
                              ResourceNodeInfo* resource) {
  TrackConsumption<ResourceNodeInfo>(
      func_args_list, *resource,
      [&func_args_list](ResourceNodeInfo& node_info, FunctionInfo* func,
                        Node* node, const Edge* /*edge*/,
                        bool is_end_consumer) {
        VLOG(3) << "TrackResourceConsumption: " << node->name();
        auto consumer = std::make_unique<ResourceNodeInfo>();
        consumer->node_info.func = func;
        consumer->node_info.node = node;
        consumer->node_info.is_end_consumer = is_end_consumer;
        if (!is_end_consumer)
          TrackResourceConsumption(func_args_list, consumer.get());
        node_info.consumers.insert(std::move(consumer));
      });
}

// Get the consumers for a particular const op.
void TrackConstConsumption(const FuncArgsList& func_args_list,
                           ConstNodeInfo* const_node) {
  TrackConsumption<ConstNodeInfo>(
      func_args_list, *const_node,
      [&func_args_list](ConstNodeInfo& node_info, FunctionInfo* func,
                        Node* node, const Edge* edge, bool is_end_consumer) {
        VLOG(3) << "TrackConstConsumption: " << node->name();
        auto consumer = std::make_unique<ConstNodeInfo>();
        consumer->node_info.func = func;
        consumer->node_info.node = node;
        consumer->node_info.is_end_consumer = is_end_consumer;
        if (!is_end_consumer)
          TrackConstConsumption(func_args_list, consumer.get());
        node_info.consumers[edge] = std::move(consumer);
      });
}

// Get the variable ops and their consumers.
void AnalyseResourceVars(
    FunctionInfo* root, const FuncArgsList& func_args_list,
    ::gtl::linked_hash_set<std::unique_ptr<ResourceNodeInfo>>* resource_vars) {
  for (Node* node : root->get_graph()->nodes()) {
    if (node->type_string() != "VarHandleOp") continue;
    VLOG(3) << "AnalyseResourceVars: " << node->name();
    auto resource = std::make_unique<ResourceNodeInfo>();
    resource->node_info.node = node;
    resource->node_info.func = root;
    resource->node_info.is_end_consumer = false;
    TrackResourceConsumption(func_args_list, resource.get());
    resource_vars->insert(std::move(resource));
  }
}

// Get the const ops and their consumers.
absl::Status AnalyseConsts(
    FunctionInfo* root, const FuncArgsList& func_args_list,
    ::gtl::linked_hash_map<Node*, std::unique_ptr<ConstNodeInfo>>*
        const_infos) {
  std::vector<FunctionInfo*> funcs;
  TF_RETURN_IF_ERROR(root->Find(
      [&func_args_list, &const_infos](FunctionInfo* func) {
        for (Node* node : func->get_graph()->nodes()) {
          if (node->type_string() != "Const") continue;
          if (const_infos->contains(node)) continue;
          VLOG(3) << "AnalyseConsts: " << node->name();
          auto const_info = std::make_unique<ConstNodeInfo>();
          const_info->node_info.node = node;
          const_info->node_info.func = func;
          const_info->node_info.is_end_consumer = false;
          TrackConstConsumption(func_args_list, const_info.get());
          const_infos->emplace(node, std::move(const_info));
        }
        return false;
      },
      &funcs));
  return absl::OkStatus();
}

void BuildVarHandleInfo(ResourceNodeInfo* resource, VarHandleInfo* info) {
  info->resource_var = resource;
  Node* node = resource->node_info.node;
  if (node->type_string() == "AssignVariableOp") {
    // (Initializer / RestoreV2) --> AssignVariableOp
    for (const Edge* assign_in_edge : GetInEdges(node)) {
      if (assign_in_edge->src()->output_type(assign_in_edge->src_output()) !=
              DT_RESOURCE &&
          !assign_in_edge->IsControlEdge()) {
        info->assign_node_to_rval_edge_map[resource] = assign_in_edge;
        break;
      }
    }
  } else if (node->type_string() == "ReadVariableOp" ||
             node->type_string() == "ResourceGather" ||
             node->type_string() == "ResourceGatherNd") {
    // VarHandleOp --> ReadVariableOp --> (SaveV2, Identity, etc.)
  } else if (node->type_string() ==
             absl::StrCat("Ad", "BrainResourceDeserialize")) {
    info->resource_deserialize_node = resource;
  } else {
    if (node->type_string() == "_Arg") {
      info->resource_args.insert(resource);
    }
    for (const auto& consumer_ptr : resource->consumers) {
      BuildVarHandleInfo(consumer_ptr.get(), info);
    }
  }
}

void BuildVarHandleCollection(
    const ::gtl::linked_hash_set<std::unique_ptr<ResourceNodeInfo>>&
        resource_vars,
    const std::function<bool(Node*)>& condition_fn,
    VarHandleCollection* var_handles) {
  for (const auto& it : resource_vars) {
    if (!condition_fn(it->node_info.node)) continue;
    VarHandleInfo info = VarHandleInfo();
    BuildVarHandleInfo(it.get(), &info);
    (*var_handles)[it->node_info.node] = info;
  }
}

void FilterVarHandleCollectionByFuncScope(const FunctionInfo& scope_func,
                                          VarHandleCollection* var_handles) {
  std::vector<Node*> var_to_remove;
  for (const auto& it : *var_handles) {
    bool has_usage_in_scope = false;
    for (ResourceNodeInfo* consumer : it.second.consumers) {
      if (consumer->node_info.func->IsDescendantOfInclusive(scope_func)) {
        has_usage_in_scope = true;
        break;
      }
    }
    if (!has_usage_in_scope) var_to_remove.push_back(it.first);
  }
  for (Node* var_handle_node : var_to_remove)
    var_handles->erase(var_handle_node);
}

absl::Status DuplicateConsts(ConnectedComponent& connected_component,
                             FunctionInfo* func) {
  // Get all the consts and their consumers.
  FunctionInfo* root_func = func->GetRoot();
  FuncArgsList func_args_list;
  TF_RETURN_IF_ERROR(GetFuncArgsList(root_func, &func_args_list));
  ::gtl::linked_hash_map<Node*, std::unique_ptr<ConstNodeInfo>> const_infos;
  TF_RETURN_IF_ERROR(AnalyseConsts(root_func, func_args_list, &const_infos));

  for (const auto& [node, const_info] : const_infos) {
    if (connected_component.node_ids.contains(node->id())) continue;
    // Only duplicate constants that are both used on the CPU and in an XLA
    // cluster.
    bool has_consumer_inside_of_cc = false;
    bool has_consumer_outside_of_cc = false;
    for (const auto& [_, consumer] : const_info->consumers) {
      if (connected_component.node_ids.contains(
              consumer->node_info.node->id())) {
        has_consumer_inside_of_cc = true;
      } else {
        has_consumer_outside_of_cc = true;
      }
    }
    if (!has_consumer_inside_of_cc || !has_consumer_outside_of_cc) continue;

    // Duplicate existing const op and clear any device assignment it might
    // have.
    const NodeInfo& node_info = const_info->node_info;
    Graph* graph = node_info.func->get_graph();
    NodeDef node_def = node->def();
    *node_def.mutable_name() = absl::StrCat(node->name(), "/const_dup");
    *node_def.mutable_device() = "";
    TF_ASSIGN_OR_RETURN(Node * duplicate, graph->AddNode(node_def));

    // If the Const op is used in a sub function on the TPU, it shouldn't
    // be added to the connected_component's node_ids. They only contain
    // node_ids in the top level future TPU function.
    if (func->name() == node_info.func->name()) {
      connected_component.node_ids.insert(duplicate->id());
    }

    // Remove existing input edge to any consumers in the XLA cluster and
    // replace it with an edge to the duplicated const.
    for (const auto& [edge, consumer] : const_infos[node]->consumers) {
      if (!connected_component.node_ids.contains(
              consumer->node_info.node->id()))
        continue;
      if (connected_component.input_edges.contains(edge))
        connected_component.input_edges.erase(edge);
      TF_RETURN_IF_ERROR(
          graph->UpdateEdge(duplicate, 0, edge->dst(), edge->dst_input()));
    }

    VLOG(3) << "Duplicated const: " << node->name();
  }
  return absl::OkStatus();
}

// Check if a function graph is eligible for partition.
bool IsPartitionEligible(
    FunctionInfo* func, const FunctionLibraryDefinition& flib_def,
    const ::gtl::linked_hash_set<string>& disallowed_nodes,
    absl::string_view xla_jit_device) {
  bool has_partitioned_call = false;
  bool has_tpu_compatible_partitioned_call = false;
  bool is_save_or_restore_call = false;

  std::vector<Node*> post_order;
  GetPostOrder(*func->get_graph(), &post_order, NodeComparatorID());
  ::gtl::linked_hash_map<Node*, bool> compatibility;
  TF_CHECK_OK(GetTpuCompatibility(flib_def, post_order, disallowed_nodes, func,
                                  &compatibility, xla_jit_device));

  for (Node* node : func->get_graph()->nodes()) {
    if (node->type_string() == "PartitionedCall" ||
        node->type_string() == "StatefulPartitionedCall") {
      has_partitioned_call = true;
      if (compatibility.at(node)) has_tpu_compatible_partitioned_call = true;
    }
    if ((node->type_string() == "SaveV2" ||
         node->type_string() == "RestoreV2") &&
        !func->IsRoot())
      is_save_or_restore_call = true;
  }

  // A FuncGraph is eligible for partition if the graph
  // 1) doesn't contain StatefulPartitionedCall or PartitionedCall; Or
  // 2) contains a TPU-compatible StatefulPartitionedCall or PartitionedCall.
  if ((!has_partitioned_call || has_tpu_compatible_partitioned_call) &&
      !is_save_or_restore_call)
    return true;
  return false;
}

// Traverse the function tree to find an eligible partition candidate
// which has larger overhead than the current candidate.
// `func_names` allows users to specify functions to partition: Only functions
// in `func_names` will be selected.
// If empty all functions are taken into consideration.
FunctionInfo* FindPartitionCandidate(
    FunctionInfo* root_func, const ::gtl::linked_hash_set<string>& func_names,
    const ::gtl::linked_hash_set<string>& disallowed_nodes,
    absl::string_view xla_jit_device) {
  FunctionInfo* candidate = nullptr;
  FunctionLibraryDefinition flib_def = root_func->get_graph()->flib_def();

  std::vector<FunctionInfo*> stack;
  stack.push_back(root_func);
  while (!stack.empty()) {
    FunctionInfo* info = stack.back();
    stack.pop_back();
    if (IsPartitionEligible(info, flib_def, disallowed_nodes, xla_jit_device)) {
      VLOG(1) << info->name() << " is eligible";
      // Find a candidate with larger overhead.
      if ((candidate == nullptr || info->overhead() > candidate->overhead()) &&
          // If name matches user-specified function for partition,
          // or function name is not specified (empty string).
          (func_names.empty() || func_names.contains(info->name())))
        candidate = info;
    }
    std::vector<FunctionInfo*> children_ptrs;
    TF_CHECK_OK(info->get_children_ptrs(&children_ptrs));
    for (FunctionInfo* child : children_ptrs) stack.push_back(child);
  }
  return candidate;
}

bool IsIndependentConnectedComponentsOnTpu(const ConnectedComponent& cluster,
                                           const ConnectedComponent& candidate,
                                           const Predecessors& predecessors) {
  ::gtl::linked_hash_set<Node*> input_nodes, output_nodes;
  for (const Edge* edge : cluster.input_edges) {
    input_nodes.insert(edge->src());
  }
  for (const Edge* edge : cluster.output_edges) {
    output_nodes.insert(edge->dst());
  }
  for (const Edge* edge : candidate.input_edges) {
    if (IsDescendantOf(edge->src(), output_nodes, predecessors)) {
      VLOG(3) << "Cannot merge candidate (overhead " << candidate.overhead
              << ") with cluster (overhead " << cluster.overhead
              << ") because candidate's input node " << edge->src()->name()
              << " is descendant of cluster.";
      return false;
    }
  }
  for (const Edge* edge : candidate.output_edges) {
    if (IsPredecessorOf(edge->dst(), input_nodes, predecessors)) {
      VLOG(3) << "Cannot merge candidate (overhead " << candidate.overhead
              << ") with cluster (overhead " << cluster.overhead
              << ") because candidate's output node " << edge->dst()->name()
              << " is descendant of cluster.";
      return false;
    }
  }
  return true;
}

::gtl::linked_hash_set<const Edge*> GetEdgesIntersection(
    ::gtl::linked_hash_set<const Edge*> edges1,
    ::gtl::linked_hash_set<const Edge*> edges2) {
  std::set<const Edge*, EdgeComparatorID> intersection;
  std::set<const Edge*, EdgeComparatorID> e1(edges1.begin(), edges1.end());
  std::set<const Edge*, EdgeComparatorID> e2(edges2.begin(), edges2.end());

  std::set_intersection(e1.begin(), e1.end(), e2.begin(), e2.end(),
                        std::inserter(intersection, intersection.end()),
                        EdgeComparatorID());
  return ::gtl::linked_hash_set<const Edge*>(intersection.begin(),
                                             intersection.end());
}

// Handle the cases where input edges of one connected component are the output
// edges of another connected component.
absl::Status HandleConnectedComponentOverlap(
    Graph* graph, std::vector<ConnectedComponent>& tpu_clusters) {
  if (tpu_clusters.size() == 1) return absl::OkStatus();

  // Identify the problematic edges.
  ::gtl::linked_hash_set<const Edge*> input_edges;
  ::gtl::linked_hash_set<const Edge*> output_edges;
  for (ConnectedComponent& cluster : tpu_clusters) {
    input_edges.insert(cluster.input_edges.begin(), cluster.input_edges.end());
    output_edges.insert(cluster.output_edges.begin(),
                        cluster.output_edges.end());
    output_edges.insert(cluster.nonsplit_output_edges.begin(),
                        cluster.nonsplit_output_edges.end());
  }
  ::gtl::linked_hash_set<const Edge*> overlap_edges =
      GetEdgesIntersection(input_edges, output_edges);

  // Replace each problematic edge with two new edges. One edge will be the new
  // output edge and the other will be the new input edge.
  ::gtl::linked_hash_map<const Edge*, std::pair<const Edge*, const Edge*>>
      updated_edges;
  for (const Edge* edge : overlap_edges) {
    const Edge* updated_input_edge;
    const Edge* updated_output_edge;
    if (edge->IsControlEdge()) {
      // If the problematic edge is a control edge, replace it with two edges
      // connected by a NoOp.
      // src -> dst      becomes     src -> NoOp -> dst.
      TF_ASSIGN_OR_RETURN(
          Node * noop,
          NodeBuilder(absl::StrCat(edge->src()->name(), "_noop_", edge->id()),
                      "NoOp")
              .Finalize(graph));
      updated_output_edge = graph->AddControlEdge(edge->src(), noop);
      updated_input_edge = graph->AddControlEdge(noop, edge->dst());

    } else {
      // If the problematic edge is not a control edge, replace it with two
      // edges connected by an Identity op.
      // src -> dst      becomes     src -> Identity -> dst.
      Node* src = edge->src();
      int src_output = edge->src_output();
      TF_ASSIGN_OR_RETURN(
          Node * identity,
          NodeBuilder(absl::StrCat(src->name(), "_identity_", edge->id()),
                      "Identity")
              .Attr("T", src->output_type(src_output))
              .Input(src, src_output)
              .Finalize(graph));
      updated_input_edge =
          graph->AddEdge(identity, 0, edge->dst(), edge->dst_input());
      updated_output_edge = *identity->in_edges().begin();
    }
    updated_edges.emplace(
        std::piecewise_construct, std::forward_as_tuple(edge),
        std::forward_as_tuple(updated_output_edge, updated_input_edge));
  }

  // Update the connected components to use the new edges instead of the
  // problematic overlapping edges.
  for (ConnectedComponent& cluster : tpu_clusters) {
    ::gtl::linked_hash_set<const Edge*> input_edges_to_update =
        GetEdgesIntersection(cluster.input_edges, overlap_edges);
    for (const auto& edge : input_edges_to_update) {
      cluster.input_edges.insert(updated_edges[edge].second);
      cluster.input_edges.erase(edge);
    }
    ::gtl::linked_hash_set<const Edge*> output_edges_to_update =
        GetEdgesIntersection(cluster.output_edges, overlap_edges);
    for (const auto& edge : output_edges_to_update) {
      cluster.output_edges.insert(updated_edges[edge].first);
      cluster.output_edges.erase(edge);
    }
    ::gtl::linked_hash_set<const Edge*> nonsplit_output_edges_to_update =
        GetEdgesIntersection(cluster.nonsplit_output_edges, overlap_edges);
    for (const auto& edge : nonsplit_output_edges_to_update) {
      cluster.nonsplit_output_edges.insert(updated_edges[edge].first);
      cluster.nonsplit_output_edges.erase(edge);
    }
  }

  for (const Edge* edge : overlap_edges) {
    graph->RemoveEdge(edge);
  }

  return absl::OkStatus();
}

// Build function tree and search for partition candidate.
absl::Status GetGraphForPartition(
    Graph* graph, FunctionInfo* root_func, FunctionInfo** partition_candidate,
    ::gtl::linked_hash_map<string, int>* func_overhead,
    const ::gtl::linked_hash_set<string>& func_names,
    const ::gtl::linked_hash_set<string>& disallowed_nodes,
    absl::string_view xla_jit_device) {
  // Build FunctionInfo tree.
  root_func->set_graph(graph);
  TF_RETURN_IF_ERROR(root_func->Build(graph->flib_def(), func_overhead));

  // Seek partition candidate in tree.
  *partition_candidate =
      FindPartitionCandidate(root_func, func_names, disallowed_nodes,
                             xla_jit_device);
  if (*partition_candidate == nullptr) *partition_candidate = root_func;
  if (!func_names.empty() &&
      !func_names.contains((*partition_candidate)->name())) {
    std::string func_names_list = absl::StrJoin(func_names, ", ");
    return errors::InvalidArgument(
        "Could not find a TPU eligible function with the name(s) specified via "
        "function_alias and/or partition_func_names. Function name(s) provided "
        "via function_alias and/or partition_func_names are: ",
        func_names_list,
        " whereas partition candidate function identified is: ",
        (*partition_candidate)->name());
  }
  LOG(INFO) << (*partition_candidate)->name() << " selected for partition.";

  return absl::OkStatus();
}

bool VarHandleInfo::IsConsumedByFunction(FunctionInfo* func) const {
  for (ResourceNodeInfo* consumer : this->consumers) {
    if (consumer->node_info.func->IsDescendantOfInclusive(*func)) return true;
  }
  return false;
}
}  // namespace orbax
}  // namespace tensorflow
