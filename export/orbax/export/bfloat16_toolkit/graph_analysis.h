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

#ifndef THIRD_PARTY_PY_ORBAX_EXPORT_BFLOAT16_TOOLKIT_GRAPH_ANALYSIS_H_
#define THIRD_PARTY_PY_ORBAX_EXPORT_BFLOAT16_TOOLKIT_GRAPH_ANALYSIS_H_

#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "third_party/absl/status/status.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/absl/strings/string_view.h"
#include "orbax/export/bfloat16_toolkit/function_tree.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/tpu/tpu_defs.h"
#include "util/gtl/linked_hash_map.h"
#include "util/gtl/linked_hash_set.h"

namespace tensorflow {
namespace orbax {

constexpr char kIdentity[] = "Identity";
constexpr char kVariable[] = "Variable";
constexpr char kVariableV2[] = "VariableV2";
constexpr char kSaveV2[] = "SaveV2";

// { node -> { predecessor_node -> distance } }
using Predecessors =
    ::gtl::linked_hash_map<Node*, ::gtl::linked_hash_map<Node*, int>>;

// call_node_in_parent_graph -> (func -> (arg_index -> arg_node))
using FuncArgsList = ::gtl::linked_hash_map<
    Node*,
    ::gtl::linked_hash_map<FunctionInfo*, ::gtl::linked_hash_map<int, Node*>>>;

struct ConnectedComponent {
  int overhead = 0;  // Overhead of ops contained in nodes of this component.
  ::gtl::linked_hash_set<const Edge*> input_edges;
  ::gtl::linked_hash_set<const Edge*> output_edges;
  ::gtl::linked_hash_set<const Edge*> nonsplit_output_edges;
  ::gtl::linked_hash_set<int> node_ids;

  ConnectedComponent(int o, ::gtl::linked_hash_set<const Edge*> ie,
                     ::gtl::linked_hash_set<const Edge*> oe,
                     ::gtl::linked_hash_set<const Edge*> ns_oe,
                     ::gtl::linked_hash_set<int> nids)
      : overhead(o),
        input_edges(ie),
        output_edges(oe),
        nonsplit_output_edges(ns_oe),
        node_ids(nids) {}
  ConnectedComponent(int* o, ::gtl::linked_hash_set<const Edge*>* ie,
                     ::gtl::linked_hash_set<const Edge*>* oe,
                     ::gtl::linked_hash_set<const Edge*>* ns_oe,
                     ::gtl::linked_hash_set<int>* nids)
      : overhead(*o),
        input_edges(*ie),
        output_edges(*oe),
        nonsplit_output_edges(*ns_oe),
        node_ids(*nids) {}
  ConnectedComponent()
      : overhead(0),
        input_edges(::gtl::linked_hash_set<const Edge*>()),
        output_edges(::gtl::linked_hash_set<const Edge*>()),
        nonsplit_output_edges(::gtl::linked_hash_set<const Edge*>()),
        node_ids(::gtl::linked_hash_set<int>()) {}

  // Merges ConnectedComponent `other` into `this`.
  void Merge(const ConnectedComponent& other) {
    this->overhead += other.overhead;
    this->input_edges.insert(other.input_edges.begin(),
                             other.input_edges.end());
    this->output_edges.insert(other.output_edges.begin(),
                              other.output_edges.end());
    this->nonsplit_output_edges.insert(other.nonsplit_output_edges.begin(),
                                       other.nonsplit_output_edges.end());
    this->node_ids.insert(other.node_ids.begin(), other.node_ids.end());
  }
};

class CompareByOverhead {
 public:
  bool operator()(const ConnectedComponent& a, const ConnectedComponent& b) {
    if (a.overhead < b.overhead)
      return true;
    else if (a.overhead > b.overhead)
      return false;
    else if (a.input_edges.size() + a.output_edges.size() +
                 a.nonsplit_output_edges.size() <
             b.input_edges.size() + b.output_edges.size() +
                 b.nonsplit_output_edges.size())
      return true;
    else if (a.input_edges.size() + a.output_edges.size() +
                 a.nonsplit_output_edges.size() >
             b.input_edges.size() + b.output_edges.size() +
                 b.nonsplit_output_edges.size())
      return false;
    else
      return false;
  }
};

struct NodeInfo {
  // The node described by this node info.
  Node* node;
  // The function the node is contained in.
  FunctionInfo* func;
  // Whether or not this node is an end consumer. Examples of end consumers
  // for variables are ops like ReadVariableOp or ResourceGather.
  bool is_end_consumer;
};

struct ResourceNodeInfo {
  NodeInfo node_info;
  // Four possible kinds of nodes can be contained in `node`:
  // 1. VarHandleOp: In root function
  // 2. _Arg node with "T=DT_RESOURCE"
  // 3. End consumers: ReadVariableOp / ResourceGather etc. / AssignVariableOp /
  // (SaveV2 will be preceded by ReadVariableOp.)
  ::gtl::linked_hash_set<std::unique_ptr<ResourceNodeInfo>> consumers;
};

struct VarHandleInfo {
  ResourceNodeInfo* resource_var;
  // { key: ResourceNodeInfo of AssignVariableOp,
  //   val: Edge (RestoreV2/Initializer) -> AssignVariableOp }
  ::gtl::linked_hash_map<ResourceNodeInfo*, const Edge*>
      assign_node_to_rval_edge_map;
  ::gtl::linked_hash_set<ResourceNodeInfo*> consumers;
  ::gtl::linked_hash_set<ResourceNodeInfo*> resource_args;
  // copybara:strip_begin
  ResourceNodeInfo* resource_deserialize_node;

  VarHandleInfo() : resource_deserialize_node(nullptr) {}
  // copybara:strip_end

  bool IsConsumedByFunction(FunctionInfo* func) const;
};

using VarHandleCollection = ::gtl::linked_hash_map<Node*, VarHandleInfo>;

absl::StatusOr<OutputTensor> ParseOutputTensorFromString(const Graph& g,
                                                         const string& tensor);

bool IsCompileTimeConstant(
    Node* node, ::gtl::linked_hash_map<Node*, bool>* compile_time_constant,
    const ::gtl::linked_hash_map<Node*, bool>& compatibility);

void EnforceCompileTimeConstantConstraints(
    Node* node, ::gtl::linked_hash_map<Node*, bool>* compile_time_constant,
    ::gtl::linked_hash_map<Node*, bool>* compatibility);

bool IsCallNodeTPUCompatible(Node* node, FunctionInfo* func_info, int depth);

bool IsTPUCompatible(Node* node, FunctionInfo* func_info, int depth,
                     absl::string_view xla_jit_device);

Status GetTpuCompatibility(
    const FunctionLibraryDefinition& flib_def, const std::vector<Node*>& nodes,
    const ::gtl::linked_hash_set<string>& disallowed_nodes,
    FunctionInfo* func_info, ::gtl::linked_hash_map<Node*, bool>* compatibility,
    absl::string_view xla_jit_device = DEVICE_TPU_XLA_JIT);

void GetPredecessors(const Graph* g,
                     const std::vector<Node*>& reverse_post_order,
                     Predecessors* predecessors);

void GetTpuDependency(
    const Graph* g, const std::vector<Node*>& reverse_post_order,
    const Predecessors& predecessors,
    const ::gtl::linked_hash_map<Node*, bool>& compatibility,
    ::gtl::linked_hash_set<std::pair<Node*, Node*>>* completely_depend_on);

bool IsDescendantOf(Node* node, const ::gtl::linked_hash_set<Node*>& node_set,
                    const Predecessors& predecessors);

bool IsPredecessorOf(Node* node, const ::gtl::linked_hash_set<Node*>& node_set,
                     const Predecessors& predecessors);

bool HasDescendant(Node* node, Graph* g,
                   const std::function<bool(Node*)>& condition_fn,
                   const Predecessors& predecessors);

bool HasPredecessor(Node* node, const std::function<bool(Node*)>& condition_fn,
                    const Predecessors& predecessors);

bool IsValidInputEdge(const Edge* e);

bool IsValidOutputEdge(const Edge* e);

// Identifies input_tensors, output_tensors and compute overhead.
Status AnalyseConnectedComponent(
    const Graph* g, ::gtl::linked_hash_set<int>* node_ids,
    ::gtl::linked_hash_set<const Edge*>* input_edges,
    ::gtl::linked_hash_set<const Edge*>* output_edges,
    ::gtl::linked_hash_set<const Edge*>* nonsplit_output_edges, int* overhead,
    const ::gtl::linked_hash_map<string, int>& func_overhead = {});

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
                       bool enforce_subgraph_convexity);

Status AnalyseGraph(
    Graph* g, const FunctionLibraryDefinition& flib_def,
    FunctionInfo* func_info, std::vector<Node*>* reverse_post_order,
    ::gtl::linked_hash_map<Node*, bool>* compatibility,
    Predecessors* predecessors,
    ::gtl::linked_hash_set<std::pair<Node*, Node*>>* tpu_complete_dependency,
    const ::gtl::linked_hash_set<string>& disallowed_nodes = {},
    absl::string_view xla_jit_device = DEVICE_TPU_XLA_JIT);

Status GetConnectedComponentsOnTpu(
    Graph* g, const std::vector<OutputTensor>& input_tensors,
    const std::vector<OutputTensor>& output_tensors,
    const ::gtl::linked_hash_map<Node*, bool>& compatibility,
    const Predecessors& predecessors,
    const ::gtl::linked_hash_set<std::pair<Node*, Node*>>&
        tpu_complete_dependency,
    std::vector<ConnectedComponent>* connected_components,
    const ::gtl::linked_hash_map<string, int>& func_overhead,
    bool enforce_subgraph_convexity);

Status GetEdgeOverhead(Graph* g, const FunctionLibraryDefinition& flib_def,
                       int unknown_dim_weight, int edge_penalty,
                       ::gtl::linked_hash_map<const Edge*, int>* edge_overhead);

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
    int unknown_dim_weight,
    const ::gtl::linked_hash_map<string, int>& func_overhead = {});

void GetMergeCandidates(
    const Graph* g, const ::gtl::linked_hash_set<int>& tpu_node_ids,
    const Predecessors& predecessors,
    const ::gtl::linked_hash_set<Node*>& original_input_nodes,
    ::gtl::linked_hash_map<Node*, ::gtl::linked_hash_set<Node*>>*
        merge_candidates);

void GetMergeCandidatesByType(
    const Graph* g, const ::gtl::linked_hash_set<string>& merge_node_types,
    const ::gtl::linked_hash_set<int>& tpu_node_ids,
    const Predecessors& predecessors,
    const ::gtl::linked_hash_set<Node*>& original_input_nodes,
    ::gtl::linked_hash_map<Node*, ::gtl::linked_hash_set<Node*>>*
        merge_candidates);

Status MaybeFindBetterCut(
    Graph* g, const FunctionLibraryDefinition& flib_def,
    ConnectedComponent* candidate, const std::vector<Node*>& reverse_post_order,
    const ::gtl::linked_hash_map<Node*, bool>& compatibility,
    const Predecessors& predecessors,
    const ::gtl::linked_hash_set<std::pair<Node*, Node*>>&
        tpu_complete_dependency,
    std::vector<ConnectedComponent>* connected_components,
    int edge_unknown_dim_weight, int edge_penalty, bool get_merge_node_by_type,
    bool enforce_subgraph_convexity,
    const ::gtl::linked_hash_map<string, int>& func_overhead = {});

Status GetFuncArgsList(FunctionInfo* func, FuncArgsList* func_args_list);

void TrackResourceConsumption(const FuncArgsList& func_args_list,
                              ResourceNodeInfo* resource);

void AnalyseResourceVars(
    FunctionInfo* root, const FuncArgsList& func_args_list,
    ::gtl::linked_hash_set<std::unique_ptr<ResourceNodeInfo>>* resource_vars);

void BuildVarHandleInfo(ResourceNodeInfo* resource, VarHandleInfo* info);

void BuildVarHandleCollection(
    const ::gtl::linked_hash_set<std::unique_ptr<ResourceNodeInfo>>&
        resource_vars,
    const std::function<bool(Node*)>& condition_fn,
    VarHandleCollection* var_handles);

void FilterVarHandleCollectionByFuncScope(const FunctionInfo& func,
                                          VarHandleCollection* var_handles);

// Duplicate any const ops that are not in this connected component but they
// are used as inputs to this connected component.
Status DuplicateConsts(ConnectedComponent& component, FunctionInfo* func);

void GetFunctionCallNames(Graph* g, ::gtl::linked_hash_set<string>* func_names,
                          const string& op_type);

bool IsPartitionEligible(
    FunctionInfo* func, const FunctionLibraryDefinition& flib_def,
    const ::gtl::linked_hash_set<string>& disallowed_nodes = {},
    absl::string_view xla_jit_device = DEVICE_TPU_XLA_JIT);

FunctionInfo* FindPartitionCandidate(
    FunctionInfo* root_func, const ::gtl::linked_hash_set<string>& func_names,
    const ::gtl::linked_hash_set<string>& disallowed_nodes = {},
    absl::string_view xla_jit_device = DEVICE_TPU_XLA_JIT);

// Whether two ConnectedComponentsOnTpu are independent on each other, in the
// sense that:
// 1. None of `candidate`'s input node is descendant of any of `cluster`'s
// output nodes;
// 2. None of `candidate`'s output node is precessor of any of `cluster`'s
// output nodes.
bool IsIndependentConnectedComponentsOnTpu(const ConnectedComponent& cluster,
                                           const ConnectedComponent& candidate,
                                           const Predecessors& predecessors);

// Handle the cases where input edges of one connected component are the output
// edges of another connected component.
absl::Status HandleConnectedComponentOverlap(
    Graph* graph, std::vector<ConnectedComponent>& tpu_clusters);

absl::Status GetGraphForPartition(
    Graph* g, FunctionInfo* root_func, FunctionInfo** partition_candidate,
    ::gtl::linked_hash_map<string, int>* func_overhead,
    const ::gtl::linked_hash_set<string>& func_names,
    const ::gtl::linked_hash_set<string>& disallowed_nodes = {},
    absl::string_view xla_jit_device = DEVICE_TPU_XLA_JIT);

}  // namespace orbax
}  // namespace tensorflow

#endif  // THIRD_PARTY_PY_ORBAX_EXPORT_BFLOAT16_TOOLKIT_GRAPH_ANALYSIS_H_
