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

#include "orbax/export/bfloat16_tookit/function_tree.h"

#include <deque>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "base/logging.h"
#include "third_party/absl/container/btree_set.h"
#include "third_party/absl/container/flat_hash_map.h"
#include "third_party/absl/flags/flag.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/strings/match.h"
#include "third_party/absl/strings/str_cat.h"
#include "third_party/absl/strings/str_join.h"
#include "third_party/absl/types/optional.h"
#include "orbax/export/bfloat16_tookit/utils.h"
#include "tensorflow/compiler/xla/tsl/platform/errors.h"
#include "tensorflow/compiler/xla/tsl/platform/statusor.h"
#include "tensorflow/core/framework/attr_value.proto.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.proto.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/dump_graph.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"
#include "util/gtl/linked_hash_map.h"

// TODO(b/275392289): Add long term fix that solves b/275392289 by default
// for Converter V2 models. This emergency fix is needed to unblock an LLM
// launch.
ABSL_FLAG(bool, experimental_use_global_op_registry_for_obm, false,
          "Whether to use the Global OpRegistry when creating new graphs "
          "instead of passing the full FunctionLibraryDefinition which uses "
          "significantly more memory. This flag is off by default because "
          "while it does reduce memory, it can cause crashes.");

namespace tensorflow {
namespace orbax {

namespace {
bool NodeIsPrunable(Node* node, bool consider_sink = true) {
  if (!node->IsOp()) {
    return false;
  }
  if (node->type_string() == "NoOp") {
    // NoOps are often special so let's not prune these.
    return false;
  }
  if (node->IsRetval()) {
    return false;
  }
  for (Node* out : node->out_nodes()) {
    if (out->IsSink()) {
      if (consider_sink) {
        return false;
        break;
      }
    } else if (out->name() != "NoOp") {
      return false;
      break;
    }
  }
  return true;
}
}  // namespace

FunctionInfo::FunctionInfo(const string& name) {
  root_graph_ = nullptr;
  node_in_parent_graph_ = nullptr;
  name_ = name;
  name_new_ = "";
  overhead_ = 0;
  parent_ = nullptr;
  children_ = std::vector<std::unique_ptr<FunctionInfo>>();
  mutated_ = false;
  deleted_ = false;
}

Graph* FunctionInfo::get_graph() {
  if (this->IsRoot()) return root_graph_;
  return graph_.get();
}

Graph* FunctionInfo::set_graph(std::unique_ptr<Graph>* graph) {
  graph_ = std::move(*graph);
  if (IsRoot()) root_graph_ = graph_.get();
  return graph_.get();
}

Graph* FunctionInfo::set_graph(Graph* graph) {
  CHECK(IsRoot());
  root_graph_ = graph;
  return root_graph_;
}

FunctionInfo* FunctionInfo::GetRoot() {
  if (IsRoot()) return this;
  return this->parent()->GetRoot();
}

Node* FunctionInfo::GetNodeInRootGraph() {
  CHECK(!IsRoot());
  if (this->parent()->IsRoot()) return this->node_in_parent_graph();
  return this->parent()->GetNodeInRootGraph();
}

absl::Status FunctionInfo::PruneUp() {
  if (IsRoot()) {
    // If we are at root, just remove dangling nodes.
    // Post-order will guarantee that each node is visited before its
    // inputs, hence it is sufficient to prune on the fly (after pruning each
    // node, its inputs may be made prunable, and we'll detect that when we
    // visit them later.)
    std::vector<Node*> nodes_in_post_order;
    GetPostOrder(*get_graph(), &nodes_in_post_order, NodeComparatorID());
    for (Node* node : nodes_in_post_order) {
      if (NodeIsPrunable(node, /*consider_sink=*/false) &&
          !node->IsFunctionCall()) {
        VLOG(4) << "Removing root node: " << node->name();
        get_graph()->RemoveNode(node);
      }
    }
    return absl::OkStatus();
  }
  // We are not root yet; prune non-arg dangling nodes, and mark the args that
  // we will need to prune. We need to keep track the "index" of each arg node,
  // so that later on when we remove args, we can re-index the remaining ones
  // correctly.
  std::vector<Node*> input_arg_nodes;
  for (Node* node : get_graph()->op_nodes()) {
    if (node->IsArg()) {
      int index;
      TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "index", &index));
      if (input_arg_nodes.size() < index + 1) {
        input_arg_nodes.resize(index + 1);
      }
      input_arg_nodes[index] = node;
    }
  }
  // Again, visit all nodes and prune the prunable ones. For arg nodes, we also
  // need to mark the indices of the ones we pruned.
  absl::btree_set<int> args_to_remove{};
  std::vector<Node*> nodes_in_post_order;
  GetPostOrder(*get_graph(), &nodes_in_post_order, NodeComparatorID());
  for (Node* node : nodes_in_post_order) {
    if (NodeIsPrunable(node)) {
      if (node->IsArg()) {
        int index;
        TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "index", &index));
        args_to_remove.insert(index);
      }
      VLOG(4) << "Removing node: " << node->name();
      get_graph()->RemoveNode(node);
    }
  }
  // Calculate the new indices of the remaining args. Example:
  // Old args: [a, b, c, d]
  // New args: [a, c, d]
  // Old arg "b" would have an index of 1, but now c should have 1. So the old
  // index of "2" maps to the new index of "1", and so on.
  std::vector<Node*> new_input_arg_nodes;
  absl::flat_hash_map<int, int> args_to_keep_to_new_pos;
  for (int i = 0; i < input_arg_nodes.size(); i++) {
    if (args_to_remove.find(i) == args_to_remove.end()) {
      new_input_arg_nodes.push_back(input_arg_nodes[i]);
      int next_new_arg_pos = args_to_keep_to_new_pos.size();
      args_to_keep_to_new_pos.emplace(i, next_new_arg_pos);
    }
  }
  // Apply the new indices to the remaining arg nodes. When we reconstruct the
  // function def from this graph, these nodes are used to create the right
  // function signature.
  for (int i = 0; i < new_input_arg_nodes.size(); i++) {
    graph_transforms::SetNodeAttr("index", i,
                                  new_input_arg_nodes[i]->mutable_def());
  }
  TF_RETURN_IF_ERROR(MarkMutated("_prune_up"));

  // We also need to prune the args of the call-site op in the parent graph,
  // before recursing up to handle the effect of the dropped edges in the parent
  // function's graph.
  Node* callsite = node_in_parent_graph();
  VLOG(4) << "Looking at the callsite op " << callsite
          << "\nWant to remove a few things: \n"
          << "["
          << absl::StrJoin(args_to_remove.begin(), args_to_remove.end(), ", ")
          << "]";
  VLOG(4) << "======= BEFORE PRUNING ========== \n" << callsite->DebugString();

  // We should not prune away any control inputs since they are not visible in
  // the child function graph (hence should not be affected by any pruning we do
  // in the child graph).
  std::vector<NodeBuilder::NodeOut> data_inputs_to_keep;
  std::vector<Node*> control_inputs;
  for (auto edge : callsite->in_edges()) {
    if (edge->IsControlEdge()) {
      control_inputs.push_back(edge->src());
      continue;
    }
    if (args_to_keep_to_new_pos.contains(edge->dst_input())) {
      // Here we need to ensure that the input tensors are correctly connected -
      // the edges that connect to pruned args should be omitted, and the other
      // edges should be re-targeted towards the new arg indices.
      int input_pos = args_to_keep_to_new_pos[edge->dst_input()];
      if (data_inputs_to_keep.size() < input_pos + 1) {
        data_inputs_to_keep.resize(input_pos + 1);
      }
      data_inputs_to_keep[input_pos] =
          NodeBuilder::NodeOut(edge->src(), edge->src_output());
    }
  }
  // Here a new call op is created w/ the same name as the call op we are
  // replacing (this is important since the call op may be referenced elsewhere,
  // e.g. in SavedModel signature defs).
  NodeBuilder builder(callsite->name(), callsite->type_string());
  builder.Device(callsite->requested_device())
      .ControlInputs(control_inputs)
      // We replace the inputs w/ the re-indexed pruned list here.
      .Input(data_inputs_to_keep);
  // We will copy all op attrs except for Tin which was already created during
  // the previous ".Input" call.
  for (auto& [name, attr] : callsite->attrs()) {
    if (name != "Tin") builder.Attr(name, attr);
  }
  TF_ASSIGN_OR_RETURN(Node * new_callsite,
                      builder.Finalize(parent()->get_graph()));
  set_node_in_parent_graph(new_callsite);
  // We hook up the new call op outputs into the same outputs of the old call
  // op, then remove the old op.
  std::vector<const Edge*> callsite_out_edges{callsite->out_edges().begin(),
                                              callsite->out_edges().end()};
  for (const Edge* edge : callsite_out_edges) {
    parent()->get_graph()->AddEdge(new_callsite, edge->src_output(),
                                   edge->dst(), edge->dst_input());
    parent()->get_graph()->RemoveEdge(edge);
  }
  VLOG(4) << "======= AFTER PRUNING ========== \n"
          << new_callsite->DebugString();
  parent()->get_graph()->RemoveNode(callsite);
  // Now the call op in the parent graph has been replaced w/ one that depends
  // on fewer inputs, so additional nodes may have become prunable in the parent
  // function; hence we recurse up.
  return parent()->PruneUp();
}

absl::Status FunctionInfo::get_children_ptrs(
    std::vector<FunctionInfo*>* children_ptrs) {
  children_ptrs->reserve(this->children_.size());
  for (int i = 0; i < this->children_.size(); ++i) {
    if (this->children_.at(i)->IsDeleted()) continue;
    children_ptrs->push_back(this->children_.at(i).get());
  }
  return absl::OkStatus();
}

absl::StatusOr<FunctionInfo*> FunctionInfo::get_child(Node* call_node) {
  for (auto& child : this->children_) {
    if (child->IsDeleted()) continue;
    if (child->node_in_parent_graph() == call_node) return child.get();
  }
  return errors::NotFound(
      absl::StrCat("Could not find a child function called by node ",
                   call_node->DebugString()));
}

absl::Status FunctionInfo::BuildChild(
    const FunctionLibraryDefinition& flib_def,
    ::gtl::linked_hash_map<string, int>* func_overhead, Node* node,
    string attr_name, string func_name, absl::optional<string> new_func_name) {
  if (!new_func_name.has_value()) {
    new_func_name = func_name;
  }
  this->children_.push_back(
      std::make_unique<FunctionInfo>(new_func_name.value()));
  FunctionInfo* child = this->children_.back().get();
  // Extract function's graph and other info.
  std::unique_ptr<Graph> func_graph;
  if (absl::GetFlag(FLAGS_experimental_use_global_op_registry_for_obm)) {
    func_graph = std::make_unique<Graph>(OpRegistry::Global());
  } else {
    func_graph = std::make_unique<Graph>(flib_def);
  }
  TF_RETURN_IF_ERROR(ExtractGraphFromFunction(
      flib_def, func_name, func_graph.get(), &child->control_ret_));
  // Fill in function info.
  // Must set parent before calling set_graph(), so that the check for
  // IsRoot() in set_graph() returns false.
  child->parent_ = this;
  child->set_graph(&func_graph);
  child->node_in_parent_graph_ = node;
  child->func_attr_ = attr_name;
  // Build child.
  TF_RETURN_IF_ERROR(child->Build(flib_def, func_overhead));
  // Update overhead.
  this->overhead_ += child->overhead_;
  return absl::OkStatus();
}

// Build function tree recursively.
absl::Status FunctionInfo::Build(
    const FunctionLibraryDefinition& flib_def,
    ::gtl::linked_hash_map<string, int>* func_overhead) {
  uint unique_name_counter = 0;

  this->overhead_ = 0;
  for (Node* node : this->get_graph()->nodes()) {
    // Accumulate overhead for root.
    this->overhead_ += GetNodeOverhead(node);
    // Build FunctionInfo for functions.
    // IsFunctionCall() returns true for three NodeClass types:
    // 1. NC_PARTITIONED_CALL. This excludes TPUPartitionedCall currently. We
    //    explicitly captures all *PartitionedCalls.
    // 2. NC_FUNCTION_OP. This includes While, If, and customized functions.
    // 3. NC_SYMBOLIC_GRADIENT. We don't support this in inference.
    // Therefore we have to manually address each case without using
    // IsFunctionCall.
    if (node->type_string() == "SymbolicGradient") continue;
    std::vector<string> attr_names;
    if (absl::StrContains(node->type_string(), "PartitionedCall") ||
        node->type_string() == "BatchFunction") {
      attr_names.push_back("f");
    } else if (node->type_string() == "While" ||
               node->type_string() == "StatelessWhile") {
      attr_names.push_back("body");
      attr_names.push_back("cond");
    } else if (node->type_string() == "If" ||
               node->type_string() == "StatelessIf") {
      attr_names.push_back("then_branch");
      attr_names.push_back("else_branch");
    }
    for (const auto& attr_name : attr_names) {
      const string& func_name = node->attrs().Find(attr_name)->func().name();
      TF_RETURN_IF_ERROR(
          BuildChild(flib_def, func_overhead, node, attr_name, func_name));
    }

    // Customized functions.
    if (flib_def.Contains(node->type_string())) {
      string new_func_name =
          absl::StrCat(node->type_string(), "_", unique_name_counter++);
      TF_RETURN_IF_ERROR(BuildChild(flib_def, func_overhead, node, "",
                                    node->type_string(), new_func_name));
    }
  }
  if (func_overhead != nullptr)
    (*func_overhead)[this->name()] = this->overhead_;

  VLOG(3) << "Function " << this->name() << ", overhead=" << this->overhead_
          << ", parent is " << (this->IsRoot() ? "null" : this->parent_->name())
          << ", node "
          << (this->IsRoot() ? "null" : node_in_parent_graph()->name());
  return absl::OkStatus();
}

absl::Status FunctionInfo::UpdateFuncNodeAttr() {
  if (func_attr().empty()) {
    VLOG(3) << "Update customized function " << this->name_new_;
    this->node_in_parent_graph_->set_name(this->name_new_);
  } else {
    VLOG(3) << "Update attribute " << this->func_attr() << " for function "
            << this->name_new_;
    NameAttrList func_attr;
    func_attr.set_name(this->name_new_);
    this->node_in_parent_graph_->ClearAttr(this->func_attr());
    this->node_in_parent_graph_->AddAttr(this->func_attr(), func_attr);
  }
  return absl::OkStatus();
}

// Update function library with mutated functions, and
// Update function call nodes in parent graph to point to new functions.
absl::Status FunctionInfo::LazyUpdate(FunctionDefLibrary* fdef_lib) {
  if (this->IsDeleted() || !this->IsMutated()) return absl::OkStatus();
  VLOG(3) << "Updating FunctionDef: " << this->name();
  for (const std::unique_ptr<FunctionInfo>& child : this->children_) {
    TF_RETURN_IF_ERROR(child.get()->LazyUpdate(fdef_lib));
  }
  if (!this->IsRoot()) {
    // Update function node attribute to point to new function name.
    TF_RETURN_IF_ERROR(UpdateFuncNodeAttr());

    // Add optimized function if it is not yet in function library.
    bool has_optimized_fdef = false;
    for (const FunctionDef& fdef : fdef_lib->function()) {
      if (fdef.has_signature() && fdef.signature().name() == this->name_new_) {
        has_optimized_fdef = true;
        break;
      }
    }
    if (!has_optimized_fdef) {
      auto control_ret_node_names =
          [&](const Node* node) -> absl::optional<std::string> {
        if (this->control_ret_.contains(node->name())) {
          return node->name();
        }
        return absl::nullopt;
      };
      TF_RETURN_IF_ERROR(
          GraphToFunctionDef(*(this->get_graph()), this->name_new_,
                             /*control_ret=*/control_ret_node_names,
                             /*fdef=*/fdef_lib->add_function()));
    }
  }
  // Flip mutated_ bit.
  this->mutated_ = false;
  VLOG(1) << DumpGraphToFile(this->name_new_, *this->get_graph());
  return absl::OkStatus();
}

// Mark the current function as mutated, and retrospectively mark all its
// ancestors until root.
absl::Status FunctionInfo::MarkMutated(const string& suffix) {
  this->name_new_ = absl::StrCat(this->name(), suffix);
  this->mutated_ = true;
  if (!this->IsRoot()) TF_RETURN_IF_ERROR(this->parent_->MarkMutated(suffix));
  return absl::OkStatus();
}

absl::Status FunctionInfo::RebuildChildren(
    const FunctionLibraryDefinition& flib_def,
    ::gtl::linked_hash_map<string, int>* func_overhead) {
  children_.clear();
  return Build(flib_def, func_overhead);
}

void FunctionInfo::MarkDeleted() {
  this->deleted_ = true;
  for (int i = 0; i < this->children_.size(); ++i) {
    this->children_.at(i)->MarkDeleted();
  }
}

bool FunctionInfo::IsDescendantOfInclusive(const FunctionInfo& other) const {
  const FunctionInfo* func = this;
  while (!func->IsRoot() && func != &other) func = func->parent();
  return (func == &other);
}

absl::Status FunctionInfo::Find(
    const std::function<bool(FunctionInfo*)>& filter_fn,
    std::vector<FunctionInfo*>* result) {
  std::deque<FunctionInfo*> func_queue;
  func_queue.push_back(this);
  while (!func_queue.empty()) {
    FunctionInfo* func = func_queue.front();
    func_queue.pop_front();
    if (filter_fn(func)) {
      result->push_back(func);
    }
    for (int i = 0; i < func->children_.size(); ++i) {
      func_queue.push_back(func->children_.at(i).get());
    }
  }
  return absl::OkStatus();
}

absl::Status FunctionInfo::RewriteGraphRecursively(
    std::string rewrite_suffix,
    const std::function<bool(FunctionInfo*)>& filter_fn,
    const std::function<absl::Status(FunctionInfo*)>& rewrite_fn) {
  // Rewrite func graph.
  TF_RETURN_IF_ERROR(rewrite_fn(this));
  TF_RETURN_IF_ERROR(MarkMutated(rewrite_suffix));

  // Recursively rewrite children graphs.
  std::vector<FunctionInfo*> children_ptrs;
  TF_RETURN_IF_ERROR(get_children_ptrs(&children_ptrs));
  for (FunctionInfo* child : children_ptrs) {
    // Non-standard function requires special handling, e.g. Flow input
    // _Arg.
    if (filter_fn(child)) continue;
    TF_RETURN_IF_ERROR(
        child->RewriteGraphRecursively(rewrite_suffix, filter_fn, rewrite_fn));
  }
  return absl::OkStatus();
}

bool FunctionInfo::IsTopLevelGpuFunction() const {
  if (this->IsRoot()) return false;

  Node* node = this->node_in_parent_graph();
  return node->type_string() == "StatefulPartitionedCall" &&
         GetBooleanAttribute(node, "_XlaMustCompile") &&
         absl::StrContains(node->requested_device(), "GPU");
}

bool FunctionInfo::ExecutesOnGpu() const {
  const FunctionInfo* func = this;
  while (!func->IsRoot()) {
    if (func->IsTopLevelGpuFunction()) return true;
    func = func->parent();
  }
  return false;
}

bool FunctionInfo::IsTopLevelTpuFunction() const {
  if (this->IsRoot()) return false;

  return this->node_in_parent_graph()->type_string() == "TPUPartitionedCall";
}

bool FunctionInfo::ExecutesOnTpu() const {
  const FunctionInfo* func = this;
  while (!func->IsRoot()) {
    if (func->IsTopLevelTpuFunction()) return true;
    func = func->parent();
  }
  return false;
}

bool FunctionInfo::IsTopLevelXlaFunction() {
  if (this->IsRoot()) return false;
  if (this->IsTopLevelGpuFunction() || this->IsTopLevelTpuFunction())
    return true;
  if (this->node_in_parent_graph()->type_string() != "StatefulPartitionedCall")
    return false;

  bool has_replicate_metadata = false;
  bool has_replicated_input = false;
  bool has_replicated_output = false;
  for (const Node* node : this->get_graph()->nodes()) {
    if (node->type_string() == "TPUReplicateMetadata")
      has_replicate_metadata = true;
    if (node->type_string() == "TPUReplicatedInput")
      has_replicated_input = true;
    if (node->type_string() == "TPUReplicatedOutput")
      has_replicated_output = true;

    // If all three ops are found, return true early.
    if (has_replicate_metadata && has_replicated_input && has_replicated_output)
      return true;
  }

  // If it hasn't already returned, at least some op's missing. Return false.
  return false;
}

bool FunctionInfo::ExecutesOnXlaDevice() {
  FunctionInfo* func = this;
  while (!func->IsRoot()) {
    if (func->IsTopLevelXlaFunction()) return true;
    func = func->parent();
  }
  return false;
}

}  // namespace orbax
}  // namespace tensorflow
