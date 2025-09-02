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

#ifndef THIRD_PARTY_PY_ORBAX_EXPORT_BFLOAT16_TOOLKIT_FUNCTION_TREE_H_
#define THIRD_PARTY_PY_ORBAX_EXPORT_BFLOAT16_TOOLKIT_FUNCTION_TREE_H_

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "third_party/absl/status/status.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/absl/types/optional.h"
#include "tensorflow/compiler/xla/tsl/platform/status.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.proto.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/types.h"
#include "util/gtl/linked_hash_map.h"
#include "util/gtl/linked_hash_set.h"

namespace tensorflow {
namespace orbax {

// FunctionInfo is a class for recording tree-structured function call hierarchy
// in tensorflow graphs. A FunctionInfo instance is a node in that
// tree structure, representing a function call.
class FunctionInfo {
 public:
  explicit FunctionInfo(const string& name);

  // Build function tree recursively.
  absl::Status Build(const FunctionLibraryDefinition& flib_def,
                     ::gtl::linked_hash_map<string, int>* func_overhead);

  absl::Status BuildChild(const FunctionLibraryDefinition& flib_def,
                          ::gtl::linked_hash_map<string, int>* func_overhead,
                          Node* node, string attr_name, string func_name,
                          absl::optional<string> new_func_name = absl::nullopt);

  // Recursively update function graphs and the function library.
  absl::Status LazyUpdate(FunctionDefLibrary* fdef_lib);

  // Recursively mark a function and its ancestors as mutated.
  absl::Status MarkMutated(const string& suffix);

  // Transfers ownership of a Graph to this FunctionInfo object.
  // This is for ordinary function calls.
  Graph* set_graph(std::unique_ptr<Graph>* graph);

  // Sets `root_graph_` to point to top-level graph, which is owned by a
  // Converter instance. This is for root node only.
  Graph* set_graph(Graph* graph);

  // Returns pointer to the Graph this FunctionInfo represents.
  Graph* get_graph();

  // Returns pointer to root function.
  FunctionInfo* GetRoot();

  // For non-root functions, return call node in root graph.
  Node* GetNodeInRootGraph();

  // Will remove all unused nodes in this function, then remove the function
  // parameters if they become unused, then remove the arguments passed into the
  // callsite op in the parent function graph, then invoke itself for the parent
  // functions, recursing all the way up.
  absl::Status PruneUp();

  // Get a vector of pointers to children functions.
  absl::Status get_children_ptrs(std::vector<FunctionInfo*>* children_ptrs);

  // Get the child called by the given call op.
  absl::StatusOr<FunctionInfo*> get_child(Node* call_node);

  // Update corresponding attr in call node in parent graph.
  absl::Status UpdateFuncNodeAttr();

  // Rebuild children functions according to the updated graph.
  absl::Status RebuildChildren(
      const FunctionLibraryDefinition& flib_def,
      ::gtl::linked_hash_map<string, int>* func_overhead);

  // Update graph in a FunctionInfo with a new one, and
  // update all its children's node_in_parent_graph_. (Normal functions)
  Graph* UpdateFuncGraph(std::unique_ptr<Graph>* graph) {
    return UpdateFuncGraphImpl<std::unique_ptr<Graph>*>(graph);
  }
  // Update graph in a FunctionInfo with a new one, and
  // update all its children's node_in_parent_graph_. (Root function)
  Graph* UpdateFuncGraph(Graph* graph) {
    return UpdateFuncGraphImpl<Graph*>(graph);
  }

  // Recursively mark a function and its descendants as deleted.
  void MarkDeleted();

  bool IsDescendantOfInclusive(const FunctionInfo& other) const;

  // Find function from descendants. Results returned pre-order.
  absl::Status Find(const std::function<bool(FunctionInfo*)>& filter_fn,
                    std::vector<FunctionInfo*>* result);

  absl::Status RewriteGraphRecursively(
      std::string rewrite_suffix,
      const std::function<bool(FunctionInfo*)>& filter_fn,
      const std::function<absl::Status(FunctionInfo*)>& rewrite_fn);

  // Returns true if a function is the top level GPU function, which is defined
  // to be a function called with _XlaMustCompile attribute and GPU device
  // assignment. A function called with _XlaMustCompile attribute and GPU device
  // assignment nested within another such function will still return true.
  bool IsTopLevelGpuFunction() const;
  // Returns true if a function will execute on the GPU. Similar to
  // IsTopLevelGpuFunction above but it also returns true if this function is a
  // descendent of another GPU function.
  bool ExecutesOnGpu() const;

  // Returns true if a function is the top level TPU function, which is defined
  // to be a function called with TPUPartitionedCall. A function called with
  // TPUPartitionedCall nested within another such function will still return
  // true.
  bool IsTopLevelTpuFunction() const;
  // Returns true if a function will execute on the TPU. Similar to
  // IsTopLevelTpuFunction above but it also returns true if this function is a
  // descendent of another TPU function.
  bool ExecutesOnTpu() const;

  // Returns true if a function is the top level XLA function, which is defined
  // to be either top level GPU/TPU function or a device-agnostic XLA function,
  // which is called with StatefulPartitionedCall which in turn contains
  // TPUReplicateMetadata, TPUReplicatedInput, and TPUReplicatedOutput ops.
  bool IsTopLevelXlaFunction();
  // Returns true if a function will execute on the XLA device. Similar to
  // IsTopLevelXlaFunction above but it also returns true if this function is a
  // descendent of another XLA function.
  bool ExecutesOnXlaDevice();

  const string& name() const { return name_; }
  const string& func_attr() const { return func_attr_; }
  Node* node_in_parent_graph() const { return node_in_parent_graph_; }
  void set_node_in_parent_graph(Node* n) { node_in_parent_graph_ = n; }
  void clear_control_ret() { control_ret_.clear(); }
  FunctionInfo* parent() const { return parent_; }
  bool IsRoot() const { return (parent_ == nullptr); }
  int overhead() const { return overhead_; }
  bool IsMutated() const { return mutated_; }
  bool IsDeleted() const { return deleted_; }

 private:
  // One function call correspond to one graph, where the top-level tensorflow
  // graph is also treated as a special function call for simplicity.
  // The graph corresponding to this function call is captured by `graph_` (for
  // ordinary functions), or `root_graph_` (for top-level graph). Each instance
  // of FunctionInfo has ownership of the unique_ptr to the graph, except
  // the top-level graph which is owned by a Converter instance.
  // Calling get_graph() method will return a Graph* pointer regardless of
  // whether this instance represents a function or the top-level graph.
  Graph* root_graph_;
  std::unique_ptr<Graph> graph_;

  // The node that references this function in the graph of its parent.
  Node* node_in_parent_graph_;

  // Attribute name of function.
  // For standard functions, it should be "f";
  // For "While", it should be "body" or "cond";
  // For "If", it should be "then_branch" or "else_branch".
  string func_attr_;

  // A modified function will be appended to function library with a new name.
  string name_;
  string name_new_;

  // Overhead of the function graph, computed as sum of overhead of all nodes.
  int overhead_;

  // Pointer to parent in function call hierarchy. If parent is a nullptr,
  // this is the 'root' which represents the top-level tensorflow graph.
  FunctionInfo* parent_;

  // Unique pointers to children in function call hierarchy.
  std::vector<std::unique_ptr<FunctionInfo>> children_;

  // When a function's graph is modified, the `mutated_` bit of the function
  // and all its ancestors will be set.
  // When updating a function, we first update all its descendants with their
  // `mutated_` bit set recursively, and then the function itself.
  bool mutated_;

  // We cannot delete a function when still traversing the tree. But in further
  // processing of the tree, we must treat it as if the function has been
  // deleted. This tag allows for the check.
  bool deleted_;

  // Control return node names in original FunctionDef.
  ::gtl::linked_hash_set<string> control_ret_;

  template <typename T>
  Graph* UpdateFuncGraphImpl(T graph);
};

template <typename T>
Graph* FunctionInfo::UpdateFuncGraphImpl(T graph) {
  ::gtl::linked_hash_map<std::string, FunctionInfo*> call_node_names;
  std::vector<FunctionInfo*> children_ptrs;
  TF_CHECK_OK(this->get_children_ptrs(&children_ptrs));
  for (FunctionInfo* child : children_ptrs) {
    call_node_names[child->node_in_parent_graph()->name()] = child;
  }
  Graph* new_graph = this->set_graph(graph);
  for (Node* node : new_graph->nodes()) {
    if (call_node_names.contains(node->name())) {
      call_node_names.at(node->name())->set_node_in_parent_graph(node);
    }
  }
  return new_graph;
}

}  // namespace orbax
}  // namespace tensorflow

#endif  // THIRD_PARTY_PY_ORBAX_EXPORT_BFLOAT16_TOOLKIT_FUNCTION_TREE_H_
