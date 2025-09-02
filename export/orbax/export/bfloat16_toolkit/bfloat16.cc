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

#include "orbax/export/bfloat16_toolkit/bfloat16.h"

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "base/logging.h"
#include "third_party/absl/algorithm/container.h"
#include "third_party/absl/flags/flag.h"
#include "third_party/absl/log/check.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/strings/match.h"
#include "third_party/absl/strings/str_cat.h"
#include "third_party/absl/strings/string_view.h"
#include "orbax/export/bfloat16_toolkit/converter_options.proto.h"
#include "orbax/export/bfloat16_toolkit/function_tree.h"
#include "orbax/export/bfloat16_toolkit/graph_analysis.h"
#include "orbax/export/bfloat16_toolkit/optim_utils.h"
#include "orbax/export/bfloat16_toolkit/utils.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/passes.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/tsl/platform/errors.h"
#include "tensorflow/compiler/xla/tsl/platform/status.h"
#include "tensorflow/compiler/xla/tsl/platform/statusor.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/framework/attr_value.proto.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.proto.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.proto.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/bfloat16.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/meta_graph.proto.h"
#include "tensorflow/core/tpu/tpu_defs.h"
#include "tensorflow/core/util/dump_graph.h"
#include "util/gtl/linked_hash_map.h"
#include "util/gtl/linked_hash_set.h"

// TODO(b/241160425): Make removing all chained casts the default.
ABSL_FLAG(bool, experimental_remove_all_chained_casts_for_obm, false,
          "Whether to remove all chained casts versus just chained casts "
          "that share the same name.");

namespace tensorflow {
namespace orbax {

constexpr absl::string_view kXlaOutsideCompilationAttr =
    "_xla_outside_compilation";

bool RewriteGraphRecursivelyFilterFn(FunctionInfo* func) {
  return !IsStandardFunctionNode(func->node_in_parent_graph()) &&
         func->node_in_parent_graph()->type_string() != "While" &&
         func->node_in_parent_graph()->type_string() != "If" &&
         func->node_in_parent_graph()->type_string() != "StatelessWhile" &&
         func->node_in_parent_graph()->type_string() != "StatelessIf";
}

absl::StatusOr<std::string> FindTPUClusterName(const Edge* edge) {
  const AttrValue* attr = edge->src()->attrs().Find("_tpu_replicate");
  if (attr != nullptr && attr->has_s()) return attr->s();

  // TPUReplicatedInput nodes usually don't have the "_tpu_replicate" attribute.
  // So we find the destination node for the TPU cluster name.
  if ((edge->src()->type_string() == "TPUReplicatedInput")) {
    attr = edge->dst()->attrs().Find("_tpu_replicate");
    if (attr != nullptr && attr->has_s()) return attr->s();
    return errors::NotFound(
        "Could not find a TPU cluster name for the Cast node connecting the "
        "TPUReplicatedInput node.");
  }

  return std::string();
}

// Input node must have already been converted to src_dtype.
// Output node doesn't matter.
absl::Status CastFloatTensor(Graph* g, Node* node, int src_output,
                             DataType src_dtype = DT_FLOAT,
                             DataType dst_dtype = DT_BFLOAT16) {
  Node* cast_node = nullptr;
  std::vector<const Edge*> old_edges;
  for (const Edge* edge : GetOutEdges(node)) {
    if (edge->IsControlEdge()) continue;
    // Only cast float tensors.
    // This graph is before attributes converted to bfloat16.
    if (edge->src()->output_type(edge->src_output()) == src_dtype &&
        // Only convert the specified tensor.
        edge->src_output() == src_output)
      old_edges.push_back(edge);
  }
  for (const Edge* edge : old_edges) {
    if (cast_node == nullptr) {
      // Build Cast Node. Only one cast node will be built for one tensor.
      NodeBuilder cast_builder(
          absl::StrCat(
              node->name(),
              (dst_dtype == DT_BFLOAT16 ? "/CastToBfloat16_" : "/CastToFloat_"),
              edge->src_output()),
          "Cast");
      cast_builder.Attr("DstT", dst_dtype)
          .Attr("SrcT", src_dtype)
          .Attr("Truncate", false)
          .Input(node, edge->src_output());

      TF_ASSIGN_OR_RETURN(std::string cluster_name, FindTPUClusterName(edge));
      if (!cluster_name.empty())
        cast_builder.Attr("_tpu_replicate", cluster_name);

      // Some rare models have outside compilation tags that need to be
      // propagated so the cast is also outside compiled.
      const AttrValue* attr = node->attrs().Find(kXlaOutsideCompilationAttr);
      if (attr != nullptr && attr->has_s()) {
        cast_builder.Attr(kXlaOutsideCompilationAttr, attr->s());
      }
      TF_RETURN_IF_ERROR(cast_builder.Finalize(g, &cast_node));
      VLOG(4) << "Adding Cast node: " << cast_node->name();
    }
    g->RemoveEdge(edge);
    g->AddEdge(cast_node, 0, edge->dst(), edge->dst_input());
  }
  return absl::OkStatus();
}

void ConvertFunctionCallNodeToBfloat16(Node* node) {
  CHECK(IsStandardFunctionNode(node));
  // Tin.
  std::vector<DataType> dtypes_in;
  // For TPUPartitionedCall, the last input is TPUOrdinal, which is not in Tin.
  // Sometimes DT_RESOURCE could be in Tcaptured, sometimes in Tin.
  // Therefore we directly take length of original list.
  int num_dtypes = node->attrs().Find("Tin")->list().type_size();
  for (int i = 0; i < num_dtypes; ++i) {
    if (node->input_type(i) == DT_FLOAT)
      dtypes_in.push_back(DT_BFLOAT16);
    else
      dtypes_in.push_back(node->input_type(i));
  }
  DataTypeVector dtype_vec_in;
  dtype_vec_in.assign(dtypes_in.begin(), dtypes_in.end());
  node->ClearAttr("Tin");
  node->AddAttr("Tin", dtype_vec_in);
  VLOG(4) << "Converting attribute Tin to DT_BFLOAT16 for " << node->name();

  // Tout.
  std::vector<DataType> dtypes_out;
  for (int i = 0; i < node->num_outputs(); ++i) {
    if (node->output_type(i) == DT_FLOAT)
      dtypes_out.push_back(DT_BFLOAT16);
    else
      dtypes_out.push_back(node->output_type(i));
  }
  DataTypeVector dtype_vec_out;
  dtype_vec_out.assign(dtypes_out.begin(), dtypes_out.end());
  node->ClearAttr("Tout");
  node->AddAttr("Tout", dtype_vec_out);
  VLOG(4) << "Converting attribute Tout to DT_BFLOAT16 for " << node->name();
}

void ConvertNodeToBfloat16(Node* node) {
  if (IsStandardFunctionNode(node)) {
    ConvertFunctionCallNodeToBfloat16(node);
    return;
  }
  if (node->type_string() == "Cast") return;

  // Gather list of attributes that should be updated.
  ::gtl::linked_hash_set<std::string> datatypes;
  ::gtl::linked_hash_map<std::string, DataTypeVector> datatype_vectors;
  for (auto& attr_value : node->attrs()) {
    std::string name = attr_value.first;
    const AttrValue& attr = attr_value.second;
    // Only one data type attached.
    if (attr.has_type() && attr.type() == DT_FLOAT) {
      datatypes.insert(name);
    }
    // A list of data types attached.
    if (attr.has_list() && attr.list().type_size() > 0) {
      auto original_dtypes = attr.list();
      std::vector<DataType> dtypes;
      for (int i = 0; i < original_dtypes.type_size(); ++i) {
        DataType dtype_original = original_dtypes.type(i);
        if (dtype_original == DT_FLOAT)
          dtypes.push_back(DT_BFLOAT16);
        else
          dtypes.push_back(dtype_original);
      }
      DataTypeVector& dtype_vec = datatype_vectors[name];
      dtype_vec.assign(dtypes.begin(), dtypes.end());
    }
  }

  // Update attributes.
  for (auto& datatype : datatypes) {
    node->ClearAttr(datatype.data());
    node->AddAttr(datatype.data(), DT_BFLOAT16);
    VLOG(4) << "Converting attribute " << datatype << " to DT_BFLOAT16 for "
               << node->name();
  }
  for (auto& datatype_vector : datatype_vectors) {
    absl::string_view name = datatype_vector.first;
    node->ClearAttr(name.data());
    node->AddAttr(name.data(), datatype_vector.second);
    VLOG(4) << "Converting attribute (list of "
               << datatype_vector.second.size() << ") " << name
               << " to DT_BFLOAT16 for " << node->name();
  }
}

// copybara:strip_begin
absl::Status BuildMirrorBfloat16Variable(Graph* graph, Node* var_handle_node,
                                         Node* brainserver_init) {
  // 1. Build ReadVariableOp to read Float variable.
  Node* read_for_cast = nullptr;
  TF_RETURN_IF_ERROR(
      NodeBuilder(absl::StrCat(var_handle_node->name(), "/ReadVariableOp_cast"),
                  "ReadVariableOp")
          .Input(var_handle_node, 0)
          .Attr("dtype", DT_FLOAT)
          .Finalize(graph, &read_for_cast));

  // 2. Build Cast node.
  Node* cast_node = nullptr;
  TF_RETURN_IF_ERROR(
      NodeBuilder(absl::StrCat(var_handle_node->name(), "/CastToBfloat16"),
                  "Cast")
          .Attr("DstT", DT_BFLOAT16)
          .Attr("SrcT", DT_FLOAT)
          .Attr("Truncate", false)
          .Input(read_for_cast, 0)
          .Finalize(graph, &cast_node));

  // 3. Build Mirroring VarHandleOp.
  Node* var_handle_bfloat16 = nullptr;
  string var_handle_name(absl::StrCat(var_handle_node->name(), "_bfloat16"));
  auto var_handle_builder =
      NodeBuilder(var_handle_name, "VarHandleOp")
          .Attr("_class", {absl::StrCat("loc@", var_handle_name)})
          .Attr("container", "")
          .Attr("dtype", DT_BFLOAT16)
          .Attr("shape", var_handle_node->attrs().Find("shape")->shape())
          .Attr("shared_name", var_handle_name);
  const AttrValue* xla_sharding_attr =
      var_handle_node->attrs().Find("_XlaSharding");
  if (xla_sharding_attr != nullptr) {
    var_handle_builder.Attr("_XlaSharding", xla_sharding_attr->s());
  }
  const AttrValue* xla_sharding_attr_v2 =
      var_handle_node->attrs().Find("_XlaShardingV2");
  if (xla_sharding_attr_v2 != nullptr) {
    var_handle_builder.Attr("_XlaShardingV2", xla_sharding_attr_v2->s());
  }

  TF_RETURN_IF_ERROR(var_handle_builder.Finalize(graph, &var_handle_bfloat16));

  // 4. Build AssignVariableOp.
  Node* assign_node;
  TF_RETURN_IF_ERROR(
      NodeBuilder(absl::StrCat(var_handle_name, "/AssignVariableOp"),
                  "AssignVariableOp")
          .Input(var_handle_bfloat16)
          .Input(cast_node, 0)
          .Attr("dtype", DT_BFLOAT16)
          .Finalize(graph, &assign_node));

  // 5. Connect AssignVariableOp to brainserver_init.
  graph->AddControlEdge(assign_node, brainserver_init);

  // 6. Replace original var_handle_node's connection to tpu_partitioned_call.
  std::vector<const Edge*> edges_to_tpu_call;
  for (const Edge* edge : GetOutEdges(var_handle_node)) {
    if (edge->dst()->type_string() == "TPUPartitionedCall") {
      edges_to_tpu_call.push_back(edge);
    }
  }
  for (const Edge* edge : edges_to_tpu_call) {
    graph->RemoveEdge(edge);
    graph->AddEdge(var_handle_bfloat16, 0, edge->dst(), edge->dst_input());
  }

  return absl::OkStatus();
}
// copybara:strip_end

absl::Status VarFloatToBfloat16(FunctionInfo* bfloat16_func,
                                const VarHandleCollection& var_handles,
                                const ConverterOptions& options) {
  Graph* root_graph = bfloat16_func->GetRoot()->get_graph();
  // copybara:strip_begin
  Node* brainserver_init = nullptr;
  constexpr char kBrainServerInitOpName[] = "brainserver_init";
  constexpr char kBrainServerInitOpType[] = "NoOp";
  constexpr char kBrainServerInitOpDevice[] = "/job:localhost/device:CPU:0";
  // copybara:strip_end
  for (const auto& it : var_handles) {
    Node* var_handle_node = it.first;
    const VarHandleInfo& info = it.second;
    VLOG(4) << "Processing " << var_handle_node->name();

    // copybara:strip_begin
    if (!options.orbax_bfloat16_ckpt() &&
        info.resource_deserialize_node != nullptr) {
      if (brainserver_init == nullptr) {
        // If the user-graph already has a 'brainserver_init' NoOp node, just
        // append our new init ops to that one.
        for (auto* n : root_graph->nodes()) {
          if (n->name() == kBrainServerInitOpName) {
            if (n->op_def().name() != kBrainServerInitOpType ||
                n->requested_device() != kBrainServerInitOpDevice) {
              return tensorflow::errors::InvalidArgument(absl::StrCat(
                  "The input graph already has a node called \"",
                  kBrainServerInitOpName, "\", but it is not an ",
                  kBrainServerInitOpType,
                  " node (tf.group) or it is not assigned to device ",
                  kBrainServerInitOpDevice,
                  ", so we cannot append the init ops we need to create to it. "
                  "Node definition: \n",
                  n->DebugString()));
            }
            brainserver_init = n;
          }
        }
      }
      if (brainserver_init == nullptr) {
        // Instantiate NoOp for brainserver_init.
        TF_RETURN_IF_ERROR(
            NodeBuilder(kBrainServerInitOpName, kBrainServerInitOpType)
                .Device(kBrainServerInitOpDevice)
                .Finalize(root_graph, &brainserver_init));
      }
      TF_RETURN_IF_ERROR(BuildMirrorBfloat16Variable(
          root_graph, var_handle_node, brainserver_init));
      for (ResourceNodeInfo* consumer : info.consumers) {
        if (consumer->node_info.func->IsDescendantOfInclusive(*bfloat16_func)) {
          // ReadVariableOp / ResourceGather dtype DT_FLOAT -> DT_BFLOAT16.
          ConvertNodeToBfloat16(consumer->node_info.node);
        }
      }
      continue;
    }
    // copybara:strip_end

    // 1. VarHandleOp dtype DT_FLOAT -> DT_BFLOAT16.
    ConvertNodeToBfloat16(var_handle_node);

    // 2. Process AssignVariableOp.
    for (auto it : info.assign_node_to_rval_edge_map) {
      ResourceNodeInfo* assign_resource_node_info = it.first;
      const Edge* rval_edge = it.second;

      // 2.1 AssignVariableOp dtype DT_FLOAT -> DT_BFLOAT16.
      const NodeInfo& assign_node_info = assign_resource_node_info->node_info;
      ConvertNodeToBfloat16(assign_node_info.node);

      // 2.2 Insert Cast node before Assign.
      if (!options.convert_checkpoint_bfloat16() ||
          (rval_edge->src()->type_string() != "RestoreV2" &&
           !absl::StrContains(assign_node_info.func->name(),
                              "traced_restore"))) {
        TF_RETURN_IF_ERROR(CastFloatTensor(assign_node_info.func->get_graph(),
                                           rval_edge->src(),
                                           rval_edge->src_output()));
      }

      TF_RETURN_IF_ERROR(assign_node_info.func->MarkMutated(kOptimFuncSuffix));
    }

    // 3. Process ReadVariableOp / ResourceGather outside bfloat16 scope.

    for (ResourceNodeInfo* consumer : info.consumers) {
      // 3.1 ReadVariableOp / ResourceGather dtype DT_FLOAT -> DT_BFLOAT16.
      const NodeInfo& consumer_node_info = consumer->node_info;
      ConvertNodeToBfloat16(consumer_node_info.node);

      // 3.2 If out of bfloat16 scope, cast all output tensors
      // DT_BFLOAT16 -> DT_FLOAT.

      bool outside_bfloat16_scope = false;
      if (!consumer_node_info.func->IsDescendantOfInclusive(*bfloat16_func)) {
        for (int o = 0; o < consumer_node_info.node->num_outputs(); ++o) {
          TF_RETURN_IF_ERROR(CastFloatTensor(
              consumer_node_info.func->get_graph(), consumer_node_info.node, o,
              /*src_dtype=*/DT_BFLOAT16,
              /*dst_dtype=*/DT_FLOAT));
        }
        outside_bfloat16_scope = true;
      }
      // If an op is not inside the bfloat16 scope, mark the mutated function
      // with a different suffix. This supports the case when the same function
      // is called both inside the bfloat16 scope and outside the bfloat16
      // scope. In this case, there should now be two copies of the function:
      // one that is entirely bfloat16 and one that immediately casts the output
      // of ReadVariableOps to float32. These functions must have two different
      // mutated names or else the Converter will assume they are still the same
      // function and write only one back.
      if (outside_bfloat16_scope) {
        TF_RETURN_IF_ERROR(
            consumer_node_info.func->MarkMutated(kOutsideOptimFuncSuffix));
      } else {
        TF_RETURN_IF_ERROR(
            consumer_node_info.func->MarkMutated(kOptimFuncSuffix));
      }
    }
  }
  return absl::OkStatus();
}

absl::Status ConvertXlaCallModuleOpNodesToBfloat16(Graph* g) {
  for (Node* node : g->nodes()) {
    if (node->type_string() != "XlaCallModule") continue;
    const AttrValue* attr = node->attrs().Find("module");
    if (!attr || !attr->has_s()) {
      continue;
    }
    TF_ASSIGN_OR_RETURN(
        std::string converted_serialized_module,
        mlir::quant::stablehlo::ConvertSerializedStableHloModuleToBfloat16(
            attr->s()));
    node->ClearAttr("module");
    node->AddAttr("module", converted_serialized_module);
    VLOG(4) << "Converting XlaCallModuleOp serialized module to bfloat16 for "
            << node->name();
  }
  return absl::OkStatus();
}

absl::Status ConvertCastNodesToBfloat16(Graph* g) {
  for (Node* node : g->nodes()) {
    if (node->type_string() != "Cast") continue;
    // Src of Cast will be changed to bfloat16 later.
    if (node->attrs().Find("SrcT")->type() == DT_FLOAT) {
      node->ClearAttr("SrcT");
      node->AddAttr("SrcT", DT_BFLOAT16);
      VLOG(4) << "Converting SrcT to DT_BFLOAT16 for " << node->name();
    }
    // Dst of Cast will be changed to bfloat16 later.
    if (node->attrs().Find("DstT")->type() == DT_FLOAT) {
      node->ClearAttr("DstT");
      node->AddAttr("DstT", DT_BFLOAT16);
      VLOG(4) << "Converting DstT to DT_BFLOAT16 for " << node->name();
    }
  }
  return absl::OkStatus();
}

absl::Status CastConstNodesToBfloat16(Graph* g) {
  for (Node* const_node : g->nodes()) {
    if (const_node->type_string() != "Const") continue;
    // Const node has only 1 output.
    TF_RETURN_IF_ERROR(CastFloatTensor(g, const_node, 0));
  }
  return absl::OkStatus();
}

absl::Status CastNonVariableArgToBfloat16(Graph* g) {
  for (Node* arg_node : g->nodes()) {
    if (arg_node->type_string() != "_Arg" ||
        arg_node->attrs().Find("T")->type() == DT_RESOURCE)
      continue;
    for (int o = 0; o < arg_node->num_outputs(); ++o)
      TF_RETURN_IF_ERROR(CastFloatTensor(g, arg_node, o));
  }
  return absl::OkStatus();
}

absl::Status ConvertGraphNodesToBfloat16(
    Graph* g, const std::function<bool(Node*)>& filter_condition) {
  for (Node* node : g->nodes()) {
    if (filter_condition(node)) continue;
    ConvertNodeToBfloat16(node);
  }
  return absl::OkStatus();
}

absl::Status CastSrcOfRetvalToBfloat16(
    Graph* g, const ::gtl::linked_hash_set<int>& bfloat16_arg_ret) {
  for (Node* ret_node : g->nodes()) {
    if (ret_node->type_string() != "_Retval") continue;
    if (bfloat16_arg_ret.contains(ret_node->id())) {
      VLOG(4) << "Skipping DT_FLOAT cast for intended "
              << "DT_BFLOAT16 Ret " << ret_node->name();
      continue;
    }
    std::vector<const Edge*> edges_to_cast;
    for (const Edge* edge : GetInEdges(ret_node)) {
      if (edge->IsControlEdge()) continue;
      edges_to_cast.push_back(edge);
    }
    for (const Edge* edge : edges_to_cast) {
      TF_RETURN_IF_ERROR(CastFloatTensor(g, edge->src(), edge->src_output(),
                                         /*src_dtype=*/DT_BFLOAT16,
                                         /*dst_dtype=*/DT_FLOAT));
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<Node*> CastFloatEdge(Graph* graph, const Edge* edge,
                                    DataType src_dtype,
                                    DataType desired_dst_dtype) {
  std::string new_dst_dtype_name =
      desired_dst_dtype == DT_FLOAT ? "Float" : "Bfloat16";

  static int i = 0;
  NodeBuilder cast_builder(absl::StrCat(edge->src()->name(), "/CastTo",
                                        new_dst_dtype_name, "_", i++),
                           "Cast");
  cast_builder.Attr("DstT", desired_dst_dtype)
      .Attr("SrcT", src_dtype)
      .Attr("Truncate", false)
      .Input(edge->src(), edge->src_output());

  TF_ASSIGN_OR_RETURN(std::string cluster_name, FindTPUClusterName(edge));
  if (!cluster_name.empty()) cast_builder.Attr("_tpu_replicate", cluster_name);

  // Some rare models have outside compilation tags that need to be propagated
  // so the cast is also outside compiled.
  const AttrValue* attr = edge->src()->attrs().Find(kXlaOutsideCompilationAttr);
  if (attr != nullptr && attr->has_s()) {
    cast_builder.Attr(kXlaOutsideCompilationAttr, attr->s());
  }

  Node* cast_node = nullptr;
  TF_RETURN_IF_ERROR(cast_builder.Finalize(graph, &cast_node));
  VLOG(4) << "Adding Cast node: " << cast_node->name();

  graph->RemoveEdge(edge);
  graph->AddEdge(cast_node, 0, edge->dst(), edge->dst_input());

  return cast_node;
}

absl::Status WrapNodeFloatValue(
    Graph* graph, Node* node,
    ::gtl::linked_hash_map<std::pair<Node*, int>, Node*>*
        output_tensor_to_cast_node,
    ::gtl::linked_hash_map<std::pair<Node*, int>, Node*>*
        input_tensor_to_cast_node,
    DataType node_dtype, DataType graph_dtype) {
  std::vector<const Edge*> old_in_edges;
  std::vector<const Edge*> old_out_edges;
  for (const Edge* edge : GetOutEdges(node)) {
    if (edge->IsControlEdge()) continue;
    // node_dtype (node) -> graph_dtype
    if (edge->dst()->input_type(edge->dst_input()) != graph_dtype ||
        edge->src()->output_type(edge->src_output()) != node_dtype)
      continue;
    old_out_edges.push_back(edge);
  }
  for (const Edge* edge : GetInEdges(node)) {
    if (edge->IsControlEdge()) continue;
    // graph_dtype -> node_dtype (node).
    if (edge->src()->output_type(edge->src_output()) != graph_dtype ||
        edge->dst()->input_type(edge->dst_input()) != node_dtype)
      continue;
    old_in_edges.push_back(edge);
  }
  for (const Edge* edge : old_out_edges) {
    Node* cast_node = nullptr;
    if (!output_tensor_to_cast_node->contains(
            std::make_pair(edge->src(), edge->src_output()))) {
      // Build Cast Node.
      TF_ASSIGN_OR_RETURN(cast_node,
                          CastFloatEdge(graph, edge, node_dtype, graph_dtype));
      // Cache the Cast Node so it can be reused later.
      (*output_tensor_to_cast_node)[std::make_pair(
          edge->src(), edge->src_output())] = cast_node;

    } else {
      // Reuse existing Cast Node.
      cast_node = output_tensor_to_cast_node->at(
          std::make_pair(edge->src(), edge->src_output()));
      graph->RemoveEdge(edge);
      graph->AddEdge(cast_node, 0, edge->dst(), edge->dst_input());
    }
  }
  for (const Edge* edge : old_in_edges) {
    Node* cast_node = nullptr;
    if (!input_tensor_to_cast_node->contains(
            std::make_pair(edge->src(), edge->dst_input()))) {
      // Build Cast Node.
      TF_ASSIGN_OR_RETURN(cast_node,
                          CastFloatEdge(graph, edge, graph_dtype, node_dtype));
      // Cache the Cast Node so it can be reused later.
      (*input_tensor_to_cast_node)[std::make_pair(
          edge->src(), edge->dst_input())] = cast_node;

    } else {
      // Reuse existing Cast Node.
      cast_node = input_tensor_to_cast_node->at(
          std::make_pair(edge->src(), edge->dst_input()));
      graph->RemoveEdge(edge);
      graph->AddEdge(cast_node, 0, edge->dst(), edge->dst_input());
    }
  }
  return absl::OkStatus();
}

absl::Status PartiallyWrapNodeFloatValue(
    Graph* graph, Node* node,
    ::gtl::linked_hash_map<std::pair<Node*, int>, Node*>*
        output_tensor_to_cast_node,
    ::gtl::linked_hash_map<std::pair<Node*, int>, Node*>*
        input_tensor_to_cast_node) {
  // Convert to bfloat16 whenever possible but then make the edges match
  ::gtl::linked_hash_map<const Edge*, std::pair<DataType, DataType>>
      old_in_edges;
  ::gtl::linked_hash_map<const Edge*, std::pair<DataType, DataType>>
      old_out_edges;

  // For this node's out edges, if its out datatype does not match the
  // destinations's datatype, record it so it can be updated.
  for (const Edge* edge : GetOutEdges(node)) {
    if (edge->IsControlEdge()) continue;
    DataType edge_src_dtype = edge->src()->output_type(edge->src_output());
    DataType edge_dst_dtype = edge->dst()->input_type(edge->dst_input());
    if (edge_src_dtype == edge_dst_dtype) continue;
    old_out_edges.emplace(
        std::piecewise_construct, std::forward_as_tuple(edge),
        std::forward_as_tuple(edge_src_dtype, edge_dst_dtype));
  }
  // For this node's in edges, if its in datatype does not match the
  // source's datatype, record it so it can be updated.
  for (const Edge* edge : GetInEdges(node)) {
    if (edge->IsControlEdge()) continue;
    DataType edge_src_dtype = edge->src()->output_type(edge->src_output());
    DataType edge_dst_dtype = edge->dst()->input_type(edge->dst_input());
    if (edge_src_dtype == edge_dst_dtype) continue;
    old_in_edges.emplace(std::piecewise_construct, std::forward_as_tuple(edge),
                         std::forward_as_tuple(edge_src_dtype, edge_dst_dtype));
  }

  // For every out edge that needs a cast, reuse an existing cast or create
  // a new one.
  for (const auto& [edge, datatypes] : old_out_edges) {
    auto [src_dtype, desired_dst_dtype] = datatypes;
    Node* cast_node = nullptr;
    auto key = std::make_pair(edge->src(), edge->src_output());
    // Check if an existing Cast Node can be reused.
    if (output_tensor_to_cast_node->contains(key)) {
      DataType cached_cast_dst_dtype =
          output_tensor_to_cast_node->at(key)->attrs().Find("DstT")->type();
      if (cached_cast_dst_dtype == desired_dst_dtype) {
        cast_node = output_tensor_to_cast_node->at(key);
        graph->RemoveEdge(edge);
        graph->AddEdge(cast_node, 0, edge->dst(), edge->dst_input());
        continue;
      }
    }
    // Build Cast Node.
    TF_ASSIGN_OR_RETURN(
        cast_node, CastFloatEdge(graph, edge, src_dtype, desired_dst_dtype));
    // Cache the Cast Node so it can be reused later.
    (*output_tensor_to_cast_node)[key] = cast_node;
  }
  // For every in edge that needs a cast, reuse an existing cast or create
  // a new one.
  for (const auto& [edge, datatypes] : old_in_edges) {
    DataType src_dtype = datatypes.first;
    DataType desired_dst_dtype = datatypes.second;
    Node* cast_node = nullptr;
    auto key = std::make_pair(edge->src(), edge->dst_input());
    // Check if an existing Cast Node can be reused.
    auto it = input_tensor_to_cast_node->find(key);
    if (it != input_tensor_to_cast_node->end()) {
      DataType cached_cast_dst_dtype = it->second->attrs().Find("DstT")->type();
      if (cached_cast_dst_dtype == desired_dst_dtype) {
        // Reuse existing Cast Node.
        cast_node = it->second;
        graph->RemoveEdge(edge);
        graph->AddEdge(cast_node, 0, edge->dst(), edge->dst_input());
        continue;
      }
    }
    // Build Cast Node.
    TF_ASSIGN_OR_RETURN(
        cast_node, CastFloatEdge(graph, edge, src_dtype, desired_dst_dtype));
    // Cache the Cast Node so it can be reused later.
    (*input_tensor_to_cast_node)[key] = cast_node;
  }

  return absl::OkStatus();
}

absl::Status WrapBfloat16IncompatibleNodes(
    Graph* g, const ::gtl::linked_hash_set<string>& filterlist) {
  ::gtl::linked_hash_map<std::pair<Node*, int>, Node*>
      output_tensor_to_cast_node;
  ::gtl::linked_hash_map<std::pair<Node*, int>, Node*>
      input_tensor_to_cast_node;
  for (Node* node : g->nodes()) {
    // Check if there are any nodes that should be partially wrapped.
    // For now, only FastSparseTensorDenseMatMul is supported by partial
    // wrapping.
    if (node->type_string() == "FastSparseTensorDenseMatMul" &&
        node->attrs().Find("T")->type() == DT_FLOAT) {
      VLOG(3) << "Partially wrapping Bfloat16-incompatible node"
              << node->name();
      node->ClearAttr("T");
      node->AddAttr("T", DT_BFLOAT16);
      TF_RETURN_IF_ERROR(PartiallyWrapNodeFloatValue(
          g, node, &output_tensor_to_cast_node, &input_tensor_to_cast_node));
      continue;
    }
    if (filterlist.contains(node->type_string())) {
      VLOG(3) << "Wrapping BFloat16-incompatible node " << node->name();
      TF_RETURN_IF_ERROR(WrapNodeFloatValue(
          g, node, &output_tensor_to_cast_node, &input_tensor_to_cast_node,
          /* node_dtype = */ DT_FLOAT,
          /* graph_dtype = */ DT_BFLOAT16));
    }
  }
  return absl::OkStatus();
}

void ConvertWhileNodeToBfloat16(Node* node,
                                ::gtl::linked_hash_set<int>* flow_input_index) {
  std::vector<DataType> dtypes;
  for (int i = 0; i < node->num_inputs(); ++i) {
    if (node->input_type(i) == DT_FLOAT) {
      Node* input_node = nullptr;
      TF_CHECK_OK(node->input_node(i, &input_node));
      // TensorArray* op's flow output has unchangeable dtype float.
      // TensorArray* op's handle output doesn't need taken care of
      // because they are of dtype resource.
      if (absl::StrContains(input_node->type_string(), "TensorArray")) {
        bool is_flow_input = false;
        for (const Edge* edge : GetOutEdges(input_node))
          if (edge->dst() == node && edge->dst_input() == i &&
              absl::StrContains(
                  input_node->op_def().output_arg(edge->src_output()).name(),
                  "flow")) {
            dtypes.push_back(node->input_type(i));
            flow_input_index->insert(i);
            is_flow_input = true;
            break;
          }
        if (!is_flow_input) dtypes.push_back(DT_BFLOAT16);
      } else {
        dtypes.push_back(DT_BFLOAT16);
      }
    } else {
      dtypes.push_back(node->input_type(i));
    }
  }
  DataTypeVector dtype_vec;
  dtype_vec.assign(dtypes.begin(), dtypes.end());
  node->ClearAttr("T");
  node->AddAttr("T", dtype_vec);

  VLOG(4) << "Converting While attribute T to DT_BFLOAT16 for " << node->name();
}

void ConvertNonFlowArgsAndRetvalsToBfloat16(
    Graph* g, const ::gtl::linked_hash_set<int>& flow_input_index) {
  // While body has exactly same number / dtype of input / output.
  for (Node* node : g->nodes()) {
    if (node->type_string() == "_Arg") {
      if (!flow_input_index.contains(node->attrs().Find("index")->i())) {
        // Flow input should not be converted.
        ConvertNodeToBfloat16(node);
      } else {  // Convert Identity node back to float.
        for (const Edge* edge : GetOutEdges(node)) {
          if (edge->dst()->type_string() == "Identity" &&
              edge->dst()->attrs().Find("T")->type() == DT_BFLOAT16) {
            edge->dst()->ClearAttr("T");
            edge->dst()->AddAttr("T", DT_FLOAT);
          }
        }
      }
    }
    if (node->type_string() == "_Retval" &&
        !flow_input_index.contains(node->attrs().Find("index")->i())) {
      // Flow output should not be converted.
      ConvertNodeToBfloat16(node);
    }
  }
}

absl::Status WhileOpFloatToBfloat16(Node* while_node,
                                    const std::vector<FunctionInfo*>& funcs) {
  ::gtl::linked_hash_set<int> flow_input_index;
  ConvertWhileNodeToBfloat16(while_node, &flow_input_index);
  // Convert _Arg and _Retval in "body" and "cond".
  for (FunctionInfo* body_or_cond : funcs) {
    // _Arg and _Retval nodes and corresponding identities.
    ConvertNonFlowArgsAndRetvalsToBfloat16(body_or_cond->get_graph(),
                                           flow_input_index);
    TF_RETURN_IF_ERROR(body_or_cond->MarkMutated(kOptimFuncSuffix));
  }
  return absl::OkStatus();
}

void ConvertIfNodeToBfloat16(Node* node) {
  // Tin.
  std::vector<DataType> dtypes_in;
  // Input 0 is Tcond. Tin starts from 1.
  for (int i = 1; i < node->num_inputs(); ++i) {
    if (node->input_type(i) == DT_FLOAT)
      dtypes_in.push_back(DT_BFLOAT16);
    else
      dtypes_in.push_back(node->input_type(i));
  }
  DataTypeVector dtype_vec_in;
  dtype_vec_in.assign(dtypes_in.begin(), dtypes_in.end());
  node->ClearAttr("Tin");
  node->AddAttr("Tin", dtype_vec_in);
  VLOG(4) << "Converting attribute Tin to DT_BFLOAT16 for " << node->name();

  // Tout.
  std::vector<DataType> dtypes_out;
  for (int i = 0; i < node->num_outputs(); ++i) {
    if (node->output_type(i) == DT_FLOAT)
      dtypes_out.push_back(DT_BFLOAT16);
    else
      dtypes_out.push_back(node->output_type(i));
  }
  DataTypeVector dtype_vec_out;
  dtype_vec_out.assign(dtypes_out.begin(), dtypes_out.end());
  node->ClearAttr("Tout");
  node->AddAttr("Tout", dtype_vec_out);
  VLOG(4) << "Converting attribute Tout to DT_BFLOAT16 for " << node->name();
}

absl::Status IfOpFloatToBfloat16(Node* if_node,
                                 const std::vector<FunctionInfo*>& funcs) {
  ConvertIfNodeToBfloat16(if_node);

  // Convert _Arg and _Retval in "then" and "else".
  for (FunctionInfo* then_or_else : funcs) {
    // _Arg and _Retval nodes.
    TF_RETURN_IF_ERROR(
        ConvertGraphNodesToBfloat16(then_or_else->get_graph(), [](Node* node) {
          return (node->type_string() != "_Arg" &&
                  node->type_string() != "_Retval");
        }));
    TF_RETURN_IF_ERROR(then_or_else->MarkMutated(kOptimFuncSuffix));
  }
  return absl::OkStatus();
}

absl::Status ControlFlowsToBFloat16(FunctionInfo* func) {
  ::gtl::linked_hash_map<Node*, std::vector<FunctionInfo*>> node_to_funcs;
  std::vector<FunctionInfo*> children_ptrs;
  TF_RETURN_IF_ERROR(func->get_children_ptrs(&children_ptrs));

  // Collect Control flow functions by call node.
  for (FunctionInfo* child : children_ptrs) {
    Node* node = child->node_in_parent_graph();
    if (node->type_string() == "While" || node->type_string() == "If" ||
        node->type_string() == "StatelessWhile" ||
        node->type_string() == "StatelessIf") {
      if (!node_to_funcs.contains(node)) {
        node_to_funcs[node] = std::vector<FunctionInfo*>();
      }
      VLOG(4) << "ControlFlow node: " << node->name()
              << ", func: " << child->name();
      node_to_funcs.at(node).push_back(child);
    }
  }

  for (const auto& it : node_to_funcs) {
    Node* node = it.first;
    const std::vector<FunctionInfo*>& funcs = it.second;
    if (node->type_string() == "While" ||
        node->type_string() == "StatelessWhile") {
      TF_RETURN_IF_ERROR(WhileOpFloatToBfloat16(node, funcs));
    } else if (node->type_string() == "If" ||
        node->type_string() == "StatelessIf") {
      TF_RETURN_IF_ERROR(IfOpFloatToBfloat16(node, funcs));
    }
  }
  return absl::OkStatus();
}

// Convert input to bfloat16, and output to float for function call nodes.
// This should be applied after call node attributes are converted to bfloat16.
absl::Status WrapFunctionCallNodeForBFloat16(FunctionInfo* bfloat16_func) {
  ::gtl::linked_hash_map<std::pair<Node*, int>, Node*>
      output_tensor_to_cast_node;
  ::gtl::linked_hash_map<std::pair<Node*, int>, Node*>
      input_tensor_to_cast_node;
  return WrapNodeFloatValue(bfloat16_func->parent()->get_graph(),
                            bfloat16_func->node_in_parent_graph(),
                            &output_tensor_to_cast_node,
                            &input_tensor_to_cast_node,
                            /* node_dtype = */ DT_BFLOAT16,
                            /* graph_dtype = */ DT_FLOAT);
}

// This does not apply recursively. It is for fixing the _Arg / _Retval in the
// top-level graph of a function whose computations and children functions
// are converted to bfloat16.
void FuncGraphArgsAndRetvalsToFloat(
    Graph* graph, const ::gtl::linked_hash_set<int>& bfloat16_arg_ret) {
  for (Node* node : graph->nodes()) {
    if ((node->type_string() == "_Arg" || node->type_string() == "_Retval") &&
        node->attrs().Find("T")->type() == DT_BFLOAT16) {
      if (bfloat16_arg_ret.contains(node->id())) {
        VLOG(4) << "Skipping DT_FLOAT conversion for intended "
                << "DT_BFLOAT16 Arg/Ret " << node->name();
        continue;
      }
      node->ClearAttr("T");
      node->AddAttr("T", DT_FLOAT);
      VLOG(4) << "Converting attribute T back to DT_FLOAT for " << node->name();
    }
  }
}

// Removes duplicate cast nodes with the same name from the graph. Duplicate
// cast nodes are unnecessary and can cause conflicts when both cast nodes share
// the same name.
absl::Status RemoveDuplicateCastNodes(Graph* graph) {
  std::vector<const Edge*> duplicate_cast_connecting_edges;
  for (Node* first_cast_node : graph->nodes()) {
    // Find two connected cast nodes.
    if (first_cast_node->type_string() != "Cast" ||
        first_cast_node->out_edges().size() > 1)
      continue;
    for (const Edge* edge : GetOutEdges(first_cast_node)) {
      const Node* second_cast_node = edge->dst();
      if (second_cast_node->type_string() != "Cast") continue;
      // TODO(b/241160425): Make removing all chained casts the default.
      if (!absl::GetFlag(FLAGS_experimental_remove_all_chained_casts_for_obm) &&
          first_cast_node->name() != second_cast_node->name())
        continue;
      duplicate_cast_connecting_edges.push_back(edge);
    }
  }

  std::set<Node*> remove_nodes;
  for (const Edge* edge : duplicate_cast_connecting_edges) {
    Node* first_cast_node = edge->src();
    Node* second_cast_node = edge->dst();
    DataType src_type = first_cast_node->attrs().Find("SrcT")->type();
    DataType dst_type = second_cast_node->attrs().Find("DstT")->type();

    std::vector<const Edge*> out_edges;
    for (const Edge* out_edge : GetOutEdges(second_cast_node)) {
      out_edges.push_back(out_edge);
    }

    if (src_type == dst_type) {
      // Cast ops only have one input cast. Connect it to all the out nodes.
      const Edge* in_edge = *(GetInEdges(first_cast_node).begin());
      for (const Edge* out_edge : out_edges) {
        graph->AddEdge(in_edge->src(), in_edge->src_output(), out_edge->dst(),
                       out_edge->dst_input());
        graph->RemoveEdge(out_edge);
      }
      graph->RemoveEdge(in_edge);

      // Remove both unnecessary casts.
      remove_nodes.insert(first_cast_node);
      remove_nodes.insert(second_cast_node);

    } else {
      // Update the first cast node to have the same DstT as the duplicate
      // cast node.
      first_cast_node->ClearAttr("DstT");
      first_cast_node->AddAttr("DstT", dst_type);

      // Remove edges from the duplicate cast node and add them to the
      // fist cast node.
      for (const Edge* edge : out_edges) {
        graph->AddEdge(first_cast_node, 0, edge->dst(), edge->dst_input());
        graph->RemoveEdge(edge);
      }

      // Remove duplicate cast node.
      remove_nodes.insert(second_cast_node);
    }
  }

  // Defer node removal to handle chains of casts greater than length two.
  for (Node* node : remove_nodes) graph->RemoveNode(node);

  return absl::OkStatus();
}

static absl::Status FixRestoreAssignDType(const Edge& restore_out_edge) {
  if (restore_out_edge.dst()->type_string() == "AssignVariableOp")
    return absl::OkStatus();
  if (restore_out_edge.dst()->type_string() != "Identity") {
    return errors::Unimplemented(
        "Unable to convert checkpoint to bfloat16 because updating the restore "
        "assign ops failed.");
  }

  // Sometimes the RestoreV2 outputs to an identity op. If this happens,
  // the identity op needs to be updated to bfloat16.
  Node* identity = restore_out_edge.dst();
  ConvertNodeToBfloat16(identity);

  // Sometimes a cast is inserted between the identity op and the
  // AssignVariable op during VarFloatToBfloat16. Update the cast to reflect
  // the new type.
  for (const Edge* identity_out_edge : identity->out_edges()) {
    if (identity_out_edge->IsControlEdge()) continue;
    if (identity_out_edge->dst()->type_string() != "Cast") continue;
    Node* cast = identity_out_edge->dst();
    cast->ClearAttr("SrcT");
    cast->AddAttr("SrcT", identity->attrs().Find("T")->type());
  }
  return absl::OkStatus();
}

static absl::Status FixRestoreDTypes(
    const ::gtl::linked_hash_map<string, Tensor>& checkpoint_tensors,
    Node* restore_node) {
  // Record the dtypes for each tensor in the checkpoint.
  ::gtl::linked_hash_map<int, DataType> checkpoint_types;
  Node* tensor_names_node = nullptr;
  TF_RETURN_IF_ERROR(restore_node->input_node(1, &tensor_names_node));
  const TensorProto& tensor =
      tensor_names_node->attrs().Find("value")->tensor();
  const auto& tensor_names = tensor.string_val();
  for (int index = 0; index < tensor_names.size(); index++) {
    const string& tensor_name = tensor_names[index];
    // 2.x saved models have a second RestoreV2 op that contains only one
    // tensor called "_CHECKPOINTABLE_OBJECT_GRAPH". This RestoreV2 op does not
    // need to be rewritten.
    if (tensor_name == "_CHECKPOINTABLE_OBJECT_GRAPH") return absl::OkStatus();
    checkpoint_types[index] = checkpoint_tensors.at(tensor_name).dtype();
  }

  // Cast any variable output from the RestoreV2 whose type does not match its
  // checkpoint type.
  std::vector<DataType> output_dtypes(restore_node->num_outputs());
  for (const Edge* edge : GetOutEdges(restore_node)) {
    // Skip over any control edges.
    if (edge->dst_input() < 0) continue;

    const DataType checkpoint_dtype = checkpoint_types.at(edge->src_output());
    const DataType dst_dtype = edge->dst()->input_type(edge->dst_input());
    if (checkpoint_dtype != dst_dtype)
      TF_RETURN_IF_ERROR(FixRestoreAssignDType(*edge));

    output_dtypes[edge->src_output()] =
        edge->dst()->input_type(edge->dst_input());
  }

  // Update the RestoreV2 dtypes.
  DataTypeVector dtype_vec;
  dtype_vec.assign(output_dtypes.begin(), output_dtypes.end());
  restore_node->ClearAttr("dtypes");
  restore_node->AddAttr("dtypes", dtype_vec);

  return absl::OkStatus();
}

static absl::Status FixSaveDTypes(
    const ::gtl::linked_hash_map<string, Tensor>& checkpoint_tensors,
    Graph* graph, Node* save_node) {
  // Record the dtypes for each tensor in the checkpoint.
  ::gtl::linked_hash_map<int, DataType> checkpoint_types;
  Node* tensor_names_node = nullptr;
  TF_RETURN_IF_ERROR(save_node->input_node(1, &tensor_names_node));
  const TensorProto& tensor =
      tensor_names_node->attrs().Find("value")->tensor();
  const auto& tensor_names = tensor.string_val();
  for (int index = 0; index < tensor_names.size(); index++) {
    const string& tensor_name = tensor_names[index];
    // 2.x saved models have a second SaveV2 op that contains only one
    // tensor called "_CHECKPOINTABLE_OBJECT_GRAPH". This SaveV2 op does not
    // need to be rewritten.
    if (tensor_name == "_CHECKPOINTABLE_OBJECT_GRAPH") return absl::OkStatus();
    checkpoint_types[index] = checkpoint_tensors.at(tensor_name).dtype();
  }

  // Cast any variable input to the SaveV2 whose type does not match its
  // checkpoint type.
  std::vector<DataType> input_dtypes(save_node->num_inputs() - 3);
  for (const Edge* edge : GetInEdges(save_node)) {
    // Skip over the non-variable inputs.
    if (absl::StrContains(edge->src()->name(), "ShardedFilename") ||
        absl::StrContains(edge->src()->name(), "tensor_names") ||
        absl::StrContains(edge->src()->name(), "shape_and_slices")) {
      continue;
    }
    // Skip over any control edges.
    if (edge->src_output() < 0) continue;

    // Calculate the adjusted index for this tensor. The first three inputs
    // to SaveV2 are ShardedFilename, tensor_names and shape_and_slices so
    // subtract three from input edge number to get the index of the tensor.
    const int index = edge->dst_input() - 3;
    const DataType src_dtype = edge->src()->output_type(edge->src_output());
    const DataType checkpoint_dtype = checkpoint_types.at(index);

    if (checkpoint_dtype == DT_BFLOAT16 && src_dtype != checkpoint_dtype)
      TF_RETURN_IF_ERROR(CastFloatTensor(graph, edge->src(), edge->src_output(),
                                         src_dtype, checkpoint_dtype));
    input_dtypes[index] = checkpoint_dtype;
  }

  // Update the SaveV2 dtypes.
  DataTypeVector dtype_vec;
  dtype_vec.assign(input_dtypes.begin(), input_dtypes.end());
  save_node->ClearAttr("dtypes");
  save_node->AddAttr("dtypes", dtype_vec);

  return absl::OkStatus();
}

absl::Status FixRestoreAndSaveDtypes(FunctionInfo* func,
                                     const ConverterOptions& options) {
  // Read the newly updated checkpoint tensors. These tensors will be used
  // to determine which ops surrounding RestoreV2 and SaveV2 need to be updated.
  ::gtl::linked_hash_map<string, Tensor> checkpoint_tensors;
  TF_RETURN_IF_ERROR(ReadVariablesAsTensors(
      options.output_variables_filename_prefix(), &checkpoint_tensors));

  // If any RestoreV2 or SaveV2 ops are in this graph, update them.
  std::vector<Node*> restore_and_save_nodes;
  for (Node* node : func->get_graph()->nodes()) {
    restore_and_save_nodes.push_back(node);
  }
  for (Node* node : restore_and_save_nodes) {
    if (node->type_string() == "RestoreV2")
      TF_RETURN_IF_ERROR(FixRestoreDTypes(checkpoint_tensors, node));
    if (node->type_string() == "SaveV2") {
      TF_RETURN_IF_ERROR(
          FixSaveDTypes(checkpoint_tensors, func->get_graph(), node));
    }
  }

  // Recursively search for more RestoreV2 and SaveV2 ops.
  std::vector<FunctionInfo*> children_ptrs;
  TF_RETURN_IF_ERROR(func->get_children_ptrs(&children_ptrs));
  for (FunctionInfo* child : children_ptrs) {
    if (absl::StrContains(child->name(), "traced_restore") ||
        absl::StrContains(child->name(), "traced_save")) {
      TF_RETURN_IF_ERROR(FixRestoreAndSaveDtypes(child, options));
    }
  }

  TF_RETURN_IF_ERROR(func->MarkMutated(kOptimFuncSuffix));
  return absl::OkStatus();
}

// Convert checkpoint values to BFloat16.
absl::Status CheckpointToBFloat16(const ConverterOptions& options,
                                  FunctionInfo* bfloat16_func) {
  // Collect the variables that are used in the bfloat16_func.
  FuncArgsList func_args_list;
  TF_RETURN_IF_ERROR(
      GetFuncArgsList(bfloat16_func->GetRoot(), &func_args_list));
  ::gtl::linked_hash_set<std::unique_ptr<ResourceNodeInfo>> resource_vars;
  VarHandleCollection var_handles;
  AnalyseResourceVars(bfloat16_func->GetRoot(), func_args_list, &resource_vars);
  BuildVarHandleCollection(
      resource_vars,
      [](Node* node) {
        return (node->attrs().Find("dtype")->type() == DT_BFLOAT16);
      },
      &var_handles);
  FilterVarHandleCollectionByFuncScope(*bfloat16_func, &var_handles);

  // Create a mapping from tensor names in the checkpoint to their associated
  // variables names in the graph. This mapping is needed when updating the
  // checkpoint.
  //
  // For 2.x saved models, the names in the checkpoint can be unrelated to the
  // variable names which makes creating this mapping more difficult. The best
  // way to create this mapping is to utilize the ordering of the variables from
  // the RestoreV2 op. The RestoreV2 op outputs to variables. The ordering of
  // the variables directly maps to the ordering of the variables in
  // the checkpoint. So for example, if the RestoreV2 op's third output edge is
  // to my_variable, my_variable is also the third tensor in the checkpoint.
  ::gtl::linked_hash_map<string, Node*> checkpoint_variables;
  for (const auto& it : var_handles) {
    Node* variable = it.first;
    if (variable->attrs().Find("dtype")->type() != DT_BFLOAT16) continue;

    const VarHandleInfo& var_handle_info = it.second;
    const Edge* restore_edge = nullptr;
    for (const auto& ite : var_handle_info.assign_node_to_rval_edge_map) {
      std::queue<const Edge*> expansion_entries;
      expansion_entries.push(ite.second);

      while (!expansion_entries.empty()) {
        const Edge* edge = expansion_entries.front();
        expansion_entries.pop();
        if (edge->IsControlEdge()) continue;

        if (edge->src()->type_string() == "RestoreV2") {
          restore_edge = edge;
          break;
        }
        for (const Edge* in_edge : GetInEdges(edge->src()))
          expansion_entries.push(in_edge);
      }
      if (restore_edge == nullptr) continue;

      Node* restore = restore_edge->src();
      int index = restore_edge->src_output();

      Node* tensor_names_node = nullptr;
      TF_RETURN_IF_ERROR(restore->input_node(1, &tensor_names_node));

      const TensorProto& tensor =
          tensor_names_node->attrs().Find("value")->tensor();
      const auto& tensor_names = tensor.string_val();
      checkpoint_variables[tensor_names[index]] = variable;
    }
    if (restore_edge == nullptr)
      LOG(WARNING) << "Unable update the stored checkpoint value for "
                   << variable->name() << " to bfloat16 because no "
                   << "associated RestoreV2 op could be found. This can "
                   << "happen if the variable's initial value is not stored "
                   << "in the checkpoint.";
  }

  // Read Tensor values from saved params.
  ::gtl::linked_hash_map<string, Tensor> var_tensors;
  TF_RETURN_IF_ERROR(ReadVariablesAsTensors(
      options.input_variables_filename_prefix(), &var_tensors));

  // Cast tensor values to bfloat16.
  std::vector<string> var_names;
  for (const auto& it : checkpoint_variables) {
    if (var_tensors.contains(it.first)) {
      var_names.push_back(it.first);
    }
  }
  for (const string& var_name : var_names) {
    // Skip if the variable is already in bfloat16 (e.g. when using mixed
    // precision).
    if (var_tensors[var_name].dtype() == DT_BFLOAT16) continue;
    Tensor bfloat16_tensor(DT_BFLOAT16, var_tensors[var_name].shape());
    for (int i = 0; i < var_tensors[var_name].NumElements(); ++i) {
      bfloat16_tensor.flat<tensorflow::bfloat16>()(i) =
          static_cast<tensorflow::bfloat16>(
              var_tensors[var_name].flat<float>()(i));
    }
    var_tensors.erase(var_name);
    var_tensors[var_name] = bfloat16_tensor;
    VLOG(3) << "Rewrite " << var_name << " in checkpoint to "
            << DataTypeString(var_tensors[var_name].dtype());
  }

  // Write back tensors to saved model variables.
  TF_RETURN_IF_ERROR(WriteTensorsToVariables(
      options.output_variables_filename_prefix(), var_tensors));

  return absl::OkStatus();
}

absl::Status GenerateFilterlists(
    FunctionInfo* bfloat16_func, const ConverterOptions& options,
    bool auto_generate_filterlist,
    ::gtl::linked_hash_set<string>& tpu_filterlist,
    ::gtl::linked_hash_set<string>& cpu_filterlist) {
  ::gtl::linked_hash_set<string> general_filterlist({
      // clang-format off
    "Assert",  // Unexpected behaviour if try to modify Assert.
    "Bucketize",
    "Const",
    "Dequantize",  // TODO(lzr): Support Dequantize to Bfloat16.
    "FakeQuantWithMinMaxVars",
    "FusedBatchNorm",
    "FusedBatchNormV3",
    "MatrixSolve",
    "NonMaxSuppressionV5",
    // Parse example related ops: "Tdense" does not support bfloat16.
    "ParseExample",
    "ParseExampleDataset",
    "ParseExampleV2",
    "ParseSequenceExample",
    "ParseSingleExample",
    "ParseSingleSequenceExample",
    "Qr",
    "RFFT",
    "Range",
    "ReadVariableOp",
    "ResizeArea",
    "ResizeBilinear",
    // Resource-related ops are handled together with VarHandleOp.
    "ResourceGather",
    "ResourceGatherNd",
    "SparseMatMul",
    "SparseSegmentSqrtN",
    "StatelessWhile",  // StatelessWhile is handled with some special logic.
    "TopKWithUnique",
    // The following Xla ops have functions in their attributes. Rewriting the
    // dtypes will cause mismatch between the op and the function.
    "XlaIf",
    "XlaReduce",
    "XlaReduceWindow",
    "XlaScatter",
    "XlaSelectAndScatter",
    "XlaVariadicReduce",
    "XlaVariadicReduceV2",
    "XlaVariadicSort",
    "XlaWhile",
    "While",  // While is handled with some special logic.
      // clang-format on
  });

  general_filterlist.insert(options.bfloat16_filterlist().begin(),
                            options.bfloat16_filterlist().end());

  tpu_filterlist.insert(general_filterlist.begin(), general_filterlist.end());
  cpu_filterlist.insert(general_filterlist.begin(), general_filterlist.end());

  if (options.bfloat16_scope() == ::orbax::BFloat16Scope::ALL &&
      !options.convert_checkpoint_bfloat16()) {
    // RestoreV2 is in bfloat16 scope under ALL mode.
    cpu_filterlist.insert("RestoreV2");
    // SaveV2 is in bfloat16 scope under ALL mode.
    cpu_filterlist.insert("SaveV2");
  }

  if (!auto_generate_filterlist) return absl::OkStatus();

  // Auto-generate filter list by checking kernel definitions.
  absl::string_view tpu_xla_jit_device = DEVICE_TPU_XLA_JIT;
  absl::string_view cpu_xla_jit_device = DEVICE_CPU_XLA_JIT;
  std::vector<FunctionInfo*> funcs;
  TF_RETURN_IF_ERROR(bfloat16_func->Find(
      [&](FunctionInfo* func) {
        bool executes_on_tpu = func->ExecutesOnTpu();
        absl::string_view xla_jit_device =
            executes_on_tpu ? tpu_xla_jit_device : cpu_xla_jit_device;
        auto& filterlist = executes_on_tpu ? tpu_filterlist : cpu_filterlist;

        for (Node* node : func->get_graph()->nodes()) {
          if (filterlist.contains(node->type_string())) continue;

          // Get a list of the current float attributes.
          ::gtl::linked_hash_set<std::string> float_attrs;
          for (auto& [name, attr] : node->attrs()) {
            // Only one data type attached.
            if (attr.has_type() && attr.type() == DT_FLOAT) {
              float_attrs.insert(name);
            }
            // A list of data types attached.
            if (attr.has_list() && attr.list().type_size() > 0) {
              auto original_dtypes = attr.list();
              for (int i = 0; i < original_dtypes.type_size(); ++i) {
                DataType dtype_original = original_dtypes.type(i);
                if (dtype_original == DT_FLOAT) {
                  float_attrs.insert(name);
                  break;
                }
              }
            }
          }
          // Get the kernel def of the node.
          const KernelDef* kernel_def = nullptr;
          absl::Status s = FindKernelDef(DeviceType(xla_jit_device),
                                         node->def(), &kernel_def, nullptr);
          // If the node does not have a kernel registration, skip it.
          if (!s.ok()) continue;

          // Check if the kernel def supports updating the float types to
          // bfloat16.
          for (const auto& constraint : kernel_def->constraint()) {
            if (!float_attrs.contains(constraint.name())) continue;
            if (!absl::c_linear_search(
                    constraint.allowed_values().list().type(), DT_BFLOAT16)) {
              filterlist.insert(node->type_string());
            }
          }
        }
        return false;
      },
      &funcs));

  return absl::OkStatus();
}

absl::Status UpdateTensorInfo(
    TensorInfo& tensor_info,
    const ::gtl::linked_hash_set<std::string>& bf16_node_names) {
  // Remove suffix, e.g. StatefulPartitionedCall:0 -> StatefulPartitionedCall.
  const std::string node_name =
      tensor_info.name().substr(0, tensor_info.name().find(':'));

  // Update float tensor that corresponds to a bfloat16 node.
  // At this point, all nodes referenced in bf16_node_names will have had
  // their float types converted to bfloat16. Here, the tensors in the
  // SignatureDef corresponding to these same nodes are also converted for
  // consistency. Note that each node may have multiple types, and only the
  // float tensors should be updated.
  if (bf16_node_names.contains(node_name) && tensor_info.dtype() == DT_FLOAT) {
    tensor_info.set_dtype(DT_BFLOAT16);
  }
  return absl::OkStatus();
}

// Updates the model's SignatureDef float tensors to bfloat16 to reflect the
// equivalent change in the model's signature during the bfloat16 optimization.
absl::Status UpdateSignatureDef(
    FunctionInfo* root_func,
    std::map<std::string, SignatureDef>* signature_def) {
  // Skip updating the SignatureDef if none is passed.
  // The Inference Converter V2 passes it and V1 does not.
  if (signature_def == nullptr) {
    return absl::OkStatus();
  }

  // Find all nodes in root that have bfloat16 as one of their types.
  ::gtl::linked_hash_set<std::string> bf16_node_names;
  for (Node* node : root_func->get_graph()->nodes()) {
    if (IsNodeBFloat16(node)) {
      bf16_node_names.insert(node->name());
    }
  }

  // Update the signature inputs and outputs.
  for (auto& [_, signature_value] : *signature_def) {
    for (auto& [_, tensor_info] : *signature_value.mutable_inputs()) {
      TF_RETURN_IF_ERROR(UpdateTensorInfo(tensor_info, bf16_node_names));
    }
    for (auto& [_, tensor_info] : *signature_value.mutable_outputs()) {
      TF_RETURN_IF_ERROR(UpdateTensorInfo(tensor_info, bf16_node_names));
    }
  }
  return absl::OkStatus();
}

absl::Status FuncFloatToBFloat16(
    FunctionInfo* bfloat16_func, const ConverterOptions& options,
    std::map<string, tensorflow::SignatureDef>* signature_def,
    bool auto_generate_filterlist) {
  LOG(INFO) << "BFloat16 scope function: " << bfloat16_func->name();
  // 0. Generate filterlist.
  ::gtl::linked_hash_set<string> tpu_filterlist;
  ::gtl::linked_hash_set<string> cpu_filterlist;
  TF_RETURN_IF_ERROR(GenerateFilterlists(bfloat16_func, options,
                                         auto_generate_filterlist,
                                         tpu_filterlist, cpu_filterlist));

  // 1. Convert Var-related nodes.
  // TODO(b/240491117): Only upcast variables that are inputs to high precision
  // nodes.
  if (std::find(options.bfloat16_filterlist().begin(),
                options.bfloat16_filterlist().end(),
                "VarHandleOp") == options.bfloat16_filterlist().end()) {
    // 1.1 Collect Var info.
    FuncArgsList func_args_list;
    TF_RETURN_IF_ERROR(
        GetFuncArgsList(bfloat16_func->GetRoot(), &func_args_list));
    ::gtl::linked_hash_set<std::unique_ptr<ResourceNodeInfo>> resource_vars;
    VarHandleCollection var_handles;
    AnalyseResourceVars(bfloat16_func->GetRoot(), func_args_list,
                        &resource_vars);
    BuildVarHandleCollection(
        resource_vars,
        [](Node* node) {
          return (node->attrs().Find("dtype")->type() == DT_FLOAT);
        },
        &var_handles);
    FilterVarHandleCollectionByFuncScope(*bfloat16_func, &var_handles);

    // 1.2 Convert Var-related components to bfloat16.
    TF_RETURN_IF_ERROR(VarFloatToBfloat16(bfloat16_func, var_handles, options));
  }

  // 2. Convert Cast nodes in TPU subgraph to bfloat16.
  TF_RETURN_IF_ERROR(bfloat16_func->RewriteGraphRecursively(
      kOptimFuncSuffix, RewriteGraphRecursivelyFilterFn,
      [](FunctionInfo* func) {
        return ConvertCastNodesToBfloat16(func->get_graph());
      }));

  // 3. Cast Const, Arg in TPU subgraph
  TF_RETURN_IF_ERROR(bfloat16_func->RewriteGraphRecursively(
      kOptimFuncSuffix, RewriteGraphRecursivelyFilterFn,
      [](FunctionInfo* func) {
        return CastConstNodesToBfloat16(func->get_graph());
      }));
  if (!options.bfloat16_cast_outside()) {
    TF_RETURN_IF_ERROR(
        CastNonVariableArgToBfloat16(bfloat16_func->get_graph()));
  }

  // Keep track of DT_BFLOAT16 Inputs/Outputs. These should not be converted to
  // DT_FLOAT during the optimization pass.
  ::gtl::linked_hash_set<int> bfloat16_arg_ret;
  for (const auto& node : bfloat16_func->get_graph()->nodes()) {
    if ((node->IsArg() && node->attrs().Find("T")->type() == DT_BFLOAT16) ||
        (node->IsRetval() && node->attrs().Find("T")->type() == DT_BFLOAT16)) {
      bfloat16_arg_ret.insert(node->id());
    }
  }

  // 4. Convert XlaCallModule Op node to bfloat16.
  TF_RETURN_IF_ERROR(bfloat16_func->RewriteGraphRecursively(
      kOptimFuncSuffix, RewriteGraphRecursivelyFilterFn,
      [&](FunctionInfo* func) {
        auto& filterlist =
            func->ExecutesOnTpu() ? tpu_filterlist : cpu_filterlist;
        if (filterlist.contains("XlaCallModule")) {
          VLOG(4) << "Skip XlaCallModule Op, node is in filterlist";
          return absl::OkStatus();
        }
        return ConvertXlaCallModuleOpNodesToBfloat16(func->get_graph());
      }));

  // 5. Convert node attributes to bfloat16.
  TF_RETURN_IF_ERROR(bfloat16_func->RewriteGraphRecursively(
      kOptimFuncSuffix, RewriteGraphRecursivelyFilterFn,
      [&](FunctionInfo* func) {
        auto& filterlist =
            func->ExecutesOnTpu() ? tpu_filterlist : cpu_filterlist;
        return ConvertGraphNodesToBfloat16(func->get_graph(), [&](Node* node) {
          // If the node type is in filterlist, or if the node is an _Arg or
          // _Retval node for While / If. _Arg and _Retval nodes for While / If
          // will be specially handled later to avoid touching flow inputs so
          // skip it here.
          return (
              filterlist.contains(node->type_string()) ||
              ((node->type_string() == "_Arg" ||
                node->type_string() == "_Retval") &&
               (func->node_in_parent_graph()->type_string() == "While" ||
                func->node_in_parent_graph()->type_string() == "If" ||
                func->node_in_parent_graph()->type_string() ==
                    "StatelessWhile" ||
                func->node_in_parent_graph()->type_string() == "StatelessIf")));
        });
      }));

  // 6. Process _Arg / _Retval DataType.
  // _Arg / _Retval attributes are converted to bfloat16. If cast needs to
  // happen inside function, changes in attributes need to be reverted; If not,
  // the function call node needs to be wrapped with type casts.
  if (!bfloat16_func->IsRoot()) {
    if (options.bfloat16_cast_outside() &&
        IsStandardFunctionNode(bfloat16_func->node_in_parent_graph())) {
      ConvertFunctionCallNodeToBfloat16(bfloat16_func->node_in_parent_graph());
      TF_RETURN_IF_ERROR(WrapFunctionCallNodeForBFloat16(bfloat16_func));
    } else {
      // If bfloat16 cast should happen in function, fix _Arg / _Retval.
      FuncGraphArgsAndRetvalsToFloat(bfloat16_func->get_graph(),
                                     bfloat16_arg_ret);
    }
  }

  // 7. Convert Control-flow nodes (While / If) to bfloat16.
  TF_RETURN_IF_ERROR(bfloat16_func->RewriteGraphRecursively(
      kOptimFuncSuffix, RewriteGraphRecursivelyFilterFn,
      ControlFlowsToBFloat16));

  // 8. Cast to bfloat16 before _Retval, if bfloat16_cast_in_func.
  if (!options.bfloat16_cast_outside()) {
    TF_RETURN_IF_ERROR(CastSrcOfRetvalToBfloat16(bfloat16_func->get_graph(),
                                                 bfloat16_arg_ret));
  }

  // 9. Wrap bfloat16-incompatible ops.
  TF_RETURN_IF_ERROR(bfloat16_func->RewriteGraphRecursively(
      kOptimFuncSuffix, RewriteGraphRecursivelyFilterFn,
      [&](FunctionInfo* func) {
        auto& filterlist =
            func->ExecutesOnTpu() ? tpu_filterlist : cpu_filterlist;
        return WrapBfloat16IncompatibleNodes(func->get_graph(), filterlist);
      }));

  // 10. Remove duplicate casts.
  TF_RETURN_IF_ERROR(bfloat16_func->RewriteGraphRecursively(
      kOptimFuncSuffix, RewriteGraphRecursivelyFilterFn,
      [](FunctionInfo* func) {
        return RemoveDuplicateCastNodes(func->get_graph());
      }));

  // 11. Convert Checkpoint to BFloat16.
  if (options.convert_checkpoint_bfloat16()) {
    TF_RETURN_IF_ERROR(CheckpointToBFloat16(options, bfloat16_func));
    TF_RETURN_IF_ERROR(
        FixRestoreAndSaveDtypes(bfloat16_func->GetRoot(), options));
  }

  // 12. Update the SignatureDef.
  // Only applicable when the optimized function is the root of the graph.
  // Note: this behavior is exclusive to the Inference Converter V2. V1 does
  // not update a model's SignatureDef upon bfloat16 optimization.
  if (bfloat16_func->IsRoot()) {
    TF_RETURN_IF_ERROR(
        UpdateSignatureDef(bfloat16_func->GetRoot(), signature_def));
  }

  return absl::OkStatus();
}

absl::Status ApplyBFloat16Optimization(
    const DeviceAgnosticBFloat16Scope bfloat16_scope,
    const ConverterOptions& options, FunctionInfo* func) {
  // Determine which function to apply bfloat16 optimization
  FunctionInfo* bfloat16_func = nullptr;
  if (bfloat16_scope == DeviceAgnosticBFloat16Scope::kDevice) {
    bfloat16_func = func;
  } else if (bfloat16_scope == DeviceAgnosticBFloat16Scope::kBatch) {
    if (func->IsRoot())
      return errors::InvalidArgument(
          "Cannot apply the bfloat16 optimization to the BatchFunction when "
          "there is no BatchFunction in the model. Either add the "
          "BatchFunction or change the BFloat16Scope.");
    // Search for BatchFunction in tpu_func's ancestors.
    FunctionInfo* batch_func = func->parent();
    while (!batch_func->IsRoot() &&
           batch_func->node_in_parent_graph()->type_string() !=
               "BatchFunction") {
      batch_func = batch_func->parent();
    }
    if (batch_func->IsRoot())
      return errors::InvalidArgument(
          "Cannot apply the bfloat16 optimization to the BatchFunction when "
          "there is no BatchFunction in the model. Either add the "
          "BatchFunction or change the BFloat16Scope.");
    bfloat16_func = batch_func;
  } else if (bfloat16_scope == DeviceAgnosticBFloat16Scope::kAll) {
    bfloat16_func = func->GetRoot();
  } else if (bfloat16_scope == DeviceAgnosticBFloat16Scope::kOther) {
    if (options.bfloat16_func_prefix().empty()) {
      return tensorflow::errors::FailedPrecondition(
          "Name prefix of FunctionDef to be converted to bfloat16",
          "needs to be unspecified.");
    }
    std::vector<FunctionInfo*> bfloat16_func_candidates;
    TF_RETURN_IF_ERROR(func->GetRoot()->Find(
        [&](FunctionInfo* f) {
          return absl::StartsWith(f->name(), options.bfloat16_func_prefix());
        },
        &bfloat16_func_candidates));
    if (bfloat16_func_candidates.empty()) {
      return tensorflow::errors::NotFound("FunctionDef name with prefix ",
                                          options.bfloat16_func_prefix(),
                                          " not found.");
    }
    bfloat16_func = bfloat16_func_candidates.front();
  } else {
    return tensorflow::errors::InvalidArgument("Invalid BFloat16Scope.");
  }
  TF_RETURN_IF_ERROR(FuncFloatToBFloat16(bfloat16_func, options));
  VLOG(3) << DumpGraphToFile(
      absl::StrCat(bfloat16_func->name(), "_after_bfloat16_optim"),
      *bfloat16_func->get_graph());
  return absl::OkStatus();
}

absl::Status ApplyBfloat16OptimizationV2(
    ::orbax::BFloat16OptimizationOptions bfloat16_options,
    FunctionInfo* xla_function_info,
    std::map<string, tensorflow::SignatureDef>* signature_def) {
  LOG(INFO) << "Applying bfloat16 optimization to "
            << xla_function_info->name();

  if (!bfloat16_options.skip_safety_checks()) {
    absl::Status s =
        ValidateGraphHasNoBFloat16Ops(xla_function_info->get_graph());
    if (!s.ok()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Encountered the following error when screening for bfloat16 ops ",
          "which can cause problems with the bfloat16 optimization. To ",
          "disable this set BFloat16OptimizationOptions's skip_safety_check. ",
          "Error:\n\"", s.message(), "\""));
    }
  }

  if (bfloat16_options.preserve_signature() &&
      bfloat16_options.scope() != BFloat16OptimizationOptions::ALL) {
    return absl::InvalidArgumentError(
        "The preserve_signature flag is only supported for the ALL scope. All "
        "other scopes preserve the signature by default. Set "
        "preserve_signature to false, or change the scope of the bfloat16 "
        "optimization.");
  }

  std::vector<FunctionInfo*> bfloat16_function_infos;
  ConverterOptions converter_options_v1;
  switch (bfloat16_options.scope()) {
    case BFloat16OptimizationOptions::DEFAULT:
    case BFloat16OptimizationOptions::TPU:
      bfloat16_function_infos.push_back(xla_function_info);
      converter_options_v1.set_bfloat16_scope(::orbax::BFloat16Scope::TPU);
      break;
    case BFloat16OptimizationOptions::ALL:
      converter_options_v1.set_bfloat16_scope(::orbax::BFloat16Scope::ALL);
      if (bfloat16_options.preserve_signature()) {
        TF_RETURN_IF_ERROR(xla_function_info->GetRoot()->get_children_ptrs(
            &bfloat16_function_infos));
      } else {
        bfloat16_function_infos.push_back(xla_function_info->GetRoot());
      }
      break;
    case BFloat16OptimizationOptions::GPU:
      bfloat16_function_infos.push_back(xla_function_info);
      converter_options_v1.set_gpu_bfloat16_scope(ConverterOptions::GPU);
      break;
    default:
      return absl::UnimplementedError(absl::StrCat(
          "Unexpected value for bfloat16_scope ", bfloat16_options.scope()));
  }

  // The only case when there may be more than one function to be converted to
  // bfloat16 is if the model signature should be preserved, in which case we
  // convert all children of root instead of root itself.
  if (bfloat16_options.preserve_signature()) {
    TF_RET_CHECK(!bfloat16_function_infos.empty());
  } else {
    TF_RET_CHECK(bfloat16_function_infos.size() == 1);
  }

  *converter_options_v1.mutable_bfloat16_filterlist() =
      bfloat16_options.filterlist();

  if (bfloat16_options.enable_cast_outside() &&
      converter_options_v1.bfloat16_scope() == ::orbax::BFloat16Scope::ALL) {
    return absl::InvalidArgumentError(
        "Cannot enable bfloat16 enable_cast_outside if the bfloat16 scope is "
        "not TPU or GPU.");
  }
  converter_options_v1.set_bfloat16_cast_outside(
      bfloat16_options.enable_cast_outside());

  for (FunctionInfo* bfloat16_function_info : bfloat16_function_infos) {
    TF_RET_CHECK(bfloat16_function_info != nullptr);

    TF_RETURN_IF_ERROR(FuncFloatToBFloat16(
        bfloat16_function_info, converter_options_v1, signature_def, true));

    TF_RETURN_IF_ERROR(bfloat16_function_info->MarkMutated(kOptimFuncSuffix));

    VLOG(3) << DumpGraphToFile("after_bfloat16_optimization_model",
                               *bfloat16_function_info->get_graph());
  }

  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<FunctionInfo>> GetFunctionInfoFromGraphDef(
    GraphDef& graph_def) {
  // Convert graph def to graph. graph will be passed to root_func which will
  // take ownership of the unique_ptr.
  auto graph(std::make_unique<Graph>(OpRegistry::Global()));
  ::tensorflow::ImportGraphDefOptions opts;
  opts.validate_shape = false;
  TF_RETURN_IF_ERROR(
      ImportGraphDef(opts, graph_def, graph.get(), nullptr, nullptr));

  // Build function tree.
  ::gtl::linked_hash_map<string, int> func_overhead;
  const FunctionLibraryDefinition flib_def = graph->flib_def();
  auto root_func = std::make_unique<FunctionInfo>("root");
  root_func->set_graph(&graph);
  TF_RETURN_IF_ERROR(root_func->Build(flib_def, &func_overhead));

  return root_func;
}

}  // namespace orbax
}  // namespace tensorflow
