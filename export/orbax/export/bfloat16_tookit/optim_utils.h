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

#ifndef THIRD_PARTY_PY_ORBAX_EXPORT_BFLOAT16_TOOKIT_OPTIM_UTILS_H_
#define THIRD_PARTY_PY_ORBAX_EXPORT_BFLOAT16_TOOKIT_OPTIM_UTILS_H_

#include "third_party/absl/status/status.h"
#include "third_party/absl/strings/string_view.h"
#include "orbax/export/bfloat16_tookit/converter_options.proto.h"
#include "orbax/export/bfloat16_tookit/function_tree.h"
#include "tensorflow/compiler/jit/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/types.h"
#include "util/gtl/linked_hash_map.h"

namespace tensorflow {
namespace orbax {

constexpr char kOptimFuncSuffix[] = "_optim";
constexpr char kOutsideOptimFuncSuffix[] = "_outside_optim";

inline constexpr absl::string_view kDataTypeAttrs[] = {
    "dtype", "T", "Tidx", "Tparams", "LhsT", "RhsT", "Tin", "Tout"};

absl::Status ReadVariablesAsTensors(
    const string& input_prefix,
    ::gtl::linked_hash_map<string, Tensor>* var_tensors);

absl::Status WriteTensorsToVariables(
    const string& output_prefix,
    const ::gtl::linked_hash_map<string, Tensor>& var_tensors);

bool IsStandardFunctionNode(Node* node);

absl::Status InferShapesRecursively(FunctionInfo* func,
                                    GraphShapeInfo* shape_info);

// Replaces TPUPartitionedCall with StatefulPartitionedCall.
absl::Status RewriteTPUPartitionedCall(FunctionInfo* tpu_func);

bool IsNodeBFloat16(Node* node);

absl::Status ValidateGraphHasNoBFloat16Ops(Graph* graph);

}  // namespace orbax
}  // namespace tensorflow

#endif  // THIRD_PARTY_PY_ORBAX_EXPORT_BFLOAT16_TOOKIT_OPTIM_UTILS_H_
