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

#ifndef THIRD_PARTY_PY_ORBAX_EXPORT_BFLOAT16_TOOKIT_BFLOAT16_H_
#define THIRD_PARTY_PY_ORBAX_EXPORT_BFLOAT16_TOOKIT_BFLOAT16_H_

#include <map>

#include "third_party/absl/status/status.h"
#include "orbax/export/bfloat16_tookit/converter_options.proto.h"
#include "orbax/export/bfloat16_tookit/function_tree.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/meta_graph.proto.h"

namespace tensorflow {
namespace orbax {

using ConverterOptions = ::orbax::ConverterOptions;

// Convert a Function to bfloat16.
absl::Status FuncFloatToBFloat16(
    FunctionInfo* bfloat16_func, const ConverterOptions& options,
    std::map<string, tensorflow::SignatureDef>* signature_def = nullptr,
    bool auto_generate_filterlist = false);

// An internal enum to adapt to both TPU BFloat16Scope and GPUBFloat16Scope
enum class DeviceAgnosticBFloat16Scope { kDevice, kBatch, kAll, kOther };

// Convert a Function or its ancester to bfloat16, depending on bfloat16_scope
absl::Status ApplyBFloat16Optimization(
    DeviceAgnosticBFloat16Scope bfloat16_scope, const ConverterOptions& options,
    FunctionInfo* bfloat16_func);

}  // namespace orbax
}  // namespace tensorflow

#endif  // THIRD_PARTY_PY_ORBAX_EXPORT_BFLOAT16_TOOKIT_BFLOAT16_H_
