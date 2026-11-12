/* Copyright 2026 The Orbax Authors.

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

#include <map>
#include <string>
#include <vector>

#include "third_party/absl/status/status.h"
#include "third_party/absl/status/statusor.h"
#include "orbax/export/bfloat16_toolkit/converter_options.proto.h"
#include "orbax/export/bfloat16_toolkit/converter_options_v2.proto.h"
#include "orbax/export/bfloat16_toolkit/function_tree.h"
#include "third_party/pybind11/include/pybind11/cast.h"
#include "third_party/pybind11/include/pybind11/gil.h"
#include "third_party/pybind11/include/pybind11/pybind11.h"
#include "third_party/pybind11/include/pybind11/pytypes.h"
#include "third_party/pybind11/include/pybind11/stl.h"
#include "third_party/pybind11_abseil/import_status_module.h"
#include "third_party/pybind11_abseil/status_caster.h"
#include "third_party/pybind11_abseil/statusor_caster.h"
#include "third_party/pybind11_protobuf/native_proto_caster.h"
#include "tensorflow/core/framework/graph.proto.h"
#include "tensorflow/core/protobuf/meta_graph.proto.h"

namespace tensorflow::orbax {
namespace {

PYBIND11_MODULE(bfloat16, m) {
  pybind11_protobuf::ImportNativeProtoCasters();
  pybind11::google::ImportStatusModule();
  pybind11::module_::import(
      "orbax.export.bfloat16_toolkit.converter_options_v2_pb2");
  pybind11::module_::import(
      "orbax.export.bfloat16_toolkit.python.function_tree");

  m.doc() = "Python bindings for bfloat16";

  pybind11::enum_<DeviceAgnosticBFloat16Scope>(m, "DeviceAgnosticBFloat16Scope")
      .value("DEVICE", DeviceAgnosticBFloat16Scope::kDevice)
      .value("BATCH", DeviceAgnosticBFloat16Scope::kBatch)
      .value("ALL", DeviceAgnosticBFloat16Scope::kAll)
      .value("OTHER", DeviceAgnosticBFloat16Scope::kOther)
      .export_values();

  m.def("apply_bfloat16_optimization", &ApplyBFloat16Optimization,
        "Convert a Function or its ancestor to bfloat16, depending on "
        "bfloat16_scope",
        pybind11::arg("bfloat16_scope"), pybind11::arg("options"),
        pybind11::arg("bfloat16_func").none(false),
        pybind11::call_guard<pybind11::gil_scoped_release>());

  m.def(
      "apply_bfloat16_optimization_v2",
      [](const BFloat16OptimizationOptions& bfloat16_options,
         FunctionInfo* xla_function_info,
         std::map<std::string, SignatureDef> signature_def,
         const InternalOptions& internal_options)
          -> absl::StatusOr<std::map<std::string, SignatureDef>> {
        if (absl::Status status =
                ApplyBfloat16OptimizationV2(bfloat16_options, xla_function_info,
                                            &signature_def, internal_options);
            !status.ok()) {
          return status;
        }
        return signature_def;
      },
      "Applies V2 bfloat16 optimization and returns the updated SignatureDef "
      "map.",
      pybind11::arg("bfloat16_options"),
      pybind11::arg("xla_function_info").none(false),
      pybind11::arg("signature_def") = std::map<std::string, SignatureDef>(),
      pybind11::arg("internal_options") = InternalOptions(),
      pybind11::call_guard<pybind11::gil_scoped_release>());
  m.def("get_function_info_from_graph_def", &GetFunctionInfoFromGraphDef,
        "Given a GraphDef, create a FunctionInfo that can be used to interact "
        "with the V1 converter. Changes to the FunctionInfo will only be "
        "preserved if UpdateGraphDefUsingFunctionInfo is called.",
        pybind11::arg("graph_def"),
        pybind11::call_guard<pybind11::gil_scoped_release>());
  m.def("update_graph_def_using_function_info",
        &UpdateGraphDefUsingFunctionInfo,
        "Writes changes made to the FunctionInfo tree back to a GraphDef.",
        pybind11::arg("function_info").none(false),
        pybind11::call_guard<pybind11::gil_scoped_release>());
  m.def(
      "get_xla_function_info",
      [](FunctionInfo& function_info) -> absl::StatusOr<pybind11::list> {
        absl::StatusOr<std::vector<FunctionInfo*>> xla_function_infos;
        {
          pybind11::gil_scoped_release release;
          xla_function_infos = GetXlaFunctionInfo(function_info);
        }
        if (!xla_function_infos.ok()) {
          return xla_function_infos.status();
        }
        return pybind11::cast(
            *xla_function_infos,
            pybind11::return_value_policy::reference_internal,
            pybind11::cast(&function_info,
                           pybind11::return_value_policy::reference));
      },
      "Finds top-level TPU or XLA GPU functions in the FunctionInfo tree.",
      pybind11::arg("function_info").none(false));
}

}  // namespace
}  // namespace tensorflow::orbax
