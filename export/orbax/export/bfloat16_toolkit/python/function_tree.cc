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

#include "orbax/export/bfloat16_toolkit/function_tree.h"

#include <string>
#include <vector>

#include "third_party/absl/status/status.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/pybind11/include/pybind11/cast.h"
#include "third_party/pybind11/include/pybind11/pybind11.h"
#include "third_party/pybind11/include/pybind11/stl.h"
#include "third_party/pybind11_abseil/import_status_module.h"
#include "third_party/pybind11_abseil/statusor_caster.h"
#include "third_party/pybind11_protobuf/native_proto_caster.h"
#include "tensorflow/core/framework/graph.proto.h"
#include "tensorflow/core/graph/graph.h"

namespace tensorflow::orbax {
namespace {

PYBIND11_MODULE(function_tree, m) {
  pybind11::google::ImportStatusModule();
  pybind11_protobuf::ImportNativeProtoCasters();

  pybind11::class_<Graph>(m, "Graph", pybind11::module_local())
      .def(
          "to_graph_def",
          [](const Graph& self, bool include_flib_def,
             bool include_debug_info) {
            GraphDef graph_def;
            self.ToGraphDef(&graph_def, include_flib_def, include_debug_info);
            return graph_def;
          },
          "Returns a GraphDef proto representation of the graph.",
          pybind11::arg("include_flib_def") = true,
          pybind11::arg("include_debug_info") = false);

  pybind11::class_<FunctionInfo>(m, "FunctionInfo")
      .def(pybind11::init<const std::string&>())
      .def("name", &FunctionInfo::name)
      .def("is_top_level_tpu_function", &FunctionInfo::IsTopLevelTpuFunction)
      .def("is_top_level_gpu_function", &FunctionInfo::IsTopLevelGpuFunction)
      .def(
          "get_children",
          [](FunctionInfo& self) -> absl::StatusOr<std::vector<FunctionInfo*>> {
            std::vector<FunctionInfo*> children;
            if (absl::Status status = self.get_children_ptrs(&children);
                !status.ok()) {
              return status;
            }
            return children;
          },
          pybind11::return_value_policy::reference_internal)
      .def("get_graph", &FunctionInfo::get_graph,
           // Ties the returned Graph pointer's lifetime to its parent
           // FunctionInfo.
           pybind11::return_value_policy::reference_internal);
}

}  // namespace
}  // namespace tensorflow::orbax
