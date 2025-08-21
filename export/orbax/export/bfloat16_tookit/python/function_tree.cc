#include "orbax/export/bfloat16_tookit/function_tree.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>

#include "third_party/pybind11/include/pybind11/cast.h"
#include "third_party/pybind11/include/pybind11/pybind11.h"
#include "third_party/pybind11/include/pybind11/stl.h"
#include "third_party/pybind11_abseil/absl_casters.h"
#include "third_party/pybind11_abseil/status_casters.h"
#include "third_party/pybind11_protobuf/native_proto_caster.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/graph/graph.h"

namespace tensorflow::orbax {
namespace {

PYBIND11_MODULE(function_tree, m) {
  pybind11::google::ImportStatusModule();
  pybind11_protobuf::ImportNativeProtoCasters();

  pybind11::class_<Graph>(m, "Graph")
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
      .def("get_graph", &FunctionInfo::get_graph,
           // This is the crucial part!
           // It tells pybind11 that the returned Graph pointer's lifetime
           // is tied to (internal to) its parent FunctionInfo object.
           pybind11::return_value_policy::reference_internal);
}

}  // namespace
}  // namespace tensorflow::orbax
