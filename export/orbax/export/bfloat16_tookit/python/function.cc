#include "tensorflow/core/framework/function.h"

#include <pybind11/pybind11.h>

#include <memory>

#include "third_party/pybind11_protobuf/native_proto_caster.h"
#include "tensorflow/core/framework/graph.proto.h"  // copybara:strip
#include "tensorflow/core/framework/op.h"  // copybara:strip

namespace tensorflow::orbax {
namespace {

PYBIND11_MODULE(function, m) {
  pybind11_protobuf::ImportNativeProtoCasters();
  // copybara:strip_begin(internal)
  pybind11::class_<FunctionLibraryDefinition>(m, "FunctionLibraryDefinition")
      .def(pybind11::init([](const GraphDef& graph_def) {
             return std::make_unique<FunctionLibraryDefinition>(
                 OpRegistry::Global(), graph_def);
           }),
           pybind11::arg("graph_def"));
  // copybara:strip_end
}

}  // namespace
}  // namespace tensorflow::orbax
