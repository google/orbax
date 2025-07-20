#include "orbax/export/bfloat16_tookit/function_tree.h"

#include <pybind11/pybind11.h>

#include <string>

#include "third_party/pybind11/include/pybind11/pybind11.h"

namespace tensorflow::orbax {
namespace {

PYBIND11_MODULE(function_tree, m) {
  pybind11::class_<FunctionInfo>(m, "FunctionInfo")
      .def(pybind11::init<const std::string&>());
}

}  // namespace
}  // namespace tensorflow::orbax
