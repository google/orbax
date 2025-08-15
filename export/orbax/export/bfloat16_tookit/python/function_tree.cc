#include "orbax/export/bfloat16_tookit/function_tree.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <string>

#include "third_party/absl/status/status.h"
#include "third_party/pybind11/include/pybind11/cast.h"
#include "third_party/pybind11/include/pybind11/pybind11.h"
#include "third_party/pybind11/include/pybind11/stl.h"
#include "third_party/pybind11_abseil/absl_casters.h"

namespace tensorflow::orbax {
namespace {

PYBIND11_MODULE(function_tree, m) {
  pybind11::class_<FunctionInfo>(m, "FunctionInfo")
      .def(pybind11::init<const std::string&>())
      .def(
          "lazy_update",
          [](FunctionInfo& self, const pybind11::bytes& fdef_lib_bytes) {
            // Create a FunctionDefLibrary object
            FunctionDefLibrary fdef_lib;

            // Deserialize the protobuf message from Python bytes
            std::string serialized_str = fdef_lib_bytes;
            if (!fdef_lib.ParseFromString(serialized_str)) {
              throw std::runtime_error(
                  "Failed to parse FunctionDefLibrary protobuf from bytes.");
            }

            // Call the original C++ method with a pointer to the object
            // The status is checked and converted to a Python exception if it's
            // not OK.
            absl::Status status = self.LazyUpdate(&fdef_lib);
            if (!status.ok()) {
              throw std::runtime_error(status.ToString());
            }

            // Since LazyUpdate might modify the fdef_lib, we serialize it back
            // and return it to the Python side.
            return pybind11::bytes(fdef_lib.SerializeAsString());
          },
          R"doc(
          Recursively update function graphs and the function library.

          Args:
              fdef_lib_bytes: A bytes object containing the serialized FunctionDefLibrary protobuf message.

          Returns:
              A bytes object containing the serialized, potentially modified, FunctionDefLibrary.

          Raises:
              RuntimeError: If the protobuf parsing fails or the C++ method returns a non-OK status.
      )doc",
          pybind11::arg("fdef_lib_bytes"))
      .def("name", &FunctionInfo::name,
           pybind11::return_value_policy::reference);
}

}  // namespace
}  // namespace tensorflow::orbax
