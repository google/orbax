# Copyright 2025 The Orbax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from absl.testing import absltest
from orbax.export.bfloat16_toolkit import converter_options_pb2
from orbax.export.bfloat16_toolkit import converter_options_v2_pb2
from orbax.export.bfloat16_toolkit.python import bfloat16
from orbax.export.bfloat16_toolkit.python import function_tree
from absl.testing import absltest
from .third_party.pybind11_abseil import status


class BFloat16Test(googletest.TestCase):

  # Only add a simple test to check the binding can be called.
  def test_apply_bfloat16_optimization_raise_execption(self):
    bfloat16_func = function_tree.FunctionInfo("root")
    options = converter_options_pb2.ConverterOptions()
    with self.assertRaisesRegex(
        status.StatusNotOk,
        "Cannot apply the bfloat16 optimization to the BatchFunction when there"
        " is no BatchFunction in the model. Either add the BatchFunction or"
        " change the BFloat16Scope.",
    ):
      bfloat16.apply_bfloat16_optimization(
          bfloat16.BATCH, options, bfloat16_func
      )

  # Only add a simple test to check the binding can be called.
  def test_apply_bfloat16_optimization_v2(self):
    xla_function_info = function_tree.FunctionInfo("root")
    options = converter_options_v2_pb2.BFloat16OptimizationOptions()
    options.scope = converter_options_v2_pb2.BFloat16OptimizationOptions.TPU
    options.preserve_signature = True
    # The safety check fails in the test environment due to unsupported CPU
    # instructions.
    options.skip_safety_checks = True
    with self.assertRaisesRegex(
        status.StatusNotOk,
        "The preserve_signature flag is only supported for the ALL scope. All"
        " other scopes preserve the signature by default. Set"
        " preserve_signature to false, or change the scope of the bfloat16"
        " optimization.",
    ):
      bfloat16.apply_bfloat16_optimization_v2(options, xla_function_info, {})

  def test_get_function_info_from_graph_def(self):
    graph_def = graph_pb2.GraphDef()
    function_info = bfloat16.get_function_info_from_graph_def(graph_def)
    self.assertIsInstance(function_info, function_tree.FunctionInfo)


if __name__ == "__main__":
  absltest.main()
