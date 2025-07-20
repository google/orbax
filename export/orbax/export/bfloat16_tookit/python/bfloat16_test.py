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
from orbax.export.bfloat16_tookit import converter_options_pb2
from orbax.export.bfloat16_tookit.python import bfloat16
from orbax.export.bfloat16_tookit.python import function_tree
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


if __name__ == "__main__":
  absltest.main()
