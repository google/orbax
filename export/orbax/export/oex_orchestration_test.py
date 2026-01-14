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

from collections.abc import Set
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from orbax.export import oex_orchestration
from orbax.export import typing as oex_typing
from orbax.export.data_processors import data_processor_base
from orbax.export.modules import obm_module
from orbax.export.protos import oex_orchestration_pb2
import tensorflow as tf


@tf.function
def tf_fn(a):
  return a


def tf_t(shape, name=None, dtype=tf.float32):
  return tf.TensorSpec(shape=shape, dtype=dtype, name=name)


class TestProcessor(data_processor_base.DataProcessor):

  def __init__(
      self,
      name: str = "",
      input_keys: Set[str] = frozenset(),
      output_keys: Set[str] = frozenset(),
  ):
    super().__init__(name=name, input_keys=input_keys, output_keys=output_keys)

  def prepare(self, input_signature):
    pass


class MockObmModule:

  def __init__(self, apply_fn_map):
    self.apply_fn_map = apply_fn_map


def _get_apply_fn_info(input_keys, output_keys):
  return oex_typing.ApplyFnInfo(
      apply_fn=lambda p, x: x,
      input_keys=input_keys,
      output_keys=output_keys,
  )


class OexOrchestrationTest(parameterized.TestCase):
  # Dummy test to make copybara happy, will be removed once all the obm
  # dependencies are OSSed.
  def test_dummy(self):
    assert True
    pass


if __name__ == "__main__":
  absltest.main()
