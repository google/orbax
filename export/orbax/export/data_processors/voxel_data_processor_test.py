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

"""Tests for VoxelDataProcessor."""

from typing import Dict
from unittest import mock

from absl.testing import absltest
import jax
import jax.numpy as jnp
from orbax.experimental.model.core.python import function
from orbax.export.data_processors import voxel_data_processor


class _VoxelModule:
  """A mock VoxelModule for testing."""

  def get_output_signature(
      self, input_signature: Dict[str, jax.Array]
  ) -> Dict[str, jax.Array]:
    return {
        'output': jnp.zeros(
            shape=input_signature['input'].shape,
            dtype=input_signature['input'].dtype,
        )
    }

  def export_plan(self):
    plan_proto = mock.Mock()
    plan_proto.SerializeToString.return_value = b'test plan'
    return plan_proto


class VoxelDataProcessorTest(absltest.TestCase):

  def test_property_raises_error_without_calling_prepare(self):
    processor = voxel_data_processor.VoxelDataProcessor(_VoxelModule())
    with self.assertRaisesWithLiteralMatch(
        RuntimeError,
        '`prepare()` must be called before accessing this property.',
    ):
      _ = processor.input_signature

    with self.assertRaisesWithLiteralMatch(
        RuntimeError,
        '`prepare()` must be called before accessing this property.',
    ):
      _ = processor.output_signature

    with self.assertRaisesWithLiteralMatch(
        RuntimeError,
        '`prepare()` must be called before accessing this property.',
    ):
      _ = processor.obm_function

  def test_prepare_succeeds(self):
    processor = voxel_data_processor.VoxelDataProcessor(_VoxelModule())
    input_signature = {'input': jax.ShapeDtypeStruct((1, 3), jnp.float32)}
    processor.prepare(input_signature)

    self.assertIsNotNone(processor.obm_function)
    self.assertEqual(
        voxel_data_processor._to_shlo_tensor_spec(processor.input_signature),
        {'input': function.ShloTensorSpec(shape=(1, 3), dtype=jnp.float32)},
    )
    self.assertEqual(
        voxel_data_processor._to_shlo_tensor_spec(processor.output_signature),
        {'output': function.ShloTensorSpec(shape=(1, 3), dtype=jnp.float32)},
    )


if __name__ == '__main__':
  absltest.main()
