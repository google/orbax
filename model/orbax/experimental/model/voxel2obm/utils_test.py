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

"""Tests for utils."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from orbax.experimental.model import core as obm
from orbax.experimental.model.voxel2obm import utils
from .learning.brain.experimental.jax_data.python import voxel_tensor_spec


class UtilsTest(parameterized.TestCase):

  def test_obm_spec_to_voxel_signature(self):
    obm_spec = {
        'a': obm.ShloTensorSpec(shape=(1, 2), dtype=obm.ShloDType.i32),
        'b': obm.ShloTensorSpec(shape=(3,), dtype=obm.ShloDType.f32),
    }
    voxel_sig = utils.obm_spec_to_voxel_signature(obm_spec)
    expected_voxel_sig = {
        'a': voxel_tensor_spec.VoxelTensorSpec(
            shape=(1, 2), dtype=np.dtype(np.int32)
        ),
        'b': voxel_tensor_spec.VoxelTensorSpec(
            shape=(3,), dtype=np.dtype(np.float32)
        ),
    }

    self.assertEqual(voxel_sig['a'], expected_voxel_sig['a'])
    self.assertEqual(voxel_sig['b'], expected_voxel_sig['b'])

  def test_voxel_signature_to_obm_spec(self):
    voxel_sig = {
        'a': voxel_tensor_spec.VoxelTensorSpec(
            shape=(1, 2), dtype=np.dtype(np.int32)
        ),
        'b': voxel_tensor_spec.VoxelTensorSpec(
            shape=(3,), dtype=np.dtype(np.float32)
        ),
    }
    obm_spec = utils.voxel_signature_to_obm_spec(voxel_sig)
    expected_obm_spec = {
        'a': obm.ShloTensorSpec(shape=(1, 2), dtype=obm.ShloDType.i32),
        'b': obm.ShloTensorSpec(shape=(3,), dtype=obm.ShloDType.f32),
    }
    self.assertEqual(obm_spec, expected_obm_spec)

  def test_voxel_to_obm_dtype_error(self):
    with self.assertRaisesRegex(
        ValueError,
        "Expected a numpy.dtype, got <class 'int'> of type <class 'type'>",
    ):
      utils._voxel_to_obm_dtype(int)


if __name__ == '__main__':
  absltest.main()
