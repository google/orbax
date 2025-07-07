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

# /orbax/checkpoint/experimental/v1/_src/layout/safetensors_layout_test.py
import unittest
from absl.testing import absltest
from etils import epath
import numpy as np
from orbax.checkpoint import v1 as ocp
from orbax.checkpoint.experimental.v1._src.layout import safetensors_layout
import safetensors.numpy

SafetensorsLayout = safetensors_layout.SafetensorsLayout
np_save_file = safetensors.numpy.save_file
training = ocp.training


class SafetensorsLayoutTest(
    absltest.TestCase, unittest.IsolatedAsyncioTestCase
):

  def setUp(self):
    super().setUp()
    self.test_dir = self.create_tempdir()
    self.orbax_path = epath.Path(self.test_dir.full_path) / 'test_checkpoint'
    self.safetensors_path = (
        epath.Path(self.test_dir.full_path) / 'test_checkpoint.safetensors'
    )
    self.layout = SafetensorsLayout()

    # Create a mock SafeTensors and Orbax checkpoint
    self.object_to_save = {
        'a': np.array(3 * [1, 2, 3]),
        'b': np.arange(0, 1, 0.2),
    }
    np_save_file(self.object_to_save, self.safetensors_path)
    with training.Checkpointer(self.orbax_path) as ckptr:
      ckptr.save_pytree(0, pytree=self.object_to_save)

  async def test_valid_safetensors_checkpoint(self):
    self.assertTrue(await self.layout.validate(path=self.safetensors_path))

  async def test_invalid_safetensors_checkpoint_orbax(self):
    self.assertFalse(await self.layout.validate(path=self.orbax_path / '0'))

  async def test_validate_fails_not_file(self):
    self.assertFalse(
        await self.layout.validate(path=epath.Path(self.test_dir.full_path))
    )

  async def test_validate_fails_wrong_suffix(self):
    wrong_suffix_path = (
        epath.Path(self.test_dir.full_path) / 'test_checkpoint.txt'
    )
    self.assertFalse(await self.layout.validate(path=wrong_suffix_path))

  async def test_load_safetensors_checkpoint(self):
    restored_checkpointables_await = await self.layout.load(
        directory=self.safetensors_path
    )
    restored_checkpointables = await restored_checkpointables_await

    np.testing.assert_array_equal(
        restored_checkpointables['a'], self.object_to_save['a']
    )
    # TODO(zachmeyers): Use assert_array_equal once we can control precision.
    np.testing.assert_allclose(
        restored_checkpointables['b'], self.object_to_save['b']
    )


if __name__ == '__main__':
  absltest.main()
