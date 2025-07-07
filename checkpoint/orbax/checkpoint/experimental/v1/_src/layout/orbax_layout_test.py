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

# /orbax/checkpoint/experimental/v1/_src/layout/orbax_layout_test.py
import unittest
from absl.testing import absltest
from etils import epath
import numpy as np
from orbax.checkpoint import v1 as ocp
from orbax.checkpoint.experimental.v1._src.handlers import registration
from orbax.checkpoint.experimental.v1._src.layout import orbax_layout
import safetensors.numpy

np_save_file = safetensors.numpy.save_file
OrbaxLayout = orbax_layout.OrbaxLayout
training = ocp.training


class LayoutTest(absltest.TestCase, unittest.IsolatedAsyncioTestCase):

  def setUp(self):
    super().setUp()
    self.test_dir = self.create_tempdir()
    self.orbax_path = epath.Path(self.test_dir.full_path) / 'test_checkpoint'
    self.safetensors_path = epath.Path(self.test_dir.full_path) / 'test_checkpoint.safetensors'  # pylint: disable=line-too-long
    self.layout = OrbaxLayout(registration.local_registry())

    # Create a mock SafeTensors and Orbax checkpoint
    self.object_to_save = {'a': np.array(3 * [1, 2, 3]), 'b': np.arange(0, 1, 0.2)}  # pylint: disable=line-too-long
    np_save_file(self.object_to_save, self.safetensors_path)
    with training.Checkpointer(self.orbax_path) as ckptr:
      orbax_saved = ckptr.save_pytree(0, pytree=self.object_to_save)
      print(orbax_saved)

  async def test_valid_orbax_checkpoint(self):
    self.assertTrue(await self.layout.validate(path=self.orbax_path / '0'))

  async def test_invalid_orbax_checkpoint(self):
    self.assertFalse(await self.layout.validate(path=self.safetensors_path))

  async def test_validate_fails_not_directory(self):
    # This tests `format_utils.validate_checkpoint_directory`
    invalid_path = epath.Path(self.orbax_path / '1')
    self.assertFalse(await self.layout.validate(path=invalid_path))

  async def test_validate_fails_no_metadata_file(self):
    # This tests `format_utils.validate_checkpoint_metadata`
    metadata_path = self.orbax_path / '0' / '_CHECKPOINT_METADATA'
    # TODO(zachmeyers): update to orbax.checkpoint once the CL is submitted.
    self.assertTrue(metadata_path.exists())
    metadata_path.rmtree()  # Remove the metadata file
    self.assertFalse(await self.layout.validate(path=self.orbax_path / '0'))

  async def test_validate_fails_tmp_directory(self):
    tmp_path = epath.Path(self.test_dir.full_path) / 'test_checkpoint.tmp'
    tmp_path.mkdir()
    self.assertFalse(await self.layout.validate(path=tmp_path))

  async def test_load_orbax_checkpoint(self):
    restored_checkpointables_await = await self.layout.load(
        directory=self.orbax_path / '0'
    )
    restored_checkpointables = await restored_checkpointables_await
    np.testing.assert_array_equal(
        restored_checkpointables['pytree']['a'], self.object_to_save['a']
    )
    np.testing.assert_array_equal(
        restored_checkpointables['pytree']['b'], self.object_to_save['b']
    )

if __name__ == '__main__':
  absltest.main()
