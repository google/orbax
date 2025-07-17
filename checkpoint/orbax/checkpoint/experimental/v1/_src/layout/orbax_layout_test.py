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

import unittest
from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import numpy as np
from orbax.checkpoint import test_utils
from orbax.checkpoint.experimental.v1._src.handlers import composite_handler
from orbax.checkpoint.experimental.v1._src.layout import orbax_layout
from orbax.checkpoint.experimental.v1._src.saving import saving
import safetensors.numpy

np_save_file = safetensors.numpy.save_file
OrbaxLayout = orbax_layout.OrbaxLayout


class OrbaxLayoutTest(unittest.IsolatedAsyncioTestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.test_dir = self.create_tempdir()
    self.orbax_path = epath.Path(self.test_dir.full_path) / 'test_checkpoint'
    self.safetensors_path = (
        epath.Path(self.test_dir.full_path) / 'test_checkpoint.safetensors'
    )
    self.layout = OrbaxLayout()

    # Create a mock SafeTensors and Orbax checkpoint
    self.object_to_save = {
        'a': np.array(3 * [1, 2, 3], dtype=np.int32),
        'b': np.array([0, 1, 0.2], dtype=np.float32),
    }
    np_save_file(self.object_to_save, self.safetensors_path)
    saving.save_pytree(self.orbax_path / '0', self.object_to_save)

  def test_valid_orbax_checkpoint(self):
    self.assertTrue(self.layout.validate(path=self.orbax_path / '0'))

  def test_invalid_orbax_checkpoint(self):
    self.assertFalse(self.layout.validate(path=self.safetensors_path))

  def test_validate_fails_not_directory(self):
    # This tests `format_utils.validate_checkpoint_directory`
    invalid_path = epath.Path(self.orbax_path / '1')
    self.assertFalse(self.layout.validate(path=invalid_path))

  def test_validate_fails_no_metadata_file(self):
    # This tests `format_utils.validate_checkpoint_metadata`
    indicator_path = (
        self.orbax_path
        / '0'
        / composite_handler.ORBAX_CHECKPOINT_INDICATOR_FILE
    )
    indicator_path.rmtree()  # Remove the indicator file
    metadata_path = self.orbax_path / '0' / '_CHECKPOINT_METADATA'
    self.assertTrue(metadata_path.exists())
    metadata_path.rmtree()  # Remove the metadata file
    self.assertFalse(self.layout.validate(path=self.orbax_path / '0'))

  def test_validate_fails_tmp_directory(self):
    # This simulates a temporary directory created by Orbax (should fail)
    test_utils.save_fake_tmp_dir(self.orbax_path, 0, 'test_checkpoint.tmp')
    self.assertFalse(
        self.layout.validate(
            path=epath.Path(self.test_dir.full_path) / 'test_checkpoint.tmp'
        )
    )

  async def test_load_orbax_checkpoint(self):
    restored_checkpointables_await = await self.layout.load(
        directory=self.orbax_path / '0'
    )
    restored_checkpointables = await restored_checkpointables_await
    test_utils.assert_tree_equal(
        self, restored_checkpointables['pytree'], self.object_to_save
    )


if __name__ == '__main__':
  absltest.main()
