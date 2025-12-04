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
from orbax.checkpoint._src.checkpointers import checkpointer
from orbax.checkpoint._src.handlers import composite_checkpoint_handler
from orbax.checkpoint._src.handlers import standard_checkpoint_handler
from orbax.checkpoint.experimental.v1._src.layout import orbax_layout
from orbax.checkpoint.experimental.v1._src.layout import orbax_v0_layout
from orbax.checkpoint.experimental.v1._src.layout import registry
from orbax.checkpoint.experimental.v1._src.saving import saving


get_checkpoint_layout_pytree = registry.get_checkpoint_layout_pytree
CheckpointLayoutEnum = registry.CheckpointLayoutEnum


class PyTreeCheckpointableResolutionTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):

  def setUp(self):
    super().setUp()
    self.root_directory = epath.Path(self.create_tempdir())
    self.v1_directory = self.root_directory / 'v1'
    saving.save_pytree(
        self.v1_directory,
        {'a': 1, 'b': 2},
    )
    self.v0_directory = self.root_directory / 'v0'
    ckptr = checkpointer.Checkpointer(
        composite_checkpoint_handler.CompositeCheckpointHandler()
    )
    ckptr.save(
        self.v0_directory,
        composite_checkpoint_handler.CompositeArgs(
            state=standard_checkpoint_handler.StandardSaveArgs({'a': 1, 'b': 2})
        ),
    )

  async def test_root_directory(self):
    # Root directory is not a valid checkpoint.
    with self.assertRaises(registry.InvalidLayoutError):
      await get_checkpoint_layout_pytree(
          self.root_directory, CheckpointLayoutEnum.ORBAX, None
      )

  async def test_v1_valid_name(self):
    layout, name = await get_checkpoint_layout_pytree(
        self.v1_directory, CheckpointLayoutEnum.ORBAX, 'pytree'
    )
    self.assertIsInstance(layout, orbax_layout.OrbaxLayout)
    self.assertEqual(layout.path, self.v1_directory)
    self.assertEqual(name, 'pytree')
    self.assertTrue(await orbax_layout.has_indicator_file(self.v1_directory))

  @parameterized.parameters([None, 'state', 'params'])
  async def test_v1_invalid_name(self, checkpointable_name):
    with self.assertRaises(registry.InvalidLayoutError):
      await get_checkpoint_layout_pytree(
          self.v1_directory, CheckpointLayoutEnum.ORBAX, checkpointable_name
      )

  async def test_v0_valid_name(self):
    layout, name = await get_checkpoint_layout_pytree(
        self.v0_directory, CheckpointLayoutEnum.ORBAX, 'state'
    )
    self.assertIsInstance(layout, orbax_v0_layout.OrbaxV0Layout)
    self.assertEqual(layout.path, self.v0_directory)
    self.assertEqual(name, 'state')
    self.assertFalse(await orbax_layout.has_indicator_file(self.v0_directory))

  @parameterized.parameters([None, 'params'])
  async def test_v0_invalid_name(self, checkpointable_name):
    with self.assertRaises(registry.InvalidLayoutError):
      await get_checkpoint_layout_pytree(
          self.v0_directory, CheckpointLayoutEnum.ORBAX, checkpointable_name
      )

  async def test_v1_direct_path(self):
    layout, name = await get_checkpoint_layout_pytree(
        self.v1_directory / 'pytree', CheckpointLayoutEnum.ORBAX, None
    )
    self.assertIsInstance(layout, orbax_v0_layout.OrbaxV0Layout)
    self.assertFalse(
        await orbax_layout.has_indicator_file(self.v1_directory / 'pytree')
    )
    self.assertEqual(layout.path, self.v1_directory / 'pytree')
    self.assertIsNone(name)

  async def test_v0_direct_path(self):
    layout, name = await get_checkpoint_layout_pytree(
        self.v0_directory / 'state', CheckpointLayoutEnum.ORBAX, None
    )
    self.assertIsInstance(layout, orbax_v0_layout.OrbaxV0Layout)
    self.assertFalse(
        await orbax_layout.has_indicator_file(self.v0_directory / 'state')
    )
    self.assertEqual(layout.path, self.v0_directory / 'state')
    self.assertIsNone(name)

  async def test_v1_missing_indicator_file(self):
    (self.v1_directory / orbax_layout.ORBAX_CHECKPOINT_INDICATOR_FILE).unlink()
    with self.assertRaises(registry.InvalidLayoutError):
      await get_checkpoint_layout_pytree(
          self.v1_directory, CheckpointLayoutEnum.ORBAX, None
      )

  async def test_v0_checkpoint_path(self):
    with self.assertRaises(registry.InvalidLayoutError):
      await get_checkpoint_layout_pytree(
          self.v0_directory, CheckpointLayoutEnum.ORBAX, None
      )

  async def test_v1_checkpoint_path_missing_pytree_metadata(self):
    (self.v1_directory / orbax_layout.ORBAX_CHECKPOINT_INDICATOR_FILE).unlink()
    (self.v1_directory / 'pytree' / '_METADATA').unlink()
    with self.assertRaises(registry.InvalidLayoutError):
      await get_checkpoint_layout_pytree(
          self.v1_directory, CheckpointLayoutEnum.ORBAX, None
      )

  async def test_v0_checkpoint_path_missing_pytree_metadata(self):
    (self.v0_directory / 'state' / '_METADATA').unlink()
    with self.assertRaises(registry.InvalidLayoutError):
      await get_checkpoint_layout_pytree(
          self.v0_directory, CheckpointLayoutEnum.ORBAX, None
      )

  async def test_v0_flat(self):
    flat_directory = self.root_directory / 'v0_flat'
    ckptr = checkpointer.Checkpointer(
        standard_checkpoint_handler.StandardCheckpointHandler()
    )
    ckptr.save(
        flat_directory,
        standard_checkpoint_handler.StandardSaveArgs({'a': 1, 'b': 2}),
    )

    layout, name = await get_checkpoint_layout_pytree(
        flat_directory, CheckpointLayoutEnum.ORBAX, None
    )
    self.assertIsInstance(layout, orbax_v0_layout.OrbaxV0Layout)
    self.assertEqual(layout.path, flat_directory)
    self.assertIsNone(name)

  async def test_v1_flat_errors(self):
    flat_directory = self.root_directory / 'v1_flat'
    ckptr = checkpointer.Checkpointer(
        standard_checkpoint_handler.StandardCheckpointHandler()
    )
    ckptr.save(
        flat_directory,
        standard_checkpoint_handler.StandardSaveArgs({'a': 1}),
    )
    # Mutlitate to become V1 by adding indicator file
    (flat_directory / orbax_layout.ORBAX_CHECKPOINT_INDICATOR_FILE).touch()

    self.assertTrue(await orbax_layout.has_indicator_file(flat_directory))

    # V1 checkpoints cannot be flat
    with self.assertRaises(registry.InvalidLayoutError):
      await get_checkpoint_layout_pytree(
          flat_directory, CheckpointLayoutEnum.ORBAX, None
      )


if __name__ == '__main__':
  absltest.main()
