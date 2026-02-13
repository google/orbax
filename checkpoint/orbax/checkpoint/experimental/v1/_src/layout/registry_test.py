# Copyright 2026 The Orbax Authors.
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
from orbax.checkpoint import args
from orbax.checkpoint._src.checkpointers import checkpointer
from orbax.checkpoint._src.handlers import composite_checkpoint_handler
from orbax.checkpoint._src.handlers import standard_checkpoint_handler
from orbax.checkpoint.checkpoint_manager import CheckpointManager
from orbax.checkpoint.experimental.v1._src.layout import orbax_layout
from orbax.checkpoint.experimental.v1._src.layout import orbax_v0_layout
from orbax.checkpoint.experimental.v1._src.layout import registry
from orbax.checkpoint.experimental.v1._src.saving import saving


get_checkpoint_layout_pytree = registry.get_checkpoint_layout_pytree
CheckpointLayoutEnum = registry.CheckpointLayoutEnum


class PyTreeCheckpointableResolutionAsyncTest(
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
    layout = await get_checkpoint_layout_pytree(
        self.v1_directory, CheckpointLayoutEnum.ORBAX, 'pytree'
    )
    self.assertIsInstance(layout, orbax_layout.OrbaxLayout)
    self.assertTrue(await orbax_layout.has_indicator_file(self.v1_directory))

  @parameterized.parameters([None, 'state', 'params'])
  async def test_v1_invalid_name(self, checkpointable_name):
    with self.assertRaises(registry.InvalidLayoutError):
      await get_checkpoint_layout_pytree(
          self.v1_directory, CheckpointLayoutEnum.ORBAX, checkpointable_name
      )

  async def test_v0_valid_name(self):
    layout = await get_checkpoint_layout_pytree(
        self.v0_directory, CheckpointLayoutEnum.ORBAX, 'state'
    )
    self.assertIsInstance(layout, orbax_v0_layout.OrbaxV0Layout)
    self.assertFalse(await orbax_layout.has_indicator_file(self.v0_directory))

  @parameterized.parameters([None, 'params'])
  async def test_v0_invalid_name(self, checkpointable_name):
    with self.assertRaises(registry.InvalidLayoutError):
      await get_checkpoint_layout_pytree(
          self.v0_directory, CheckpointLayoutEnum.ORBAX, checkpointable_name
      )

  async def test_v1_direct_path(self):
    with self.assertRaises(registry.InvalidLayoutError):
      await get_checkpoint_layout_pytree(
          self.v1_directory / 'pytree', CheckpointLayoutEnum.ORBAX, None
      )

  async def test_v0_direct_path(self):
    layout = await get_checkpoint_layout_pytree(
        self.v0_directory / 'state', CheckpointLayoutEnum.ORBAX, None
    )
    self.assertIsInstance(layout, orbax_v0_layout.OrbaxV0Layout)
    self.assertFalse(
        await orbax_layout.has_indicator_file(self.v0_directory / 'state')
    )

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

    layout = await get_checkpoint_layout_pytree(
        flat_directory, CheckpointLayoutEnum.ORBAX, None
    )
    self.assertIsInstance(layout, orbax_v0_layout.OrbaxV0Layout)

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


class IsOrbaxCheckpointTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.root_directory = epath.Path(self.create_tempdir())
    self.v1_directory = self.root_directory / 'v1'
    saving.save_pytree(
        self.v1_directory,
        {'a': 1, 'b': 2},
    )
    self.composite_dir = self.root_directory / 'composite_checkpoint'
    mngr = CheckpointManager(self.composite_dir)
    mngr.save(
        0,
        args=args.Composite(state=args.StandardSave({'a': 1, 'b': 2})),
    )
    mngr.wait_until_finished()

  def test_is_orbax_checkpoint(self):
    self.assertTrue(registry.is_orbax_checkpoint(self.v1_directory))
    self.assertTrue(orbax_layout.is_orbax_v1_checkpoint(self.v1_directory))
    self.assertTrue(registry.is_orbax_checkpoint(self.composite_dir / '0'))
    self.assertTrue(
        orbax_v0_layout.is_orbax_v0_checkpoint(self.composite_dir / '0')
    )


if __name__ == '__main__':
  absltest.main()
