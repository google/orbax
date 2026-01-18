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
from orbax.checkpoint.experimental.v1._src.layout import checkpoint_layout
from orbax.checkpoint.experimental.v1._src.layout import orbax_layout
from orbax.checkpoint.experimental.v1._src.layout import registry
from orbax.checkpoint.experimental.v1._src.saving import saving


try_resolve_pytree_checkpointable = registry._try_resolve_pytree_checkpointable


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
    layout = orbax_layout.OrbaxLayout(self.root_directory)
    self.assertFalse(layout.has_indicator_file)
    with self.assertRaisesRegex(
        ValueError, 'failed to resolve a checkpointable name'
    ):
      await try_resolve_pytree_checkpointable(layout, None)

  @parameterized.product(checkpointable_name=['state', 'params', None])
  async def test_v1(self, checkpointable_name):
    layout = orbax_layout.OrbaxLayout(self.v1_directory)
    self.assertTrue(layout.has_indicator_file)
    resolved_layout, name = await try_resolve_pytree_checkpointable(
        layout, checkpointable_name
    )
    self.assertIs(layout, resolved_layout)
    self.assertEqual(name, checkpointable_name)

  @parameterized.product(checkpointable_name=['state', 'params', None])
  async def test_v0(self, checkpointable_name):
    # Note: params resolves even though it doesn't exist.
    expected_name = (
        checkpointable_name if checkpointable_name is not None else 'state'
    )
    layout = orbax_layout.OrbaxLayout(self.v0_directory)
    self.assertFalse(layout.has_indicator_file)
    resolved_layout, resolved_name = await try_resolve_pytree_checkpointable(
        layout, checkpointable_name
    )
    self.assertIs(layout, resolved_layout)
    self.assertEqual(resolved_name, expected_name)

  async def test_v1_direct_path(self):
    layout = orbax_layout.OrbaxLayout(self.v1_directory / 'pytree')
    self.assertFalse(layout.has_indicator_file)
    resolved_layout, name = await try_resolve_pytree_checkpointable(
        layout, None
    )
    self.assertEqual(resolved_layout.path, self.v1_directory)
    self.assertEqual(name, 'pytree')

  async def test_v0_direct_path(self):
    layout = orbax_layout.OrbaxLayout(self.v0_directory / 'state')
    self.assertFalse(layout.has_indicator_file)
    resolved_layout, name = await try_resolve_pytree_checkpointable(
        layout, None
    )
    self.assertEqual(resolved_layout.path, self.v0_directory)
    self.assertEqual(name, 'state')

  async def test_v1_missing_indicator_file(self):
    (self.v1_directory / orbax_layout.ORBAX_CHECKPOINT_INDICATOR_FILE).unlink()
    layout = orbax_layout.OrbaxLayout(self.v1_directory)
    self.assertFalse(layout.has_indicator_file)
    resolved_layout, name = await try_resolve_pytree_checkpointable(
        layout, None
    )
    self.assertIs(layout, resolved_layout)
    self.assertEqual(name, 'pytree')

  async def test_v0_checkpoint_path(self):
    layout = orbax_layout.OrbaxLayout(self.v0_directory)
    self.assertFalse(layout.has_indicator_file)
    resolved_layout, name = await try_resolve_pytree_checkpointable(
        layout, None
    )
    self.assertIs(layout, resolved_layout)
    self.assertEqual(name, 'state')

  async def test_v1_checkpoint_path_missing_pytree_metadata(self):
    (self.v1_directory / orbax_layout.ORBAX_CHECKPOINT_INDICATOR_FILE).unlink()
    (self.v1_directory / 'pytree' / '_METADATA').unlink()
    layout = orbax_layout.OrbaxLayout(self.v1_directory)
    self.assertFalse(layout.has_indicator_file)
    with self.assertRaises(checkpoint_layout.InvalidLayoutError):
      await try_resolve_pytree_checkpointable(layout, None)

  async def test_v0_checkpoint_path_missing_pytree_metadata(self):
    (self.v0_directory / 'state' / '_METADATA').unlink()
    layout = orbax_layout.OrbaxLayout(self.v0_directory)
    self.assertFalse(layout.has_indicator_file)
    with self.assertRaises(checkpoint_layout.InvalidLayoutError):
      await try_resolve_pytree_checkpointable(layout, None)


if __name__ == '__main__':
  absltest.main()
