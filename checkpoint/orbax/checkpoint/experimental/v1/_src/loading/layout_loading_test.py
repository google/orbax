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

import asyncio
import time
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import numpy as np
from orbax.checkpoint import test_utils
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.context import options as options_lib
from orbax.checkpoint.experimental.v1._src.layout import checkpoint_layout
from orbax.checkpoint.experimental.v1._src.loading import loading
from orbax.checkpoint.experimental.v1._src.metadata import types as metadata_types
from orbax.checkpoint.experimental.v1._src.saving import saving
import safetensors.numpy

np_save_file = safetensors.numpy.save_file
InvalidLayoutError = checkpoint_layout.InvalidLayoutError


class LayoutLoadingTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.test_dir = self.create_tempdir()
    self.test_dir_safetensors = self.create_tempdir()
    self.orbax_path = epath.Path(self.test_dir.full_path) / 'test_checkpoint'
    self.orbax_pytree_path = (
        epath.Path(self.test_dir.full_path) / 'test_checkpoint_pytree'
    )
    self.orbax_checkpointables_path = (
        epath.Path(self.test_dir.full_path) / 'test_checkpoint_checkpointables'
    )
    self.safetensors_path = (
        epath.Path(self.test_dir.full_path) / 'test_checkpoint.safetensors'
    )

    # Create a mock SafeTensors and Orbax checkpoint pytree
    self.object_to_save = {
        'a': np.array(3 * [1, 2, 3], dtype=np.int32),
        'b': np.array([0, 1, 0.2], dtype=np.float32),
    }
    np_save_file(self.object_to_save, self.safetensors_path)
    saving.save_pytree(self.orbax_pytree_path, self.object_to_save)

    # Create a mock Orbax checkpoint checkpointables
    self.checkpointables_to_save = {
        'pytree_a': self.object_to_save,
        'pytree_b': self.object_to_save,
    }
    saving.save_checkpointables(
        self.orbax_checkpointables_path, self.checkpointables_to_save
    )

  def test_load_safetensors_checkpoint(self):
    with context_lib.Context(
        checkpoint_layout=options_lib.CheckpointLayout.SAFETENSORS
    ):
      pytree = loading.load_pytree(self.safetensors_path)
    self.assertIsInstance(pytree, dict)
    np.testing.assert_array_equal(pytree['a'], self.object_to_save['a'])
    # TODO(b/430651483)
    np.testing.assert_allclose(pytree['b'], self.object_to_save['b'])

  def test_load_orbax_pytree_checkpoint(self):
    pytree = loading.load_pytree(self.orbax_pytree_path)
    test_utils.assert_tree_equal(self, self.object_to_save, pytree)

  def test_load_orbax_checkpointables_checkpoint(self):
    loaded = loading.load_checkpointables(self.orbax_checkpointables_path)
    test_utils.assert_tree_equal(self, self.checkpointables_to_save, loaded)

  @parameterized.parameters(
      (options_lib.CheckpointLayout.ORBAX,),
  )
  def test_load_bad_path_orbax_ckpt(self, layout_enum):
    # User provides a directory of Orbax checkpoints, not specific one.
    with context_lib.Context(checkpoint_layout=layout_enum):
      with self.assertRaises(InvalidLayoutError):
        loading.load_pytree(
            epath.Path(self.test_dir.full_path),
        )

  @parameterized.parameters(
      (options_lib.CheckpointLayout.SAFETENSORS,),
  )
  def test_load_bad_path_safetensors_ckpt(self, layout_enum):
    # User provides a empty directory of SafeTensors checkpoints, not a file.
    with context_lib.Context(checkpoint_layout=layout_enum):
      with self.assertRaises(InvalidLayoutError):
        loading.load_pytree(
            epath.Path(self.test_dir_safetensors.full_path),
        )

  def test_load_safetensors_ckpt_from_dir(self):
    safetensors_dir = epath.Path(self.test_dir_safetensors.full_path)
    safetensors_path = safetensors_dir / 'model.safetensors'
    np_save_file(self.object_to_save, safetensors_path)
    with context_lib.Context(
        checkpoint_layout=options_lib.CheckpointLayout.SAFETENSORS
    ):
      pytree = loading.load_pytree(safetensors_dir)
    self.assertIsInstance(pytree, dict)
    np.testing.assert_array_equal(pytree['a'], self.object_to_save['a'])
    np.testing.assert_allclose(pytree['b'], self.object_to_save['b'])

  def test_nonexistent_path(self):
    # User provides a path that does not exist.
    with self.assertRaises(InvalidLayoutError):
      loading.load_pytree(
          epath.Path(self.test_dir.full_path) / 'nonexistent_path',
      )

  def test_load_pytree_with_checkpoint_metadata(self):
    abstract_pytree = self.object_to_save
    metadata = metadata_types.CheckpointMetadata(
        path=self.orbax_pytree_path, metadata=abstract_pytree
    )

    loaded = loading.load_pytree(
        self.orbax_pytree_path, abstract_pytree=metadata
    )
    test_utils.assert_tree_equal(self, self.object_to_save, loaded)

  def test_load_checkpointables_with_checkpoint_metadata(self):
    metadata = metadata_types.CheckpointMetadata(
        path=self.orbax_checkpointables_path,
        metadata=self.checkpointables_to_save,
    )

    loaded = loading.load_checkpointables(
        self.orbax_checkpointables_path, abstract_checkpointables=metadata
    )
    test_utils.assert_tree_equal(self, self.checkpointables_to_save, loaded)

  @parameterized.parameters(
      (options_lib.CheckpointLayout.SAFETENSORS,),
      (options_lib.CheckpointLayout.ORBAX,),
  )
  def test_load_pytree_async(self, layout: options_lib.CheckpointLayout):
    original_finalize_load = loading._LoadPyTreeResponse._finalize_load

    async def sleep_and_load(*args, **kwargs):
      await asyncio.sleep(2)
      return await original_finalize_load(*args, **kwargs)

    self.enter_context(
        mock.patch.object(
            loading._LoadPyTreeResponse,
            '_finalize_load',
            new=sleep_and_load,
        )
    )

    pytree = self.object_to_save
    if layout == options_lib.CheckpointLayout.SAFETENSORS:
      directory = self.safetensors_path
    else:
      directory = self.orbax_pytree_path

    with context_lib.Context(checkpoint_layout=layout):
      if layout != options_lib.CheckpointLayout.SAFETENSORS:
        with self.assertRaises(NotImplementedError):
          loading.load_pytree_async(directory)
        return

      start = time.time()
      response = loading.load_pytree_async(directory)

    self.assertLess(time.time() - start, 1)
    loaded = response.result()
    self.assertGreater(time.time() - start, 2)
    test_utils.assert_tree_equal(self, pytree, loaded)

  # TODO(b/431045454): Add tests for abstract_checkpointables.

  def test_load_auto_resolution_mode_orbax(self):
    with context_lib.Context(
        checkpoint_layout=options_lib.CheckpointLayout.ORBAX
    ):
      loaded_orbax = loading.load_pytree(
          self.orbax_pytree_path,
          checkpointable_name=checkpoint_layout.AUTO_CHECKPOINTABLE_KEY,
      )
    test_utils.assert_tree_equal(self, self.object_to_save, loaded_orbax)

  def test_load_auto_resolution_mode_safetensors(self):
    with context_lib.Context(
        checkpoint_layout=options_lib.CheckpointLayout.SAFETENSORS
    ):
      loaded_safe = loading.load_pytree(
          self.safetensors_path,
          checkpointable_name=checkpoint_layout.AUTO_CHECKPOINTABLE_KEY,
      )
    test_utils.assert_tree_equal(self, self.object_to_save, loaded_safe)

  def test_load_auto_multiple_checkpointables_priority(self):
    # Save a checkpoint structure containing multiple checkpointable names.
    checkpointables = {
        'analytics': {'a': np.array([1, 2, 3])},
        'pytree': {'a': np.array([1, 2, 3])},
        'state': {'b': np.array([4, 5, 6])},
    }
    multiple_path = epath.Path(self.test_dir.full_path) / 'multi_checkpoint'
    saving.save_checkpointables(multiple_path, checkpointables)

    # Triggering AUTO loading mode should prioritize resolving 'pytree'.
    with context_lib.Context(
        checkpoint_layout=options_lib.CheckpointLayout.ORBAX
    ):
      loaded = loading.load_pytree(multiple_path)

    test_utils.assert_tree_equal(self, checkpointables['pytree'], loaded)

  def test_load_auto_non_pytree_fallback(self):
    # Save a checkpoint that intentionally omits the standard 'pytree' key.
    custom_checkpointables = {
        'custom_state': {'c': np.array([7, 8, 9])},
    }
    fallback_path = epath.Path(self.test_dir.full_path) / 'fallback_checkpoint'
    saving.save_checkpointables(fallback_path, custom_checkpointables)

    with context_lib.Context(
        checkpoint_layout=options_lib.CheckpointLayout.ORBAX
    ):
      loaded = loading.load_pytree(
          fallback_path,
          checkpointable_name=checkpoint_layout.AUTO_CHECKPOINTABLE_KEY,
      )

    # Returns the first valid checkpointable name alphabetically.
    test_utils.assert_tree_equal(
        self, custom_checkpointables['custom_state'], loaded
    )


if __name__ == '__main__':
  absltest.main()
