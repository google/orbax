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
    metadata = metadata_types.CheckpointMetadata(metadata=abstract_pytree)

    loaded = loading.load_pytree(
        self.orbax_pytree_path, abstract_pytree=metadata
    )
    test_utils.assert_tree_equal(self, self.object_to_save, loaded)

  def test_load_checkpointables_with_checkpoint_metadata(self):
    metadata = metadata_types.CheckpointMetadata(
        metadata=self.checkpointables_to_save
    )

    loaded = loading.load_checkpointables(
        self.orbax_checkpointables_path, abstract_checkpointables=metadata
    )
    test_utils.assert_tree_equal(self, self.checkpointables_to_save, loaded)

  # TODO(b/431045454): Add tests for abstract_checkpointables.


if __name__ == '__main__':
  absltest.main()
