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

from absl.testing import absltest
from etils import epath
import numpy as np
from orbax.checkpoint.experimental.v1._src.layout import safetensors_layout
from orbax.checkpoint.experimental.v1._src.loading import loading
from orbax.checkpoint.experimental.v1._src.saving import saving
import safetensors.numpy

SafetensorsLayout = safetensors_layout.SafetensorsLayout
np_save_file = safetensors.numpy.save_file


class LayoutLoadingTest(absltest.TestCase):

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
        'a': np.array(3 * [1, 2, 3], dtype=np.int32),
        'b': np.array([0, 1, 0.2], dtype=np.float32),
    }
    np_save_file(self.object_to_save, self.safetensors_path)
    saving.save_pytree(self.orbax_path, self.object_to_save)

  def test_load_safetensors_checkpoint(self):
    loaded_checkpointables = loading.load_checkpointables(
        self.safetensors_path, abstract_checkpointables=None
    )
    self.assertIsInstance(loaded_checkpointables, dict)
    np.testing.assert_array_equal(
        loaded_checkpointables['a'], self.object_to_save['a']
    )
    # TODO(b/430651483)
    np.testing.assert_allclose(
        loaded_checkpointables['b'], self.object_to_save['b']
    )

  def test_load_orbax_checkpoint(self):
    loaded_checkpointables = loading.load_checkpointables(
        self.orbax_path, abstract_checkpointables=None
    )
    self.assertIsInstance(loaded_checkpointables, dict)
    np.testing.assert_array_equal(
        loaded_checkpointables['pytree']['a'], self.object_to_save['a']
    )
    np.testing.assert_array_equal(
        loaded_checkpointables['pytree']['b'], self.object_to_save['b']
    )

  def test_load_bad_path_orbax_ckpt(self):
    # User provides a directory of Orbax checkpoints, not specific one.
    with self.assertRaises(ValueError):
      loading.load_checkpointables(
          epath.Path(self.test_dir.full_path),
          abstract_checkpointables=None,
      )

  def test_load_bad_path_safetensors_ckpt(self):
    # User provides a directory of SafeTensors checkpoints, not a file.
    for i in range(3):
      tmp_path = (
          epath.Path(self.test_dir.full_path)
          / f'{str(i)}/test_checkpoint.safetensors'
      )
      tmp_path.parent.mkdir(parents=True, exist_ok=True)
      np_save_file(self.object_to_save, tmp_path)
      with self.assertRaises(ValueError):
        loading.load_checkpointables(
            epath.Path(self.test_dir.full_path) / str(i),
            abstract_checkpointables=None,
        )

  def test_nonexistent_path(self):
    # User provides a path that does not exist.
    with self.assertRaises(ValueError):
      loading.load_checkpointables(
          epath.Path(self.test_dir.full_path) / 'nonexistent_path',
          abstract_checkpointables=None,
      )
  # TODO(b/431045454): Add tests for abstract_checkpointables.


if __name__ == '__main__':
  absltest.main()
