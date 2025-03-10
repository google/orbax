# Copyright 2024 The Orbax Authors.
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
import jax
import numpy as np
from orbax.checkpoint._src.metadata import value as value_metadata
from orbax.checkpoint.experimental.v1._src.metadata import loading as metadata_loading
from orbax.checkpoint.experimental.v1._src.metadata import types as metadata_types
from orbax.checkpoint.experimental.v1._src.saving import saving
from orbax.checkpoint.experimental.v1._src.testing import array_utils as array_test_utils


class PyTreeMetadataTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.directory = epath.Path(self.create_tempdir().full_path) / 'ckpt'
    self.pytree, self.abstract_pytree = array_test_utils.create_numpy_pytree()
    saving.save_pytree(self.directory, self.pytree)

  def _create_value_metadata(self, value):
    if isinstance(value, np.ndarray):
      storage_metadata = value_metadata.StorageMetadata(
          chunk_shape=value.shape,
          write_shape=None,
      )
      return value_metadata.ArrayMetadata(
          name='a',
          directory=self.directory,
          shape=value.shape,
          dtype=value.dtype,
          sharding=None,
          storage=storage_metadata,
      )
    elif isinstance(value, (int, float)):
      dtype = np.float64
      if isinstance(value, int):
        dtype = np.int64
      storage_metadata = value_metadata.StorageMetadata(
          chunk_shape=(),
          write_shape=(),
      )
      return value_metadata.ScalarMetadata(
          name='a',
          directory=self.directory,
          shape=(),
          dtype=dtype,
          sharding=None,
          storage=storage_metadata,
      )
    else:
      raise TypeError(f'Unsupported type: {type(value)}')

  def test_invalid_path(self):
    with self.assertRaises(FileNotFoundError):
      metadata_loading.pytree_metadata(self.directory.parent)
    with self.assertRaises(FileNotFoundError):
      metadata_loading.pytree_metadata(self.directory.parent / 'foo')

  def test_pytree_metadata(self):
    metadata = metadata_loading.pytree_metadata(self.directory)
    self.assertIsInstance(metadata, metadata_types.CheckpointMetadata)
    self.assertIsInstance(metadata.metadata, metadata_types.PyTreeMetadata)
    expected_pytree_metadata = jax.tree.map(
        self._create_value_metadata, self.pytree
    )
    self.assertEqual(expected_pytree_metadata, metadata.metadata.pytree)



if __name__ == '__main__':
  absltest.main()
