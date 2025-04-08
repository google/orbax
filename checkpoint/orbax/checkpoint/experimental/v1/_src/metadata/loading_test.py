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

import dataclasses

from absl.testing import absltest
from etils import epath
import jax
import numpy as np
from orbax.checkpoint import test_utils
from orbax.checkpoint._src.metadata import value as value_metadata
import orbax.checkpoint.experimental.v1 as ocp
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.context import options as options_lib
from orbax.checkpoint.experimental.v1._src.metadata import types as metadata_types
from orbax.checkpoint.experimental.v1._src.testing import array_utils as array_test_utils
from orbax.checkpoint.experimental.v1._src.testing import handler_utils


Foo = handler_utils.Foo
Bar = handler_utils.Bar
AbstractFoo = handler_utils.AbstractFoo
AbstractBar = handler_utils.AbstractBar


class PyTreeMetadataTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.directory = epath.Path(self.create_tempdir().full_path) / 'ckpt'
    self.pytree, self.abstract_pytree = array_test_utils.create_numpy_pytree()
    ocp.save_pytree(self.directory, self.pytree)

  def _create_value_metadata(self, value):
    if isinstance(value, np.ndarray):
      storage_metadata = value_metadata.StorageMetadata(
          chunk_shape=value.shape,
          write_shape=None,
      )
      return value_metadata.ArrayMetadata(
          name='a',
          directory=self.directory / 'pytree',
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
          directory=self.directory / 'pytree',
          shape=(),
          dtype=dtype,  # pytype: disable=wrong-arg-types
          sharding=None,
          storage=storage_metadata,
      )
    else:
      raise TypeError(f'Unsupported type: {type(value)}')

  def test_invalid_path(self):
    with self.assertRaises(FileNotFoundError):
      ocp.pytree_metadata(self.directory.parent)
    with self.assertRaises(FileNotFoundError):
      ocp.pytree_metadata(self.directory.parent / 'foo')

  def test_pytree_metadata(self):
    metadata = ocp.pytree_metadata(self.directory)
    self.assertIsInstance(metadata, metadata_types.CheckpointMetadata)
    expected_pytree_metadata = jax.tree.map(
        self._create_value_metadata, self.pytree
    )
    self.assertEqual(expected_pytree_metadata, metadata.metadata)

  def test_load_with_metadata(self):
    metadata = ocp.pytree_metadata(self.directory)

    def _set_numpy_cast_type(x):
      if isinstance(x, np.ndarray):
        return x.astype(np.int16)
      elif isinstance(x, value_metadata.ArrayMetadata) and not isinstance(
          x, value_metadata.ScalarMetadata
      ):
        return dataclasses.replace(x, dtype=np.int16)
      else:
        return x

    metadata = dataclasses.replace(
        metadata, metadata=jax.tree.map(_set_numpy_cast_type, metadata.metadata)
    )
    expected_pytree = jax.tree.map(_set_numpy_cast_type, self.pytree)

    with self.subTest('pytree_metadata'):
      loaded_pytree = ocp.load_pytree(self.directory, metadata.metadata)
      test_utils.assert_tree_equal(self, expected_pytree, loaded_pytree)
    with self.subTest('full_metadata'):
      loaded_pytree = ocp.load_pytree(self.directory, metadata)
      test_utils.assert_tree_equal(self, expected_pytree, loaded_pytree)



class CheckpointablesMetadataTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.directory = epath.Path(self.create_tempdir().full_path) / 'ckpt'
    checkpointables_options = (
        options_lib.CheckpointablesOptions.create_with_handlers(
            handler_utils.FooHandler,
            handler_utils.BarHandler,
        )
    )
    self.enter_context(
        context_lib.Context(checkpointables_options=checkpointables_options)
    )
    checkpointables = {
        'foo': Foo(1, 'foo'),
        'bar': Bar(2, 'bar'),
    }
    ocp.save_checkpointables(self.directory, checkpointables)

  def test_checkpointables_metadata(self):
    metadata = ocp.checkpointables_metadata(self.directory)
    self.assertIsInstance(metadata, metadata_types.CheckpointMetadata)
    self.assertIsInstance(metadata.metadata, dict)
    self.assertIsInstance(metadata.metadata['foo'], AbstractFoo)
    self.assertIsInstance(metadata.metadata['bar'], AbstractBar)

  def test_load_with_metadata(self):
    path1 = epath.Path(self.create_tempdir().full_path) / 'ckpt1'
    path2 = epath.Path(self.create_tempdir().full_path) / 'ckpt2'

    ocp.save_checkpointables(path1, {'foo': Foo(1, 'foo')})
    ocp.save_checkpointables(path2, {'bar': Bar(3, 'bar')})

    metadata = ocp.checkpointables_metadata(path1)
    self.assertSameElements(metadata.metadata.keys(), ['foo'])
    metadata = dataclasses.replace(metadata, metadata={'bar': AbstractFoo()})
    loaded = ocp.load_checkpointables(path2, metadata)
    self.assertSameElements(loaded.keys(), ['bar'])
    self.assertEqual(Foo(3, 'bar'), loaded['bar'])



if __name__ == '__main__':
  absltest.main()
