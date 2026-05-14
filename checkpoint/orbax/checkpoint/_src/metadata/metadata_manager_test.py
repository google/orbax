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

"""Unit tests for MetadataManager."""

import unittest
from unittest import mock

from absl.testing import absltest
from etils import epath
import jax.numpy as jnp
from orbax.checkpoint._src.metadata import array_metadata_store as array_metadata_store_lib
from orbax.checkpoint._src.metadata import metadata_manager
from orbax.checkpoint._src.metadata import tree as tree_metadata
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.path import format_utils
from orbax.checkpoint._src.serialization import ocdbt_utils
from orbax.checkpoint._src.serialization import type_handler_registry as type_handler_registry_lib
from orbax.checkpoint._src.serialization import type_handlers



class MetadataManagerTest(absltest.TestCase, unittest.IsolatedAsyncioTestCase):

  def setUp(self):
    super().setUp()
    self.directory = epath.Path(self.create_tempdir().full_path)
    self.options = tree_metadata.PYTREE_METADATA_OPTIONS
    self.manager = metadata_manager.MetadataManager()
    self.registry = type_handler_registry_lib.GLOBAL_TYPE_HANDLER_REGISTRY

  async def test_write_and_read_metadata_file(self):
    typestr = type_handler_registry_lib.get_param_typestr(
        0, self.registry, self.options
    )
    param_infos = {
        'x': type_handlers.ParamInfo(
            name='x',
            parent_dir=self.directory,
            skip_deserialize=False,
            value_typestr=typestr,
        ),
        'y': type_handlers.ParamInfo(
            name='y',
            parent_dir=self.directory,
            skip_deserialize=False,
            value_typestr=typestr,
        ),
    }
    save_args = {
        'x': type_handlers.SaveArgs(),
        'y': type_handlers.SaveArgs(),
    }
    custom_metadata = {'step': 10}

    internal_tree_metadata = tree_metadata.InternalTreeMetadata.build(
        param_infos,
        save_args=save_args,
        use_ocdbt=False,
        use_zarr3=False,
        custom_metadata=custom_metadata,
        pytree_metadata_options=self.options,
    )
    await self.manager.write_metadata_file(
        self.directory,
        internal_tree_metadata,
        primary_host=0,
        pytree_metadata_options=self.options,
    )

    metadata_path = self.directory / format_utils.PYTREE_METADATA_FILE
    self.assertTrue(metadata_path.exists())

    actual = await self.manager.read_metadata_file(
        self.directory, pytree_metadata_options=self.options
    )
    self.assertEqual(actual.custom_metadata, custom_metadata)
    tree = actual.as_nested_tree()
    self.assertIn('x', tree)
    self.assertIn('y', tree)

  async def test_read_metadata_file_not_found(self):
    with self.assertRaises(FileNotFoundError):
      await self.manager.read_metadata_file(
          self.directory, pytree_metadata_options=self.options
      )

  async def test_finalize_async(self):
    mock_store = mock.MagicMock(spec=array_metadata_store_lib.Store)
    mock_validator = mock.MagicMock(spec=array_metadata_store_lib.Validator)

    with mock.patch.object(
        array_metadata_store_lib,
        'validate_all_array_metadatas',
        new_callable=mock.AsyncMock,
    ) as mock_validate, mock.patch.object(
        ocdbt_utils,
        'merge_ocdbt_per_process_files',
        new_callable=mock.AsyncMock,
    ) as mock_merge:
      await self.manager.finalize_async(
          self.directory,
          array_metadata_store=mock_store,
          primary_host=0,
          array_metadata_validator=mock_validator,
          use_zarr3=False,
          enable_post_merge_validation=True,
      )

      mock_validate.assert_awaited_once_with(
          mock_validator, mock_store, self.directory
      )
      mock_merge.assert_awaited_once()

  async def test_get_param_infos_with_write_shape(self):
    mock_store = mock.MagicMock(spec=array_metadata_store_lib.Store)

    mock_array_metadata = mock.MagicMock()
    mock_array_metadata.param_name = 'x'
    mock_array_metadata.write_shape = (10, 20)
    mock_store.read = mock.AsyncMock(return_value=[mock_array_metadata])

    typestr = type_handler_registry_lib.get_param_typestr(
        jnp.zeros((10, 20), dtype=jnp.float32),
        self.registry,
        self.options,
    )
    param_infos = {
        'x': type_handlers.ParamInfo(
            name='x',
            parent_dir=self.directory,
            skip_deserialize=False,
            value_typestr=typestr,
        ),
    }

    with mock.patch.object(multihost, 'process_index', return_value=0):
      updated_param_infos = await self.manager.get_param_infos_with_write_shape(
          param_infos,
          self.directory,
          array_metadata_store=mock_store,
          primary_host=0,
      )

    self.assertEqual(updated_param_infos['x'].write_shape, (10, 20))


if __name__ == '__main__':
  absltest.main()
