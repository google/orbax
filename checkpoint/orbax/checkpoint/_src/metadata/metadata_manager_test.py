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
from absl.testing import parameterized
from etils import epath
import jax
import jax.numpy as jnp
from orbax.checkpoint._src.metadata import array_metadata_store as array_metadata_store_lib
from orbax.checkpoint._src.metadata import metadata_manager
from orbax.checkpoint._src.metadata import tree as tree_metadata
from orbax.checkpoint._src.multihost import dispatchers
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.multihost import pathways as pathways_multihost
from orbax.checkpoint._src.path import format_utils
from orbax.checkpoint._src.serialization import ocdbt_utils
from orbax.checkpoint._src.serialization import type_handler_registry as type_handler_registry_lib
from orbax.checkpoint._src.serialization import type_handlers
from orbax.checkpoint._src.tree import types as tree_types


class MetadataManagerTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):

  def setUp(self):
    super().setUp()
    self.directory = epath.Path(self.create_tempdir().full_path)
    self.options = tree_metadata.PYTREE_METADATA_OPTIONS
    self.registry = type_handler_registry_lib.GLOBAL_TYPE_HANDLER_REGISTRY

  def test_init_properties(self):
    mock_store = mock.MagicMock(spec=array_metadata_store_lib.Store)
    mock_validator = mock.MagicMock(spec=array_metadata_store_lib.Validator)
    manager = metadata_manager.MetadataManager(
        primary_host=0,
        pytree_metadata_options=self.options,
        array_metadata_store=mock_store,
        array_metadata_validator=mock_validator,
        use_zarr3=True,
        enable_post_merge_validation=False,
    )
    self.assertEqual(manager.primary_host, 0)
    self.assertEqual(manager.pytree_metadata_options, self.options)
    self.assertEqual(manager.array_metadata_store, mock_store)
    self.assertEqual(manager.array_metadata_validator, mock_validator)
    self.assertTrue(manager.use_zarr3)
    self.assertFalse(manager.enable_post_merge_validation)

  async def test_write_and_read_metadata_file(self):
    manager = metadata_manager.MetadataManager(
        primary_host=0,
        pytree_metadata_options=self.options,
    )
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
    custom_metadata: tree_types.JsonType = {'step': 10}

    internal_tree_metadata = tree_metadata.InternalTreeMetadata.build(
        param_infos,
        save_args=save_args,
        use_ocdbt=False,
        use_zarr3=False,
        custom_metadata=custom_metadata,
        pytree_metadata_options=self.options,
    )
    await manager.write_metadata_file(
        self.directory,
        internal_tree_metadata,
    )

    metadata_path = self.directory / format_utils.PYTREE_METADATA_FILE
    self.assertTrue(metadata_path.exists())

    actual = await manager.read_metadata_file(self.directory)
    self.assertEqual(actual.custom_metadata, custom_metadata)
    tree = actual.as_nested_tree()
    self.assertIn('x', tree)
    self.assertIn('y', tree)

  async def test_read_metadata_file_not_found(self):
    manager = metadata_manager.MetadataManager(
        pytree_metadata_options=self.options,
    )
    with self.assertRaises(FileNotFoundError):
      await manager.read_metadata_file(self.directory)

  async def test_finalize_async(self):
    mock_store = mock.MagicMock(spec=array_metadata_store_lib.Store)
    mock_validator = mock.MagicMock(spec=array_metadata_store_lib.Validator)

    manager = metadata_manager.MetadataManager(
        primary_host=0,
        array_metadata_store=mock_store,
        array_metadata_validator=mock_validator,
        use_zarr3=False,
        enable_post_merge_validation=True,
    )

    with mock.patch.object(
        array_metadata_store_lib,
        'validate_all_array_metadatas',
        new_callable=mock.AsyncMock,
    ) as mock_validate, mock.patch.object(
        ocdbt_utils,
        'merge_ocdbt_per_process_files',
        new_callable=mock.AsyncMock,
    ) as mock_merge:
      await manager.finalize_async(self.directory)

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

    manager = metadata_manager.MetadataManager(
        primary_host=0,
        array_metadata_store=mock_store,
        pytree_metadata_options=self.options,
    )

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
      updated_param_infos = await manager.get_param_infos_with_write_shape(
          param_infos,
          self.directory,
      )

    self.assertEqual(updated_param_infos['x'].write_shape, (10, 20))

  def _make_param_infos(self):
    typestr = type_handler_registry_lib.get_param_typestr(
        jnp.zeros((10, 20), dtype=jnp.float32),
        self.registry,
        self.options,
    )
    return {
        'x': type_handlers.ParamInfo(
            name='x',
            parent_dir=self.directory,
            skip_deserialize=False,
            value_typestr=typestr,
        ),
    }

  def _make_store(self):
    mock_store = mock.MagicMock(spec=array_metadata_store_lib.Store)
    mock_array_metadata = mock.MagicMock()
    mock_array_metadata.param_name = 'x'
    mock_array_metadata.write_shape = (10, 20)
    mock_store.read = mock.AsyncMock(return_value=[mock_array_metadata])
    return mock_store

  @parameterized.named_parameters(
      # Worker 0 was not part of the save (e.g. its slice is down); worker 8
      # was elected and wrote the shared metadata.
      dict(testcase_name='worker_0_absent', elected=8),
      # Non-elastic / all slices up: worker 0 remains the worker primary host.
      dict(testcase_name='worker_0_present', elected=0),
  )
  async def test_get_param_infos_with_write_shape_reads_worker_primary_host(
      self, elected
  ):
    mock_store = self._make_store()
    manager = metadata_manager.MetadataManager(
        primary_host=0,
        array_metadata_store=mock_store,
        pytree_metadata_options=self.options,
    )
    dispatcher = mock.create_autospec(
        dispatchers.ColocatedPythonDispatcher, instance=True
    )
    devices = tuple(jax.devices()[:1])

    with mock.patch.object(
        pathways_multihost, 'worker_primary_host', return_value=elected
    ) as mock_elect, mock.patch.object(
        multihost, 'process_index', return_value=0
    ):
      updated_param_infos = await manager.get_param_infos_with_write_shape(
          self._make_param_infos(),
          self.directory,
          dispatcher=dispatcher,
          devices=devices,
      )

    self.assertEqual(updated_param_infos['x'].write_shape, (10, 20))
    mock_elect.assert_called_once_with(dispatcher, devices, 0)
    # Only the worker primary host's `ArrayMetadata` is read.
    mock_store.read.assert_awaited_once_with(
        self.directory, process_index=elected
    )

  async def test_get_param_infos_with_write_shape_raises_if_elected_missing(
      self,
  ):
    mock_store = self._make_store()
    mock_store.read = mock.AsyncMock(return_value=None)
    manager = metadata_manager.MetadataManager(
        primary_host=0,
        array_metadata_store=mock_store,
        pytree_metadata_options=self.options,
    )
    dispatcher = mock.create_autospec(
        dispatchers.ColocatedPythonDispatcher, instance=True
    )
    with mock.patch.object(
        pathways_multihost, 'worker_primary_host', return_value=8
    ), mock.patch.object(multihost, 'process_index', return_value=0):
      with self.assertRaisesRegex(
          ValueError, 'No ArrayMetadata found for process_index=8'
      ):
        await manager.get_param_infos_with_write_shape(
            self._make_param_infos(),
            self.directory,
            dispatcher=dispatcher,
            devices=tuple(jax.devices()[:1]),
        )

  @parameterized.named_parameters(
      # Multi-controller saves have no dispatcher.
      dict(testcase_name='no_dispatcher', use_dispatcher=False, devices=1),
      # A dispatcher but no jax.Array leaves (e.g. only numpy/scalars).
      dict(testcase_name='no_devices', use_dispatcher=True, devices=0),
  )
  async def test_get_param_infos_with_write_shape_reads_current_process(
      self, use_dispatcher, devices
  ):
    mock_store = self._make_store()
    manager = metadata_manager.MetadataManager(
        primary_host=0,
        array_metadata_store=mock_store,
        pytree_metadata_options=self.options,
    )
    dispatcher = None
    if use_dispatcher:
      dispatcher = mock.create_autospec(
          dispatchers.ColocatedPythonDispatcher, instance=True
      )
    with mock.patch.object(
        pathways_multihost, 'worker_primary_host'
    ) as mock_elect, mock.patch.object(
        multihost, 'process_index', return_value=0
    ):
      await manager.get_param_infos_with_write_shape(
          self._make_param_infos(),
          self.directory,
          dispatcher=dispatcher,
          devices=tuple(jax.devices()[:devices]),
      )
    mock_elect.assert_not_called()
    mock_store.read.assert_awaited_once_with(self.directory, process_index=0)



if __name__ == '__main__':
  absltest.main()
