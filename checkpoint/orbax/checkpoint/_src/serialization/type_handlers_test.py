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
import dataclasses
import sys
import threading
from typing import Any, Optional
import unittest

from absl import flags
from absl.testing import parameterized
from etils import epath
import jax
from jax.experimental import layout
from jax.experimental import mesh_utils
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint import test_utils
from orbax.checkpoint._src.arrays import sharding as arrays_sharding_lib
from orbax.checkpoint._src.metadata import array_metadata_store as array_metadata_store_lib
from orbax.checkpoint._src.multihost import dispatchers
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.multihost import multislice
from orbax.checkpoint._src.path import async_path
from orbax.checkpoint._src.path import atomicity
from orbax.checkpoint._src.serialization import jax_array_handlers
from orbax.checkpoint._src.serialization import limits
from orbax.checkpoint._src.serialization import ocdbt_utils
from orbax.checkpoint._src.serialization import replica_slices
from orbax.checkpoint._src.serialization import serialization
from orbax.checkpoint._src.serialization import tensorstore_utils as ts_utils
from orbax.checkpoint._src.serialization import type_handler_registry
from orbax.checkpoint._src.serialization import type_handlers
from orbax.checkpoint._src.serialization import types
from orbax.checkpoint._src.sharding_utils import make_single_device_sharding
from orbax.checkpoint._src.testing import multiprocess_test
from orbax.checkpoint._src.tree import utils as tree_utils

import tensorstore as ts

mock = unittest.mock
PyTree = Any
ParamInfo = type_handlers.ParamInfo
SaveArgs = type_handlers.SaveArgs
SingleReplicaArrayRestoreArgs = type_handlers.SingleReplicaArrayRestoreArgs
ArrayRestoreArgs = type_handlers.ArrayRestoreArgs
if jax.__version_info__ >= (0, 6, 2):
  Format = layout.Format
else:
  Format = layout.Layout
if jax.__version_info__ >= (0, 6, 3):
  DLL = layout.Layout
else:
  DLL = layout.DeviceLocalLayout  # type: ignore
PLACEHOLDER = type_handlers.PLACEHOLDER


FLAGS = flags.FLAGS

jax.config.update('jax_enable_x64', True)


async def _create_param_save_dir(param_info: ParamInfo):
  path = param_info.parent_dir / param_info.name
  if path is None or param_info.is_ocdbt_checkpoint:
    return
  if jax.process_index() == 0:
    await async_path.mkdir(path, parents=True)


async def _create_param_save_dirs(param_infos):
  await asyncio.gather(
      *jax.tree.flatten(
          jax.tree.map(
              _create_param_save_dir,
              param_infos,
          )
      )[0]
  )
  test_utils.sync_global_processes(
      'SingleReplicaArrayHandlerTest:create_param_save_dirs'
  )


def get_param_info(
    name: str,
    path: epath.Path,
    is_ocdbt: Optional[bool] = False,
    ts_context: Optional[ts.Context] = None,
    raise_array_data_missing_error: bool = True,
):
  return type_handlers.ParamInfo(
      name=name,
      parent_dir=path,
      is_ocdbt_checkpoint=is_ocdbt,
      ts_context=ts_context,
      raise_array_data_missing_error=raise_array_data_missing_error,
  )


def get_replica_pids(rep_id: int, mesh: jax.sharding.Mesh):
  """Return host and device IDs from specified replica from the mesh."""
  replica_devices = np.take(mesh.devices, rep_id, axis=0).flatten()
  pids = set([d.process_index for d in replica_devices])
  ids = set([d.id for d in replica_devices])
  return ids, pids


def per_host_write_size(value: Any) -> int:
  if not isinstance(value, jax.Array) and multihost.process_index() != 0:
    return 0

  if isinstance(value, np.ndarray):
    return value.size * value.dtype.itemsize
  elif isinstance(value, jax.Array):
    return sum(
        shard.data.size * shard.data.dtype.itemsize
        for shard in value.addressable_shards
        if shard.replica_id == 0
    )
  elif isinstance(value, str):
    return len(value)
  else:
    return sys.getsizeof(value)


def per_host_read_size(value: Any) -> int:
  if isinstance(value, np.ndarray):
    return value.size * value.dtype.itemsize
  elif isinstance(value, jax.Array):
    return sum(
        shard.data.size * shard.data.dtype.itemsize
        for shard in value.addressable_shards
    )
  elif isinstance(value, str):
    return len(value)
  else:
    return sys.getsizeof(value)


def per_host_size(value: Any) -> int:
  if isinstance(value, np.ndarray):
    return (
        value.size * value.dtype.itemsize
        if multihost.process_index() == 0
        else 0
    )
  elif isinstance(value, jax.Array):
    shards = value.addressable_shards
    total = 0
    for shard in shards:
      if shard.replica_id == 0:
        total += shard.data.size * shard.data.dtype.itemsize
    return total
  elif isinstance(value, str):
    return len(value) if multihost.process_index() == 0 else 0
  else:
    return sys.getsizeof(value) if multihost.process_index() == 0 else 0


class SerializationTest(
    unittest.IsolatedAsyncioTestCase,
    multiprocess_test.MultiProcessTest,
    parameterized.TestCase,
):
  """Captures aspects of serialization relevant to type_handlers."""

  def setUp(self):
    super().setUp()
    self.directory = epath.Path(
        self.multiprocess_create_tempdir(name='serialization_test')
    )
    assert jax.device_count() == 8
    assert jax.process_count() == 2

    test_utils.set_tensorstore_driver_for_test()

    test_utils.sync_global_processes('SerializationTest:setup_complete')

  def tearDown(self):
    test_utils.sync_global_processes('SerializationTest:tests_complete')
    super().tearDown()

  @parameterized.parameters(
      (0,),
      (1,),
  )
  async def test_array_serialization(self, primary_host):
    handler = type_handlers.ArrayHandler(primary_host=primary_host)
    sharding = jax.sharding.NamedSharding(
        mesh=jax.sharding.Mesh(
            devices=np.asarray(jax.devices()).reshape(2, 4),
            axis_names=('x', 'y'),
        ),
        spec=jax.sharding.PartitionSpec('y'),
    )
    # Array has 4 unique shards (A, B, C, D), each with 2 replicas. Importantly,
    # each host has all shards A, B, C, and D, but host 0 will have all shards
    # with replica_id=0, and host 1 will have all shards with replica_id=1. We
    # must ensure that the array is correctly saved even when the primary
    # host is selected as 1 (with all the non-primary shards).
    arr = jax.device_put(np.arange(32), sharding)
    info = get_param_info('a', self.directory)
    futures = await handler.serialize([arr], [info])
    for f in futures:
      f.result()
    test_utils.sync_global_processes('test_array_serialization:serialized')

    restored = await handler.deserialize(
        [info], [ArrayRestoreArgs(sharding=sharding)]
    )
    test_utils.assert_array_equal(self, arr, restored[0])

  @parameterized.product(
      enable_replica_parallel_separate_folder=(True, False),
      use_ocdbt=(True, False),
  )
  async def test_array_serialization_replicated_folder(
      self, enable_replica_parallel_separate_folder, use_ocdbt
  ):
    handler = type_handlers.ArrayHandler(
        primary_host=0,
        use_replica_parallel=True,
        enable_replica_parallel_separate_folder=enable_replica_parallel_separate_folder,
    )

    # build shardings
    mesh = jax.sharding.Mesh(
        devices=np.asarray(jax.devices()).reshape(2, 4),
        axis_names=('x', 'y'),
    )
    full_replicated_sharding = jax.sharding.NamedSharding(
        mesh=mesh,
        spec=jax.sharding.PartitionSpec(),  # Fully replicated
    )
    partial_replicated_sharding = jax.sharding.NamedSharding(
        mesh=mesh,
        spec=jax.sharding.PartitionSpec('x'),  # Fully replicated
    )
    sharded_sharding = jax.sharding.NamedSharding(
        mesh=mesh,
        spec=jax.sharding.PartitionSpec('x', 'y'),  # Fully replicated
    )

    # build arrays
    full_replicated_arr = jax.device_put(
        np.arange(32).reshape(4, 8), full_replicated_sharding
    )
    partial_replicated_arr = jax.device_put(
        np.arange(32).reshape(4, 8), partial_replicated_sharding
    )
    sharded_arr = jax.device_put(np.arange(32).reshape(4, 8), sharded_sharding)

    if use_ocdbt:
      ts_context = ts_utils.get_ts_context(use_ocdbt=use_ocdbt)
    else:
      ts_context = None

    # build ParamInfos
    full_replicated_info = get_param_info(
        'full_replicated',
        self.directory,
        is_ocdbt=use_ocdbt,
        ts_context=ts_context,
    )
    partial_replicated_info = get_param_info(
        'partial_replicated',
        self.directory,
        is_ocdbt=use_ocdbt,
        ts_context=ts_context,
    )
    sharded_replicated_info = get_param_info(
        'sharded_replicated',
        self.directory,
        is_ocdbt=use_ocdbt,
        ts_context=ts_context,
    )

    futures = await handler.serialize(
        [full_replicated_arr, partial_replicated_arr, sharded_arr],
        [
            full_replicated_info,
            partial_replicated_info,
            sharded_replicated_info,
        ],
    )

    for f in futures:
      f.result()
    test_utils.sync_global_processes(
        'test_array_serialization_replicated_folder:serialized'
    )

    if use_ocdbt:
      await ocdbt_utils.merge_ocdbt_per_process_files(
          self.directory,
          ts_context=ts_utils.get_ts_context(use_ocdbt=use_ocdbt),
          use_zarr3=False,
      )
      test_utils.sync_global_processes(
          'local_serialization:merge_ocdbt_complete'
      )

    replicated_dirs = list(
        self.directory.glob('*' + ts_utils.REPLICA_SUBDIR_SUFFIX + '*')
    )
    if enable_replica_parallel_separate_folder and use_ocdbt:
      self.assertNotEmpty(replicated_dirs)
    else:
      self.assertEmpty(replicated_dirs)

    restored = await handler.deserialize(
        [
            full_replicated_info,
            partial_replicated_info,
            sharded_replicated_info,
        ],
        [
            ArrayRestoreArgs(sharding=full_replicated_sharding),
            ArrayRestoreArgs(sharding=partial_replicated_sharding),
            ArrayRestoreArgs(sharding=sharded_sharding),
        ],
    )

    test_utils.assert_array_equal(self, full_replicated_arr, restored[0])
    test_utils.assert_array_equal(self, partial_replicated_arr, restored[1])
    test_utils.assert_array_equal(self, sharded_arr, restored[2])

  async def test_array_deserialization_with_custom_layout(self):
    handler = type_handlers.ArrayHandler()
    mesh = jax.sharding.Mesh(
        devices=np.asarray(jax.devices()).reshape(4, 2),
        axis_names=('x', 'y'),
    )
    np_inp = np.arange(32).reshape(8, 4)
    s = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('x', 'y'))
    arr = jax.device_put(np_inp, s)

    info = get_param_info('a', self.directory)
    futures = await handler.serialize([arr], [info])
    for f in futures:
      f.result()
    test_utils.sync_global_processes('test_array_serialization:serialized')

    arr_layout = arrays_sharding_lib.get_device_local_layout(arr)
    in_layout = Format(arr_layout, sharding=s)
    restored = await handler.deserialize(
        [info], [ArrayRestoreArgs(sharding=in_layout)]
    )
    self.assertEqual(restored[0].format, arr.format)

    # flip the layout
    custom_layout = Format(
        DLL(
            arr_layout.major_to_minor[::-1],
            arr_layout.tiling,
        ),
        sharding=arr.sharding,
    )
    restored = await handler.deserialize(
        [info], [ArrayRestoreArgs(sharding=custom_layout)]
    )

    self.assertEqual(restored[0].format, custom_layout)
    self.assertEqual(
        arr_layout.major_to_minor,
        arrays_sharding_lib.get_device_local_layout(restored[0]).major_to_minor[
            ::-1
        ],
    )

  @parameterized.product(
      use_ocdbt=(True, False),
      raise_array_data_missing_error=(True, False),
      use_zarr3=(True, False),
  )
  async def test_local_serialization(
      self, use_ocdbt, raise_array_data_missing_error, use_zarr3
  ):
    self.assertEqual(multihost.process_count(), 2)
    directory = self.directory / f'process_{multihost.process_index()}'
    directory.mkdir(parents=False, exist_ok=False)

    handler = type_handlers.ArrayHandler(
        primary_host=None, replica_id=None, use_replica_parallel=False
    )
    sharding = jax.sharding.NamedSharding(
        mesh=jax.sharding.Mesh(
            devices=np.asarray(jax.devices()),
            axis_names=('x',),
        ),
        spec=jax.sharding.PartitionSpec('x'),
    )
    # 8 shards, each of length 4.
    arr = jax.device_put(np.arange(32, dtype=np.int32), sharding)
    zeros_arr = jax.device_put(np.zeros((32,), dtype=np.int32), sharding)
    ts_context = ts_utils.get_ts_context(use_ocdbt=use_ocdbt)
    info = get_param_info(
        'a',
        directory,
        is_ocdbt=use_ocdbt,
        ts_context=ts_context,
        raise_array_data_missing_error=raise_array_data_missing_error,
    )
    futures = await handler.serialize([arr], [info])
    for f in futures:
      f.result()
    test_utils.sync_global_processes('test_array_serialization:serialized')
    if use_ocdbt:
      await ocdbt_utils.merge_ocdbt_per_process_files(
          directory, ts_context=ts_context, use_zarr3=use_zarr3
      )
      test_utils.sync_global_processes(
          'local_serialization:merge_ocdbt_complete'
      )

    restored = await handler.deserialize(
        [info], [ArrayRestoreArgs(sharding=sharding)]
    )
    test_utils.assert_array_equal(self, arr, restored[0])

    orig_get_device_to_index_map = serialization._get_device_to_index_map

    def shuffled_get_device_to_index_map(global_shape, sharding):
      device_to_index_map = orig_get_device_to_index_map(global_shape, sharding)
      processes = [d.process_index for d in device_to_index_map.keys()]
      assert processes == sorted(processes)
      devices = list(device_to_index_map.keys())
      devices.reverse()
      return dict(zip(devices, device_to_index_map.values()))

    with mock.patch.object(
        serialization,
        '_get_device_to_index_map',
        new=shuffled_get_device_to_index_map,
    ):
      if raise_array_data_missing_error:
        with self.assertRaisesRegex(
            Exception, 'Encountered error while reading array index'
        ):
          await handler.deserialize(
              [info], [ArrayRestoreArgs(sharding=sharding)]
          )
      else:
        restored = await handler.deserialize(
            [info], [ArrayRestoreArgs(sharding=sharding)]
        )
        test_utils.assert_array_equal(self, zeros_arr, restored[0])

  async def test_array_serialization_with_random_key(self):
    """Tests that JAX random keys are serialized and deserialized correctly."""
    store = array_metadata_store_lib.Store()
    handler = type_handlers.ArrayHandler(array_metadata_store=store)

    full_sharding = jax.sharding.NamedSharding(
        jax.sharding.Mesh(
            devices=jax.devices(),
            axis_names=('x',),
        ),
        jax.sharding.PartitionSpec('x'),
    )

    duplicated_sharding = jax.sharding.NamedSharding(
        jax.sharding.Mesh(
            devices=jax.devices(),
            axis_names=('x',),
        ),
        jax.sharding.PartitionSpec(),
    )

    arr = test_utils.create_sharded_array(
        np.arange(32),
        full_sharding.mesh,
        full_sharding.spec,
    )

    keys = [
        # new style JAX random keys
        jax.random.key(jnp.array(1, device=duplicated_sharding)),
        jax.random.key(jnp.array(2, device=duplicated_sharding), impl='rbg'),
        jax.random.key(
            jnp.array(3, device=duplicated_sharding), impl='unsafe_rbg'
        ),
        jax.random.key(
            jnp.array(4, device=duplicated_sharding), impl='threefry2x32'
        ),
        # legacy PRNG keys
        jax.random.PRNGKey(jnp.array(5, device=duplicated_sharding)),
        jax.random.PRNGKey(
            jnp.array(6, device=duplicated_sharding), impl='rbg'
        ),
        jax.random.PRNGKey(
            jnp.array(7, device=duplicated_sharding), impl='unsafe_rbg'
        ),
        jax.random.PRNGKey(
            jnp.array(8, device=duplicated_sharding), impl='threefry2x32'
        ),
    ] + [
        jax.random.split(
            jax.random.key(jnp.array(10, device=duplicated_sharding))
        )
    ]

    values = [
        arr,
    ] + keys
    infos = [
        get_param_info(f'item{i}', self.directory) for i in range(len(values))
    ]

    futures = await handler.serialize(values, infos)
    for f in futures:
      f.result()
    test_utils.sync_global_processes('test_array_serialization:serialized')

    restored = await handler.deserialize(
        infos,
        [ArrayRestoreArgs(sharding=full_sharding)]
        + [ArrayRestoreArgs(sharding=duplicated_sharding)] * len(keys),
    )

    test_utils.assert_array_equal(self, arr, restored[0])


class UtilsTest(
    unittest.IsolatedAsyncioTestCase,
    multiprocess_test.MultiProcessTest,
    parameterized.TestCase,
):

  @parameterized.named_parameters(
      dict(
          testcase_name='empty_kv_store_zarr',
          ts_params=[],
          use_zarr3=False,
          expected_error=None,
      ),
      dict(
          testcase_name='missing_zarray',
          ts_params=['a/0'],
          use_zarr3=False,
          expected_error=r'1\/1 params are missing \.zarray',
      ),
      dict(
          testcase_name='missing_zarray_shards',
          ts_params=['a/0.1.1.1', 'a/0.0.0.0'],
          use_zarr3=False,
          expected_error=r'1\/1 params are missing \.zarray',
      ),
      dict(
          testcase_name='missing_zarray_mixed',
          ts_params=['a/0.1.1.1', 'a/0.0.0.0', 'b/0', 'b/.zarray'],
          use_zarr3=False,
          expected_error=r'1\/2 params are missing \.zarray',
      ),
      dict(
          testcase_name='missing_param',
          ts_params=['a/.zarray'],
          use_zarr3=False,
          expected_error=r'1\/1 params are missing in checkpoint',
      ),
      dict(
          testcase_name='missing_param_mixed',
          ts_params=['a/.zarray', 'b/.zarray', 'b/0', 'b/1', 'c/1.1', 'c/0.0'],
          use_zarr3=False,
          expected_error=r'1\/3 params are missing in checkpoint',
      ),
      dict(
          testcase_name='empty_kv_store_zarr3',
          ts_params=[],
          use_zarr3=True,
          expected_error=None,
      ),
      dict(
          testcase_name='missing_zarray_zarr3',
          ts_params=['a/0'],
          use_zarr3=True,
          expected_error=None,  # zarr3 not yet implemented.
      ),
      dict(
          testcase_name='missing_param_zarr3',
          ts_params=['a/.zarray'],
          use_zarr3=True,
          expected_error=None,  # zarr3 not yet implemented.
      ),
  )
  async def test_validate_params(
      self, ts_params: list[str], use_zarr3: bool, expected_error: Optional[str]
  ):
    ts_kv_store = mock.create_autospec(ts.KvStore)
    ts_kv_store.list = mock.AsyncMock(
        return_value=[t.encode('utf-8') for t in ts_params]
    )
    if expected_error is not None:
      with self.assertRaisesRegex(ValueError, expected_error):
        await ocdbt_utils._validate_params(ts_kv_store, use_zarr3)
    else:
      await ocdbt_utils._validate_params(ts_kv_store, use_zarr3)


class NumpyHandlerTest(
    unittest.IsolatedAsyncioTestCase,
    multiprocess_test.MultiProcessTest,
    parameterized.TestCase,
):
  """Test class."""

  def test_memory_size(self):
    handler = type_handlers.NumpyHandler()
    if multihost.process_index() == 0:
      values = [np.arange(8, dtype=np.int32)]
    else:
      values = [np.arange(16, dtype=np.int32)]
    write_sizes, read_sizes = zip(*handler.memory_size(values))
    self.assertSequenceEqual(
        write_sizes, [per_host_write_size(v) for v in values]
    )
    self.assertSequenceEqual(
        read_sizes, [per_host_read_size(v) for v in values]
    )

  async def test_metadata(self):
    if multihost.process_index() != 0:
      self.skipTest('Only run on host 0')

    handler = type_handlers.NumpyHandler()
    values = [
        np.arange(8, dtype=np.int32),
        np.arange(12, dtype=np.float32).reshape((3, 4)),
    ]
    path = epath.Path(self.create_tempdir().full_path)
    ts_context = ts_utils.get_ts_context()
    param_infos = [
        get_param_info(
            str(i),
            path,
            is_ocdbt=True,
            ts_context=ts_context,
        )
        for i in range(len(values))
    ]
    commit_futures = await handler.serialize(values, param_infos)
    for f in commit_futures:
      f.result()
    await ocdbt_utils.merge_ocdbt_per_process_files(
        path, ts_context=ts_context, use_zarr3=False
    )

    metadatas = await handler.metadata(param_infos)
    self.assertListEqual(
        [m.shape for m in metadatas], [a.shape for a in values]
    )
    self.assertListEqual(
        [m.dtype for m in metadatas], [a.dtype for a in values]
    )
    # Storage metadata.
    chunk_shapes = []
    for m in metadatas:
      assert m.storage is not None
      chunk_shapes.append(m.storage.chunk_shape)
    self.assertListEqual(chunk_shapes, [v.shape for v in values])


class ScalarHandlerTest(parameterized.TestCase):
  """Test class."""

  def test_memory_size(self):
    handler = type_handlers.ScalarHandler()
    values = [3]
    write_sizes, read_sizes = zip(*handler.memory_size(values))
    self.assertSequenceEqual(
        write_sizes, [per_host_write_size(v) for v in values]
    )
    self.assertSequenceEqual(
        read_sizes, [per_host_read_size(v) for v in values]
    )


class StringHandlerTest(parameterized.TestCase):
  """Test class."""

  def test_memory_size(self):
    handler = type_handlers.StringHandler()
    values = ['a', 'foobar']
    write_sizes, read_sizes = zip(*handler.memory_size(values))
    self.assertSequenceEqual(
        write_sizes, [per_host_write_size(v) for v in values]
    )
    self.assertSequenceEqual(
        read_sizes, [per_host_read_size(v) for v in values]
    )


class ArrayHandlerTest(parameterized.TestCase):
  """Test class."""

  def setUp(self):
    super().setUp()
    self.pytree, _, _ = test_utils.setup_sharded_pytree()

  def test_memory_size(self):
    handler = type_handlers.ArrayHandler(use_replica_parallel=True)
    values = jax.tree.leaves(self.pytree)
    write_sizes, read_sizes = zip(*handler.memory_size(values))
    self.assertSequenceEqual(write_sizes, [32, 64, 32, 64])
    self.assertSequenceEqual(read_sizes, [256, 64, 32, 64])


class ArrayHandlerCallbackTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):

  def setUp(self):
    super().setUp()
    self.pytree, _, _ = test_utils.setup_sharded_pytree()

  class TestCallback(types.NoopSerializationStatusCallback):

    def __init__(self, priority: types.TransferPriority):
      self.priority = priority
      self.events = []

    def key_priority(self, keypath) -> types.TransferPriority:
      self.events.append(('register', tree_utils.str_keypath(keypath)))
      return self.priority

    def on_transfer_end(self, keypath) -> None:
      self.events.append(('on_transfer_end', tree_utils.str_keypath(keypath)))

    def on_write_end(self, keypath) -> None:
      self.events.append(('on_write_end', tree_utils.str_keypath(keypath)))

  def _setup_serialize_args(self):
    directory = epath.Path(self.create_tempdir().full_path)
    arr = jax.tree.leaves(self.pytree)[0]
    info = get_param_info('a', directory)
    info = info.replace(
        keypath=(jax.tree_util.DictKey('params'), jax.tree_util.DictKey('a')),
        device_host_byte_limiter=limits.LimitInFlightBytes(16),
    )
    return arr, info

  @parameterized.parameters(
      types.TransferPriority.SYNCHRONOUS,
      types.TransferPriority.ASYNCHRONOUS_DEPRIORITIZED,
  )
  async def test_callback_traversal(self, priority):
    cb = self.TestCallback(priority)
    handler = type_handlers.ArrayHandler(
        callback=cb, use_replica_parallel=False
    )
    arr, info = self._setup_serialize_args()

    futures = await handler.serialize([arr], [info])
    for f in futures:
      f.result()

    self.assertEqual(
        cb.events,
        [
            ('register', ('params', 'a')),
            ('on_transfer_end', ('params', 'a')),
            ('on_write_end', ('params', 'a')),
        ],
    )

  @parameterized.parameters(
      types.TransferPriority.SYNCHRONOUS,
      types.TransferPriority.ASYNCHRONOUS_DEPRIORITIZED,
  )
  async def test_callback_traversal_without_dispatcher(self, priority):
    cb = self.TestCallback(priority)
    handler = type_handlers.ArrayHandler(
        callback=cb, use_replica_parallel=False
    )
    arr, info = self._setup_serialize_args()

    real_transfer = replica_slices.transfer_arrays_to_host
    real_serialize = serialization.async_serialize_from_host

    def mock_transfer(*args, **kwargs):
      cb.events.append(('transfer_call', ('params', 'a')))
      return real_transfer(*args, **kwargs)

    async def mock_serialize(*args, **kwargs):
      cb.events.append(('write_call', ('params', 'a')))
      return await real_serialize(*args, **kwargs)

    with mock.patch.object(
        replica_slices, 'transfer_arrays_to_host', side_effect=mock_transfer
    ), mock.patch.object(
        serialization, 'async_serialize_from_host', side_effect=mock_serialize
    ):
      futures = await handler.serialize([arr], [info])
      for f in futures:
        f.result()

    expected_events = [
        ('register', ('params', 'a')),
        ('transfer_call', ('params', 'a')),
        ('on_transfer_end', ('params', 'a')),
        ('write_call', ('params', 'a')),
        ('on_write_end', ('params', 'a')),
    ]
    self.assertEqual(cb.events, expected_events)

  @parameterized.parameters(
      types.TransferPriority.SYNCHRONOUS,
      types.TransferPriority.ASYNCHRONOUS_DEPRIORITIZED,
  )
  async def test_callback_traversal_with_dispatcher(self, priority):
    cb = self.TestCallback(priority)
    mock_dispatcher = mock.create_autospec(dispatchers.Dispatcher)
    mock_dispatcher.device_to_host.side_effect = lambda x: x

    def mock_dispatch(*args, **kwargs):
      del args, kwargs  # Unused.
      cb.events.append(('transfer_call', ('params', 'a')))
      cb.events.append(('write_call', ('params', 'a')))
      return jnp.array([1.0])

    def mock_batches(deprioritized_params, **kwargs):
      del kwargs  # Unused.
      arrays, infos, args = zip(*deprioritized_params)
      yield list(arrays), list(infos), list(args)

    mock_dispatcher.dispatch.side_effect = mock_dispatch
    handler = type_handlers.ArrayHandler(
        dispatcher=mock_dispatcher, callback=cb
    )
    arr, info = self._setup_serialize_args()

    # Mock _get_deprioritized_batches_to_serialize to avoid the memory budget
    # calculation, which attempts to calculate distributed mem usage
    with mock.patch.object(
        jax_array_handlers,
        '_get_deprioritized_batches_to_serialize',
        side_effect=mock_batches,
    ):
      futures = await handler.serialize([arr], [info])
      for f in futures:
        f.result()

    # Note: 'write_call' appears before 'on_transfer_end' because fake_dispatch
    # mock executes fully in a single synchronous step before returning.
    # In reality, on_transfer_end would execute after the
    # host initiates the operation but before the remote write completes.
    self.assertEqual(
        cb.events,
        [
            ('register', ('params', 'a')),
            ('transfer_call', ('params', 'a')),
            ('write_call', ('params', 'a')),
            ('on_transfer_end', ('params', 'a')),
            ('on_write_end', ('params', 'a')),
        ],
    )


class PlaceholderHandlerTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):
  """Test class."""

  async def test_deserialize(self):
    handler = type_handlers.PlaceholderHandler()
    values = [PLACEHOLDER, PLACEHOLDER]
    path = epath.Path(self.create_tempdir().full_path)
    param_infos = [
        get_param_info(
            str(i),
            path,
        )
        for i in range(len(values))
    ]
    restored = await handler.deserialize(param_infos)
    self.assertListEqual(restored, values)


@dataclasses.dataclass
class SingleReplicaTestConfig:
  mesh: jax.sharding.Mesh
  np_arrays: list[np.ndarray]
  partition_specs: list[jax.sharding.PartitionSpec]
  replica_axis_index: int = 0
  primary_replica_id: int = 0
  is_ocdbt: bool = True
  set_memory_limit: Optional[str] = None
  use_replica_parallel: bool = True
  enable_write_sharding_file: bool = True
  array_metadata_store: array_metadata_store_lib.Store | None = None
  active_mesh_on_restore: bool = False

  @property
  def arrays(self) -> list[jax.Array]:
    assert len(self.np_arrays) == len(self.partition_specs)
    return [
        test_utils.create_sharded_array(arr, self.mesh, pspec)
        for arr, pspec in zip(self.np_arrays, self.partition_specs)
    ]


class SingleReplicaArrayHandlerTest(
    unittest.IsolatedAsyncioTestCase,
    multiprocess_test.MultiProcessTest,
    parameterized.TestCase,
):
  """Test class."""

  def setUp(self):
    super().setUp()

    self.directory = epath.Path(
        self.create_tempdir(name='type_handler_test').full_path
    )
    assert jax.device_count() == 8
    assert jax.process_count() == 2

    test_utils.set_tensorstore_driver_for_test()

    test_utils.sync_global_processes(
        'SingleReplicaArrayHandlerTest:setup_complete'
    )

  def tearDown(self):
    test_utils.sync_global_processes(
        'SingleReplicaArrayHandlerTest:tests_complete'
    )
    super().tearDown()

  def get_2d_arrays(
      self, array_size: int
  ) -> tuple[list[np.ndarray], list[jax.sharding.PartitionSpec]]:
    return [
        np.arange(array_size * 8).reshape((8, array_size)) * 1,
        np.arange(array_size * 16).reshape((16, array_size)) * 2,
        np.arange(2 * array_size * 8).reshape((8 * 2, array_size)) * 3,
        np.arange(3 * array_size * 16).reshape((16, 3 * array_size)) * 4,
    ], [
        jax.sharding.PartitionSpec(None, 'y'),
        jax.sharding.PartitionSpec(None, 'y'),
        jax.sharding.PartitionSpec(None, 'y'),
        jax.sharding.PartitionSpec(None, 'y'),
    ]

  def validate_values(self, restored, values):
    if not isinstance(type(restored), tuple):
      restored = tuple(restored)
    test_utils.assert_tree_equal(self, restored, tuple(values))

  async def single_replica_serialize_deserialize(
      self,
      config: SingleReplicaTestConfig,
  ):
    """Test single replica serialize and deserialize."""
    arrays = config.arrays
    mesh = config.mesh
    mesh_axes = config.partition_specs
    replica_axis_index = config.replica_axis_index
    primary_replica_id = config.primary_replica_id
    is_ocdbt = config.is_ocdbt
    set_memory_limit = config.set_memory_limit

    if not (replica_axis_index == 0 or replica_axis_index == 1):
      raise ValueError(
          f'Unsupported replica_axis_index: {replica_axis_index}. '
          'Can be 0 or 1.'
      )
    ts_context = ts_utils.get_ts_context(use_ocdbt=is_ocdbt)

    directory = epath.Path(
        self.multiprocess_create_tempdir(
            name=f'type_handler_test_w_memory_limit_{primary_replica_id}'
        )
    )
    param_infos = [
        get_param_info(
            str(i), directory, is_ocdbt=is_ocdbt, ts_context=ts_context
        )
        for i in range(len(arrays))
    ]
    if set_memory_limit is None:
      broadcast_memory_limit = None
    else:
      mem_per_leafs = [multislice.get_leaf_memory_per_device(a) for a in arrays]
      if set_memory_limit == 'half':
        broadcast_memory_limit = sum(mem_per_leafs) // 2
      elif set_memory_limit == 'leaf':
        mem_per_leafs.sort()
        # pick memory limit between the largest and second largest leaf sizes
        memory_offset = (mem_per_leafs[-1] - mem_per_leafs[-2]) // 2
        broadcast_memory_limit = mem_per_leafs[-1] - memory_offset
      else:
        raise ValueError(
            f'Unknown value for set_memory_limit: {set_memory_limit}. '
            'Can be None, `half` or `leaf`'
        )

    handler = type_handlers.SingleReplicaArrayHandler(
        replica_axis_index=replica_axis_index,
        primary_replica_id=primary_replica_id,
        broadcast_memory_limit_bytes=broadcast_memory_limit,
        use_replica_parallel=config.use_replica_parallel,
        enable_write_sharding_file=config.enable_write_sharding_file,
        array_metadata_store=config.array_metadata_store,
    )
    await _create_param_save_dirs(param_infos)  # Does nothing if is_ocdbt=True.
    commit_futures = await handler.serialize(arrays, param_infos)

    for f in commit_futures:
      f.result()
    test_utils.sync_global_processes('SingleReplicaArrayHandlerTest:serialized')
    if is_ocdbt:
      if multihost.process_index() == 0:
        await ocdbt_utils.merge_ocdbt_per_process_files(
            directory, ts_context=ts_context, use_zarr3=False
        )
      test_utils.sync_global_processes('merge_ocdbt_complete')

    restore_args = [
        test_utils.create_single_replica_restore_args(
            arr,
            mesh,
            axes,
        )
        for arr, axes in zip(arrays, mesh_axes)
    ]
    num_replicas = mesh.devices.shape[replica_axis_index]
    with mock.patch.object(
        multislice, 'slice_count', return_value=num_replicas
    ):
      if config.active_mesh_on_restore:
        if hasattr(jax, 'set_mesh'):
          with jax.set_mesh(mesh):
            restored = await handler.deserialize(param_infos, restore_args)
        else:
          with mesh:
            restored = await handler.deserialize(param_infos, restore_args)
      else:
        restored = await handler.deserialize(param_infos, restore_args)
    test_utils.sync_global_processes(
        'SingleReplicaArrayHandlerTest:deserialization_complete'
    )
    self.validate_values(restored, arrays)

    # Check the metadata reported by the handler is as expected.
    metadatas = await handler.metadata(param_infos)
    self.assertListEqual(
        [m.shape for m in metadatas], [a.shape for a in arrays]
    )
    self.assertListEqual(
        [m.dtype for m in metadatas], [a.dtype for a in arrays]
    )
    # Storage metadata.
    chunk_shapes = []
    for m in metadatas:
      assert m.storage is not None
      chunk_shapes.append(m.storage.chunk_shape)
    expected_chunk_shapes = [
        test_utils.get_expected_chunk_shape(
            a, use_replica_parallel=config.use_replica_parallel
        )
        for a in arrays
    ]
    self.assertListEqual(
        chunk_shapes,
        expected_chunk_shapes,
    )

    test_utils.sync_global_processes('done with a single test')

  @parameterized.parameters(True, False)
  async def test_single_replica_serialize_deserialize_no_ocdbt(
      self, use_replica_parallel
  ):
    mesh = jax.sharding.Mesh(
        np.asarray(jax.devices()).reshape(2, 4), ('x', 'y')
    )
    arrays = [np.arange(64).reshape(8, 8)]
    mesh_axes = [jax.sharding.PartitionSpec(None, 'y')]
    config = SingleReplicaTestConfig(
        mesh=mesh,
        np_arrays=arrays,
        partition_specs=mesh_axes,
        is_ocdbt=False,
        use_replica_parallel=use_replica_parallel,
    )
    await self.single_replica_serialize_deserialize(config)

  @parameterized.product(
      mesh_shape=((2, 4), (2, 2, 2)),
      primary_replica_id=(0, 1),
  )
  async def test_single_replica_serialize_deserialize(
      self,
      mesh_shape,
      primary_replica_id,
  ):
    assert len(mesh_shape) == 2 or len(mesh_shape) == 3
    axis_names = ('x', 'y') if len(mesh_shape) == 2 else ('x', 'y', 'z')
    mesh = jax.sharding.Mesh(
        np.asarray(jax.devices()).reshape(mesh_shape), axis_names
    )
    arrays = [
        np.arange(64).reshape(8, 8),
        np.arange(32),
        np.asarray(42),
    ]
    mesh_axes = [
        jax.sharding.PartitionSpec(None, 'y'),
        jax.sharding.PartitionSpec(None),
        jax.sharding.PartitionSpec(),
    ]
    if len(mesh_shape) == 3:
      arrays.append(np.arange(64).reshape(4, 4, 4))
      mesh_axes.append(jax.sharding.PartitionSpec(None, 'y', 'z'))
    config = SingleReplicaTestConfig(
        mesh=mesh,
        np_arrays=arrays,
        partition_specs=mesh_axes,
        primary_replica_id=primary_replica_id,
    )
    await self.single_replica_serialize_deserialize(config)

  @parameterized.product(
      ici_dcn_mesh_shapes=(((4, 1), (1, 2)), ((2, 1, 2), (1, 2, 1))),
      primary_replica_id=(0, 1),
  )
  async def test_single_replica_serialize_deserialize_replica_axis_one(
      self,
      ici_dcn_mesh_shapes,
      primary_replica_id,
  ):
    ici_mesh_shape, dcn_mesh_shape = ici_dcn_mesh_shapes
    mesh_devices = mesh_utils.create_hybrid_device_mesh(
        ici_mesh_shape, dcn_mesh_shape, process_is_granule=True
    )
    mesh_shape = mesh_devices.shape
    assert len(mesh_shape) == 2 or len(mesh_shape) == 3
    axis_names = ('x', 'y') if len(mesh_shape) == 2 else ('x', 'y', 'z')
    mesh = jax.sharding.Mesh(mesh_devices, axis_names)
    arrays = [
        np.arange(64).reshape(8, 8),
        np.arange(32),
        np.asarray(42),
    ]
    mesh_axes = [
        jax.sharding.PartitionSpec('x', None),
        jax.sharding.PartitionSpec(None),
        jax.sharding.PartitionSpec(),
    ]
    if len(mesh_shape) == 3:
      arrays.append(np.arange(64).reshape(4, 4, 4))
      mesh_axes.append(jax.sharding.PartitionSpec('x', None, 'z'))
    config = SingleReplicaTestConfig(
        mesh=mesh,
        np_arrays=arrays,
        partition_specs=mesh_axes,
        primary_replica_id=primary_replica_id,
        replica_axis_index=1,
    )
    await self.single_replica_serialize_deserialize(config)

  @parameterized.parameters(0, 1)
  async def test_single_replica_deserialize_under_active_mesh(
      self, primary_replica_id
  ):
    if len(jax.devices()) < 2:
      self.skipTest('Need at least 2 devices for this test')

    mesh = jax.sharding.Mesh(
        np.asarray(jax.devices()).reshape(2, len(jax.devices()) // 2),
        ('x', 'y'),
    )
    arrays = [np.arange(64).reshape(8, 8)]
    mesh_axes = [jax.sharding.PartitionSpec(None, 'y')]
    config = SingleReplicaTestConfig(
        mesh=mesh,
        np_arrays=arrays,
        partition_specs=mesh_axes,
        is_ocdbt=False,
        active_mesh_on_restore=True,
        primary_replica_id=primary_replica_id,
    )
    await self.single_replica_serialize_deserialize(config)

  async def test_validate_sharding_not_named_sharding_throws_error(self):
    with self.assertRaisesRegex(
        type_handlers.InvalidShardingError,
        'The provided sharding is not a NamedSharding',
    ):
      jax_array_handlers._validate_sharding_and_get_primary_replica_processes(
          replica_axis_index=0,
          primary_replica_id=0,
          sharding=make_single_device_sharding(jax.devices()[0]),
      )

  async def test_validate_sharding_all_devices_in_primary_replica_throws_error(
      self,
  ):
    mesh = jax.sharding.Mesh(
        np.asarray(jax.devices()).reshape(1, 8), ('x', 'y')
    )
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    with self.assertRaisesRegex(
        type_handlers.InvalidShardingError,
        'All devices are in the primary replica',
    ):
      jax_array_handlers._validate_sharding_and_get_primary_replica_processes(
          replica_axis_index=0,
          primary_replica_id=0,
          sharding=sharding,
      )

  async def test_validate_sharding_success(self):
    mesh = jax.sharding.Mesh(
        np.asarray(jax.devices()).reshape(2, 4), ('x', 'y')
    )
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    pids = (
        jax_array_handlers._validate_sharding_and_get_primary_replica_processes(
            replica_axis_index=0,
            primary_replica_id=0,
            sharding=sharding,
        )
    )
    self.assertEqual(pids, {0})

  async def test_single_replica_half_memory_limit(self):
    """Test setting memory limit for broadcasting to be half of pytree size."""
    array_size = 16000
    mesh = jax.sharding.Mesh(
        np.asarray(jax.devices()).reshape(2, 4), ('x', 'y')
    )
    arrays, mesh_axes = self.get_2d_arrays(array_size)
    config = SingleReplicaTestConfig(
        mesh=mesh,
        np_arrays=arrays,
        partition_specs=mesh_axes,
        set_memory_limit='half',
    )
    await self.single_replica_serialize_deserialize(config)

  async def test_leaf_larger_than_memory_limit(self):
    """Test setting memory limit to be smaller than the largest leaf size."""
    array_size = 16000
    mesh = jax.sharding.Mesh(
        np.asarray(jax.devices()).reshape(2, 4), ('x', 'y')
    )
    arrays, mesh_axes = self.get_2d_arrays(array_size)
    config = SingleReplicaTestConfig(
        mesh=mesh,
        np_arrays=arrays,
        partition_specs=mesh_axes,
        set_memory_limit='leaf',
    )
    await self.single_replica_serialize_deserialize(config)


class RegistryTest(parameterized.TestCase):

  def test_type_handler_registration(self):
    type_handler_registry.register_type_handler(
        jax.Array, type_handlers.SingleReplicaArrayHandler(), override=True
    )
    self.assertIsInstance(
        type_handler_registry.get_type_handler(jax.Array),
        type_handlers.SingleReplicaArrayHandler,
    )

  def test_missing_global_type_handler(self):
    self.assertRaises(
        ValueError, type_handler_registry.get_type_handler, 'unknown type'
    )

  def test_local_type_handler_registration(self):
    class TypeA:
      pass

    class TypeB:
      pass

    class MissingType:
      pass

    registry = type_handler_registry.create_type_handler_registry(
        (TypeA, type_handlers.SingleReplicaArrayHandler()),
    )

    registry.add(
        TypeB,
        type_handlers.SingleReplicaArrayHandler(),
        override=True,
    )
    self.assertIsInstance(
        registry.get(TypeA),
        type_handlers.SingleReplicaArrayHandler,
    )
    self.assertIsInstance(
        registry.get(TypeB),
        type_handlers.SingleReplicaArrayHandler,
    )
    self.assertRaises(
        ValueError,
        type_handler_registry.get_type_handler,
        TypeA,
    )
    self.assertRaises(
        ValueError,
        type_handler_registry.get_type_handler,
        TypeB,
    )
    self.assertRaises(
        ValueError,
        registry.get,
        MissingType,
    )


class _TrackedDeferredPath(atomicity.DeferredPath):

  def __init__(self):
    super().__init__()
    self.await_creation_entered = threading.Event()

  async def await_creation(self):
    self.await_creation_entered.set()
    return await super().await_creation()


class DeferredPathResolutionTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    test_utils.set_tensorstore_driver_for_test()
    self.directory = epath.Path(self.create_tempdir().full_path)
    self.param_name = 'a'

  def _make_unresolved_deferred_info(self):
    deferred = _TrackedDeferredPath()
    info = ParamInfo(
        name=self.param_name,
        parent_dir=deferred,
        is_ocdbt_checkpoint=False,
    )
    return info, deferred

  # TODO(nikhilbansall): Open source this test.


class ParamInfoTest(parameterized.TestCase):

  def test_replace_returns_copy_with_updated_field(self):
    path = epath.Path('/tmp/test')
    info = ParamInfo(name='param', parent_dir=path)
    new_info = info.replace(byte_limiter='limiter')
    self.assertEqual(new_info.byte_limiter, 'limiter')
    self.assertIsNone(info.byte_limiter)

  def test_replace_preserves_other_fields(self):
    path = epath.Path('/tmp/test')
    info = ParamInfo(
        name='param',
        parent_dir=path,
        use_compression=False,
        use_zarr3=True,
    )
    new_info = info.replace(byte_limiter='limiter')
    self.assertEqual(new_info.name, 'param')
    self.assertEqual(new_info.parent_dir, path)
    self.assertFalse(new_info.use_compression)
    self.assertTrue(new_info.use_zarr3)

  def test_replace_updates_multiple_fields(self):
    path = epath.Path('/tmp/test')
    new_path = epath.Path('/tmp/new')
    info = ParamInfo(name='param', parent_dir=path)
    new_info = info.replace(parent_dir=new_path, use_zarr3=True)
    self.assertEqual(new_info.parent_dir, new_path)
    self.assertTrue(new_info.use_zarr3)
    self.assertEqual(info.parent_dir, path)


if __name__ == '__main__':
  multiprocess_test.main()
