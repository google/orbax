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

import logging
import os
import time
from unittest import mock

from absl import flags
from absl.testing import parameterized
from etils import epath
import jax
import numpy as np
from orbax.checkpoint import test_utils
from orbax.checkpoint import utils
from orbax.checkpoint._src.handlers import pytree_checkpoint_handler
from orbax.checkpoint._src.multihost import dispatchers
from orbax.checkpoint._src.serialization import jax_array_handlers
from orbax.checkpoint._src.serialization import pathways_handler_registry
from orbax.checkpoint._src.serialization import type_handler_registry
from orbax.checkpoint._src.serialization import types as serialization_types
from orbax.checkpoint._src.serialization import worker_memory_utils
from orbax.checkpoint._src.tree import utils as tree_utils

from .learning.deepmind.jax.ocean.remote_python import rp
from .pyglib.contrib.g3_multiprocessing import g3_multiprocessing
from absl.testing import absltest

USE_COLOCATED_PYTHON = flags.DEFINE_boolean(
    'use_colocated_python',
    False,
    'Whether to use colocated Python.',
)

FLAGS = flags.FLAGS
PyTreeCheckpointHandler = test_utils.PyTreeCheckpointHandler
PyTreeSaveArgs = pytree_checkpoint_handler.PyTreeSaveArgs
PyTreeRestoreArgs = pytree_checkpoint_handler.PyTreeRestoreArgs
ArrayRestoreArgs = pytree_checkpoint_handler.ArrayRestoreArgs
ParamInfo = serialization_types.ParamInfo
SaveArgs = serialization_types.SaveArgs


def _get_dispatcher(use_colocated_python: bool):
  return (
      dispatchers.ColocatedPythonDispatcher()
      if use_colocated_python
      else dispatchers.RemotePythonDispatcher()
  )


def _get_actual_worker_memory_usage(
    arr: jax.Array, use_colocated_python: bool
) -> dict[int, int]:
  dispatcher = _get_dispatcher(use_colocated_python=use_colocated_python)
  device_count = jax.device_count()
  sharding = jax.sharding.NamedSharding(
      jax.sharding.Mesh(jax.devices(), 'x'),
      jax.sharding.PartitionSpec(
          'x',
      ),
  )

  def _get_actual_worker_memory_usage_impl(
      a: jax.Array, sharding: jax.sharding.Sharding, device_count: int
  ) -> jax.Array:
    bytes_size = a.itemsize * sum(
        [shard.data.size for shard in a.addressable_shards]
    )
    return jax.make_array_from_callback(
        (device_count,),
        sharding,
        lambda _: np.array(bytes_size).reshape(
            1,
        ),
        dtype=np.int32,
    )

  result_specs = jax.ShapeDtypeStruct(
      (device_count,), dtype=np.int32, sharding=sharding
  )
  actual_worker_memory_usage_by_device = dispatcher.dispatch(
      _get_actual_worker_memory_usage_impl,
      input_arrays=arr,
      result_specs=result_specs,
      func_kwargs={
          'sharding': sharding,
          'device_count': device_count,
      },
  )
  jax.block_until_ready(actual_worker_memory_usage_by_device)

  device_to_worker_ids = worker_memory_utils._device_to_worker_ids(dispatcher)
  actual_worker_memory_usage = {}
  for shard in actual_worker_memory_usage_by_device.addressable_shards:
    worker_id = device_to_worker_ids[shard.device.id]
    memory_usage = np.asarray(shard.data)
    assert memory_usage.shape == (1,)
    memory_usage = int(memory_usage[0])
    if worker_id in actual_worker_memory_usage:
      assert actual_worker_memory_usage[worker_id] == memory_usage
    else:
      actual_worker_memory_usage[worker_id] = memory_usage

  return actual_worker_memory_usage


def _create_array(
    array_shape: tuple[int, ...],
    mesh_shape: tuple[int, ...],
    mesh_axes: tuple[str, ...],
    partition_axes: tuple[str | None, ...],
) -> jax.Array:
  sharding = jax.sharding.NamedSharding(
      jax.sharding.Mesh(
          np.asarray(jax.devices()).reshape(mesh_shape), mesh_axes
      ),
      jax.sharding.PartitionSpec(*partition_axes),
  )
  return jax.device_put(
      np.arange(np.prod(array_shape), dtype=np.float32).reshape(array_shape),
      device=sharding,
  )


class PathwaysMemoryUsageTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._use_colocated_python = USE_COLOCATED_PYTHON.value
    pathways_handler_registry.register_pathways_handlers(
        checkpointing_impl=pathways_handler_registry.CheckpointingImpl.from_options(
            use_colocated_python=self._use_colocated_python,
            use_remote_python=True,  # Fallback
        ),
        thinmint_testing=True,
    )
    PyTreeCheckpointHandler()
    self.assertTrue(utils.is_pathways_backend())
    self.assertTrue(rp.available())
    self.assertIsInstance(
        type_handler_registry.get_type_handler(jax.Array),
        jax_array_handlers.ArrayHandler,
    )

    self.directory = epath.Path(
        self.create_tempdir(name='checkpointing_test').full_path
    )
    test_utils.set_tensorstore_driver_for_test()

    test_utils.sync_global_processes('PathwaysMemoryUsageTest:setup_complete')

  def tearDown(self):
    test_utils.sync_global_processes('PathwaysMemoryUsageTest:tests_complete')
    super().tearDown()

  @parameterized.parameters(
      ((8,), ('x',), (None,), (1,)),
      ((8,), ('x',), (None,), (64,)),
      ((8,), ('x',), ('x',), (64,)),
      ((4, 2), ('x', 'y'), (None, None), (64, 16)),
      ((4, 2), ('x', 'y'), ('x', None), (64, 16)),
      ((4, 2), ('x', 'y'), ('x', 'y'), (64, 16)),
  )
  def test_worker_memory_usage_calculation(
      self,
      mesh_shape,
      mesh_axes,
      partition_axes,
      array_shape,
  ):
    device_to_worker_ids = worker_memory_utils._device_to_worker_ids(
        _get_dispatcher(use_colocated_python=self._use_colocated_python)
    )
    arr = _create_array(array_shape, mesh_shape, mesh_axes, partition_axes)
    actual_worker_memory_usage = _get_actual_worker_memory_usage(
        arr, self._use_colocated_python
    )
    estimated_worker_memory_usage = (
        worker_memory_utils._estimate_worker_memory_usage(
            arr, replica_id=None, device_to_worker_ids_map=device_to_worker_ids
        )
    )
    self.assertDictEqual(
        actual_worker_memory_usage, estimated_worker_memory_usage
    )

  @parameterized.parameters(
      (1, 200, ('x', 'y'), [1]),
      (1, 300, ('x', 'y'), [1]),
      (2, 200, ('x', 'y'), [1, 1]),
      (2, 300, ('x', 'y'), [1, 1]),
      (2, 513, ('x', 'y'), [2]),
      (3, 513, ('x', 'y'), [2, 1]),
      (2, 513, (None, None), [1, 1]),
  )
  def test_batching(
      self,
      num_arrays,
      device_host_max_bytes,
      partition_axes,
      expected_batch_sizes,
  ):
    mesh_shape = (4, 2)
    mesh_axes = ('x', 'y')
    # Array total size = 16 * 16 * 4 bytes = 1024 bytes.
    # Shard size = 1024 / (8 devices) = 128 bytes (fully replicated)
    # Per host, per array size = 128 * (2 workers_per_device) = 256 bytes.
    array_shape = (16, 16)

    values = [
        _create_array(array_shape, mesh_shape, mesh_axes, partition_axes)
        for _ in range(num_arrays)
    ]
    infos = [mock.Mock(spec=ParamInfo)() for _ in values]
    args = [SaveArgs() for _ in values]

    batch_idx = 0
    for batch in worker_memory_utils.next_memory_budgeted_batch(
        list(zip(values, infos, args)),
        device_host_max_bytes,
        replica_id=0,
        dispatcher=_get_dispatcher(
            use_colocated_python=self._use_colocated_python
        ),
    ):
      self.assertLen(batch, expected_batch_sizes[batch_idx])
      batch_idx += 1
    self.assertLen(expected_batch_sizes, batch_idx)

  def test_save_restore(self):
    arrays = []
    arr_size = 2**26
    num_arrays = 10
    for _ in range(num_arrays):
      sharding = jax.sharding.NamedSharding(
          jax.sharding.Mesh(jax.devices(), 'x'),
          jax.sharding.PartitionSpec(
              'x',
          ),
      )
      # ~268 MB per array
      arr = jax.device_put(
          np.arange(arr_size, dtype=np.float32), device=sharding
      )
      arrays.append(arr)
    handler = PyTreeCheckpointHandler(
        use_ocdbt=False, is_prioritized_key_fn=lambda _: False
    )
    arr_bytes = arrays[0].itemsize * arr_size
    unique_shards = jax.device_count()
    shards_per_worker = 2
    arrays_per_batch = 4
    handler._handler_impl._save_device_host_concurrent_bytes = (
        arr_bytes // unique_shards * shards_per_worker * arrays_per_batch + 1000
    )
    handler.save(self.directory, args=PyTreeSaveArgs(arrays))

    # Verify that individual param mtimes increase with each successive
    # batch. This verifies that batch `i` completes before batch `i+1` starts.
    param_mtimes = [0] * num_arrays
    for param_dir in self.directory.iterdir():
      if param_dir.is_dir() and param_dir.name.isdigit():
        mtime = param_dir.stat().mtime
        param_mtimes[int(param_dir.name)] = mtime
    prev_greatest_mtime = -1
    for i in range(0, num_arrays, arrays_per_batch):
      cur_greatest_mtime = max(param_mtimes[i : i + arrays_per_batch])
      self.assertGreaterEqual(cur_greatest_mtime, prev_greatest_mtime)
      prev_greatest_mtime = cur_greatest_mtime

    # Verify restore correctness.
    restore_args = jax.tree.map(
        lambda x: ArrayRestoreArgs(sharding=x.sharding), arrays
    )
    restored = handler.restore(
        self.directory, args=PyTreeRestoreArgs(restore_args=restore_args)
    )
    test_utils.assert_tree_equal(self, arrays, restored)
    handler.close()

  def test_save_restore_no_memory_limiting(self):
    sharding = jax.sharding.NamedSharding(
        jax.sharding.Mesh(jax.devices(), 'x'),
        jax.sharding.PartitionSpec(
            'x',
        ),
    )
    arr_size = 256
    arrays = [
        jax.device_put(np.arange(arr_size, dtype=np.float32), device=sharding)
    ]

    handler = PyTreeCheckpointHandler()
    original_get_deprioritized_batches_to_serialize = (
        jax_array_handlers._get_deprioritized_batches_to_serialize
    )
    with mock.patch.object(
        jax_array_handlers,
        '_get_deprioritized_batches_to_serialize',
        wraps=original_get_deprioritized_batches_to_serialize,
    ) as mock_get_deprioritized_batches_to_serialize:
      handler.save(self.directory, args=PyTreeSaveArgs(arrays))

      # Assert that _get_deprioritized_batches_to_serialize was not called
      mock_get_deprioritized_batches_to_serialize.assert_not_called()

    # Verify restore correctness.
    restore_args = jax.tree.map(
        lambda x: ArrayRestoreArgs(sharding=x.sharding), arrays
    )
    restored = handler.restore(
        self.directory, args=PyTreeRestoreArgs(restore_args=restore_args)
    )
    test_utils.assert_tree_equal(self, arrays, restored)
    handler.close()

  # TODO(cpgaffney): Test with an async D2H time that is artificially long.
  # Currently there is not a good way to guarantee this.
  def test_save_restore_with_prioritized_params(self):
    arrays = []
    arr_size = 2**26
    num_arrays = 10
    for _ in range(num_arrays):
      sharding = jax.sharding.NamedSharding(
          jax.sharding.Mesh(jax.devices(), 'x'),
          jax.sharding.PartitionSpec(
              'x',
          ),
      )
      # ~268 MB per array
      arr = jax.device_put(
          np.arange(arr_size, dtype=np.float32), device=sharding
      )
      arrays.append(arr)
    prioritized_keys_for_saving = [
        ('0',),
        ('2',),
        ('4',),
        ('6',),
        ('8',),
    ]
    handler = PyTreeCheckpointHandler(
        use_ocdbt=False,
        is_prioritized_key_fn=lambda key: tree_utils.str_keypath(key)
        in prioritized_keys_for_saving,
    )
    arr_bytes = arrays[0].itemsize * arr_size
    unique_shards = jax.device_count()
    shards_per_worker = 2
    arrays_per_batch = 4
    handler._handler_impl._save_device_host_concurrent_bytes = (
        arr_bytes // unique_shards * shards_per_worker * arrays_per_batch + 1000
    )
    start = time.time()
    handler.save(self.directory, args=PyTreeSaveArgs(arrays))
    end = time.time()
    logging.info('Time taken: %s seconds', end - start)

    # Verify all even params complete before odd params are started.
    param_ctimes = [0] * num_arrays
    param_mtimes = [0] * num_arrays
    for param_dir in self.directory.iterdir():
      if param_dir.is_dir() and param_dir.name.isdigit():
        ctime = os.stat(param_dir).st_ctime
        mtime = os.stat(param_dir).st_mtime
        param_mtimes[int(param_dir.name)] = mtime
        param_ctimes[int(param_dir.name)] = ctime

    self.assertLessEqual(max(param_mtimes[::2]), min(param_ctimes[1::2]))

    # Verify restore correctness.
    restore_args = jax.tree.map(
        lambda x: ArrayRestoreArgs(sharding=x.sharding), arrays
    )
    restored = handler.restore(
        self.directory, args=PyTreeRestoreArgs(restore_args=restore_args)
    )
    test_utils.assert_tree_equal(self, arrays, restored)
    handler.close()


if __name__ == '__main__':
  jax.config.parse_flags_with_absl()
  g3_multiprocessing.handle_test_main(googletest.main)
