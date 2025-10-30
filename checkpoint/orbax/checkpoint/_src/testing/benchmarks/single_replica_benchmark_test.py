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

import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import jax
import numpy as np
from orbax.checkpoint._src.checkpointers import async_checkpointer
from orbax.checkpoint._src.handlers import pytree_checkpoint_handler
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.serialization import type_handler_registry
from orbax.checkpoint._src.serialization import type_handlers
from orbax.checkpoint._src.testing.benchmarks import single_replica_benchmark
from orbax.checkpoint._src.testing.benchmarks.core import configs as benchmarks_configs
from orbax.checkpoint._src.testing.benchmarks.core import core as benchmarks_core
from orbax.checkpoint._src.testing.benchmarks.core import mesh_utils

SingleReplicaBenchmarkOptions = (
    single_replica_benchmark.SingleReplicaBenchmarkOptions
)
SingleReplicaBenchmark = single_replica_benchmark.SingleReplicaBenchmark


class SingleReplicaBenchmarkTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.benchmark = SingleReplicaBenchmark(
        checkpoint_configs=[benchmarks_configs.CheckpointConfig()],
        options=SingleReplicaBenchmarkOptions(),
    )

  @mock.patch.object(mesh_utils, 'get_local_replica_mesh', autospec=True)
  def test_construct_restore_args(self, mock_get_local_replica_mesh):
    mesh_shape = (jax.device_count(), 1)
    mesh = jax.sharding.Mesh(
        np.array(jax.devices()).reshape(mesh_shape), ('data', 'model')
    )
    sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec('data', 'model')
    )
    abstract_pytree = {
        'a': jax.ShapeDtypeStruct(
            shape=(4, 4),
            dtype=np.float32,
            sharding=sharding,
        )
    }
    mock_get_local_replica_mesh.return_value = jax.sharding.Mesh(
        np.array(jax.devices()).reshape(mesh_shape), ('data', 'model')
    )

    restore_args = self.benchmark._construct_restore_args(
        abstract_pytree, replica_axis_index=0
    )

    mock_get_local_replica_mesh.assert_called_once_with(mesh, 0)
    self.assertIsInstance(
        restore_args, pytree_checkpoint_handler.PyTreeRestoreArgs
    )
    self.assertEqual(restore_args.item, abstract_pytree)
    self.assertIn('a', restore_args.restore_args)
    restore_arg_a = restore_args.restore_args['a']
    self.assertIsInstance(
        restore_arg_a, type_handlers.SingleReplicaArrayRestoreArgs
    )
    self.assertEqual(restore_arg_a.global_shape, (4, 4))
    self.assertEqual(restore_arg_a.dtype, np.float32)
    self.assertEqual(restore_arg_a.sharding, sharding)

    single_replica_sharding = restore_arg_a.single_replica_sharding
    self.assertIsNotNone(single_replica_sharding)
    replica_mesh = single_replica_sharding.mesh
    self.assertEqual(replica_mesh.shape['data'], 1)
    self.assertEqual(replica_mesh.shape['model'], mesh.shape['model'])
    devices = replica_mesh.devices
    self.assertIsNotNone(devices)
    self.assertLen(devices.flatten(), jax.device_count())

  @mock.patch.object(async_checkpointer, 'AsyncCheckpointer', autospec=True)
  @mock.patch.object(
      type_handler_registry, 'register_type_handler', autospec=True
  )
  @mock.patch.object(
      multihost, 'is_runtime_to_distributed_ids_initialized', autospec=True
  )
  @mock.patch.object(
      multihost, 'initialize_runtime_to_distributed_ids', autospec=True
  )
  @mock.patch.object(mesh_utils, 'get_local_replica_mesh', autospec=True)
  def test_test_fn(
      self,
      mock_get_local_replica_mesh,
      mock_initialize_runtime_to_distributed_ids,
      mock_is_runtime_to_distributed_ids_initialized,
      mock_register_type_handler,
      mock_checkpointer_cls,
  ):
    mesh_shape = (jax.device_count(), 1)
    mesh = jax.sharding.Mesh(
        np.array(jax.devices()).reshape(mesh_shape), ('data', 'model')
    )
    sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec('data', 'model')
    )
    pytree = {
        'a': jax.device_put(np.arange(16).reshape((4, 4)), sharding),
    }
    mock_checkpointer = mock_checkpointer_cls.return_value
    mock_checkpointer.restore.return_value = pytree
    test_dir = os.path.join(self.create_tempdir().full_path, 'test_test_fn')
    os.makedirs(test_dir, exist_ok=True)
    context = benchmarks_core.TestContext(
        pytree=pytree,
        path=epath.Path(test_dir),
        options=SingleReplicaBenchmarkOptions(),
        mesh=mesh,
    )
    mock_get_local_replica_mesh.return_value = mesh
    mock_is_runtime_to_distributed_ids_initialized.return_value = False

    result = self.benchmark.test_fn(context)

    mock_get_local_replica_mesh.assert_called_once_with(mesh, 0)
    mock_initialize_runtime_to_distributed_ids.assert_called_once()
    mock_is_runtime_to_distributed_ids_initialized.assert_called_once()
    mock_register_type_handler.assert_called_once()
    self.assertIsInstance(result, benchmarks_core.TestResult)
    self.assertIn('save_time', result.metrics.results)
    self.assertIn('wait_until_finished_time', result.metrics.results)
    self.assertIn('restore_time', result.metrics.results)
    self.assertIn('construct_restore_args_time', result.metrics.results)
    mock_checkpointer.save.assert_called_once()
    mock_checkpointer.wait_until_finished.assert_called_once()
    mock_checkpointer.restore.assert_called_once()
    mock_checkpointer.close.assert_called_once()
    restore_args = mock_checkpointer.restore.call_args[1]['args']
    restore_arg_a = restore_args.restore_args['a']
    self.assertEqual(restore_arg_a.sharding, sharding)
    self.assertIsNotNone(restore_arg_a.single_replica_sharding)
    self.assertEqual(
        restore_arg_a.single_replica_sharding.mesh.devices.shape,
        (1, 1),
    )

  def test_generate_benchmarks(self):
    options = SingleReplicaBenchmarkOptions(
        replica_axis_index=[0, 1], primary_replica_id=[0, 1]
    )
    benchmark = SingleReplicaBenchmark(
        checkpoint_configs=[benchmarks_configs.CheckpointConfig()],
        options=options,
    )
    benchmarks = benchmark.generate()
    self.assertLen(benchmarks, 4)
    for b in benchmarks:
      self.assertIsInstance(b.options, SingleReplicaBenchmarkOptions)

  @parameterized.parameters(
      dict(
          options=SingleReplicaBenchmarkOptions(
              replica_axis_index=1,
              primary_replica_id=2,
              use_replica_parallel=False,
              broadcast_memory_limit_bytes=100,
              broadcast_memory_scaling_factor=0.5,
          )
      ),
      dict(options=SingleReplicaBenchmarkOptions()),
  )
  @mock.patch.object(async_checkpointer, 'AsyncCheckpointer', autospec=True)
  @mock.patch.object(
      type_handler_registry, 'register_type_handler', autospec=True
  )
  @mock.patch.object(type_handlers, 'SingleReplicaArrayHandler', autospec=True)
  @mock.patch.object(
      multihost, 'is_runtime_to_distributed_ids_initialized', autospec=True
  )
  @mock.patch.object(
      multihost, 'initialize_runtime_to_distributed_ids', autospec=True
  )
  @mock.patch.object(mesh_utils, 'get_local_replica_mesh', autospec=True)
  def test_test_fn_handler_options(
      self,
      mock_get_local_replica_mesh,
      mock_initialize_runtime_to_distributed_ids,
      mock_is_runtime_to_distributed_ids_initialized,
      mock_single_replica_handler,
      mock_register_type_handler,
      mock_checkpointer_cls,
      options,
  ):
    mesh_shape = (jax.device_count(), 1)
    mesh = jax.sharding.Mesh(
        np.array(jax.devices()).reshape(mesh_shape), ('data', 'model')
    )
    sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec('data', 'model')
    )
    pytree = {
        'a': jax.device_put(np.arange(16).reshape((4, 4)), sharding),
    }
    mock_checkpointer = mock_checkpointer_cls.return_value
    mock_checkpointer.restore.return_value = pytree
    test_dir = os.path.join(self.create_tempdir().full_path, 'test_test_fn')
    os.makedirs(test_dir, exist_ok=True)
    context = benchmarks_core.TestContext(
        pytree=pytree,
        path=epath.Path(test_dir),
        options=options,
        mesh=mesh,
    )
    mock_get_local_replica_mesh.return_value = mesh
    mock_is_runtime_to_distributed_ids_initialized.return_value = False

    self.benchmark.test_fn(context)

    mock_get_local_replica_mesh.assert_called_once_with(
        mesh, options.replica_axis_index
    )
    mock_initialize_runtime_to_distributed_ids.assert_called_once()
    mock_is_runtime_to_distributed_ids_initialized.assert_called_once()
    mock_register_type_handler.assert_called_once()
    mock_single_replica_handler.assert_called_once_with(
        replica_axis_index=options.replica_axis_index,
        primary_replica_id=options.primary_replica_id,
        use_replica_parallel=options.use_replica_parallel,
        broadcast_memory_limit_bytes=options.broadcast_memory_limit_bytes,
        broadcast_memory_scaling_factor=options.broadcast_memory_scaling_factor,
        dispatcher=None,
    )


if __name__ == '__main__':
  absltest.main()
