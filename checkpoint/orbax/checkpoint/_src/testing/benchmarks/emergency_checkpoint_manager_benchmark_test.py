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
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.multihost import multislice
from orbax.checkpoint._src.testing.benchmarks import emergency_checkpoint_manager_benchmark
from orbax.checkpoint._src.testing.benchmarks.core import configs as benchmarks_configs
from orbax.checkpoint._src.testing.benchmarks.core import core as benchmarks_core
from orbax.checkpoint._src.testing.benchmarks.core import mesh_utils
from orbax.checkpoint._src.testing.benchmarks.core import pytree_utils
from orbax.checkpoint.experimental.emergency import checkpoint_manager as emergency_checkpoint_manager


EcmBenchmarkOptions = emergency_checkpoint_manager_benchmark.EcmBenchmarkOptions
EmergencyCheckpointManagerBenchmark = (
    emergency_checkpoint_manager_benchmark.EmergencyCheckpointManagerBenchmark
)


class EmergencyCheckpointManagerBenchmarkTest(parameterized.TestCase):

  def test_test_fn_runs_benchmark_and_saves_metrics(self):
    mock_in_replica = self.enter_context(
        mock.patch.object(multislice, 'in_replica', autospec=True)
    )
    mock_checkpoint_manager_cls = self.enter_context(
        mock.patch.object(
            emergency_checkpoint_manager, 'CheckpointManager', autospec=True
        )
    )
    mock_sync_global_processes = self.enter_context(
        mock.patch.object(multihost, 'sync_global_processes', autospec=True)
    )
    mock_is_runtime_to_distributed_ids_initialized = self.enter_context(
        mock.patch.object(
            multihost,
            'is_runtime_to_distributed_ids_initialized',
            autospec=True,
        )
    )
    mock_initialize_runtime_to_distributed_ids = self.enter_context(
        mock.patch.object(
            multihost, 'initialize_runtime_to_distributed_ids', autospec=True
        )
    )
    mock_is_distributed_to_device_ids_initialized = self.enter_context(
        mock.patch.object(
            multihost, 'is_distributed_to_device_ids_initialized', autospec=True
        )
    )
    mock_initialize_distributed_to_device_ids = self.enter_context(
        mock.patch.object(
            multihost, 'initialize_distributed_to_device_ids', autospec=True
        )
    )
    mock_get_local_replica_mesh = self.enter_context(
        mock.patch.object(mesh_utils, 'get_local_replica_mesh', autospec=True)
    )
    benchmark = EmergencyCheckpointManagerBenchmark(
        checkpoint_config=benchmarks_configs.CheckpointConfig(),
        options=EcmBenchmarkOptions(),
    )
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
    mock_checkpoint_manager = mock_checkpoint_manager_cls.return_value
    mock_checkpoint_manager.restore.return_value = {'state': pytree}
    test_dir = os.path.join(self.create_tempdir().full_path, 'test_test_fn')
    os.makedirs(test_dir, exist_ok=True)
    context = benchmarks_core.TestContext(
        pytree=pytree,
        path=epath.Path(test_dir),
        options=EcmBenchmarkOptions(train_steps=1, local_save_interval_steps=1),
        mesh=mesh,
    )
    mock_is_runtime_to_distributed_ids_initialized.return_value = False
    mock_is_distributed_to_device_ids_initialized.return_value = False
    mock_get_local_replica_mesh.return_value = mesh
    mock_in_replica.side_effect = [True, False]

    result = benchmark.test_fn(context)

    with self.subTest('multihost initialization'):
      mock_is_runtime_to_distributed_ids_initialized.assert_called_once()
      mock_initialize_runtime_to_distributed_ids.assert_called_once()
      mock_is_distributed_to_device_ids_initialized.assert_called_once()
      mock_initialize_distributed_to_device_ids.assert_called_once()
    with self.subTest('sync and mesh calls'):
      self.assertEqual(mock_sync_global_processes.call_count, 2)
      mock_get_local_replica_mesh.assert_called_once()
    with self.subTest('benchmark result type'):
      self.assertIsInstance(result, benchmarks_core.TestResult)
    with self.subTest('metrics timings'):
      self.assertIn('create_directories', result.metrics.timings)
      self.assertIn('create_abstract_pytree', result.metrics.timings)
      self.assertIn('create_restore_args', result.metrics.timings)
      self.assertIn('create_checkpoint_manager', result.metrics.timings)
      self.assertIn('train_loop', result.metrics.timings)
      self.assertIn('save_0', result.metrics.timings)
      self.assertIn('wait_until_finished_0', result.metrics.timings)
      self.assertIn('sync_global_processes_0', result.metrics.timings)
    with self.subTest('checkpoint manager calls'):
      mock_checkpoint_manager_cls.assert_called_once()
      mock_checkpoint_manager.save.assert_called_once()
      mock_checkpoint_manager.wait_until_finished.assert_called_once()
      mock_checkpoint_manager.restore.assert_called_once()
      mock_checkpoint_manager.close.assert_called_once()
      self.assertEqual(mock_checkpoint_manager.reload.call_count, 2)

  def test_generate_benchmarks_creates_multiple_benchmark_configs(self):
    options = EcmBenchmarkOptions(
        persistent_save_interval_steps=[5, 10],
        local_save_interval_steps=[2, 4],
        replica_axis_index=[0, 1],
    )
    benchmark = EmergencyCheckpointManagerBenchmark(
        checkpoint_config=benchmarks_configs.CheckpointConfig(),
        options=options,
    )
    benchmarks = benchmark.generate()

    self.assertLen(benchmarks, 8)
    for b in benchmarks:
      self.assertIsInstance(b.options, EcmBenchmarkOptions)

  @parameterized.parameters(
      dict(
          options=EcmBenchmarkOptions(
              persistent_save_interval_steps=10,
              persistent_max_to_keep=2,
              local_save_interval_steps=3,
              local_max_to_keep=3,
              replica_axis_index=1,
              train_steps=5,
              single_host_load_and_broadcast=True,
          )
      ),
      dict(
          options=EcmBenchmarkOptions(
              train_steps=1, local_save_interval_steps=1
          )
      ),
  )
  def test_test_fn_applies_benchmark_options_correctly(self, options):
    mock_in_replica = self.enter_context(
        mock.patch.object(multislice, 'in_replica', autospec=True)
    )
    mock_checkpoint_manager_cls = self.enter_context(
        mock.patch.object(
            emergency_checkpoint_manager, 'CheckpointManager', autospec=True
        )
    )
    mock_checkpoint_manager_options_cls = self.enter_context(
        mock.patch.object(
            emergency_checkpoint_manager,
            'CheckpointManagerOptions',
            autospec=True,
        )
    )
    mock_sync_global_processes = self.enter_context(
        mock.patch.object(multihost, 'sync_global_processes', autospec=True)
    )
    mock_is_runtime_to_distributed_ids_initialized = self.enter_context(
        mock.patch.object(
            multihost,
            'is_runtime_to_distributed_ids_initialized',
            autospec=True,
        )
    )
    mock_initialize_runtime_to_distributed_ids = self.enter_context(
        mock.patch.object(
            multihost, 'initialize_runtime_to_distributed_ids', autospec=True
        )
    )
    mock_is_distributed_to_device_ids_initialized = self.enter_context(
        mock.patch.object(
            multihost, 'is_distributed_to_device_ids_initialized', autospec=True
        )
    )
    mock_initialize_distributed_to_device_ids = self.enter_context(
        mock.patch.object(
            multihost, 'initialize_distributed_to_device_ids', autospec=True
        )
    )
    mock_get_local_replica_mesh = self.enter_context(
        mock.patch.object(mesh_utils, 'get_local_replica_mesh', autospec=True)
    )
    self.benchmark = EmergencyCheckpointManagerBenchmark(
        checkpoint_config=benchmarks_configs.CheckpointConfig(),
        options=options,
    )
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
    mock_checkpoint_manager = mock_checkpoint_manager_cls.return_value
    mock_checkpoint_manager.restore.return_value = {'state': pytree}
    test_dir = os.path.join(self.create_tempdir().full_path, 'test_test_fn')
    os.makedirs(test_dir, exist_ok=True)
    context = benchmarks_core.TestContext(
        pytree=pytree,
        path=epath.Path(test_dir),
        options=options,
        mesh=mesh,
    )
    mock_is_runtime_to_distributed_ids_initialized.return_value = False
    mock_is_distributed_to_device_ids_initialized.return_value = False
    mock_get_local_replica_mesh.return_value = mesh
    mock_in_replica.side_effect = [True, False]

    self.benchmark.test_fn(context)

    with self.subTest('multihost initialization'):
      mock_is_runtime_to_distributed_ids_initialized.assert_called_once()
      mock_initialize_runtime_to_distributed_ids.assert_called_once()
      mock_is_distributed_to_device_ids_initialized.assert_called_once()
      mock_initialize_distributed_to_device_ids.assert_called_once()
    with self.subTest('options propagation'):
      mock_checkpoint_manager_cls.assert_called_once()
      mock_checkpoint_manager_options_cls.assert_called_once_with(
          local=mock.ANY,
          persistent=mock.ANY,
          replica_axis_index=options.replica_axis_index,
          single_host_load_and_broadcast=options.single_host_load_and_broadcast,
      )
      local_options = mock_checkpoint_manager_options_cls.call_args[1]['local']
      self.assertEqual(
          local_options.save_interval_steps, options.local_save_interval_steps
      )
      self.assertEqual(local_options.max_to_keep, options.local_max_to_keep)
      persistent_options = mock_checkpoint_manager_options_cls.call_args[1][
          'persistent'
      ]
      self.assertEqual(
          persistent_options.save_interval_steps,
          options.persistent_save_interval_steps,
      )
      self.assertEqual(
          persistent_options.max_to_keep, options.persistent_max_to_keep
      )
    with self.subTest('mesh setup calls'):
      mock_sync_global_processes.assert_called()
      mock_get_local_replica_mesh.assert_called_once()
      mock_in_replica.assert_called()


class HelperFunctionsTest(parameterized.TestCase):

  @mock.patch.object(
      emergency_checkpoint_manager, 'CheckpointManager', autospec=True
  )
  @mock.patch.object(
      emergency_checkpoint_manager, 'CheckpointManagerOptions', autospec=True
  )
  def test_create_checkpoint_manager(
      self,
      mock_checkpoint_manager_options_cls,
      mock_checkpoint_manager_cls,
  ):
    local_dir = epath.Path('/tmp/local')
    persistent_dir = epath.Path('/tmp/persistent')
    mesh = jax.sharding.Mesh(np.array(jax.devices()), ('data',))
    abstract_state = {'a': jax.ShapeDtypeStruct(shape=(4,), dtype=np.int32)}
    options = EcmBenchmarkOptions(
        local_save_interval_steps=2,
        local_max_to_keep=3,
        persistent_save_interval_steps=5,
        persistent_max_to_keep=6,
        replica_axis_index=1,
        single_host_load_and_broadcast=True,
    )

    emergency_checkpoint_manager_benchmark._create_checkpoint_manager(
        local_dir, persistent_dir, mesh, abstract_state, options
    )

    mock_checkpoint_manager_options_cls.assert_called_once_with(
        local=emergency_checkpoint_manager.LocalCheckpointOptions(
            save_interval_steps=2,
            max_to_keep=3,
        ),
        persistent=emergency_checkpoint_manager.PersistentCheckpointOptions(
            save_interval_steps=5,
            max_to_keep=6,
        ),
        replica_axis_index=1,
        single_host_load_and_broadcast=True,
    )
    mock_checkpoint_manager_cls.assert_called_once_with(
        local_directory=local_dir,
        persistent_directory=persistent_dir,
        global_mesh=mesh,
        abstract_state=abstract_state,
        options=mock_checkpoint_manager_options_cls.return_value,
    )

  @mock.patch.object(multihost, 'process_index', autospec=True)
  @mock.patch.object(multislice, 'in_replica', autospec=True)
  def test_is_in_replica(self, mock_in_replica, mock_process_index):
    mesh = jax.sharding.Mesh(np.array(jax.devices()), ('data',))
    replica_axis_index = 0
    replica_id = 1
    mock_process_index.return_value = 0
    mock_in_replica.return_value = True

    result = emergency_checkpoint_manager_benchmark._is_in_replica(
        mesh, replica_axis_index, replica_id
    )

    self.assertTrue(result)
    mock_in_replica.assert_called_once_with(
        0, mesh, replica_id=replica_id, replica_axis_index=replica_axis_index
    )


class RestoreAndValidateTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.manager = mock.create_autospec(
        emergency_checkpoint_manager.CheckpointManager, instance=True
    )
    self.local_directory = epath.Path(self.create_tempdir().full_path)

  @mock.patch.object(multihost, 'sync_global_processes', autospec=True)
  @mock.patch.object(pytree_utils, 'assert_pytree_equal', autospec=True)
  @mock.patch.object(pytree_utils, 'log_pytree', autospec=True)
  def test_restore_and_validate_primary_succeeds(
      self,
      mock_log_pytree,
      mock_assert_pytree_equal,
      mock_sync_global_processes,
  ):
    metrics = benchmarks_core.Metrics()
    pytree = {'a': np.array([1, 2, 3])}
    step = 0
    restore_args = {'a': mock.MagicMock()}
    self.manager.restore.return_value = {'state': pytree}
    step_dir = self.local_directory / str(step)

    emergency_checkpoint_manager_benchmark._restore_and_validate(
        self.manager,
        metrics,
        pytree,
        step,
        self.local_directory,
        is_in_primary_slice=True,
        is_in_secondary_slice=False,
        restore_args=restore_args,
    )

    mock_sync_global_processes.assert_called_once_with(f'save_completed_{step}')
    self.assertEqual(self.manager.reload.call_count, 2)
    self.manager.restore.assert_called_once()
    mock_log_pytree.assert_called_once_with('Local Restored Pytree', pytree)
    mock_assert_pytree_equal.assert_called_once_with(pytree, pytree)
    self.assertFalse(step_dir.exists())

  @mock.patch.object(multihost, 'sync_global_processes', autospec=True)
  @mock.patch.object(pytree_utils, 'assert_pytree_equal', autospec=True)
  @mock.patch.object(pytree_utils, 'log_pytree', autospec=True)
  def test_restore_and_validate_secondary_succeeds(
      self,
      mock_log_pytree,
      mock_assert_pytree_equal,
      mock_sync_global_processes,
  ):
    metrics = benchmarks_core.Metrics()
    pytree = {'a': np.array([1, 2, 3])}
    step = 0
    restore_args = {'a': mock.MagicMock()}
    self.manager.restore.return_value = {'state': pytree}
    step_dir = self.local_directory / str(step)
    backup_dir = self.local_directory / 'backup'
    step_dir.mkdir()

    emergency_checkpoint_manager_benchmark._restore_and_validate(
        self.manager,
        metrics,
        pytree,
        step,
        self.local_directory,
        is_in_primary_slice=False,
        is_in_secondary_slice=True,
        restore_args=restore_args,
    )

    mock_sync_global_processes.assert_called_once_with(f'save_completed_{step}')
    self.assertEqual(self.manager.reload.call_count, 2)
    mock_log_pytree.assert_called_once_with('Local Restored Pytree', pytree)
    mock_assert_pytree_equal.assert_called_once_with(pytree, pytree)
    self.manager.reload.assert_called()
    self.assertTrue(step_dir.exists())
    self.assertFalse(backup_dir.exists())

  @mock.patch.object(multihost, 'sync_global_processes', autospec=True)
  @mock.patch.object(pytree_utils, 'assert_pytree_equal', autospec=True)
  @mock.patch.object(pytree_utils, 'log_pytree', autospec=True)
  def test_restore_and_validate_neither_succeeds(
      self,
      mock_log_pytree,
      mock_assert_pytree_equal,
      mock_sync_global_processes,
  ):
    metrics = benchmarks_core.Metrics()
    pytree = {'a': np.array([1, 2, 3])}
    step = 0
    restore_args = {'a': mock.MagicMock()}
    self.manager.restore.return_value = {'state': pytree}
    step_dir = self.local_directory / str(step)
    backup_dir = self.local_directory / 'backup'
    step_dir.mkdir()

    emergency_checkpoint_manager_benchmark._restore_and_validate(
        self.manager,
        metrics,
        pytree,
        step,
        self.local_directory,
        is_in_primary_slice=False,
        is_in_secondary_slice=False,
        restore_args=restore_args,
    )

    mock_sync_global_processes.assert_called_once_with(f'save_completed_{step}')
    self.assertEqual(self.manager.reload.call_count, 2)
    mock_log_pytree.assert_called_once_with('Local Restored Pytree', pytree)
    mock_assert_pytree_equal.assert_called_once_with(pytree, pytree)
    self.assertTrue(step_dir.exists())
    self.assertFalse(backup_dir.exists())


if __name__ == '__main__':
  absltest.main()
