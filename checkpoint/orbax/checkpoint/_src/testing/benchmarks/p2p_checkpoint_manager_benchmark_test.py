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

import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import jax
import numpy as np
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.testing.benchmarks import p2p_checkpoint_manager_benchmark
from orbax.checkpoint._src.testing.benchmarks.core import configs as benchmarks_configs
from orbax.checkpoint._src.testing.benchmarks.core import core as benchmarks_core
from orbax.checkpoint._src.testing.benchmarks.core import mesh_utils
from orbax.checkpoint._src.testing.benchmarks.core import metric as metric_lib
from orbax.checkpoint.experimental.emergency.p2p import checkpoint_manager as p2p_checkpoint_manager
from orbax.checkpoint.experimental.emergency.p2p import options as p2p_options


P2pBenchmarkOptions = p2p_checkpoint_manager_benchmark.P2pBenchmarkOptions
P2pCheckpointManagerBenchmark = (
    p2p_checkpoint_manager_benchmark.P2pCheckpointManagerBenchmark
)


class P2pCheckpointManagerBenchmarkTest(parameterized.TestCase):

  def test_test_fn_runs_benchmark_and_saves_metrics(self):
    mock_checkpoint_manager_cls = self.enter_context(
        mock.patch.object(
            p2p_checkpoint_manager, 'CheckpointManager', autospec=True
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
    mock_pretty_log_mesh = self.enter_context(
        mock.patch.object(mesh_utils, 'pretty_log_mesh', autospec=True)
    )
    self.enter_context(
        mock.patch.object(multihost, 'process_index', return_value=0)
    )
    benchmark = P2pCheckpointManagerBenchmark(
        checkpoint_configs=[benchmarks_configs.CheckpointConfig(spec={})],
        options=P2pBenchmarkOptions(),
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
    mock_checkpoint_manager.latest_step.return_value = None
    test_dir = os.path.join(self.create_tempdir().full_path, 'test_test_fn')
    os.makedirs(test_dir, exist_ok=True)
    context = benchmarks_core.TestContext(
        pytree=pytree,
        path=epath.Path(test_dir),
        options=P2pBenchmarkOptions(train_steps=2, local_save_interval_steps=1),
        mesh=mesh,
    )
    mock_is_runtime_to_distributed_ids_initialized.return_value = False
    mock_is_distributed_to_device_ids_initialized.return_value = False

    td = epath.Path(test_dir)
    (td / 'test_local_restore' / 'local_p2p_ckpt' / 'process_0' / '1').mkdir(
        parents=True, exist_ok=True
    )
    (td / 'test_p2p_restore' / 'local_p2p_ckpt' / 'process_0' / '1').mkdir(
        parents=True, exist_ok=True
    )
    (td / 'test_gcs_restore' / 'local_p2p_ckpt' / 'process_0' / '1').mkdir(
        parents=True, exist_ok=True
    )

    result = benchmark.test_fn(context)

    with self.subTest('multihost initialization'):
      mock_is_runtime_to_distributed_ids_initialized.assert_called_once()
      mock_initialize_runtime_to_distributed_ids.assert_called_once()
      mock_is_distributed_to_device_ids_initialized.assert_called_once()
      mock_initialize_distributed_to_device_ids.assert_called_once()
    with self.subTest('sync and mesh calls'):
      mock_sync_global_processes.assert_called()
      mock_pretty_log_mesh.assert_called_once()
    with self.subTest('benchmark result type'):
      self.assertIsInstance(result, benchmarks_core.TestResult)
    with self.subTest('metrics timings'):
      self.assertIn(
          'create_abstract_pytree_time_duration', result.metrics.results
      )
      self.assertIn('create_restore_args_time_duration', result.metrics.results)
      self.assertIn(
          'test_local_restore_create_directories_time_duration',
          result.metrics.results,
      )
      self.assertIn(
          'test_local_restore_create_checkpoint_manager_time_duration',
          result.metrics.results,
      )
      self.assertIn(
          'test_local_restore_train_loop_time_duration', result.metrics.results
      )
      self.assertIn(
          'test_local_restore_save_0_time_duration', result.metrics.results
      )
      self.assertIn(
          'test_local_restore_wait_until_finished_0_time_duration',
          result.metrics.results,
      )
      self.assertIn(
          'test_local_restore_save_1_time_duration', result.metrics.results
      )
      self.assertIn(
          'test_local_restore_wait_until_finished_1_time_duration',
          result.metrics.results,
      )
      self.assertIn(
          'test_local_restore_restore_and_validate_1_time_duration',
          result.metrics.results,
      )
    with self.subTest('checkpoint manager calls'):
      num_tests = 3  # local, p2p, gcs
      self.assertEqual(
          mock_checkpoint_manager_cls.call_count,
          num_tests,
      )
      self.assertEqual(mock_checkpoint_manager.save.call_count, num_tests * 2)
      self.assertEqual(
          mock_checkpoint_manager.wait_until_finished.call_count, num_tests * 2
      )
      self.assertEqual(mock_checkpoint_manager.restore.call_count, num_tests)
      self.assertEqual(mock_checkpoint_manager.close.call_count, num_tests)
      # p2p and gcs restore cause dir deletion, so +1 reload each.
      # 1 reload after restore for each of 3 tests.
      # so 1*3 + 2 = 5 reloads.
      self.assertEqual(mock_checkpoint_manager.reload.call_count, 5)

  def test_generate_benchmarks_creates_multiple_benchmark_configs(self):
    options = P2pBenchmarkOptions(
        persistent_save_interval_steps=[5, 10],
        local_save_interval_steps=[2, 4],
        replica_axis_index=[0, 1],
    )
    benchmark = P2pCheckpointManagerBenchmark(
        checkpoint_configs=[benchmarks_configs.CheckpointConfig(spec={})],
        options=options,
    )
    benchmarks = benchmark.generate()

    self.assertLen(benchmarks, 8)
    for b in benchmarks:
      self.assertIsInstance(b.options, P2pBenchmarkOptions)

  @parameterized.parameters(
      dict(
          options=P2pBenchmarkOptions(
              persistent_save_interval_steps=10,
              persistent_max_to_keep=2,
              local_save_interval_steps=3,
              local_max_to_keep=3,
              replica_axis_index=1,
              train_steps=5,
          )
      ),
      dict(
          options=P2pBenchmarkOptions(
              train_steps=1, local_save_interval_steps=1
          )
      ),
  )
  def test_test_fn_applies_benchmark_options_correctly(self, options):
    mock_checkpoint_manager_cls = self.enter_context(
        mock.patch.object(
            p2p_checkpoint_manager, 'CheckpointManager', autospec=True
        )
    )
    mock_checkpoint_manager_options_cls = self.enter_context(
        mock.patch.object(
            p2p_options,
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
    mock_pretty_log_mesh = self.enter_context(
        mock.patch.object(mesh_utils, 'pretty_log_mesh', autospec=True)
    )
    self.benchmark = P2pCheckpointManagerBenchmark(
        checkpoint_configs=[benchmarks_configs.CheckpointConfig(spec={})],
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
    mock_checkpoint_manager.latest_step.return_value = None
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

    self.benchmark.test_fn(context)

    with self.subTest('multihost initialization'):
      mock_is_runtime_to_distributed_ids_initialized.assert_called_once()
      mock_initialize_runtime_to_distributed_ids.assert_called_once()
      mock_is_distributed_to_device_ids_initialized.assert_called_once()
      mock_initialize_distributed_to_device_ids.assert_called_once()
    with self.subTest('options propagation'):
      self.assertEqual(mock_checkpoint_manager_cls.call_count, 3)
      mock_checkpoint_manager_options_cls.assert_called_with(
          local=mock.ANY,
          persistent=mock.ANY,
          replica_axis_index=options.replica_axis_index,
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
      mock_pretty_log_mesh.assert_called_once()


class HelperFunctionsTest(parameterized.TestCase):

  @mock.patch.object(p2p_checkpoint_manager, 'CheckpointManager', autospec=True)
  @mock.patch.object(p2p_options, 'CheckpointManagerOptions', autospec=True)
  def test_create_checkpoint_manager(
      self,
      mock_checkpoint_manager_options_cls,
      mock_checkpoint_manager_cls,
  ):
    local_dir = epath.Path('/tmp/local')
    persistent_dir = epath.Path('/tmp/persistent')
    mesh = jax.sharding.Mesh(np.array(jax.devices()), ('data',))
    abstract_state = {'a': jax.ShapeDtypeStruct(shape=(4,), dtype=np.int32)}
    options = P2pBenchmarkOptions(
        local_save_interval_steps=2,
        local_max_to_keep=3,
        persistent_save_interval_steps=5,
        persistent_max_to_keep=6,
        replica_axis_index=1,
    )

    p2p_checkpoint_manager_benchmark._create_checkpoint_manager(
        local_dir, persistent_dir, mesh, abstract_state, options
    )

    mock_checkpoint_manager_options_cls.assert_called_once_with(
        local=p2p_options.LocalCheckpointOptions(
            save_interval_steps=2,
            max_to_keep=3,
        ),
        persistent=p2p_options.PersistentCheckpointOptions(
            save_interval_steps=5,
            max_to_keep=6,
        ),
        replica_axis_index=1,
    )
    mock_checkpoint_manager_cls.assert_called_once_with(
        local_directory=local_dir,
        persistent_directory=persistent_dir,
        global_mesh=mesh,
        abstract_state=abstract_state,
        options=mock_checkpoint_manager_options_cls.return_value,
    )


class DeleteCheckpointsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.manager = mock.create_autospec(
        p2p_checkpoint_manager.CheckpointManager, instance=True
    )
    self.local_directory = epath.Path(self.create_tempdir().full_path)

  @mock.patch.object(multihost, 'process_index', return_value=0)
  def test_delete_checkpoints_local_p0(self, mock_process_index):
    del mock_process_index
    step = 0
    step_dir = self.local_directory / str(step)
    step_dir.mkdir()
    self.assertTrue(step_dir.exists())
    p2p_checkpoint_manager_benchmark._delete_checkpoints(
        self.manager, step, self.local_directory, 'local_p0'
    )
    self.assertFalse(step_dir.exists())
    self.manager.reload.assert_called_once()

  @mock.patch.object(multihost, 'process_index', return_value=1)
  def test_delete_checkpoints_local_p0_non_p0(self, mock_process_index):
    del mock_process_index
    step = 0
    step_dir = self.local_directory / str(step)
    step_dir.mkdir()
    self.assertTrue(step_dir.exists())
    p2p_checkpoint_manager_benchmark._delete_checkpoints(
        self.manager, step, self.local_directory, 'local_p0'
    )
    self.assertTrue(step_dir.exists())
    self.manager.reload.assert_not_called()

  @mock.patch.object(multihost, 'process_index', return_value=0)
  def test_delete_checkpoints_local_all(self, mock_process_index):
    del mock_process_index
    step = 0
    step_dir = self.local_directory / str(step)
    step_dir.mkdir()
    self.assertTrue(step_dir.exists())
    p2p_checkpoint_manager_benchmark._delete_checkpoints(
        self.manager, step, self.local_directory, 'local_all'
    )
    self.assertFalse(step_dir.exists())
    self.manager.reload.assert_called_once()

  @mock.patch.object(multihost, 'process_index', return_value=0)
  def test_delete_checkpoints_none(self, mock_process_index):
    del mock_process_index
    step = 0
    step_dir = self.local_directory / str(step)
    step_dir.mkdir()
    self.assertTrue(step_dir.exists())
    p2p_checkpoint_manager_benchmark._delete_checkpoints(
        self.manager, step, self.local_directory, 'none'
    )
    self.assertTrue(step_dir.exists())
    self.manager.reload.assert_not_called()


class RestoreAndValidateTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.manager = mock.create_autospec(
        p2p_checkpoint_manager.CheckpointManager, instance=True
    )

  @mock.patch.object(multihost, 'sync_global_processes', autospec=True)
  @mock.patch(
      'orbax.checkpoint._src.testing.benchmarks.p2p_checkpoint_manager_benchmark.pytree_utils.assert_pytree_equal',
      autospec=True,
  )
  def test_restore_and_validate_succeeds(
      self,
      mock_assert_pytree_equal,
      mock_sync_global_processes,
  ):
    metrics = metric_lib.Metrics()
    pytree = {'a': np.array([1, 2, 3])}
    abstract_pytree = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), pytree
    )
    step = 0
    restore_args = {'a': mock.MagicMock()}
    self.manager.restore.return_value = {'state': pytree}

    p2p_checkpoint_manager_benchmark._restore_and_validate(
        self.manager,
        metrics,
        pytree,
        abstract_pytree,
        step,
        restore_args=restore_args,
    )

    mock_sync_global_processes.assert_called_once_with('save_completed_0')
    self.manager.reload.assert_called_once()
    self.manager.restore.assert_called_once()
    mock_assert_pytree_equal.assert_called_once_with(pytree, pytree)


if __name__ == '__main__':
  absltest.main()
