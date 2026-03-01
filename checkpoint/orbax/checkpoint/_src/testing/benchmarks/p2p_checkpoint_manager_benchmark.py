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

"""Benchmarks for P2P CheckpointManager.

This module contains benchmarks for
orbax.checkpoint.experimental.emergency.p2p.checkpoint_manager.CheckpointManager.
"""

from collections.abc import Sequence
import dataclasses
import inspect
from typing import Any
from absl import logging
from etils import epath
import jax
from orbax.checkpoint._src.handlers import pytree_checkpoint_handler
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.serialization import type_handlers
from orbax.checkpoint._src.testing.benchmarks.core import core as benchmarks_core
from orbax.checkpoint._src.testing.benchmarks.core import mesh_utils
from orbax.checkpoint._src.testing.benchmarks.core import metric as metric_lib
from orbax.checkpoint._src.testing.benchmarks.core import pytree_utils
from orbax.checkpoint._src.tree import utils
from orbax.checkpoint.experimental.emergency.p2p import args as p2p_args_lib
from orbax.checkpoint.experimental.emergency.p2p import checkpoint_manager as p2p_checkpoint_manager
from orbax.checkpoint.experimental.emergency.p2p import options as p2p_options


# ==============================================================================
# 1. Define the Options Dataclass
# ==============================================================================
@dataclasses.dataclass(frozen=True)
class P2pBenchmarkOptions(benchmarks_core.BenchmarkOptions):
  """Configuration options for benchmarks targeting P2P CheckpointManager.

  Attributes:
    persistent_save_interval_steps: The interval at which persistent checkpoints
      should be saved.
    persistent_max_to_keep: The maximum number of persistent checkpoints to
      keep.
    local_save_interval_steps: The interval at which local checkpoints should be
      saved.
    local_max_to_keep: The maximum number of local checkpoints to keep.
    replica_axis_index: The index of the replica axis in the global mesh.
    train_steps: The number of training steps to run.
  """

  persistent_save_interval_steps: int | Sequence[int] = 5
  persistent_max_to_keep: int | Sequence[int] = 5
  local_save_interval_steps: int | Sequence[int] = 2
  local_max_to_keep: int | Sequence[int] = 2
  replica_axis_index: int | Sequence[int] = 0
  train_steps: int | Sequence[int] = 10
  experimental_use_distributed_id_for_mesh_consistency: (
      bool | Sequence[bool]
  ) = True
  experimental_orbax_use_distributed_process_id: bool | Sequence[bool] = True


# ==============================================================================
# 2. Implement the Benchmark Generator
# ==============================================================================
def _create_checkpoint_manager(
    local_directory: epath.Path,
    persistent_directory: epath.Path,
    global_mesh: jax.sharding.Mesh,
    abstract_state: Any,
    options: P2pBenchmarkOptions,
) -> p2p_checkpoint_manager.CheckpointManager:
  """Creates an P2P CheckpointManager."""
  return p2p_checkpoint_manager.CheckpointManager(
      local_directory=local_directory,
      persistent_directory=persistent_directory,
      global_mesh=global_mesh,
      abstract_state=abstract_state,
      options=p2p_options.CheckpointManagerOptions(
          local=p2p_options.LocalCheckpointOptions(
              save_interval_steps=options.local_save_interval_steps,
              max_to_keep=options.local_max_to_keep,
          ),
          persistent=p2p_options.PersistentCheckpointOptions(
              save_interval_steps=options.persistent_save_interval_steps,
              max_to_keep=options.persistent_max_to_keep,
          ),
          replica_axis_index=options.replica_axis_index,
      ),
  )


def _restore_and_validate(
    manager: p2p_checkpoint_manager.CheckpointManager,
    metrics: metric_lib.Metrics,
    pytree: Any,
    step: int,
    local_directory: epath.Path,
    restore_args: Any,
    test_name: str = '',
    delete_before_restore: str = 'local_p0',
):
  """Restores a checkpoint and validates it."""
  prefix = f'{test_name}_' if test_name else ''
  # Wait for save to complete on all hosts.
  with metrics.measure(f'{prefix}sync_global_processes_{step}'):
    multihost.sync_global_processes(f'{prefix}save_completed_{step}')

  step_dir = local_directory / str(step)
  if delete_before_restore == 'local_p0':
    if multihost.process_index() == 0 and step_dir.exists():
      logging.info(
          'Process 0: removing local checkpoint to trigger P2P restore.'
      )
      step_dir.rmtree()
      manager.reload()
  elif delete_before_restore == 'local_all':
    if step_dir.exists():
      logging.info(
          'All processes: removing local checkpoint to trigger GCS restore.'
      )
      step_dir.rmtree()
      manager.reload()
  elif delete_before_restore == 'none':
    logging.info('Skipping deletion of local checkpoint for local restore.')
  else:
    raise ValueError(
        f'Invalid delete_before_restore: {delete_before_restore}'
    )

  logging.info('Not using restore args: %r', restore_args)

  with metrics.measure(f'{prefix}restore_{step}'):
    restored = manager.restore(
        step,
        args=p2p_args_lib.Composite(
            state=pytree_checkpoint_handler.PyTreeRestoreArgs(
                restore_args=restore_args
            )
        ),
    )['state']
  pytree_utils.log_pytree('Restored Pytree', restored)
  logging.info('Assert Restored Pytree')
  pytree_utils.assert_pytree_equal(pytree, restored)
  with metrics.measure(f'{prefix}reload_after_restore_{step}'):
    manager.reload()


@benchmarks_core.benchmark_options(P2pBenchmarkOptions)
class P2pCheckpointManagerBenchmark(benchmarks_core.BenchmarksGenerator):
  """A generator for benchmarking P2P CheckpointManager."""

  def _run_test(
      self,
      test_name: str,
      context: benchmarks_core.TestContext,
      metrics: metric_lib.Metrics,
      abstract_pytree: Any,
      restore_args: Any,
      delete_before_restore: str = 'local_p0',
  ):
    """Runs a single test case."""
    logging.info('Running test: %s', test_name)
    pytree = context.pytree
    persistent_directory = context.path / test_name / 'persistent_p2p_ckpt'
    if context.local_path is not None:
      local_path = epath.Path(context.local_path) / test_name / 'local_p2p_ckpt'
      local_directory = epath.Path(local_path)
    else:
      local_directory = (
          context.path
          / test_name
          / 'local_p2p_ckpt'
          / f'process_{multihost.process_index()}'
      )
    options = context.options
    mesh = context.mesh
    assert isinstance(options, P2pBenchmarkOptions)

    with metrics.measure(f'{test_name}_create_directories'):
      if jax.process_index() == 0:
        persistent_directory.mkdir(parents=True, exist_ok=True)
      local_directory.mkdir(parents=True, exist_ok=True)
      multihost.sync_global_processes(f'{test_name}_create_directories')

    with metrics.measure(f'{test_name}_create_checkpoint_manager'):
      manager = _create_checkpoint_manager(
          local_directory=local_directory,
          persistent_directory=persistent_directory,
          global_mesh=mesh,
          abstract_state=abstract_pytree,
          options=options,
      )

    step = manager.latest_step()
    if step is not None:
      logging.info('Latest step in test %s: %d', test_name, step)

      with metrics.measure(f'{test_name}_restore_and_validate_{step}'):
        _restore_and_validate(
            manager,
            metrics,
            pytree,
            step,
            local_directory,
            restore_args,
            test_name=test_name,
            delete_before_restore=delete_before_restore,
        )

    start_step = step + 1 if step is not None else 0
    with metrics.measure(f'{test_name}_train_loop'):
      for step in range(start_step, options.train_steps):
        logging.info('Test %s: Training step %d', test_name, step)
        with metrics.measure(f'{test_name}_save_{step}'):
          manager.save(
              step,
              args=p2p_args_lib.Composite(
                  state=pytree_checkpoint_handler.PyTreeSaveArgs(pytree)
              ),
          )
        with metrics.measure(f'{test_name}_wait_until_finished_{step}'):
          manager.wait_until_finished()

        if step % options.local_save_interval_steps == 0 and step != 0:
          with metrics.measure(f'{test_name}_restore_and_validate_{step}'):
            _restore_and_validate(
                manager,
                metrics,
                pytree,
                step,
                local_directory,
                restore_args,
                test_name=test_name,
                delete_before_restore=delete_before_restore,
            )

    manager.close()

  def test_local_restore(
      self,
      context: benchmarks_core.TestContext,
      metrics: metric_lib.Metrics,
      abstract_pytree: Any,
      restore_args: Any,
  ):
    self._run_test(
        'test_local_restore',
        context,
        metrics,
        abstract_pytree,
        restore_args,
        delete_before_restore='none',
    )

  def test_p2p_restore(
      self,
      context: benchmarks_core.TestContext,
      metrics: metric_lib.Metrics,
      abstract_pytree: Any,
      restore_args: Any,
  ):
    self._run_test(
        'test_p2p_restore',
        context,
        metrics,
        abstract_pytree,
        restore_args,
        delete_before_restore='local_p0',
    )

  def test_gcs_restore(
      self,
      context: benchmarks_core.TestContext,
      metrics: metric_lib.Metrics,
      abstract_pytree: Any,
      restore_args: Any,
  ):
    self._run_test(
        'test_gcs_restore',
        context,
        metrics,
        abstract_pytree,
        restore_args,
        delete_before_restore='local_all',
    )

  def test_fn(
      self, context: benchmarks_core.TestContext
  ) -> benchmarks_core.TestResult:
    """The core test logic for a single save/restore cycle."""
    metrics = metric_lib.Metrics()
    pytree = context.pytree
    options = context.options
    mesh = context.mesh
    assert isinstance(options, P2pBenchmarkOptions)

    if mesh is None:
      raise ValueError(
          'Mesh must be provided for P2pCheckpointManagerBenchmark'
      )
    if not multihost.is_runtime_to_distributed_ids_initialized():
      multihost.initialize_runtime_to_distributed_ids()

    if not multihost.is_distributed_to_device_ids_initialized():
      multihost.initialize_distributed_to_device_ids()

    mesh_utils.pretty_log_mesh('Global Mesh: ', mesh)

    with metrics.measure('create_abstract_pytree'):
      abstract_pytree = jax.tree.map(utils.to_shape_dtype_struct, pytree)
      logging.info('abstract_pytree: %r', abstract_pytree)

    with metrics.measure('create_restore_args'):
      restore_args = type_handlers.SingleReplicaArrayRestoreArgs()
      logging.info('restore_args: %r', restore_args)

    tests_to_run = []
    for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
      if name.startswith('test_') and name != 'test_fn':
        tests_to_run.append(method)

    for test in tests_to_run:
      test(context, metrics, abstract_pytree, restore_args)

    return benchmarks_core.TestResult(metrics=metrics)
