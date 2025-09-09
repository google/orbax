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

"""Benchmarks for orbax.checkpoint.experimental.emergency.checkpoint_manager.CheckpointManager."""

from collections.abc import Sequence
import dataclasses
from typing import Any
from absl import flags
from absl import logging
from etils import epath
import jax
from orbax.checkpoint import checkpoint_utils
from orbax.checkpoint._src.handlers import composite_checkpoint_handler
from orbax.checkpoint._src.handlers import pytree_checkpoint_handler
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.multihost import multislice
from orbax.checkpoint._src.testing.benchmarks.core import core as benchmarks_core
from orbax.checkpoint._src.testing.benchmarks.core import mesh_utils
from orbax.checkpoint._src.testing.benchmarks.core import pytree_utils
from orbax.checkpoint._src.tree import utils
from orbax.checkpoint.experimental.emergency import checkpoint_manager as emergency_checkpoint_manager


# ==============================================================================
# 1. Define the Options Dataclass
# ==============================================================================
@dataclasses.dataclass(frozen=True)
class EcmBenchmarkOptions(benchmarks_core.BenchmarkOptions):
  """Configuration options for benchmarks targeting EmergencyCheckpointManager.

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
  use_shard_map_broadcast: bool | Sequence[bool] = True
  single_host_load_and_broadcast: bool | Sequence[bool] = True
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
    options: EcmBenchmarkOptions,
) -> emergency_checkpoint_manager.CheckpointManager:
  """Creates an EmergencyCheckpointManager."""
  return emergency_checkpoint_manager.CheckpointManager(
      local_directory=local_directory,
      persistent_directory=persistent_directory,
      global_mesh=global_mesh,
      abstract_state=abstract_state,
      options=emergency_checkpoint_manager.CheckpointManagerOptions(
          local=emergency_checkpoint_manager.LocalCheckpointOptions(
              save_interval_steps=options.local_save_interval_steps,
              max_to_keep=options.local_max_to_keep,
          ),
          persistent=emergency_checkpoint_manager.PersistentCheckpointOptions(
              save_interval_steps=options.persistent_save_interval_steps,
              max_to_keep=options.persistent_max_to_keep,
          ),
          replica_axis_index=options.replica_axis_index,
          use_shard_map_broadcast=options.use_shard_map_broadcast,
          single_host_load_and_broadcast=options.single_host_load_and_broadcast,
      ),
  )


def _is_in_replica(
    mesh: jax.sharding.Mesh, replica_axis_index: int, replica_id: int
) -> bool:
  """Returns whether the current process is in the given replica."""
  return multislice.in_replica(
      multihost.process_index(),
      mesh,
      replica_id=replica_id,
      replica_axis_index=replica_axis_index,
  )


def _restore_and_validate(
    manager: emergency_checkpoint_manager.CheckpointManager,
    metrics: benchmarks_core.Metrics,
    pytree: Any,
    step: int,
    local_directory: epath.Path,
    is_in_primary_slice: bool,
    is_in_secondary_slice: bool,
    restore_args: Any,
):
  """Restores a checkpoint and validates it."""
  # Wait for save to complete on all hosts.
  with metrics.time(f"sync_global_processes_{step}"):
    multihost.sync_global_processes(f"save_completed_{step}")

  # Remove local checkpoint on secondary slice.
  if not is_in_primary_slice:
    assert (local_directory / str(step)).exists()
  if is_in_secondary_slice:
    (local_directory / str(step)).rename(local_directory / "backup")
    logging.info("Removing secondary slice checkpoint at step %d", step)
  with metrics.time(f"reload_first_time_{step}"):
    manager.reload()
  with metrics.time(f"restore_{step}"):
    restored = manager.restore(
        step,
        args=composite_checkpoint_handler.CompositeArgs(
            state=pytree_checkpoint_handler.PyTreeRestoreArgs(
                restore_args=restore_args
            )
        ),
    )["state"]
  pytree_utils.log_pytree("Local Restored Pytree", restored)
  logging.info("Assert Local Restored Pytree")
  pytree_utils.assert_pytree_equal(pytree, restored)

  # Put back secondary slice local checkpoint.
  if is_in_secondary_slice:
    (local_directory / "backup").rename(local_directory / str(step))
    logging.info("Putting back secondary slice checkpoint at step %d", step)
  with metrics.time(f"reload_second_time_{step}"):
    manager.reload()


@benchmarks_core.benchmark_options(EcmBenchmarkOptions)
class EmergencyCheckpointManagerBenchmark(benchmarks_core.BenchmarksGenerator):
  """A generator for benchmarking EmergencyCheckpointManager."""

  # TODO: b/381928709 - Add a way to run the benchmark with automatic restart
  # and resume for ECM.
  def test_fn(
      self, context: benchmarks_core.TestContext
  ) -> benchmarks_core.TestResult:
    """The core test logic for a single save/restore cycle."""
    metrics = benchmarks_core.Metrics()
    pytree = context.pytree
    persistent_directory = context.path / "persistent_replica_ckpt"
    local_directory = (
        context.path
        / "local_replica_ckpt"
        / f"process_{multihost.process_index()}"
    )
    options = context.options
    mesh = context.mesh
    assert isinstance(options, EcmBenchmarkOptions)

    if mesh is None:
      raise ValueError(
          "Mesh must be provided for EmergencyCheckpointManagerBenchmark"
      )
    flags.FLAGS.experimental_use_distributed_id_for_mesh_consistency = (
        options.experimental_use_distributed_id_for_mesh_consistency
    )
    flags.FLAGS.experimental_orbax_use_distributed_process_id = (
        options.experimental_orbax_use_distributed_process_id
    )
    if not multihost.is_runtime_to_distributed_ids_initialized():
      multihost.initialize_runtime_to_distributed_ids()

    if not multihost.is_distributed_to_device_ids_initialized():
      multihost.initialize_distributed_to_device_ids()

    mesh_utils.pretty_log_mesh("Global Mesh: ", mesh)
    mesh_utils.pretty_log_mesh(
        "Local Replica Mesh: ",
        mesh_utils.get_local_replica_mesh(mesh, options.replica_axis_index),
    )

    with metrics.time("create_directories"):
      if jax.process_index() == 0:
        persistent_directory.mkdir(parents=True)
      local_directory.mkdir(parents=True)
      multihost.sync_global_processes("create directories")

    with metrics.time("create_abstract_pytree"):
      abstract_pytree = jax.tree.map(utils.to_shape_dtype_struct, pytree)
      logging.info("abstract_pytree: %r", abstract_pytree)

    with metrics.time("create_restore_args"):
      restore_args = checkpoint_utils.construct_restore_args(abstract_pytree)
      logging.info("restore_args: %r", restore_args)

    with metrics.time("create_checkpoint_manager"):
      manager = _create_checkpoint_manager(
          local_directory=local_directory,
          persistent_directory=persistent_directory,
          global_mesh=mesh,
          abstract_state=abstract_pytree,
          options=options,
      )

    is_in_primary_slice = _is_in_replica(
        mesh,
        options.replica_axis_index,
        emergency_checkpoint_manager._PRIMARY_REPLICA_ID,  # pylint: disable=protected-access
    )
    logging.info(
        "process_index=%d, is_in_primary_slice: %r",
        multihost.process_index(),
        is_in_primary_slice,
    )

    is_in_secondary_slice = _is_in_replica(
        mesh,
        options.replica_axis_index,
        emergency_checkpoint_manager._SECONDARY_REPLICA_ID,  # pylint: disable=protected-access
    )
    logging.info(
        "process_index=%d, is_in_secondary_slice: %r",
        multihost.process_index(),
        is_in_secondary_slice,
    )

    with metrics.time("train_loop"):
      for step in range(options.train_steps):
        logging.info("Training step %d", step)
        with metrics.time(f"save_{step}"):
          manager.save(
              step,
              args=composite_checkpoint_handler.CompositeArgs(
                  state=pytree_checkpoint_handler.PyTreeSaveArgs(pytree)
              ),
          )
        with metrics.time(f"wait_until_finished_{step}"):
          manager.wait_until_finished()

        if step % options.local_save_interval_steps == 0:
          with metrics.time(f"restore_and_validate_{step}"):
            _restore_and_validate(
                manager,
                metrics,
                pytree,
                step,
                local_directory,
                is_in_primary_slice,
                is_in_secondary_slice,
                restore_args,
            )

    manager.close()
    return benchmarks_core.TestResult(metrics=metrics)
