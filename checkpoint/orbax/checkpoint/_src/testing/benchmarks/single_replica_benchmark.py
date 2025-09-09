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

"""Benchmarks for orbax.checkpoint._src.serialization.type_handlers.SingleReplicaArrayHandler."""
from collections.abc import Sequence
import dataclasses
from typing import Any
from absl import logging
import jax
from orbax.checkpoint._src.checkpointers import async_checkpointer
from orbax.checkpoint._src.handlers import pytree_checkpoint_handler
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.serialization import type_handlers
from orbax.checkpoint._src.testing.benchmarks.core import core as benchmarks_core
from orbax.checkpoint._src.testing.benchmarks.core import mesh_utils
from orbax.checkpoint._src.testing.benchmarks.core import pytree_utils
from orbax.checkpoint._src.tree import utils


# ==============================================================================
# 1. Define the Options Dataclass
# ==============================================================================
@dataclasses.dataclass(frozen=True)
class SingleReplicaBenchmarkOptions(benchmarks_core.BenchmarkOptions):
  """Configuration options for benchmarks targeting SingleReplicaArrayHandler.

  Attributes:
    replica_axis_index: The index in mesh_axes that represents the replica axis.
    primary_replica_id: The replica ID to use as the primary for loading.
    use_replica_parallel: Whether to parallelize saving across replicas.
    broadcast_memory_limit_bytes: The maximum memory to use for broadcasting.
    broadcast_memory_scaling_factor: The scaling factor to use for broadcasting.
  """

  replica_axis_index: int | Sequence[int] = 0
  primary_replica_id: int | Sequence[int] = 0
  use_replica_parallel: bool | Sequence[bool] = True
  broadcast_memory_limit_bytes: int | Sequence[int] | None = None
  broadcast_memory_scaling_factor: float | Sequence[float] = 0.75
  use_shard_map: bool | Sequence[bool] = False


# ==============================================================================
# 2. Implement the Benchmark Generator
# ==============================================================================
@benchmarks_core.benchmark_options(SingleReplicaBenchmarkOptions)
class SingleReplicaBenchmark(benchmarks_core.BenchmarksGenerator):
  """A generator for benchmarking SingleReplicaArrayHandler."""

  def _construct_restore_args(
      self, abstract_pytree: Any, replica_axis_index: int
  ) -> pytree_checkpoint_handler.PyTreeRestoreArgs:
    """Constructs the restore args for a single replica restore."""

    def map_to_pspec(data):
      pspec = data.sharding.spec
      mesh = data.sharding.mesh
      replica_mesh = mesh_utils.get_local_replica_mesh(mesh, replica_axis_index)
      mesh_utils.pretty_log_mesh("Replica mesh: ", replica_mesh)
      single_replica_sharding = jax.sharding.NamedSharding(replica_mesh, pspec)

      return type_handlers.SingleReplicaArrayRestoreArgs(
          sharding=jax.sharding.NamedSharding(mesh, pspec),
          single_replica_sharding=single_replica_sharding,
          global_shape=data.shape,
          dtype=data.dtype,
      )

    restore_args = jax.tree_util.tree_map(map_to_pspec, abstract_pytree)
    return pytree_checkpoint_handler.PyTreeRestoreArgs(
        item=abstract_pytree, restore_args=restore_args
    )

  def test_fn(
      self, context: benchmarks_core.TestContext
  ) -> benchmarks_core.TestResult:
    """The core test logic for a single save/restore cycle."""
    metrics = benchmarks_core.Metrics()
    pytree = context.pytree
    save_path = context.path / "single_replica_ckpt"
    options = context.options
    mesh = context.mesh
    assert isinstance(options, SingleReplicaBenchmarkOptions)

    if mesh is None:
      raise ValueError("Mesh must be provided for SingleReplicaBenchmark")

    if not multihost.is_runtime_to_distributed_ids_initialized():
      multihost.initialize_runtime_to_distributed_ids()

    type_handlers.register_type_handler(
        jax.Array,
        type_handlers.SingleReplicaArrayHandler(
            replica_axis_index=options.replica_axis_index,
            primary_replica_id=options.primary_replica_id,
            use_replica_parallel=options.use_replica_parallel,
            broadcast_memory_limit_bytes=options.broadcast_memory_limit_bytes,
            broadcast_memory_scaling_factor=options.broadcast_memory_scaling_factor,
            use_shard_map=options.use_shard_map,
        ),
        override=True,
    )

    handler = pytree_checkpoint_handler.PyTreeCheckpointHandler()
    checkpointer = async_checkpointer.AsyncCheckpointer(handler)

    with metrics.time("save"):
      checkpointer.save(
          save_path, args=pytree_checkpoint_handler.PyTreeSaveArgs(pytree)
      )

    with metrics.time("wait_until_finished"):
      checkpointer.wait_until_finished()

    abstract_pytree = jax.tree.map(utils.to_shape_dtype_struct, pytree)
    logging.info("abstract_pytree: %s", abstract_pytree)

    with metrics.time("restore"):
      with metrics.time("construct_restore_args"):
        restore_args = self._construct_restore_args(
            abstract_pytree,
            options.replica_axis_index,
        )
        logging.info("restore_args: %s", restore_args)

      restored_pytree = checkpointer.restore(
          save_path,
          args=restore_args,
      )
      pytree_utils.log_pytree("Restored Pytree", restored_pytree)
      # Verify the restored pytree is the same as the original pytree.
      pytree_utils.assert_pytree_equal(pytree, restored_pytree)

    checkpointer.close()
    return benchmarks_core.TestResult(metrics=metrics)
