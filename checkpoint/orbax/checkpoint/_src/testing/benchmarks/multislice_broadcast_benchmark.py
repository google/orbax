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

"""Benchmarks for orbax.checkpoint._src.multihost.multislice."""

from collections.abc import Sequence
import dataclasses
from absl import flags
from absl import logging
import jax
import jax.numpy as jnp
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.multihost import multislice
from orbax.checkpoint._src.testing.benchmarks.core import core as benchmarks_core
from orbax.checkpoint._src.testing.benchmarks.core import mesh_utils
from orbax.checkpoint._src.testing.benchmarks.core import metric as metric_lib
from orbax.checkpoint._src.testing.benchmarks.core import pytree_utils


# ==============================================================================
# 1. Define the Options Dataclass
# ==============================================================================
@dataclasses.dataclass(frozen=True)
class MultisliceBroadcastBenchmarkOptions(benchmarks_core.BenchmarkOptions):
  """Configuration options for benchmarks targeting Multilice Operations.

  Attributes:
    replica_axis_index: The index of the replica axis in the global mesh.
  """

  replica_axis_index: int | Sequence[int] = 0


# ==============================================================================
# 2. Implement the Benchmark Generator
# ==============================================================================
@benchmarks_core.benchmark_options(MultisliceBroadcastBenchmarkOptions)
class MultisliceBroadcastBenchmark(benchmarks_core.BenchmarksGenerator):
  """A generator for benchmarking Multislice Operations."""

  def test_fn(
      self, context: benchmarks_core.TestContext
  ) -> benchmarks_core.TestResult:
    """The core test logic multislice operations."""
    metrics = metric_lib.Metrics()
    options = context.options
    mesh = context.mesh
    assert isinstance(options, MultisliceBroadcastBenchmarkOptions)

    if mesh is None:
      raise ValueError("Mesh must be provided for MultisliceBroadcastBenchmark")

    flags.FLAGS.experimental_orbax_use_distributed_process_id = True
    if not multihost.is_runtime_to_distributed_ids_initialized():
      multihost.initialize_runtime_to_distributed_ids()

    if not multihost.is_distributed_to_device_ids_initialized():
      multihost.initialize_distributed_to_device_ids()

    local_replica_mesh = mesh_utils.get_local_replica_mesh(
        mesh, options.replica_axis_index
    )
    mesh_utils.pretty_log_mesh("Global Mesh: ", mesh)
    mesh_utils.pretty_log_mesh(
        "Local Replica Mesh: ",
        local_replica_mesh,
    )

    with metrics.time("process_spans_multiple_replicas"):
      if multislice.process_spans_multiple_replicas(
          mesh, replica_axis_index=options.replica_axis_index
      ):
        logging.warning(
            "Process spans multiple replicas. Mesh: %r, replica_axis_index: %r",
            mesh,
            options.replica_axis_index,
        )

    is_source_replica = multislice.in_replica(
        multihost.process_index(),
        mesh,
        replica_id=0,
        replica_axis_index=options.replica_axis_index,
    )
    logging.info(
        "Process %d is in source replica: %s",
        multihost.process_index(),
        is_source_replica,
    )
    expected_arr_list = []
    single_replica_arr_list = []

    def _fn(expected_arr):
      expected_arr_list.append(expected_arr)
      if is_source_replica:
        arr = jax.device_get(expected_arr)
      else:
        arr = jnp.zeros(expected_arr.shape, dtype=expected_arr.dtype)
      arr_single_replica_sharding = jax.sharding.NamedSharding(
          local_replica_mesh, expected_arr.sharding.spec
      )
      arr_single_replica = jax.device_put(arr, arr_single_replica_sharding)
      pytree_utils.log_pytree("Single Replica Array", arr_single_replica)
      single_replica_arr_list.append(arr_single_replica)

    jax.tree.map(_fn, context.pytree)
    with metrics.time("broadcast_array"):
      broadcasted_tuple, num_broadcasts = (
          multislice.broadcast_one_replica_to_all(
              tuple(single_replica_arr_list),
              mesh,
              replica_axis_index=options.replica_axis_index,
              is_source=is_source_replica,
              memory_limit_bytes=1024 * 1024 * 1024,  # 1GB
          )
      )
    expected_tuple = tuple(expected_arr_list)
    logging.info("Number of broadcasts: %d", num_broadcasts)
    pytree_utils.log_pytree("Expected arrays", expected_tuple)
    pytree_utils.log_pytree("Broadcasted arrays", broadcasted_tuple)
    pytree_utils.assert_pytree_equal(expected_tuple, broadcasted_tuple)
    return benchmarks_core.TestResult(metrics=metrics)
