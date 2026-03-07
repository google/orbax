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

"""Benchmarks for V1 free functions."""

from __future__ import annotations

import dataclasses
import pprint
from typing import Any

from absl import logging
from etils import epath
import jax
from orbax.checkpoint import v1 as ocp
from orbax.checkpoint._src.multihost import multislice
from orbax.checkpoint._src.testing.benchmarks.core import checkpoint_generation
from orbax.checkpoint._src.testing.benchmarks.core import core as benchmarks_core
from orbax.checkpoint._src.testing.benchmarks.core import metric as metric_lib
from orbax.checkpoint._src.testing.benchmarks.v1 import benchmark


# ==============================================================================
# 1. Define the Options Dataclass for this specific benchmark
# ==============================================================================
@dataclasses.dataclass(frozen=True)
class RestoreAndBroadcastBenchmarkOptions(benchmark.BenchmarkOptions):
  """Configuration options for benchmarks targeting ReshardingBenchmark.

  See parent class.

  Attributes:
    reference_checkpoint_path: The path to the reference checkpoint. This
      dictates the structure of the checkpoint to be restored.
    reference_sharding_path: The path to the reference sharding config. This
      dictates the shardings used for restoration. Note that this sharding
      config is for a *single replica*. The benchmark should be configured with
      DCN parallelism, and the test harness will replicate the sharding config
      to the multiple replicas dictated by the mesh.
  """

  reference_checkpoint_path: str | None = None
  reference_sharding_path: str | None = None
  use_load_and_broadcast: bool = True

  def is_valid(self) -> bool:
    if self.reference_checkpoint_path is None:
      return False
    if self.reference_sharding_path is None:
      return False
    return super().is_valid()


def _get_single_replica_abstract_state(
    context: ocp.Context,
    global_mesh: jax.sharding.Mesh,
    reference_checkpoint_path: epath.Path,
    reference_sharding_path: epath.Path,
):
  """Returns the abstract state for a single replica."""
  with ocp.Context(context=context):
    metadata = ocp.pytree_metadata(reference_checkpoint_path)
    # Abstract tree has shardings on a single replica.
    return checkpoint_generation.get_abstract_state_from_sharding_config(
        reference_sharding_path,
        metadata.metadata,
        devices=multislice.replica_devices(
            global_mesh, replica_id=0, replica_axis_index=0
        ).tolist(),
    )


def _get_abstract_state(
    context: ocp.Context,
    global_mesh: jax.sharding.Mesh,
    single_replica_abstract_state: Any,
):
  """Returns the abstract state for all replicas."""
  with ocp.Context(context=context):
    # Blow shardings up to all replicas.
    def _multi_replica_sharding(abstract_arr: jax.ShapeDtypeStruct):
      logging.info(
          "Original (single-replica) sharding: %s", abstract_arr.sharding
      )
      assert isinstance(abstract_arr.sharding, jax.sharding.NamedSharding)
      single_replica_mesh = abstract_arr.sharding.mesh
      single_replica_partition_spec = abstract_arr.sharding.spec
      multi_replica_sharding = jax.sharding.NamedSharding(
          jax.sharding.Mesh(
              devices=global_mesh.devices.reshape(
                  -1, *single_replica_mesh.devices.shape
              ),
              axis_names=["replica", *single_replica_mesh.axis_names],
          ),
          spec=jax.sharding.PartitionSpec(*single_replica_partition_spec),
      )
      logging.info("Multi-replica sharding: %s", multi_replica_sharding)
      return jax.ShapeDtypeStruct(
          shape=abstract_arr.shape,
          dtype=abstract_arr.dtype,
          sharding=multi_replica_sharding,
      )

    return jax.tree.map(
        _multi_replica_sharding,
        single_replica_abstract_state,
    )


# ==============================================================================
# 2. Implement the Benchmark Generator
# ==============================================================================
@benchmarks_core.benchmark_options(RestoreAndBroadcastBenchmarkOptions)
class RestoreAndBroadcastBenchmark(benchmarks_core.BenchmarksGenerator):
  """A concrete generator for restore and broadcast benchmarks."""

  def test_fn(
      self, context: benchmarks_core.TestContext
  ) -> benchmarks_core.TestResult:
    """The core test logic for a single save/restore cycle.

    This function is called for each combination of options generated by the
    framework. It uses the `context.options` to configure the handler
    dynamically for each run.

    Args:
      context: The test context containing the pytree, path, and options.

    Returns:
      The test result containing the metrics.
    """
    metrics = metric_lib.Metrics()
    assert context.pytree is None
    options = context.options
    assert isinstance(options, RestoreAndBroadcastBenchmarkOptions)
    assert options.reference_checkpoint_path is not None
    assert options.reference_sharding_path is not None
    assert context.mesh is not None

    logging.info("Benchmark options: %s", pprint.pformat(options))
    metrics_to_measure = benchmark.get_metrics_to_measure(options)

    reference_checkpoint_path = epath.Path(
        options.reference_checkpoint_path
    )
    reference_sharding_path = epath.Path(
        options.reference_sharding_path
    )

    if context.mesh.devices.ndim != 2:
      raise ValueError(
          "Found mesh with unexpected number of dimensions:"
          f" {context.mesh.ndim}"
      )
    if [str(axis) for axis in context.mesh.axis_names] != ["replica", "model"]:
      raise ValueError(
          f"Found mesh with unexpected axis names: {context.mesh.axis_names}"
      )

    single_replica_abstract_pytree = _get_single_replica_abstract_state(
        context=options.context,
        global_mesh=context.mesh,
        reference_checkpoint_path=reference_checkpoint_path,
        reference_sharding_path=reference_sharding_path,
    )
    abstract_pytree = _get_abstract_state(
        context=options.context,
        global_mesh=context.mesh,
        single_replica_abstract_state=single_replica_abstract_pytree,
    )

    with ocp.Context(context=options.context):
      if options.enable_trace:
        jax.profiler.start_trace(context.path / "trace_load")
      with metrics.measure("load", metrics_to_measure):
        restored_pytree = ocp.load_pytree(
            reference_checkpoint_path, abstract_pytree
        )
      benchmark.clear_pytree(restored_pytree)
      if options.enable_trace:
        jax.profiler.stop_trace()

    return benchmarks_core.TestResult(metrics=metrics)
