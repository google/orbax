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

"""Benchmarks for Snapshotter scale up and scale down."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import gc
import pprint

from absl import logging
import jax
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint import v1 as ocp
from orbax.checkpoint._src.testing.benchmarks.core import core as benchmarks_core
from orbax.checkpoint._src.testing.benchmarks.core import metric as metric_lib
from orbax.checkpoint._src.testing.benchmarks.v1 import benchmark
from orbax.checkpoint.experimental.v1._src.training.pathways import snapshotter


@dataclasses.dataclass(frozen=True)
class SnapshotterBenchmarkOptions(benchmark.BenchmarkOptions):
  """Configuration options for Snapshotter scale up and scale down benchmark.

  Attributes:
    num_scale_down_slices: Number of slices to load into for scale-down testing.
    num_scale_up_slices: Number of slices to load into for scale-up testing.
    num_savings: The number of times to repeat the benchmark cycle.
  """

  num_scale_down_slices: int | Sequence[int] = 1
  num_scale_up_slices: int | Sequence[int] = 2
  num_savings: int = 1


@benchmarks_core.benchmark_options(SnapshotterBenchmarkOptions)
class SnapshotterBenchmark(benchmarks_core.BenchmarksGenerator):
  """A concrete generator for Snapshotter scale up and scale down benchmarks."""

  def test_fn(
      self, context: benchmarks_core.TestContext
  ) -> benchmarks_core.TestResult:
    """The test logic measuring Snapshotter save, scale down load, and scale up load.

    Args:
      context: The test context containing paths and options.

    Returns:
      The test result containing measured metrics.
    """
    metrics = metric_lib.Metrics()
    options = context.options
    assert isinstance(options, SnapshotterBenchmarkOptions)

    logging.info("SnapshotterBenchmark options: %s", pprint.pformat(options))
    metrics_to_measure = benchmark.get_metrics_to_measure(options)

    all_devices = jax.devices()
    unique_slices = sorted(
        list(set(getattr(d, "slice_index", 0) for d in all_devices))
    )
    total_slices = max(1, len(unique_slices))

    # TODO(nikhilbansall): we should use real models in the future
    def create_sharded_state(num_target_slices: int):
      target_slice_indices = unique_slices[
          : min(num_target_slices, total_slices)
      ]
      target_devices = [
          d
          for d in all_devices
          if getattr(d, "slice_index", 0) in target_slice_indices
      ]
      actual_slices = max(1, len(target_slice_indices))
      devices_arr = np.asarray(target_devices)
      mesh = jax.sharding.Mesh(
          devices_arr.reshape((actual_slices, -1)), ("replica", "model")
      )
      sharding = jax.sharding.NamedSharding(
          mesh, jax.sharding.PartitionSpec(None, "model")
      )
      shape = (actual_slices * 16, (len(target_devices) // actual_slices) * 64)
      pytree = {
          "weights": jax.device_put(
              jnp.ones(shape, dtype=jnp.float32), sharding
          ),
      }
      abstract_pytree = jax.tree.map(ocp.arrays.to_shape_dtype_struct, pytree)
      return pytree, abstract_pytree

    for step in range(options.num_savings):
      logging.info("SnapshotterBenchmark: Starting Step: %s", step)

      # 1. Take snapshot of full PyTree (e.g. 2 slices)
      pytree_full, abstract_full = create_sharded_state(
          options.num_scale_up_slices
      )
      _, abstract_down = create_sharded_state(options.num_scale_down_slices)

      logging.info("save: abstract pytree:\n%s", pprint.pformat(abstract_full))

      snapshot_mgr = snapshotter.Snapshotter(replica_axis_index=0)

      save_trace = context.trace_path(f"save_{step}")
      if save_trace is not None:
        jax.profiler.start_trace(str(save_trace))
      with metrics.measure("save", metrics_to_measure):
        snapshot_mgr.save(step, pytree_full)
        snapshot_mgr._queue.join()  # pylint: disable=protected-access
      if save_trace is not None:
        jax.profiler.stop_trace()

      # 2. Test elastic scale down (restoring onto fewer slices)
      logging.info(
          "load_scale_down: abstract target pytree:\n%s",
          pprint.pformat(abstract_down),
      )
      load_down_trace = context.trace_path(f"load_scale_down_{step}")
      if load_down_trace is not None:
        jax.profiler.start_trace(str(load_down_trace))
      with metrics.measure("load_scale_down", metrics_to_measure):
        restored_down = snapshot_mgr.load(abstract_down)
        jax.block_until_ready(restored_down)
      if load_down_trace is not None:
        jax.profiler.stop_trace()

      abstract_restored_down = jax.tree.map(
          ocp.arrays.to_shape_dtype_struct, restored_down
      )
      logging.info(
          "load_scale_down: abstract restored pytree:\n%s",
          pprint.pformat(abstract_restored_down),
      )

      benchmark.clear_pytree(restored_down)
      benchmark.clear_pytree(pytree_full)

      # 3. Test scale up (saving on fewer slices and loading onto more slices)
      pytree_down, _ = create_sharded_state(options.num_scale_down_slices)
      snapshot_mgr_up = snapshotter.Snapshotter(replica_axis_index=0)

      with metrics.measure("save_scale_down_state", metrics_to_measure):
        snapshot_mgr_up.save(step + 1000, pytree_down)
        snapshot_mgr_up._queue.join()  # pylint: disable=protected-access

      logging.info(
          "load_scale_up: abstract target pytree:\n%s",
          pprint.pformat(abstract_full),
      )
      load_up_trace = context.trace_path(f"load_scale_up_{step}")
      if load_up_trace is not None:
        jax.profiler.start_trace(str(load_up_trace))
      with metrics.measure("load_scale_up", metrics_to_measure):
        restored_up = snapshot_mgr_up.load(abstract_full)
        jax.block_until_ready(restored_up)
      if load_up_trace is not None:
        jax.profiler.stop_trace()

      abstract_restored_up = jax.tree.map(
          ocp.arrays.to_shape_dtype_struct, restored_up
      )
      logging.info(
          "load_scale_up: abstract restored pytree:\n%s",
          pprint.pformat(abstract_restored_up),
      )

      benchmark.clear_pytree(restored_up)
      benchmark.clear_pytree(pytree_down)
      gc.collect()

    return benchmarks_core.TestResult(metrics=metrics)
