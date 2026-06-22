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

from collections.abc import Sequence
import dataclasses
import pprint
from typing import Any

from absl import logging
import jax
from orbax.checkpoint import v1 as ocp
from orbax.checkpoint._src.testing.benchmarks.core import core as benchmarks_core
from orbax.checkpoint._src.testing.benchmarks.core import metric as metric_lib


def get_metrics_to_measure(options: BenchmarkOptions) -> list[str]:
  """Returns the list of metrics to measure.

  Cheap captures (time, rss, jax_monitoring, device_memory, tensorstore)
  are always on. Tracemalloc is opt-in because its per-allocation
  snapshots have measurable runtime overhead.

  Args:
    options: Benchmark options; tracemalloc is added when
      metric_tracemalloc_enabled is set.

  Returns:
    The metric names to capture for each measured operation.
  """
  metrics = metric_lib.default_metrics()
  if options.metric_tracemalloc_enabled:
    metrics.append("tracemalloc")
  return metrics


# ==============================================================================
# 1. Define the Options Dataclass for this specific benchmark
# ==============================================================================
@dataclasses.dataclass(frozen=True)
class BenchmarkOptions(benchmarks_core.BenchmarkOptions):
  """Configuration options for benchmarks targeting BenchmarkHandler.

  Each attribute can be a single value or a list of values to create
  a parameter sweep.

  Attributes:
    async_enabled: Whether to use async checkpointing.
    use_ocdbt: Whether to use ocdbt.
    use_zarr3: Whether to use zarr3.
    use_compression: Whether to use compression.
    save_concurrent_gb: The number of concurrent GB to use for saving.
    restore_concurrent_gb: The number of concurrent GB to use for restoring.
    metric_tracemalloc_enabled: Whether to enable the tracemalloc metric (opt-in
      because per-allocation snapshots are expensive).
    use_load_and_broadcast: Whether to use load and broadcast.
    use_replica_parallel: Whether to use replica parallel.
    enable_replica_parallel_separate_folder: Whether to enable replica parallel
      separate folder.
  """

  async_enabled: bool | Sequence[bool] = True
  use_ocdbt: bool | Sequence[bool] = True
  use_zarr3: bool | Sequence[bool] = True
  use_compression: bool | Sequence[bool] = True
  save_concurrent_gb: int | None | Sequence[int | None] = None
  restore_concurrent_gb: int | None | Sequence[int | None] = None
  metric_tracemalloc_enabled: bool = False
  use_load_and_broadcast: bool | Sequence[bool] = False
  use_replica_parallel: bool | Sequence[bool] = False
  enable_replica_parallel_separate_folder: bool | Sequence[bool] = False
  chunk_byte_size: int | None | Sequence[int | None] = None

  def is_valid(self) -> bool:
    assert isinstance(self.use_replica_parallel, bool)
    assert isinstance(self.enable_replica_parallel_separate_folder, bool)
    if self.enable_replica_parallel_separate_folder and (
        not self.use_replica_parallel or not self.use_ocdbt
    ):
      return False
    return True

  @property
  def context(self) -> ocp.Context:
    ctx = ocp.Context()
    ctx.array.saving.storage_options.chunk_byte_size = self.chunk_byte_size
    ctx.array.saving.use_ocdbt = self.use_ocdbt
    ctx.array.saving.use_zarr3 = self.use_zarr3
    ctx.array.saving.use_replica_parallel = self.use_replica_parallel
    ctx.array.saving.use_compression = self.use_compression
    ctx.array.saving.enable_replica_parallel_separate_folder = (
        self.enable_replica_parallel_separate_folder
    )
    ctx.array.loading.use_load_and_broadcast = self.use_load_and_broadcast
    ctx.memory.write_concurrent_bytes = (
        self.save_concurrent_gb * 1024**3
        if self.save_concurrent_gb is not None
        else None
    )
    ctx.memory.read_concurrent_bytes = (
        self.restore_concurrent_gb * 1024**3
        if self.restore_concurrent_gb is not None
        else None
    )
    return ctx


def clear_pytree(pytree: Any) -> Any:
  """Clears the pytree to free up memory."""
  return jax.tree.map(
      lambda x: x.delete() if isinstance(x, jax.Array) else None, pytree
  )


# ==============================================================================
# 2. Implement the Benchmark Generator
# ==============================================================================
@benchmarks_core.benchmark_options(BenchmarkOptions)
class Benchmark(benchmarks_core.BenchmarksGenerator):
  """A concrete generator for `orbax.checkpoint.BenchmarkHandler`.

  This class provides the specific test logic for benchmarking the
  BenchmarkHandler with various configurations.
  """

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
    pytree = context.pytree
    abstract_pytree = jax.tree.map(ocp.arrays.to_shape_dtype_struct, pytree)
    save_path = context.path / "ckpt"
    options = context.options
    assert isinstance(options, BenchmarkOptions)

    logging.info("Benchmark options: %s", pprint.pformat(options))
    metrics_to_measure = get_metrics_to_measure(options)

    save_trace = context.trace_path("save")
    load_trace = context.trace_path("load")

    with ocp.Context(context=options.context):
      if save_trace is not None:
        jax.profiler.start_trace(str(save_trace))
      with metrics.measure("save", metrics_to_measure):
        if options.async_enabled:
          f = ocp.save_async(save_path, pytree)
          f.result()
        else:
          ocp.save(save_path, pytree)
      context.pytree = clear_pytree(context.pytree)
      if save_trace is not None:
        jax.profiler.stop_trace()

      if load_trace is not None:
        jax.profiler.start_trace(str(load_trace))
      with metrics.measure("load", metrics_to_measure):
        restored_pytree = ocp.load(save_path, abstract_state=abstract_pytree)
      clear_pytree(restored_pytree)
      if load_trace is not None:
        jax.profiler.stop_trace()

    return benchmarks_core.TestResult(metrics=metrics)
