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


def _metrics_to_measure(options: V1BenchmarkOptions) -> list[str]:
  """Returns the list of metrics to measure."""
  metrics = ["time", "rss", "io"]
  if options.metric_tracemalloc_enabled:
    metrics.append("tracemalloc")
  if options.metric_tensorstore_enabled:
    metrics.append("tensorstore")
  return metrics


# ==============================================================================
# 1. Define the Options Dataclass for this specific benchmark
# ==============================================================================
@dataclasses.dataclass(frozen=True)
class V1BenchmarkOptions(benchmarks_core.BenchmarkOptions):
  """Configuration options for benchmarks targeting V1BenchmarkHandler.

  Each attribute can be a single value or a list of values to create
  a parameter sweep.

  Attributes:
    async_enabled: Whether to use async checkpointing.
    use_ocdbt: Whether to use ocdbt.
    use_zarr3: Whether to use zarr3.
    use_compression: Whether to use compression.
    save_concurrent_gb: The number of concurrent GB to use for saving.
    restore_concurrent_gb: The number of concurrent GB to use for restoring.
    metric_tracemalloc_enabled: Whether to enable tracemalloc metric.
    metric_tensorstore_enabled: Whether to enable tensorstore metric.
    use_replica_parallel: Whether to use replica parallel.
    enable_replica_parallel_separate_folder: Whether to enable replica parallel
      separate folder.
    enable_trace: Whether to enable trace.
  """

  async_enabled: bool | Sequence[bool] = True
  use_ocdbt: bool | Sequence[bool] = True
  use_zarr3: bool | Sequence[bool] = True
  use_compression: bool | Sequence[bool] = True
  save_concurrent_gb: int | None | Sequence[int | None] = None
  restore_concurrent_gb: int | None | Sequence[int | None] = None
  metric_tracemalloc_enabled: bool = False
  metric_tensorstore_enabled: bool = False
  use_replica_parallel: bool | Sequence[bool] = False
  enable_replica_parallel_separate_folder: bool | Sequence[bool] = False
  enable_trace: bool = False

  def is_valid(self):
    assert isinstance(self.use_replica_parallel, bool)
    assert isinstance(self.enable_replica_parallel_separate_folder, bool)
    if self.enable_replica_parallel_separate_folder and (
        not self.use_replica_parallel or not self.use_ocdbt
    ):
      return False
    return True

  @property
  def context(self) -> ocp.Context:
    return ocp.Context(
        array_options=ocp.options.ArrayOptions(
            saving=ocp.options.ArrayOptions.Saving(
                use_ocdbt=self.use_ocdbt,
                use_zarr3=self.use_zarr3,
                use_replica_parallel=self.use_replica_parallel,
                use_compression=self.use_compression,
                enable_replica_parallel_separate_folder=self.enable_replica_parallel_separate_folder,
                concurrent_bytes=self.save_concurrent_gb * 1024**3
                if self.save_concurrent_gb is not None
                else None,
            ),
            loading=ocp.options.ArrayOptions.Loading(
                concurrent_bytes=self.restore_concurrent_gb * 1024**3
                if self.restore_concurrent_gb is not None
                else None,
            ),
        ),
    )


# ==============================================================================
# 2. Implement the Benchmark Generator
# ==============================================================================
@benchmarks_core.benchmark_options(V1BenchmarkOptions)
class V1Benchmark(benchmarks_core.BenchmarksGenerator):
  """A concrete generator for `orbax.checkpoint.V1BenchmarkHandler`.

  This class provides the specific test logic for benchmarking the
  V1BenchmarkHandler with various configurations.
  """

  def _clear_pytree(self, pytree: Any) -> Any:
    """Clears the pytree to free up memory."""
    return jax.tree.map(
        lambda x: x.delete() if isinstance(x, jax.Array) else None, pytree
    )

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
    assert isinstance(options, V1BenchmarkOptions)

    logging.info("Benchmark options: %s", pprint.pformat(options))
    metrics_to_measure = _metrics_to_measure(options)

    with ocp.Context(context=options.context):
      if options.enable_trace:
        jax.profiler.start_trace(context.path / "trace_save")
      if options.async_enabled:
        with metrics.measure("save_blocking", metrics_to_measure):
          f = ocp.save_pytree_async(save_path, pytree)
        with metrics.measure("save_background", metrics_to_measure):
          f.result()
      else:
        with metrics.measure("save_blocking", metrics_to_measure):
          ocp.save_pytree(save_path, pytree)
        with metrics.measure("save_background", metrics_to_measure):
          pass
      context.pytree = self._clear_pytree(context.pytree)
      if options.enable_trace:
        jax.profiler.stop_trace()

      if options.enable_trace:
        jax.profiler.start_trace(context.path / "trace_load")
      with metrics.measure("load", metrics_to_measure):
        restored_pytree = ocp.load_pytree(save_path, abstract_pytree)
      self._clear_pytree(restored_pytree)
      if options.enable_trace:
        jax.profiler.stop_trace()

    return benchmarks_core.TestResult(metrics=metrics)
