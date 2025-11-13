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

"""Benchmarks for orbax.checkpoint.PyTreeCheckpointHandler."""

from collections.abc import Sequence
import dataclasses
import functools
import pprint
from typing import Any
from unittest import mock

from absl import logging
import jax
import orbax.checkpoint as ocp
from orbax.checkpoint._src.testing.benchmarks.core import core as benchmarks_core
from orbax.checkpoint._src.testing.benchmarks.core import metric as metric_lib


def _metrics_to_measure(options: "PyTreeCheckpointOptions") -> list[str]:
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
class PyTreeCheckpointOptions(benchmarks_core.BenchmarkOptions):
  """Configuration options for benchmarks targeting PyTreeCheckpointHandler.

  Each attribute can be a single value or a list of values to create
  a parameter sweep.

  Attributes:
    async_enabled: Whether to use async checkpointer.
    use_ocdbt: Whether to use OCPDBT for checkpointing.
    use_zarr3: Whether to use Zarr3 for checkpointing.
    use_compression: Whether to use compression for checkpointing.
    save_concurrent_gb: The number of concurrent GB to use for saving.
    restore_concurrent_gb: The number of concurrent GB to use for restoring.
    metric_tracemalloc_enabled: Whether to enable tracemalloc metrics.
    metric_tensorstore_enabled: Whether to enable tensorstore metrics.
  """

  async_enabled: bool | Sequence[bool] = True
  use_ocdbt: bool | Sequence[bool] = True
  use_zarr3: bool | Sequence[bool] = False
  use_compression: bool | Sequence[bool] = True
  save_concurrent_gb: int | None | Sequence[int | None] = None
  restore_concurrent_gb: int | None | Sequence[int | None] = None
  metric_tracemalloc_enabled: bool = False
  metric_tensorstore_enabled: bool = False
  use_replica_parallel: bool | Sequence[bool] = False
  enable_replica_parallel_separate_folder: bool | Sequence[bool] = False
  use_jax_array_handler: bool | Sequence[bool] = True
  use_colocated_python: bool | Sequence[bool] = False
  save_device_host_concurrent_gb: int | None | Sequence[int | None] = None

  def is_valid(self):
    assert isinstance(self.use_replica_parallel, bool)
    assert isinstance(self.enable_replica_parallel_separate_folder, bool)
    assert isinstance(self.use_jax_array_handler, bool)
    assert isinstance(self.use_colocated_python, bool)

    if self.enable_replica_parallel_separate_folder and (
        not self.use_replica_parallel or not self.use_ocdbt
    ):
      return False
    if not ocp.multihost.is_pathways_backend() and (
        self.use_colocated_python or not self.use_jax_array_handler
    ):
      return False
    if not self.use_jax_array_handler and self.use_replica_parallel:
      return False
    return True


# ==============================================================================
# 2. Implement the Benchmark Generator
# ==============================================================================
@benchmarks_core.benchmark_options(PyTreeCheckpointOptions)
class PyTreeCheckpointBenchmark(benchmarks_core.BenchmarksGenerator):
  """A concrete generator for `orbax.checkpoint.PyTreeCheckpointHandler`.

  This class provides the specific test logic for benchmarking the
  PyTreeCheckpointHandler with various configurations.
  """

  def _clear_pytree(self, pytree: Any) -> Any:
    """Clears the pytree to free up memory."""
    return jax.tree.map(
        lambda x: x.delete() if isinstance(x, jax.Array) else None, pytree
    )

  def register_array_type_handler(self, options: PyTreeCheckpointOptions):
    if not ocp.multihost.is_pathways_backend():
      array_handler = ocp.type_handlers.ArrayHandler(
          use_replica_parallel=options.use_replica_parallel,
          enable_replica_parallel_separate_folder=options.enable_replica_parallel_separate_folder,
      )
      logging.info("Registering MC-JAX array type handler")
      ocp.type_handlers.register_type_handler(
          jax.Array,
          array_handler,
          override=True,
      )
    else:
      if options.use_jax_array_handler:
        if options.use_persistence_array_handler:
          ocp.type_handlers.register_pathways_handlers(
              use_persistence_array_handler=options.use_persistence_array_handler,
          )
        else:
          ocp.type_handlers.register_pathways_handlers(
              use_colocated_python=options.use_colocated_python,
              use_replica_parallel=options.use_replica_parallel,
              enable_replica_parallel_separate_folder=options.enable_replica_parallel_separate_folder,
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
    save_path = context.path / "pytree"
    options = context.options
    assert isinstance(options, PyTreeCheckpointOptions)

    logging.info("Benchmark options: %s", pprint.pformat(options))

    self.register_array_type_handler(options)

    handler = ocp.PyTreeCheckpointHandler(
        use_ocdbt=options.use_ocdbt,
        use_zarr3=options.use_zarr3,
        use_compression=options.use_compression,
        save_concurrent_gb=options.save_concurrent_gb,
        restore_concurrent_gb=options.restore_concurrent_gb,
        save_device_host_concurrent_gb=options.save_device_host_concurrent_gb,
        is_prioritized_key_fn=lambda key: "a" in ocp.tree.str_keypath(key),
    )

    if options.async_enabled:
      checkpointer = ocp.AsyncCheckpointer(handler)
    else:
      checkpointer = ocp.Checkpointer(handler)
    metrics_to_measure = _metrics_to_measure(options)

    with metrics.measure("save", metrics_to_measure):
      checkpointer.save(save_path, args=ocp.args.PyTreeSave(pytree))

    if options.async_enabled:
      with metrics.measure("wait_until_finished", metrics_to_measure):
        assert hasattr(checkpointer, "wait_until_finished")
        checkpointer.wait_until_finished()

    context.pytree = self._clear_pytree(context.pytree)

    with metrics.measure("restore", metrics_to_measure):
      checkpointer.restore(
          save_path,
          args=ocp.args.PyTreeRestore(
              item=pytree,
              restore_args=ocp.checkpoint_utils.construct_restore_args(pytree),
          ),
      )

    checkpointer.close()
    return benchmarks_core.TestResult(metrics=metrics)
