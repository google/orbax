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

"""Benchmarks for orbax.checkpoint.experimental.v1._src.training.pathways.snapshotter."""

from collections.abc import Sequence
import dataclasses
import pprint
import time
from typing import Any

from absl import logging
import jax
import orbax.checkpoint as ocp
from orbax.checkpoint._src.testing.benchmarks.core import core as benchmarks_core
from orbax.checkpoint._src.testing.benchmarks.core import metric as metric_lib
from orbax.checkpoint.experimental.v1._src.training.pathways import snapshotter


def _metrics_to_measure(options: 'SnapshotterOptions') -> list[str]:
  """Returns the list of metrics to measure."""
  metrics = ['time', 'rss']
  if options.metric_tracemalloc_enabled:
    metrics.append('tracemalloc')
  return metrics


# ==============================================================================
# 1. Define the Options Dataclass for this specific benchmark
# ==============================================================================
@dataclasses.dataclass(frozen=True)
class SnapshotterOptions(benchmarks_core.BenchmarkOptions):
  """Configuration options for benchmarks targeting Snapshotter.

  Each attribute can be a single value or a list of values to create
  a parameter sweep.

  Attributes:
    metric_tracemalloc_enabled: Whether to enable tracemalloc metrics.
  """

  metric_tracemalloc_enabled: bool | Sequence[bool] = False

  def is_valid(self):
    is_pw = ocp.multihost.is_pathways_backend()
    if not is_pw:
      logging.warning('Snapshotter benchmark requires Pathways backend.')
    return is_pw


# ==============================================================================
# 2. Implement the Benchmark Generator
# ==============================================================================
@benchmarks_core.benchmark_options(SnapshotterOptions)
class SnapshotterBenchmark(benchmarks_core.BenchmarksGenerator):
  """A concrete generator for `orbax.checkpoint.experimental.v1._src.training.pathways.snapshotter.Snapshotter`."""

  def _clear_pytree(self, pytree: Any) -> Any:
    """Clears the pytree to free up memory."""
    return jax.tree.map(
        lambda x: x.delete() if isinstance(x, jax.Array) else None, pytree
    )

  def test_fn(
      self, context: benchmarks_core.TestContext
  ) -> benchmarks_core.TestResult:
    """The core test logic for a single save/restore cycle."""
    metrics = metric_lib.Metrics()
    pytree = context.pytree
    options = context.options
    assert isinstance(options, SnapshotterOptions)

    logging.info('Benchmark options: %s', pprint.pformat(options))

    manager = snapshotter.Snapshotter()
    metrics_to_measure = _metrics_to_measure(options)

    with metrics.measure('save_pytree', metrics_to_measure):
      manager.save_pytree(step=0, state=pytree)

    abstract_state = ocp.checkpoint_utils.construct_restore_args(pytree)
    context.pytree = self._clear_pytree(context.pytree)

    logging.info('jax.devices(): %s', jax.local_devices())

    for _ in range(10):
      time.sleep(10)
      logging.info(
          'sleep 10s, process index: %d', ocp.multihost.process_index()
      )
      logging.info('jax.devices(): %s', jax.local_devices())

    with metrics.measure('load_pytree', metrics_to_measure):
      restored_pytree = manager.load_pytree(abstract_state=abstract_state)

    self._clear_pytree(restored_pytree)

    return benchmarks_core.TestResult(metrics=metrics)
