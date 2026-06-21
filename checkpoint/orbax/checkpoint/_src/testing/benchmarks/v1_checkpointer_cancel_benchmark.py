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

"""Benchmarks for Orbax V1 Checkpointer Cancellation."""

import dataclasses
import time
from typing import Sequence

from absl import logging
from orbax.checkpoint._src.testing.benchmarks.core import core as benchmarks_core
from orbax.checkpoint._src.testing.benchmarks.core import metric as metric_lib
from orbax.checkpoint.experimental.v1 import training

Checkpointer = training.Checkpointer


@dataclasses.dataclass(frozen=True)
class V1CheckpointerCancelBenchmarkOptions(benchmarks_core.BenchmarkOptions):
  """Configuration options for benchmarks targeting V1 Checkpointer Cancellation."""

  cancel_delay_secs: float | Sequence[float] = 2.0
  perform_cancellation: bool | Sequence[bool] = True


@benchmarks_core.benchmark_options(V1CheckpointerCancelBenchmarkOptions)
class V1CheckpointerCancelBenchmark(benchmarks_core.BenchmarksGenerator):
  """A benchmark for cancellation using orbax.checkpoint's V1 Checkpointer."""

  def test_fn(
      self, context: benchmarks_core.TestContext
  ) -> benchmarks_core.TestResult:
    metrics = metric_lib.Metrics()
    pytree = context.pytree
    options = context.options
    assert isinstance(options, V1CheckpointerCancelBenchmarkOptions)

    ckptr = Checkpointer(context.path)
    step = 1

    logging.info('Starting V1 Checkpointer save at step %d', step)
    start_time = time.time()

    with metrics.measure('async_save_dispatch'):
      ckptr.save_async(step, state={'array': pytree})

    dispatch_time = time.time() - start_time
    logging.info('Save dispatch completed in %.4f seconds', dispatch_time)

    if options.perform_cancellation:
      logging.info(
          'Sleeping for %.2f seconds before cancellation...',
          options.cancel_delay_secs,
      )
      time.sleep(options.cancel_delay_secs)

      logging.info('Calling ckptr.cancel(step=%d)', step)
      with metrics.measure('cancel_cost'):
        ckptr.cancel(step=step)
      logging.info('Cancellation request successfully applied.')

      with metrics.measure('async_save_dispatch'):
        ckptr.save_async(step+1, state={'array': pytree})

    ckptr.wait()
    for step_info in ckptr.checkpoints:
      logging.info('ckptr.checkpoints: %s', step_info.step)

    logging.info(
        'Waiting for background operations to finish via wait_until_finished().'
    )
    with metrics.measure('wait_until_finished'):
      ckptr._manager.wait_until_finished()  # pylint: disable=protected-access

    total_time = time.time() - start_time
    logging.info('Workflow entirely resolved in %.4f seconds.', total_time)

    return benchmarks_core.TestResult(metrics=metrics)
