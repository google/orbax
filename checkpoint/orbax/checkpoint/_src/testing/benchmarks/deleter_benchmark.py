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

"""Benchmarks for Orbax checkpoint deletion."""

import dataclasses
import time
from typing import Any, Sequence

from absl import logging
import jax
from orbax.checkpoint import args as args_lib
from orbax.checkpoint import checkpoint_manager
from orbax.checkpoint import utils
from orbax.checkpoint._src.testing.benchmarks.core import core as benchmarks_core
from orbax.checkpoint._src.testing.benchmarks.core import metric as metric_lib


@dataclasses.dataclass(frozen=True)
class DeleterBenchmarkOptions(benchmarks_core.BenchmarkOptions):
  """Configuration options for benchmarks targeting deletion."""

  save_interval_steps: int | Sequence[int] = 1
  max_to_keep: int | Sequence[int] = 1
  train_steps: int | Sequence[int] = 5
  num_deletion_threads: int | Sequence[int] = 1


@benchmarks_core.benchmark_options(DeleterBenchmarkOptions)
class DeleterBenchmark(benchmarks_core.BenchmarksGenerator):
  """A benchmark for Orbax checkpoint deletion."""

  def _get_pytree_for_restore(self, pytree: Any) -> Any:
    """Returns abstract pytree for restore."""
    return jax.tree.map(utils.to_shape_dtype_struct, pytree)

  def test_fn(
      self, context: benchmarks_core.TestContext
  ) -> benchmarks_core.TestResult:
    metrics = metric_lib.Metrics()
    pytree = context.pytree
    options = context.options
    assert isinstance(options, DeleterBenchmarkOptions)

    cm_options = checkpoint_manager.CheckpointManagerOptions(
        save_interval_steps=options.save_interval_steps,
        max_to_keep=options.max_to_keep,
        num_deletion_threads=options.num_deletion_threads,
    )
    mngr = checkpoint_manager.CheckpointManager(
        context.path, options=cm_options
    )

    json_data = {'a': 1, 'b': 'test'}
    _ = self._get_pytree_for_restore(pytree)

    save_kwargs = {
        'pytree': args_lib.StandardSave(pytree),
        'json_item': args_lib.JsonSave(json_data),
    }
    composite_args = args_lib.Composite(**save_kwargs)

    step_saved = -1
    for step in range(options.train_steps):
      logging.info('Saving checkpoint at step %d', step)
      with metrics.measure(f'save_{step}'):
        if mngr.save(step, args=composite_args):
          step_saved = step
      with metrics.measure(f'wait_until_finished_{step}'):
        mngr.wait_until_finished()
      logging.info('Finished saving checkpoint at step %d', step)

      # Manually delete the PREVIOUS step to measure deletion time!
      if step > 0:
        logging.info('Deleting checkpoint at step %d', step - 1)
        with metrics.measure(
            f'delete_step_{step-1}_threads_{options.num_deletion_threads}'
        ):
          start_time = time.time()
          # pylint: disable=protected-access
          mngr._checkpoint_deleter.delete(step - 1)
        logging.info(
            'Finished deleting checkpoint at step %d for threads %d took %f'
            ' seconds',
            step - 1,
            options.num_deletion_threads,
            time.time() - start_time,
        )

    if step_saved == -1:
      raise AssertionError('No checkpoint was saved.')

    mngr.close()
    return benchmarks_core.TestResult(metrics=metrics)
