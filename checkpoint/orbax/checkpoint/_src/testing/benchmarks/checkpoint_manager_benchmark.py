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

"""Benchmarks for orbax.checkpoint.checkpoint_manager."""

import dataclasses
from typing import Any, Sequence

from absl import logging
import jax
import numpy as np
from orbax.checkpoint import args as args_lib
from orbax.checkpoint import checkpoint_manager
from orbax.checkpoint import multihost
from orbax.checkpoint import type_handlers
from orbax.checkpoint import utils
from orbax.checkpoint._src.testing.benchmarks.core import core as benchmarks_core
from orbax.checkpoint._src.testing.benchmarks.core import metric as metric_lib
from orbax.checkpoint._src.testing.benchmarks.core import pytree_utils


@dataclasses.dataclass(frozen=True)
class CheckpointManagerBenchmarkOptions(benchmarks_core.BenchmarkOptions):
  """Configuration options for benchmarks targeting CheckpointManager."""

  save_interval_steps: int | Sequence[int] = 1
  max_to_keep: int | Sequence[int] = 1
  train_steps: int | Sequence[int] = 1


@benchmarks_core.benchmark_options(CheckpointManagerBenchmarkOptions)
class CheckpointManagerBenchmark(benchmarks_core.BenchmarksGenerator):
  """A benchmark for orbax.checkpoint.CheckpointManager."""

  def _get_pytree_for_restore(self, pytree: Any) -> Any:
    """Returns abstract pytree for restore."""
    return jax.tree.map(utils.to_shape_dtype_struct, pytree)

  # TODO(nikhilbansall): Expand to full training loop with multiple steps,
  # including a step of training.
  def test_fn(
      self, context: benchmarks_core.TestContext
  ) -> benchmarks_core.TestResult:
    metrics = metric_lib.Metrics()
    pytree = context.pytree
    options = context.options
    assert isinstance(options, CheckpointManagerBenchmarkOptions)


    cm_options = checkpoint_manager.CheckpointManagerOptions(
        save_interval_steps=options.save_interval_steps,
        max_to_keep=options.max_to_keep,
    )
    mngr = checkpoint_manager.CheckpointManager(
        context.path, options=cm_options
    )

    json_data = {'a': 1, 'b': 'test'}
    random_key = jax.random.key(0)
    np_random_key = np.random.get_state()
    pytree_for_restore = self._get_pytree_for_restore(pytree)

    save_kwargs = {
        'pytree': args_lib.StandardSave(pytree),
        'json_item': args_lib.JsonSave(json_data),
        'np_random_key': args_lib.NumpyRandomKeySave(np_random_key),
    }
    restore_kwargs = {
        'pytree': args_lib.StandardRestore(pytree_for_restore),
        'json_item': args_lib.JsonRestore(),
        'np_random_key': args_lib.NumpyRandomKeyRestore(),
    }
    if not multihost.is_pathways_backend():
      save_kwargs['jax_random_key'] = args_lib.JaxRandomKeySave(random_key)
      restore_kwargs['jax_random_key'] = args_lib.JaxRandomKeyRestore()
    composite_args = args_lib.Composite(**save_kwargs)
    restore_args = args_lib.Composite(**restore_kwargs)

    step_saved = -1
    for step in range(options.train_steps):
      logging.info('Saving checkpoint at step %d', step)
      with metrics.measure(f'save_{step}'):
        if mngr.save(step, args=composite_args):
          step_saved = step
      with metrics.measure(f'wait_until_finished_{step}'):
        mngr.wait_until_finished()
      logging.info('Finished saving checkpoint at step %d', step)

    if step_saved == -1:
      raise AssertionError('No checkpoint was saved.')

    latest_step = mngr.latest_step()
    assert latest_step == step_saved, (
        f'Expected latest step to be {step_saved}, got {latest_step}'
    )

    with metrics.measure(f'restore_{latest_step}'):
      logging.info('Restoring checkpoint at step %d', latest_step)
      restored = mngr.restore(latest_step, args=restore_args)
      logging.info('Finished restoring checkpoint at step %d', latest_step)

    with metrics.measure('correctness_check'):
      pytree_utils.assert_pytree_equal(pytree, restored['pytree'])
      assert (
          json_data == restored['json_item']
      ), f"Expected {json_data}, got {restored['json_item']}"
      if not multihost.is_pathways_backend():
        assert jax.numpy.array_equal(
            random_key, restored['jax_random_key']
        ), f"Expected {random_key}, got {restored['jax_random_key']}"
      jax.tree.map(
          np.testing.assert_equal, np_random_key, restored['np_random_key']
      )

    mngr.close()
    return benchmarks_core.TestResult(metrics=metrics)
