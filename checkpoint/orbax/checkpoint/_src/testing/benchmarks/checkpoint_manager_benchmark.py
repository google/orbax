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

"""Benchmarks for orbax.checkpoint.checkpoint_manager."""

from collections.abc import Sequence
import dataclasses
import gc
import time
from typing import Any

from absl import logging
from etils import epath
import jax
import numpy as np
import orbax.checkpoint as ocp
from orbax.checkpoint import args as args_lib
from orbax.checkpoint import checkpoint_manager
from orbax.checkpoint import multihost
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
  overwrite_existing_step: bool | Sequence[bool] = False
  enable_background_delete: bool | Sequence[bool] = True
  warmup_save: bool | Sequence[bool] = False
  verify_restore: bool | Sequence[bool] = True
  skip_sync_file_validations: bool | Sequence[bool] = False


@benchmarks_core.benchmark_options(CheckpointManagerBenchmarkOptions)
class CheckpointManagerBenchmark(benchmarks_core.BenchmarksGenerator):
  """A benchmark for orbax.checkpoint.CheckpointManager."""

  def _get_pytree_for_restore(self, pytree: Any) -> Any:
    """Returns abstract pytree for restore."""
    return jax.tree.map(utils.to_shape_dtype_struct, pytree)

  def _run_save_loop(
      self,
      mngr: checkpoint_manager.CheckpointManager,
      composite_args: args_lib.Composite,
      options: CheckpointManagerBenchmarkOptions,
      metrics: metric_lib.Metrics,
  ) -> int:
    """Runs optional warmup and the timed save loop."""
    # pylint: disable=protected-access
    overwrite = bool(options.overwrite_existing_step)
    use_warmup = bool(options.warmup_save or overwrite)
    train_steps = int(options.train_steps)  # pyrefly: ignore[bad-argument-type]
    if use_warmup:
      mngr._maybe_await_cleanup_tmp_directory()
      mngr.save(0, args=composite_args, force=True)
      mngr.wait_until_finished()
      mngr._wait_for_prev_save_duration = 0.0

    step_saved = -1
    for step_idx in range(train_steps):
      multihost.sync_global_processes(f'cm_benchmark:step_start_{step_idx}')
      step = 0 if overwrite else step_idx + int(use_warmup)
      if overwrite and hasattr(mngr, '_checkpoints'):
        # Clear in-memory step tracking while keeping physical 0/ on storage.
        mngr._checkpoints.delete_if(lambda _: True)
      logging.info('Saving checkpoint at step %d', step)
      with metrics.measure(f'save_{step_idx}'):
        if mngr.save(step, args=composite_args, force=overwrite):
          step_saved = step
      with metrics.measure(f'wait_until_finished_{step_idx}'):
        mngr.wait_until_finished()
      logging.info('Finished saving checkpoint at step %d', step)
    return step_saved

  # TODO(nikhilbansall): Expand to full training loop with multiple steps,
  # including a step of training.
  def test_fn(
      self, context: benchmarks_core.TestContext
  ) -> benchmarks_core.TestResult:
    metrics = metric_lib.Metrics()
    pytree = context.pytree
    options = context.options
    assert isinstance(options, CheckpointManagerBenchmarkOptions)

    if multihost.is_pathways_backend():
      checkpointing_impl = ocp.pathways.CheckpointingImpl.from_options(
          use_remote_python=True
      )
      ocp.pathways.register_type_handlers(
          checkpointing_impl=checkpointing_impl,
      )

    ckpt_path = context.path
    if options.warmup_save or options.overwrite_existing_step:
      attempt = int(multihost.broadcast_one_to_all(np.int64(time.time() * 1e3)))
      ckpt_path = ckpt_path / f'attempt_{attempt}'
    cm_options = checkpoint_manager.CheckpointManagerOptions(
        save_interval_steps=options.save_interval_steps,  # pyrefly: ignore[bad-argument-type]
        max_to_keep=options.max_to_keep,  # pyrefly: ignore[bad-argument-type]
        enable_background_delete=bool(options.enable_background_delete),
        file_options=ocp.options.FileOptions(
            skip_sync_file_validations=bool(options.skip_sync_file_validations)
        ),
    )
    mngr = checkpoint_manager.CheckpointManager(ckpt_path, options=cm_options)

    json_data = {'a': 1, 'b': 'test'}
    pytree_for_restore = self._get_pytree_for_restore(pytree)

    # Exclude RandomKeyCheckpointHandlers (sync directory creation blocks save).
    composite_args = args_lib.Composite(
        pytree=args_lib.StandardSave(pytree),
        json_item=args_lib.JsonSave(json_data),
    )
    restore_args = args_lib.Composite(
        pytree=args_lib.StandardRestore(pytree_for_restore),
        json_item=args_lib.JsonRestore(),
    )

    step_saved = self._run_save_loop(mngr, composite_args, options, metrics)
    if step_saved == -1:
      raise AssertionError('No checkpoint was saved.')
    if not options.verify_restore:
      mngr.close()
      for leaf in jax.tree.leaves(pytree):
        if isinstance(leaf, jax.Array) and not leaf.is_deleted():
          leaf.delete()
      gc.collect()
      return benchmarks_core.TestResult(metrics=metrics, path=ckpt_path)

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

    mngr.close()
    return benchmarks_core.TestResult(metrics=metrics, path=ckpt_path)
