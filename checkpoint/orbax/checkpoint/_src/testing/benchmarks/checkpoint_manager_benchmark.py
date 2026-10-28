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
import os
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
from orbax.checkpoint._src.testing.benchmarks.core import inventory as inventory_lib
from orbax.checkpoint._src.testing.benchmarks.core import metric as metric_lib
from orbax.checkpoint._src.testing.benchmarks.core import pytree_utils


@dataclasses.dataclass(frozen=True)
class CheckpointManagerBenchmarkOptions(benchmarks_core.BenchmarkOptions):
  """Configuration options for benchmarks targeting CheckpointManager.

  Attributes:
    save_interval_steps: Step interval between saves.
    max_to_keep: Maximum number of checkpoints to retain on storage.
    train_steps: Number of timed save iterations to execute.
    overwrite_existing_step: Benchmarks overwriting an already-populated step
      directory on storage (e.g. after preemption/restart or when a step path
      already exists). Designed to work in two phases: 1) Before the timed loop,
      `_run_save_loop` runs an untimed save at step 0 (`mngr.save(0)` +
      `wait_until_finished()`) so `0/` and all item subdirectories (`0/pytree`,
      `0/json_item`, etc.) physically exist on storage. 2) On every timed
      iteration, the benchmark targets `step = 0` and clears only the in-memory
      `mngr._checkpoints` list (`delete_if`) without calling `mngr.delete(0)`
      (which would delete `0/` on storage). This allows `mngr.save(0,
      force=True)` to bypass `StepAlreadyExistsError` and call
      `checkpointer.save(..., force=True)` on the existing `0/` directory,
      exercising `remove_existing_path` + `create_directory`.
    enable_background_delete: Forwarded to `CheckpointManagerOptions`.
    warmup_save: Runs an untimed warmup save at step 0 and pre-warms storage
      locality for steps `0..train_steps` before the timed loop. Eliminates
      first-save cold-start overhead (tmp-dir cleanup thread, XLA D2H
      compilation, TensorStore init) and storage locality LRU cache-miss skew
      when comparing fresh steps (`1..train_steps`) against
      `overwrite_existing_step=True` (step `0`). Implicitly enabled when
      `overwrite_existing_step=True`.
    verify_restore: Whether to restore the latest step and run correctness
      checks after the timed save loop. Set `False` for large-model save-only
      benchmarks to avoid allocating a second copy of the PyTree in HBM.
    enable_pinned_host_transfer: If set, configures `StandardCheckpointHandler`
      with `enable_pinned_host_transfer` via `create_default_handler_registry`.
    enable_async_host_untile: If True, configures `StandardCheckpointHandler`
      with `enable_async_host_untile` to defer host untiling to background
      worker threads.
    enable_raw_tiled_transfer: If True, configures `StandardCheckpointHandler`
      with `enable_raw_tiled_transfer` to save and restore raw tiled bytes.
    verify_mode2_resave: If True, after restoring Mode 2, saves normally and
      verifies bitwise equivalence against a normal baseline save.
  """

  save_interval_steps: int | Sequence[int] = 1
  max_to_keep: int | Sequence[int] = 1
  train_steps: int | Sequence[int] = 1
  overwrite_existing_step: bool | Sequence[bool] = False
  enable_background_delete: bool | Sequence[bool] = True
  warmup_save: bool | Sequence[bool] = False
  verify_restore: bool | Sequence[bool] = True
  enable_pinned_host_transfer: bool | None | Sequence[bool | None] = None
  enable_async_host_untile: bool | None | Sequence[bool | None] = None
  enable_raw_tiled_transfer: bool | None | Sequence[bool | None] = None
  verify_mode2_resave: bool | Sequence[bool] = False


def _clear_pytree(pytree: Any) -> None:
  """Deletes on-device array buffers in `pytree` and triggers GC."""
  for leaf in jax.tree.leaves(pytree):
    if isinstance(leaf, jax.Array) and not leaf.is_deleted():
      leaf.delete()
  gc.collect()


def _rewrap_pytree_arrays(pytree: Any) -> Any:
  """Rewraps jax.Arrays to drop cached host buffers before each save step."""
  return jax.tree.map(
      lambda x: (
          x._rewrap_with_aval_and_sharding(x.aval, x.sharding)  # pylint: disable=protected-access
          if isinstance(x, jax.Array)
          else x
      ),
      pytree,
  )


def _refresh_save_pytree(
    composite_args: args_lib.Composite,
    context: benchmarks_core.TestContext | None,
) -> None:
  """Clears cached host buffers on context.pytree and composite_args."""
  if context is not None and context.pytree is not None:
    context.pytree = _rewrap_pytree_arrays(context.pytree)
  if 'pytree' in composite_args and hasattr(composite_args['pytree'], 'item'):
    composite_args['pytree'].item = (
        context.pytree
        if context is not None and context.pytree is not None
        else _rewrap_pytree_arrays(composite_args['pytree'].item)
    )


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
      context: benchmarks_core.TestContext | None = None,
  ) -> int:
    """Runs optional warmup and the timed save loop.

    Args:
      mngr: The CheckpointManager instance under test.
      composite_args: Composite save arguments for each step.
      options: Benchmark options controlling warmup and step-overwrite behavior.
      metrics: Metrics collector for timed save and wait operations.
      context: Optional benchmark context used to resolve profiler trace paths.

    Returns:
      The last checkpoint step number saved, or -1 if no step was saved.
    """
    use_warmup = bool(options.warmup_save or options.overwrite_existing_step)
    train_steps = int(options.train_steps)  # pyrefly: ignore[bad-argument-type]
    if use_warmup:
      _refresh_save_pytree(composite_args, context)
      mngr._maybe_await_cleanup_tmp_directory()  # pylint: disable=protected-access
      mngr.save(0, args=composite_args, force=True)
      mngr.wait_until_finished()
      mngr._wait_for_prev_save_duration = 0.0  # pylint: disable=protected-access

    step_saved = -1
    trace_step_idx = min(1, train_steps - 1)
    for step_idx in range(train_steps):
      _refresh_save_pytree(composite_args, context)
      multihost.sync_global_processes(f'cm_benchmark:step_start_{step_idx}')
      if options.overwrite_existing_step:
        step = 0
      elif use_warmup:
        step = step_idx + 1
      else:
        step = step_idx
      if options.overwrite_existing_step and hasattr(mngr, '_checkpoints'):
        # Clear only in-memory step tracking while keeping physical `0/` on
        # storage (`mngr.delete(0)` would remove `0/` from disk). This lets
        # `mngr.save(0, force=True)` bypass `StepAlreadyExistsError` and
        # overwrite the existing `0/` directory via `checkpointer.save`.
        mngr._checkpoints.delete_if(lambda _: True)  # pylint: disable=protected-access
      logging.info('Saving checkpoint at step %d', step)
      trace_dir = (
          context.trace_path(f'save_{step_idx}')
          if context is not None and step_idx == trace_step_idx
          else None
      )
      if trace_dir is not None:
        jax.profiler.start_trace(str(trace_dir))
      try:
        with metrics.measure(f'save_{step_idx}'):
          if mngr.save(
              step,
              args=composite_args,
              force=bool(options.overwrite_existing_step),
          ):
            step_saved = step
        with metrics.measure(f'wait_until_finished_{step_idx}'):
          mngr.wait_until_finished()
      finally:
        if trace_dir is not None:
          jax.profiler.stop_trace()
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

    use_warmup = bool(options.warmup_save or options.overwrite_existing_step)
    ckpt_path = context.path
    if use_warmup:
      attempt = int(multihost.broadcast_one_to_all(np.int64(time.time() * 1e3)))
      ckpt_path = context.path / f'attempt_{attempt}'
    cm_options = checkpoint_manager.CheckpointManagerOptions(
        save_interval_steps=options.save_interval_steps,  # pyrefly: ignore[bad-argument-type]
        max_to_keep=options.max_to_keep,  # pyrefly: ignore[bad-argument-type]
        enable_background_delete=bool(options.enable_background_delete),
    )
    handler_registry = None
    if (
        options.enable_pinned_host_transfer is not None
        or options.enable_async_host_untile is not None
        or options.enable_raw_tiled_transfer is not None
    ):
      handler_registry = ocp.handlers.create_default_handler_registry(
          pytree=ocp.StandardCheckpointHandler(
              enable_pinned_host_transfer=(
                  bool(options.enable_pinned_host_transfer)
                  if options.enable_pinned_host_transfer is not None
                  else None
              ),
              enable_async_host_untile=(
                  bool(options.enable_async_host_untile)
                  if options.enable_async_host_untile is not None
                  else False
              ),
              enable_raw_tiled_transfer=(
                  bool(options.enable_raw_tiled_transfer)
                  if options.enable_raw_tiled_transfer is not None
                  else False
              ),
          )
      )
    mngr = checkpoint_manager.CheckpointManager(
        ckpt_path, options=cm_options, handler_registry=handler_registry
    )

    json_data = {'a': 1, 'b': 'test'}
    pytree_for_restore = self._get_pytree_for_restore(pytree)

    # Exclude RandomKeyCheckpointHandlers, which hardcode
    # `create_directories_asynchronously=False` and block the main thread on
    # directory creation during `async_save`.
    composite_args = args_lib.Composite(
        pytree=args_lib.StandardSave(pytree),
        json_item=args_lib.JsonSave(json_data),
    )
    restore_args = args_lib.Composite(
        pytree=args_lib.StandardRestore(pytree_for_restore),
        json_item=args_lib.JsonRestore(),
    )

    step_saved = self._run_save_loop(
        mngr, composite_args, options, metrics, context=context
    )
    if step_saved == -1:
      raise AssertionError('No checkpoint was saved.')
    if not options.verify_restore:
      mngr.close()
      _clear_pytree(pytree)
      return benchmarks_core.TestResult(
          metrics=metrics,
          inventory=inventory_lib.CheckpointInventory(),
      )

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

    if options.verify_mode2_resave:
      with metrics.measure('mode2_resave_verification'):
        logging.info('Starting Mode 2 resave verification...')
        normal_orig_path = ckpt_path / 'verify_normal_orig'
        normal_resaved_path = ckpt_path / 'verify_normal_resaved'
        if normal_orig_path.exists():
          normal_orig_path.rmtree()
        if normal_resaved_path.exists():
          normal_resaved_path.rmtree()
        ckptr = ocp.Checkpointer(
            ocp.StandardCheckpointHandler(
                enable_pinned_host_transfer=False,
                enable_raw_tiled_transfer=False,
            )
        )
        ckptr.save(normal_orig_path, args=args_lib.StandardSave(pytree))
        ckptr.save(
            normal_resaved_path,
            args=args_lib.StandardSave(restored['pytree']),
        )
        zeros_orig = self._get_pytree_for_restore(pytree)
        zeros_resaved = self._get_pytree_for_restore(pytree)
        norm_orig = ckptr.restore(
            normal_orig_path, args=args_lib.StandardRestore(zeros_orig)
        )
        norm_resaved = ckptr.restore(
            normal_resaved_path, args=args_lib.StandardRestore(zeros_resaved)
        )
        pytree_utils.assert_pytree_equal(norm_orig, norm_resaved)

        def _walk_files(p: epath.Path):
          for item in p.iterdir():
            if item.is_dir():
              yield from _walk_files(item)
            elif item.is_file():
              yield item

        for orig_f in _walk_files(normal_orig_path):
          if not orig_f.name.endswith(('.json', '.txt', '.pbtxt')):
            rel = orig_f.relative_to(normal_orig_path)
            resaved_f = normal_resaved_path / rel
            if resaved_f.exists() and resaved_f.is_file():
              assert (
                  orig_f.read_bytes() == resaved_f.read_bytes()
              ), f'Byte mismatch in on-disk chunk {rel}'
        ckptr.close()
        logging.info('Mode 2 resave verification passed successfully!')

    mngr.close()
    return benchmarks_core.TestResult(metrics=metrics)
