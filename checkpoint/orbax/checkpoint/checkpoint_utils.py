# Copyright 2023 The Orbax Authors.
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

"""High-level checkpoint utils provided for user convenience."""
import contextlib
import time
from typing import Any, Callable, Iterator, Optional

from absl import logging
from etils import epath
import jax
from jax.experimental import multihost_utils
import numpy as np
from orbax.checkpoint import type_handlers
from orbax.checkpoint import utils


PyTree = Any


def _lock_checkpoint(
    checkpoint_dir: epath.Path,
    step: int,
    step_prefix: Optional[str],
    step_format_fixed_length: Optional[int],
) -> bool:
  """Locks a checkpoint by writing a LOCKED directory."""
  directory = utils.get_save_directory(
      step,
      checkpoint_dir,
      step_prefix=step_prefix,
      step_format_fixed_length=step_format_fixed_length,
  )
  if not directory.exists():
    raise ValueError(f'Parent directory {directory} does not exist.')
  lockdir = utils.lockdir(directory)
  try:
    lockdir.mkdir(parents=False, exist_ok=True)
    return True
  except FileNotFoundError as e:
    logging.warning(
        'Failed to lock step: %d due to: %s. This may be attributed to'
        ' the checkpoint being cleaned up concurrently.',
        step,
        e,
    )
    return False


def _unlock_checkpoint(
    checkpoint_dir: epath.Path,
    step: int,
    step_prefix: Optional[str],
    step_format_fixed_length: Optional[int],
):
  """Removes a LOCKED directory to indicate unlocking."""
  if jax.process_index() == 0:
    directory = utils.get_save_directory(
        step,
        checkpoint_dir,
        step_prefix=step_prefix,
        step_format_fixed_length=step_format_fixed_length,
    )
    utils.lockdir(directory).unlink(missing_ok=True)


def unlock_existing_checkpoints(
    checkpoint_dir: epath.Path,
    step_prefix: Optional[str],
    step_format_fixed_length: Optional[int],
):
  """Removes LOCKED file for all existing steps, if present."""
  steps = utils.checkpoint_steps(checkpoint_dir)
  for step in steps:
    directory = utils.get_save_directory(
        step,
        checkpoint_dir,
        step_prefix=step_prefix,
        step_format_fixed_length=step_format_fixed_length,
    )
    if utils.is_locked(directory):
      _unlock_checkpoint(
          checkpoint_dir, step, step_prefix, step_format_fixed_length
      )


def _reached_desired_step(step: int, until_step: Optional[int]) -> bool:
  if step is None:
    return False
  elif until_step is None:
    return True
  elif step >= until_step:
    return True
  return False


def _wait_for_new_checkpoint(
    checkpoint_dir: epath.Path,
    until_step: Optional[int] = None,
    seconds_to_sleep: int = 1,
    timeout: Optional[int] = None,
    step_prefix: Optional[str] = None,
    step_format_fixed_length: Optional[int] = None,
) -> int:
  """See documentation for wait_for_new_checkpoint."""
  start = time.time()
  stop_time = start + timeout if timeout is not None else None

  def _sleep_and_maybe_exit():
    if stop_time is not None and time.time() + seconds_to_sleep > stop_time:
      return True
    logging.info('Sleeping for %d seconds.', seconds_to_sleep)
    time.sleep(seconds_to_sleep)
    return False

  log_str = f'Waiting for new checkpoint at {checkpoint_dir}. '
  if until_step is not None:
    log_str += f'Waiting until step {until_step} is reached. '
  if until_step is not None:
    log_str += f'Will time out after {timeout} seconds. '
  logging.info(log_str)
  result = -1
  if jax.process_index() == 0:
    while True:
      if not checkpoint_dir.exists():
        if _sleep_and_maybe_exit():
          break
        continue  # continue waiting until directory creation or timeout.
      steps = utils.checkpoint_steps(checkpoint_dir)
      checkpoint_step = max(steps) if steps else None
      if _reached_desired_step(checkpoint_step, until_step):
        if not _lock_checkpoint(
            checkpoint_dir,
            checkpoint_step,
            step_prefix,
            step_format_fixed_length,
        ):
          continue
        result = checkpoint_step
        break
      else:
        if _sleep_and_maybe_exit():
          break
  result = multihost_utils.broadcast_one_to_all(np.int32(result)).item()
  wait_duration = time.time() - start
  jax.monitoring.record_event_duration_secs(
      '/jax/orbax/checkpoint_utils/wait_duration', wait_duration
  )

  if result == -1:
    logging.info('Timed out waiting for new checkpoint. Returning -1.')
  else:
    logging.info('Found new checkpoint step: %d.', result)
  return result


@contextlib.contextmanager
def wait_for_new_checkpoint(
    checkpoint_dir: epath.Path,
    until_step: Optional[int] = None,
    seconds_to_sleep: int = 1,
    timeout: Optional[int] = None,
    step_prefix: Optional[str] = None,
    step_format_fixed_length: Optional[int] = None,
):
  """Waits until a new checkpoint file is found.

  Automatically locks any checkpoint that is returned, and unlocks the
  checkpoint when execution returns to this function.

  Args:
    checkpoint_dir: The directory in which checkpoints are saved.
    until_step: If specified, waits until a step greater than or equal to
      `until_step` has been found. If set to None (default), returns the first
      step found.
    seconds_to_sleep: The number of seconds to sleep for before looking for a
      new checkpoint.
    timeout: The maximum number of seconds to wait. If left as `None`, then the
      process will wait indefinitely.
    step_prefix: A prefix applied to step numbers (e.g. <prefix>_42).
    step_format_fixed_length: Expects to find checkpoint step directories with
      exactly this number of digits (leading zeros if necessary).

  Yields:
    a new checkpoint step, or -1 if the timeout was reached.
  """
  step = _wait_for_new_checkpoint(
      checkpoint_dir,
      until_step=until_step,
      seconds_to_sleep=seconds_to_sleep,
      timeout=timeout,
      step_prefix=step_prefix,
      step_format_fixed_length=step_format_fixed_length,
  )
  try:
    yield step
  finally:
    # Release lock on the checkpoint step.
    if step != -1:
      _unlock_checkpoint(
          checkpoint_dir, step, step_prefix, step_format_fixed_length
      )


def checkpoints_iterator(
    checkpoint_dir: epath.PathLike,
    min_interval_secs: int = 0,
    timeout: Optional[int] = None,
    timeout_fn: Optional[Callable[[], bool]] = None,
    step_prefix: Optional[str] = None,
    step_format_fixed_length: Optional[int] = None,
) -> Iterator[int]:
  """Continuously yield new checkpoint files as they appear.

  Based on the equivalent TF method.

  The iterator only checks for new checkpoints when control flow has been
  reverted to it. This means it can miss checkpoints if your code takes longer
  to run between iterations than `min_interval_secs` or the interval at which
  new checkpoints are written.

  Warning: If CheckpointManager is running in a different process for training
  and is cleaning up old checkpoints (via the `max_to_keep` argument), steps
  returned by this function may not be valid after being clean up by another
  process. In this case, `max_to_keep` should be increased (suggested value: 5)

  The `timeout` argument is the maximum number of seconds to block waiting for
  a new checkpoint.  It is used in combination with the `timeout_fn` as
  follows:

  * If the timeout expires and no `timeout_fn` was specified, the iterator
    stops yielding.
  * If a `timeout_fn` was specified, that function is called and if it returns
    a true boolean value the iterator stops yielding.
  * If the function returns a false boolean value then the iterator resumes the
    wait for new checkpoints.  At this point the timeout logic applies again.

  This behavior gives control to callers on what to do if checkpoints do not
  come fast enough or stop being generated.  For example, if callers have a way
  to detect that the training has stopped and know that no new checkpoints
  will be generated, they can provide a `timeout_fn` that returns `True` when
  the training has stopped.  If they know that the training is still going on
  they return `False` instead.

  Args:
    checkpoint_dir: The directory in which checkpoints are saved.
    min_interval_secs: The minimum number of seconds between yielding
      checkpoints.
    timeout: The maximum number of seconds to wait between checkpoints. If left
      as `None`, then the process will wait indefinitely.
    timeout_fn: Optional function to call after a timeout.  If the function
      returns True, then it means that no new checkpoints will be generated and
      the iterator will exit.  The function is called with no arguments.
    step_prefix: A prefix applied to step numbers (e.g. <prefix>_42).
    step_format_fixed_length: Expects to find checkpoint step directories with
      exactly this number of digits (leading zeros if necessary).

  Yields:
    Integer step numbers of the latest checkpoints as they arrive.
  """
  checkpoint_dir = epath.Path(checkpoint_dir)
  try:
    unlock_existing_checkpoints(
        checkpoint_dir, step_prefix, step_format_fixed_length
    )
  except FileNotFoundError as e:
    logging.warning(
        'Encountered error while unlocking existing checkpoints. Some'
        ' checkpoints may still be locked. %s.',
        e,
    )
  checkpoint_step = None
  while True:
    until_step = checkpoint_step + 1 if checkpoint_step is not None else None
    with wait_for_new_checkpoint(
        checkpoint_dir,
        until_step=until_step,
        timeout=timeout,
        step_prefix=step_prefix,
        step_format_fixed_length=step_format_fixed_length,
    ) as new_checkpoint_step:
      if new_checkpoint_step == -1:
        if not timeout_fn:
          logging.info('Timed-out waiting for a checkpoint.')
          return
        if timeout_fn():
          # The timeout_fn indicated that we are truly done.
          return
        else:
          # The timeout_fn indicated that more checkpoints may come.
          continue
      start = time.time()
      checkpoint_step = new_checkpoint_step
      yield checkpoint_step
    time_to_next_eval = start + min_interval_secs - time.time()
    if time_to_next_eval > 0:
      time.sleep(time_to_next_eval)


# TODO(b/274813763): Remove this function when no longer depended on by Flax.
def restore_args_from_target(
    mesh: jax.sharding.Mesh, target: PyTree, axes_tree: PyTree
) -> PyTree:
  """DEPRECATED, DO NOT USE.

  Creates restore_args given a target PyTree.

  This method should be used in conjunction with a CheckpointManager or
  Checkpointer that wraps a PyTreeCheckpointHandler.

  For example:

  mngr = CheckpointManager(path, Checkpointer(PyTreeCheckpointHandler()))
  restore_args = restore_args_from_target(mesh, train_state, train_state_axes)
  restore_kwargs = {'restore_args': restore_args}
  mngr.restore(..., restore_kwargs=restore_kwargs)

  OR

  mngr = CheckpointManager(path, {
      'train_state': Checkpointer(PyTreeCheckpointHandler())
  })
  restore_args = restore_args_from_target(mesh, train_state, train_state_axes)
  restore_kwargs = {'train_state': {'restore_args': restore_args} }
  mngr.restore(..., restore_kwargs=restore_kwargs)

  OR

  ckptr = Checkpointer(PyTreeCheckpointHandler())
  restore_args = restore_args_from_target(mesh, train_state, train_state_axes)
  ckptr.restore(..., restore_args=restore_args)

  If a leaf in target does is a np.ndarray, or int, or string, for example, a
  corresponding value for that leaf must be provided in axes_tree, but will be
  ignored.

  Args:
    mesh: jax.sharding.Mesh to shard arrays with.
    target: The returned value will match the structure of `target`, will be
      used to set the desired dtype and restoration shape.
    axes_tree: A PyTree matching `target` which will be used to set the
      restoration sharding.

  Returns:
    A PyTree matching target of RestoreArgs (or ArrayRestoreArgs) objects.
  """

  def _restore_args(value, axes):
    restore_type = type(value)
    dtype = None
    if hasattr(value, 'dtype'):
      dtype = value.dtype
    if isinstance(value, jax.Array):
      return type_handlers.ArrayRestoreArgs(
          restore_type=restore_type,
          mesh=mesh,
          mesh_axes=axes,
          global_shape=value.shape,
          dtype=value.dtype,
      )
    else:
      return type_handlers.RestoreArgs(restore_type=restore_type, dtype=dtype)

  return jax.tree_util.tree_map(_restore_args, target, axes_tree)


def construct_restore_args(
    target: PyTree, sharding_tree: PyTree, set_global_shape: bool = True
) -> PyTree:
  """Creates restore_args given a target PyTree.

  This method should be used in conjunction with a CheckpointManager or
  Checkpointer that wraps a PyTreeCheckpointHandler.

  For example::

    mngr = CheckpointManager(path, Checkpointer(PyTreeCheckpointHandler()))
    restore_args = construct_restore_args(train_state, train_state_sharding)
    restore_kwargs = {'restore_args': restore_args}
    mngr.restore(..., restore_kwargs=restore_kwargs)

  OR::

    mngr = CheckpointManager(path, {
        'train_state': Checkpointer(PyTreeCheckpointHandler())
    })
    restore_args = construct_restore_args(train_state, train_state_sharding)
    restore_kwargs = {'train_state': {'restore_args': restore_args} }
    mngr.restore(..., restore_kwargs=restore_kwargs)

  OR::

    ckptr = Checkpointer(PyTreeCheckpointHandler())
    restore_args = construct_restore_args(train_state, train_state_sharding)
    ckptr.restore(..., restore_args=restore_args)

  If a leaf in target does is a np.ndarray, or int, or string, for example, a
  corresponding value for that leaf must be provided in axes_tree, but will be
  ignored.

  Args:
    target: The returned value will match the structure of `target`, will be
      used to set the desired dtype and restoration shape.
    sharding_tree: A PyTree matching `target` which will be used to set the
      restoration sharding.
    set_global_shape: If true, set the `global_shape` field of ArrayRestoreArgs.

  Returns:
    A PyTree matching target of RestoreArgs (or ArrayRestoreArgs) objects.
  """

  def _restore_args(value: Any, sharding: jax.sharding.Sharding):
    restore_type = type(value)
    dtype = None
    if hasattr(value, 'dtype'):
      dtype = value.dtype
    if isinstance(value, jax.Array):
      return type_handlers.ArrayRestoreArgs(
          restore_type=restore_type,
          sharding=sharding,
          global_shape=value.shape if set_global_shape else None,
          dtype=value.dtype,
      )
    else:
      return type_handlers.RestoreArgs(restore_type=restore_type, dtype=dtype)

  return jax.tree_util.tree_map(_restore_args, target, sharding_tree)
