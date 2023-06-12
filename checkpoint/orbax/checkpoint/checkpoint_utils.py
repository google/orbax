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


def _lock_checkpoint(checkpoint_dir: epath.Path, step: int) -> bool:
  """Locks a checkpoint by writing a LOCKED directory."""
  lockdir = utils.lockdir(checkpoint_dir, step)
  try:
    lockdir.mkdir(parents=False, exist_ok=True)
    return True
  except FileNotFoundError:
    logging.info(
        'Checkpoint step: %d was cleaned up while attempting to lock.', step
    )
    return False


def _unlock_checkpoint(checkpoint_dir: epath.Path, step: int):
  """Removes a LOCKED directory to indicate unlocking."""
  utils.lockdir(checkpoint_dir, step).rmdir()


def _unlock_existing_checkpoints(checkpoint_dir: epath.Path):
  """Removes LOCKED file for all existing steps, if present."""
  steps = utils.checkpoint_steps(checkpoint_dir)
  for step in steps:
    if utils.is_locked(checkpoint_dir, step):
      _unlock_checkpoint(checkpoint_dir, step)


def _wait_for_new_checkpoint(
    checkpoint_dir: epath.Path,
    last_checkpoint_step: Optional[int],
    seconds_to_sleep: int = 1,
    timeout: Optional[int] = None,
) -> int:
  """Waits until a new checkpoint file is found.

  Args:
    checkpoint_dir: The directory in which checkpoints are saved.
    last_checkpoint_step: The last checkpoint step used or `None` if we're
      expecting a checkpoint for the first time.
    seconds_to_sleep: The number of seconds to sleep for before looking for a
      new checkpoint.
    timeout: The maximum number of seconds to wait. If left as `None`, then the
      process will wait indefinitely.

  Returns:
    a new checkpoint step, or -1 if the timeout was reached.
  """
  logging.info('Waiting for new checkpoint at %s', checkpoint_dir)
  result = -1
  if jax.process_index() == 0:
    stop_time = time.time() + timeout if timeout is not None else None
    while True:
      steps = utils.checkpoint_steps(checkpoint_dir)
      checkpoint_step = max(steps) if steps else None
      if checkpoint_step is None or checkpoint_step == last_checkpoint_step:
        if stop_time is not None and time.time() + seconds_to_sleep > stop_time:
          break
        time.sleep(seconds_to_sleep)
      else:
        if not _lock_checkpoint(checkpoint_dir, checkpoint_step):
          continue
        logging.info('Found new checkpoint step: %d', checkpoint_step)
        result = checkpoint_step
        break
  return multihost_utils.broadcast_one_to_all(np.int32(result))


@contextlib.contextmanager
def wait_for_new_checkpoint(
    checkpoint_dir: epath.Path,
    last_checkpoint_step: Optional[int],
    seconds_to_sleep: int = 1,
    timeout: Optional[int] = None,
):
  """Waits until a new checkpoint file is found.

  Automatically locks any checkpoint that is returned, and unlocks the
  checkpoint when execution returns to this function.

  Args:
    checkpoint_dir: The directory in which checkpoints are saved.
    last_checkpoint_step: The last checkpoint step used or `None` if we're
      expecting a checkpoint for the first time.
    seconds_to_sleep: The number of seconds to sleep for before looking for a
      new checkpoint.
    timeout: The maximum number of seconds to wait. If left as `None`, then the
      process will wait indefinitely.

  Yields:
    a new checkpoint step, or -1 if the timeout was reached.
  """
  step = _wait_for_new_checkpoint(
      checkpoint_dir, last_checkpoint_step, seconds_to_sleep, timeout
  )
  try:
    yield step
  finally:
    # Release lock on the checkpoint step.
    if step != -1:
      _unlock_checkpoint(checkpoint_dir, step)


def checkpoints_iterator(
    checkpoint_dir: epath.PathLike,
    min_interval_secs: int = 0,
    timeout: Optional[int] = None,
    timeout_fn: Optional[Callable[[], bool]] = None,
    unlock_existing_checkpoints: bool = True,
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
    unlock_existing_checkpoints: If True, marks any existing checkpoints as
      "unlocked", since these were most likely created by a failure during a
      previous run, which left some checkpoints as "locked", despite not being
      in use.

  Yields:
    Integer step numbers of the latest checkpoints as they arrive.
  """
  checkpoint_dir = epath.Path(checkpoint_dir)
  if unlock_existing_checkpoints:
    try:
      _unlock_existing_checkpoints(checkpoint_dir)
    except FileNotFoundError as e:
      logging.warning(
          'Encountered error while unlocking existing checkpoints. Some'
          ' checkpoints may still be locked. %s.',
          e,
      )
  checkpoint_step = None
  while True:
    with wait_for_new_checkpoint(
        checkpoint_dir, checkpoint_step, timeout=timeout
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


def construct_restore_args(target: PyTree, sharding_tree: PyTree) -> PyTree:
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
          global_shape=value.shape,
          dtype=value.dtype,
      )
    else:
      return type_handlers.RestoreArgs(restore_type=restore_type, dtype=dtype)

  return jax.tree_util.tree_map(_restore_args, target, sharding_tree)
