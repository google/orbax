# Copyright 2022 The Orbax Authors.
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
import time
from typing import Iterator, Optional

from absl import logging
from etils import epath
import jax
from jax.experimental import multihost_utils
from jax.sharding import Mesh
import numpy as np
from orbax.checkpoint import type_handlers
from orbax.checkpoint import utils


PyTree = type(jax.tree_util.tree_structure(None))


def _wait_for_new_checkpoint(
    checkpoint_dir: epath.Path,
    last_checkpoint_step: Optional[int],
    seconds_to_sleep: int = 1,
    timeout: Optional[int] = None,
) -> Optional[int]:
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
    a new checkpoint step, or None if the timeout was reached.
  """
  logging.info('Waiting for new checkpoint at %s', checkpoint_dir)
  stop_time = time.time() + timeout if timeout is not None else None
  while True:
    steps = utils.checkpoint_steps(checkpoint_dir)
    checkpoint_step = max(steps) if steps else None
    if checkpoint_step is None or checkpoint_step == last_checkpoint_step:
      if stop_time is not None and time.time() + seconds_to_sleep > stop_time:
        return None
      time.sleep(seconds_to_sleep)
    else:
      logging.info('Found new checkpoint step: %d', checkpoint_step)
      return checkpoint_step


def checkpoints_iterator(
    checkpoint_dir: epath.PathLike,
    min_interval_secs=0,
    timeout=None,
    timeout_fn=None,
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

  Yields:
    Integer step numbers of the latest checkpoints as they arrive.
  """
  checkpoint_dir = epath.Path(checkpoint_dir)
  checkpoint_step = None
  while True:
    new_checkpoint_step = 0
    if jax.process_index() == 0:
      new_checkpoint_step = (
          _wait_for_new_checkpoint(
              checkpoint_dir, checkpoint_step, timeout=timeout
          )
          or -1
      )
    new_checkpoint_step = multihost_utils.broadcast_one_to_all(
        np.int32(new_checkpoint_step)
    )
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


def restore_args_from_target(
    mesh: Mesh, target: PyTree, axes_tree: PyTree
) -> PyTree:
  """Creates restore_args given a target PyTree.

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
    mesh: Mesh to shard arrays with.
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
    if isinstance(value, jax.Array) and jax.config.jax_array:
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
