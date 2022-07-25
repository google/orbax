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

"""Utility functions for Orbax."""
import asyncio
import functools
import logging
import time
from typing import Iterator, List, Optional, Tuple

import flax.serialization
import jax
from jax.experimental import multihost_utils
import numpy as np
import tensorflow as tf
import tensorstore as ts

TMP_DIR_SUFFIX = '.orbax-checkpoint-tmp-'
CheckpointDirs = Tuple[str, str]
PyTree = type(jax.tree_structure(None))


def _wrap(func):
  """Wraps a function to make it async."""

  @functools.wraps(func)
  async def run(*args, loop=None, executor=None, **kwargs):
    if loop is None:
      loop = asyncio.get_event_loop()
    partial_func = functools.partial(func, *args, **kwargs)
    return await loop.run_in_executor(executor, partial_func)

  return run


# TODO(cpgaffney): This functionality should be provided by an external library.
async_makedirs = _wrap(tf.io.gfile.makedirs)


def register_ts_spec_for_serialization():
  # Register functions with flax.serialization to handle `ts.Spec`.
  def is_dict(s):
    return isinstance(s, (dict, flax.core.FrozenDict))

  flax.serialization.register_serialization_state(
      ts.Spec,
      ty_to_state_dict=lambda t: t.to_json(),
      # The parameter may have been written to tensorstore or msgpack.
      # If the former, a dict of the spec will be stored. If the latter it will
      # be the value itself.
      ty_from_state_dict=lambda t, s: ts.Spec(s) if is_dict(s) else s,
      override=True)


class Leaf:
  pass


def pytree_structure(directory: str) -> PyTree:
  """Reconstruct state dict from saved model format in `directory`."""

  def add_nested_key(subtree, nested_key):
    if not nested_key:
      return subtree

    current = nested_key[0]

    if len(nested_key) == 1:
      assert current not in subtree
      subtree[current] = Leaf()
      return subtree

    subkeys = nested_key[1:]
    if current not in subtree:
      subtree[current] = {}
    subtree[current] = add_nested_key(subtree[current], subkeys)
    return subtree

  keys = tf.io.gfile.listdir(directory)
  tree = {}
  for k in keys:
    tree = add_nested_key(tree, k.split('.'))
  return tree


def _rebuild_ts_specs(tree):
  """Converts any ts_spec dict leaves to ts.Spec object."""

  def is_leaf(x):
    if isinstance(x, dict):
      return set(x.keys()) >= {'driver', 'kvstore'}
    return False

  return jax.tree_map(
      lambda x: ts.Spec(x) if isinstance(x, dict) else x, tree, is_leaf=is_leaf)


def msgpack_restore(msgpack):
  """Restores tree serialized using Flax. Converts ts_spec dict to ts.Spec."""
  state_dict = flax.serialization.msgpack_restore(msgpack)
  return _rebuild_ts_specs(state_dict)


def to_state_dict(pytree):
  """Converts tree to state_dict. Converts ts_spec dict to ts.Spec."""
  state_dict = flax.serialization.to_state_dict(pytree)
  return _rebuild_ts_specs(state_dict)


def cleanup_tmp_directories(directory: str):
  """Cleanup steps in `directory` with tmp files, as these are not finalized."""
  if jax.process_index() == 0:
    tmp_files = tmp_checkpoints(directory)
    for tmp_file in tmp_files:
      if not is_checkpoint_finalized(directory, tmp_file):
        tf.io.gfile.rmtree(tf.io.gfile.join(directory, tmp_file))

  multihost_utils.sync_global_devices('cleanup_tmp_dirs')


def get_save_directory(step: int, directory: str, name: str) -> str:
  """Returns the standardized path to a save directory for a single item."""
  return tf.io.gfile.join(directory, str(step), name)


def create_tmp_directory(final_dir: str) -> str:
  """Creates a temporary directory for saving at the given path."""
  # Share a timestamp across devices.
  timestamp = multihost_utils.broadcast_one_to_all(np.int32(time.time()))
  tmp_dir = final_dir + TMP_DIR_SUFFIX + f'{timestamp}'

  if jax.process_index() == 0:
    assert not tf.io.gfile.exists(tmp_dir)
    tf.io.gfile.makedirs(tmp_dir)

  multihost_utils.sync_global_devices('make_dir')

  return tmp_dir


def is_scalar(x):
  return isinstance(x, (int, float, np.number))


def is_checkpoint_finalized(checkpoint_dir: str, file: str) -> bool:
  # <directory>/<step>.orbax-checkpoint-tmp-<timestamp>/<name>
  return TMP_DIR_SUFFIX not in tf.io.gfile.join(checkpoint_dir, file)


def checkpoint_steps(checkpoint_dir: str) -> List[int]:
  return [
      int(s)
      for s in tf.io.gfile.listdir(checkpoint_dir)
      if is_checkpoint_finalized(checkpoint_dir, s) and s.isdigit()
  ]


def tmp_checkpoints(checkpoint_dir: str) -> List[str]:
  return [
      s for s in tf.io.gfile.listdir(checkpoint_dir)
      if not is_checkpoint_finalized(checkpoint_dir, s)
  ]


def _wait_for_new_checkpoint(checkpoint_dir: str,
                             last_checkpoint_step: Optional[int],
                             seconds_to_sleep: int = 1,
                             timeout: Optional[int] = None):
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
    steps = checkpoint_steps(checkpoint_dir)
    checkpoint_step = sorted(steps)[-1] if steps else None
    if checkpoint_step is None or checkpoint_step == last_checkpoint_step:
      if stop_time is not None and time.time() + seconds_to_sleep > stop_time:
        return None
      time.sleep(seconds_to_sleep)
    else:
      logging.info('Found new checkpoint step: %d', checkpoint_step)
      return checkpoint_step


def checkpoints_iterator(checkpoint_dir: str,
                         min_interval_secs=0,
                         timeout=None,
                         timeout_fn=None) -> Iterator[int]:
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
  checkpoint_step = None
  while True:
    new_checkpoint_step = 0
    if jax.process_index() == 0:
      new_checkpoint_step = _wait_for_new_checkpoint(
          checkpoint_dir, checkpoint_step, timeout=timeout) or -1
    # None cannot be broadcast
    new_checkpoint_step = multihost_utils.broadcast_one_to_all(
        np.int32(new_checkpoint_step))
    if new_checkpoint_step == -1:
      if not timeout_fn:
        # timed out
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
