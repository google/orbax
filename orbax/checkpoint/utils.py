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

import glob
import logging
import time
from typing import Iterator, List, Optional

import flax.serialization
import jax
from jax.experimental import multihost_utils
import numpy as np
import tensorflow as tf
import tensorstore as ts

TMP_DIR_SUFFIX = ".orbax-checkpoint-tmp-"


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


def is_scalar(x):
  return isinstance(x, (int, float, np.number))


def is_checkpoint_finalized(checkpoint_dir: str, step: int) -> bool:
  # <directory>/<step>/<name>.orbax-checkpoint-tmp-<timestamp>
  tmp_dirs = glob.glob(
      tf.io.gfile.join(checkpoint_dir, str(step), "*" + TMP_DIR_SUFFIX + "*"))
  return not tmp_dirs


def _checkpoint_steps(checkpoint_dir: str) -> List[int]:
  return [int(x) for x in tf.io.gfile.listdir(checkpoint_dir) if x.isdigit()]


def checkpoint_steps(checkpoint_dir: str) -> List[int]:
  return [
      s for s in _checkpoint_steps(checkpoint_dir)
      if is_checkpoint_finalized(checkpoint_dir, s)
  ]


def tmp_checkpoint_steps(checkpoint_dir: str) -> List[int]:
  return [
      s for s in _checkpoint_steps(checkpoint_dir)
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
  logging.info("Waiting for new checkpoint at %s", checkpoint_dir)
  stop_time = time.time() + timeout if timeout is not None else None
  while True:
    steps = checkpoint_steps(checkpoint_dir)
    checkpoint_step = sorted(steps)[-1] if steps else None
    if checkpoint_step is None or checkpoint_step == last_checkpoint_step:
      if stop_time is not None and time.time() + seconds_to_sleep > stop_time:
        return None
      time.sleep(seconds_to_sleep)
    else:
      logging.info("Found new checkpoint step: %d", checkpoint_step)
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
        logging.info("Timed-out waiting for a checkpoint.")
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
