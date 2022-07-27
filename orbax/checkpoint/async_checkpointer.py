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

"""AsyncCheckpointer."""

import asyncio
import functools
import logging
from typing import Any

import jax
from jax.experimental.gda_serialization.serialization import AsyncManager
from orbax.checkpoint import utils
from orbax.checkpoint.async_checkpoint_handler import AsyncCheckpointHandler
from orbax.checkpoint.checkpointer import Checkpointer
import tensorflow as tf


def on_commit_callback(temp_ckpt_dir, final_ckpt_dir):
  if temp_ckpt_dir == final_ckpt_dir:
    with tf.io.gfile.GFile(
        tf.io.gfile.join(final_ckpt_dir, 'commit_success.txt'), 'w') as f:
      f.write(f'Checkpoint commit was successful to {final_ckpt_dir}')
  else:
    logging.info('Renaming %s to %s', temp_ckpt_dir, final_ckpt_dir)
    tf.io.gfile.rename(temp_ckpt_dir, final_ckpt_dir)
    logging.info('Finished saving checkpoint to `%s`.', final_ckpt_dir)


# TODO(b/238758658): Eliminate GDA dependency by moving AsyncManager to a
# different location.
class AsyncCheckpointer(Checkpointer, AsyncManager):
  """An asynchronous implementation of Checkpointer.

  Save operations take place in a background thread (this functionality is
  provided by AsyncManager). Users should call `wait_until_finished` to block
  until a save operation running in the background is complete.

  Like its parent, AsyncCheckpointer also makes use of an underlying
  CheckpointHandler to deal with type-specific logic.
  """

  def __init__(self, handler: AsyncCheckpointHandler):
    self._handler = handler
    AsyncManager.__init__(self)

  def save(self, directory: str, item: Any, *args, **kwargs):
    """Saves the given item to the provided directory.

    Delegates to the underlying CheckpointHandler. Ensures save operation
    atomicity. Must first block until any previous save operations running in
    the background are completed.

    Args:
      directory: a path to which to save.
      item: an object to save, supported by a CheckpointHandler.
      *args: additional args to provide to the CheckpointHandler's save method.
      **kwargs: additional keyword args to provide to the CheckpointHandler's
        save method.

    Raises:
      ValueError if the provided directory already exists.
    """
    if tf.io.gfile.exists(directory):
      raise ValueError(f'Destination {directory} already exists.')

    logging.info('Saving item to %s. Waiting for thread to finish save.',
                 directory)
    self.wait_until_finished()

    tmpdir = utils.create_tmp_directory(directory)
    # Run copy ops.
    commit_ops = asyncio.run(
        self._handler.async_save(tmpdir, item, *args, **kwargs))
    commit_ops, _ = jax.tree_flatten(commit_ops)
    commit_ops = [op for op in commit_ops if op is not None]

    self._add_futures(commit_ops)
    # Directory is the final directory
    self._start_async_commit(
        functools.partial(on_commit_callback, tmpdir, directory))
