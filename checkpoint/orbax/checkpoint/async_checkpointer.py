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

"""AsyncCheckpointer."""

import asyncio
import contextlib
import functools
import time
from typing import Any, Optional

from absl import logging
from etils import epath
import jax
from jax.experimental.array_serialization.serialization import AsyncManager
from orbax.checkpoint import utils
from orbax.checkpoint.async_checkpoint_handler import AsyncCheckpointHandler
from orbax.checkpoint.checkpointer import Checkpointer


def _on_commit_callback(temp_ckpt_dir: epath.Path, final_ckpt_dir: epath.Path,
                        checkpoint_start_time: float):
  """Finalize atomic save and record checkpoint save metrics."""
  utils.on_commit_callback(temp_ckpt_dir, final_ckpt_dir, checkpoint_start_time)
  jax.monitoring.record_event_duration_secs(
      '/jax/checkpoint/write/async/total_duration_secs',
      time.time() - checkpoint_start_time)


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

  def __init__(
      self,
      handler: AsyncCheckpointHandler,
      timeout_secs: int = 300,
      primary_host: int = 0,
  ):
    jax.monitoring.record_event('/jax/orbax/async_checkpointer/init')
    self._handler = handler
    self._primary_host = primary_host
    AsyncManager.__init__(self, timeout_secs=timeout_secs)

  def save(self,
           directory: epath.PathLike,
           item: Any,
           *args,
           force: bool = False,
           **kwargs):
    """Saves the given item to the provided directory.

    Delegates to the underlying CheckpointHandler. Ensures save operation
    atomicity. Must first block until any previous save operations running in
    the background are completed.

    This method should be called by all hosts - process synchronization and
    actions that need to be performed on only one host are managed internally.

    Args:
      directory: a path to which to save.
      item: an object to save, supported by a CheckpointHandler.
      *args: additional args to provide to the CheckpointHandler's save method.
      force: if True, allows overwriting an existing directory. May add overhead
        due to the need to delete any existing files.
      **kwargs: additional keyword args to provide to the CheckpointHandler's
        save method.

    Raises:
      ValueError if the provided directory already exists.
    """
    checkpoint_start_time = time.time()
    directory = epath.Path(directory)
    logging.info('Saving item to %s. Waiting for thread to finish save.',
                 directory)
    self.wait_until_finished()

    if directory.exists():
      if force:
        if jax.process_index() == self._primary_host:
          logging.info('Specified `force`: removing existing directory.')
          directory.rmtree()  # Post-sync handled by create_tmp_directory.
      else:
        raise ValueError(f'Destination {directory} already exists.')
    tmpdir = utils.create_tmp_directory(
        directory, primary_host=self._primary_host
    )

    # Run copy ops.
    commit_ops = asyncio.run(
        self._handler.async_save(tmpdir, item, *args, **kwargs))
    commit_ops, _ = jax.tree_util.tree_flatten(commit_ops)
    commit_ops = [op for op in commit_ops if op is not None]

    self._add_futures(commit_ops)
    # Directory is the final directory
    self._start_async_commit(
        functools.partial(_on_commit_callback, tmpdir, directory,
                          checkpoint_start_time))
    jax.monitoring.record_event_duration_secs(
        '/jax/checkpoint/write/async/blocking_duration_secs',
        time.time() - checkpoint_start_time)

  def restore(self,
              directory: epath.PathLike,
              *args,
              item: Optional[Any] = None,
              **kwargs) -> Any:
    """See superclass documentation."""
    self.wait_until_finished()
    return super().restore(directory, *args, item=item, **kwargs)

  def close(self):
    """Waits to finish any outstanding operations before closing."""
    self.wait_until_finished()
    super().close()


@contextlib.contextmanager
def async_checkpointer_context(*args, **kwargs):
  """Context manager for AsyncCheckpointer.

  Initializes AsyncCheckpointer and closes the object when the context is
  exited.

  Usage::
    with async_checkpointer_context(PyTreeCheckpointHandler()) as ckptr:
      ckptr.save(...)
      ckptr.wait_until_finished()
      ckptr.restore(...)

  Args:
    *args: Arguments to initialize AsyncCheckpointer.
    **kwargs: Keyword arguments to initialize AsyncCheckpointer.

  Yields:
    AsyncCheckpointer
  """
  ckptr = AsyncCheckpointer(*args, **kwargs)
  try:
    yield ckptr
  finally:
    ckptr.close()
