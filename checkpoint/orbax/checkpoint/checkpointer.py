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

"""Synchronous Checkpointer implementation."""

import contextlib
import time
from typing import Any, Optional

from absl import logging
from etils import epath
import jax
from orbax.checkpoint import utils
from orbax.checkpoint.abstract_checkpointer import AbstractCheckpointer
from orbax.checkpoint.checkpoint_handler import CheckpointHandler


class Checkpointer(AbstractCheckpointer):
  """A synchronous implementation of AbstractCheckpointer.

  This class saves synchronously to a given directory using an underlying
  CheckpointHandler. Atomicity of the operation is guaranteed.
  """

  def __init__(self, handler: CheckpointHandler, primary_host: int = 0):
    self._handler = handler
    self._primary_host = primary_host
    jax.monitoring.record_event('/jax/orbax/checkpointer/init')

  def save(self,
           directory: epath.PathLike,
           item: Any,
           *args,
           force: bool = False,
           **kwargs):
    """Saves the given item to the provided directory.

    Delegates to the underlying CheckpointHandler. Ensures save operation
    atomicity.

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
    logging.info('Saving item to %s.', directory)
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

    self._handler.save(tmpdir, item, *args, **kwargs)
    utils.sync_global_devices('Checkpointer:write')

    # Ensure save operation atomicity and record time saved by checkpoint.
    if jax.process_index() == self._primary_host:
      utils.on_commit_callback(tmpdir, directory, checkpoint_start_time)
    utils.sync_global_devices('Checkpointer:save')

  def restore(self,
              directory: epath.PathLike,
              *args,
              item: Optional[Any] = None,
              **kwargs) -> Any:
    """See superclass documentation."""
    directory = epath.Path(directory)
    if not directory.exists():
      raise FileNotFoundError(f'Checkpoint at {directory} not found.')
    if not utils.is_checkpoint_finalized(directory):
      raise ValueError(f'Found incomplete checkpoint at {directory}.')
    logging.info('Restoring item from %s.', directory)
    restored = self._handler.restore(directory, *args, item=item, **kwargs)
    logging.info('Finished restoring checkpoint from %s.', directory)
    return restored

  def structure(self, directory: epath.PathLike) -> Optional[Any]:
    """See superclass documentation."""
    directory = epath.Path(directory)
    try:
      return self._handler.structure(directory)
    except NotImplementedError:
      return

  def close(self):
    """Closes the underlying CheckpointHandler."""
    self._handler.close()


@contextlib.contextmanager
def checkpointer_context(*args, **kwargs):
  """Context manager for Checkpointer.

  Initializes Checkpointer and closes the object when the context is
  exited.

  Args:
    *args: Arguments to initialize Checkpointer.
    **kwargs: Keyword arguments to initialize Checkpointer.

  Usage::
    with checkpointer_context(PyTreeCheckpointHandler()) as ckptr:
      ckptr.save(...)
      ckptr.restore(...)

  Yields:
    Checkpointer
  """
  ckptr = Checkpointer(*args, **kwargs)
  try:
    yield ckptr
  finally:
    ckptr.close()
