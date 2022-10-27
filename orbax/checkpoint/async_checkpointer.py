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
from typing import Any, Union, Optional

from absl import logging
from etils import epath
import jax
from jax.experimental.gda_serialization.serialization import AsyncManager
from orbax.checkpoint import utils
from orbax.checkpoint.async_checkpoint_handler import AsyncCheckpointHandler
from orbax.checkpoint.checkpointer import Checkpointer


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

  def __init__(self, handler: AsyncCheckpointHandler, timeout_secs: int = 300):
    self._handler = handler
    AsyncManager.__init__(self, timeout_secs=timeout_secs)

  def save(self,
           directory: Union[str, epath.Path],
           item: Any,
           *args,
           force: bool = False,
           **kwargs):
    """Saves the given item to the provided directory.

    Delegates to the underlying CheckpointHandler. Ensures save operation
    atomicity. Must first block until any previous save operations running in
    the background are completed.

    Args:
      directory: a path to which to save.
      item: an object to save, supported by a CheckpointHandler.
      *args: additional args to provide to the CheckpointHandler's save method.
      force: if True, allows overwriting an existing directory.
      **kwargs: additional keyword args to provide to the CheckpointHandler's
        save method.

    Raises:
      ValueError if the provided directory already exists.
    """
    directory = epath.Path(directory)
    # TODO(cpgaffney) Support GCS + force.
    if force and utils.is_gcs_path(directory):
      raise ValueError('Force not yet supported on GCS.')
    if directory.exists():
      if force:
        logging.warning(
            'Destination %s already exists. With `force=True` it will be overwritten.',
            directory)
      else:
        raise ValueError(f'Destination {directory} already exists.')

    logging.info('Saving item to %s. Waiting for thread to finish save.',
                 directory)
    self.wait_until_finished()

    tmpdir = utils.create_tmp_directory(directory)
    # Run copy ops.
    commit_ops = asyncio.run(
        self._handler.async_save(tmpdir, item, *args, **kwargs))
    commit_ops, _ = jax.tree_util.tree_flatten(commit_ops)
    commit_ops = [op for op in commit_ops if op is not None]

    self._add_futures(commit_ops)
    # Directory is the final directory
    self._start_async_commit(
        functools.partial(utils.ensure_atomic_save, tmpdir, directory))

  def restore(self,
              directory: Union[str, epath.Path],
              *args,
              item: Optional[Any] = None,
              **kwargs) -> Any:
    """See superclass documentation."""
    self.wait_until_finished()
    return super().restore(directory, *args, item=item, **kwargs)
