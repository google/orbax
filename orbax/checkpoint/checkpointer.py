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

"""Synchronous Checkpointer implementation."""

import logging
from typing import Any, Optional, Union

from etils import epath
import jax
from jax.experimental import multihost_utils
from orbax.checkpoint import utils
from orbax.checkpoint.abstract_checkpointer import AbstractCheckpointer
from orbax.checkpoint.checkpoint_handler import CheckpointHandler


class Checkpointer(AbstractCheckpointer):
  """A synchronous implementation of AbstractCheckpointer.

  This class saves synchronously to a given directory using an underlying
  CheckpointHandler. Atomicity of the operation is guaranteed.
  """

  def __init__(self, handler: CheckpointHandler):
    self._handler = handler

  def save(self,
           directory: Union[str, epath.Path],
           item: Any,
           *args,
           force: bool = False,
           **kwargs):
    """Saves the given item to the provided directory.

    Delegates to the underlying CheckpointHandler. Ensures save operation
    atomicity.

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
    if directory.exists():
      if force:
        if jax.process_index() == 0:
          utils.rmtree(directory)
      else:
        raise ValueError(f'Destination {directory} already exists.')
    logging.info('Saving item to %s.', directory)

    tmpdir = utils.create_tmp_directory(directory)
    self._handler.save(tmpdir, item, *args, **kwargs)
    multihost_utils.sync_global_devices('Checkpointer:write')

    # Ensure save operation atomicity.
    if jax.process_index() == 0:
      utils.ensure_atomic_save(tmpdir, directory)
    multihost_utils.sync_global_devices('Checkpointer:save')

  def restore(self,
              directory: Union[str, epath.Path],
              *args,
              item: Optional[Any] = None,
              **kwargs) -> Any:
    """See superclass documentation."""
    directory = epath.Path(directory)
    if not directory.exists():
      raise FileNotFoundError(f'Checkpoint at {directory} not found.')
    if not utils.is_checkpoint_finalized(directory):
      raise ValueError(f'Found incomplete checkpoint at {directory}.')
    return self._handler.restore(directory, *args, item=item, **kwargs)

  def structure(self, directory: Union[str, epath.Path]) -> Optional[Any]:
    """See superclass documentation."""
    directory = epath.Path(directory)
    try:
      return self._handler.structure(directory)
    except NotImplementedError:
      return
