# Copyright 2024 The Orbax Authors.
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

import asyncio
import time
from typing import Any, Iterable, Optional, Type

from absl import logging
from etils import epath
from etils import epy
import jax
from orbax.checkpoint import abstract_checkpointer
from orbax.checkpoint import checkpoint_args
from orbax.checkpoint import multihost
from orbax.checkpoint import options as options_lib
from orbax.checkpoint import utils
from orbax.checkpoint._src.handlers import checkpoint_handler
from orbax.checkpoint._src.handlers import composite_checkpoint_handler
from orbax.checkpoint.metadata import checkpoint
from orbax.checkpoint.path import atomicity
from typing_extensions import Self  # for Python version < 3.11



CheckpointArgs = checkpoint_args.CheckpointArgs
register_with_handler = checkpoint_args.register_with_handler
get_legacy_handler_wrapper = (
    composite_checkpoint_handler.get_legacy_handler_wrapper
)


def construct_checkpoint_args(
    handler: checkpoint_handler.CheckpointHandler,
    for_save: bool,
    *args,
    **kwargs,
) -> checkpoint_args.CheckpointArgs:
  """Constructs `CheckpointArgs` for save or restore for the handler."""
  for arg in args:
    if isinstance(arg, checkpoint_args.CheckpointArgs):
      return arg
  for arg in kwargs.values():
    if isinstance(arg, checkpoint_args.CheckpointArgs):
      return arg
  jax.monitoring.record_event('/jax/orbax/deprecation/checkpointer_legacy_args')
  save_arg_cls, restore_arg_cls = checkpoint_args.get_registered_args_cls(
      handler
  )
  if for_save:
    return save_arg_cls(*args, **kwargs)
  else:
    return restore_arg_cls(*args, **kwargs)


class Checkpointer(
    abstract_checkpointer.AbstractCheckpointer, epy.ContextManager
):
  """A synchronous implementation of AbstractCheckpointer.

  This class saves synchronously to a given directory using an underlying
  `CheckpointHandler`. Atomicity of the operation is guaranteed.

  IMPORTANT: Async checkpointing can often be faster for saving. Strongly
  consider using `AsyncCheckpointer` instead.

  IMPORTANT: Remember that to save and restore a checkpoint, one should always
  use an `AbstractCheckpointer` coupled with a `CheckpointHandler`. The specific
  `CheckpointHandler` to use depends on the object being saved or restored.

  Basic example::

    ckptr = Checkpointer(StandardCheckpointHandler())
    args = ocp.args.StandardSave(state=pytree_of_arrays)
    ckptr.save(path, args=args)
    args = ocp.args.StandardRestore(state=abstract_pytree_target)
    ckptr.restore(path, args=args)

  Each handler includes `...SaveArgs` and `...RestoreArgs` classes that document
  what arguments are expected. When using `Checkpointer`, you can either use
  this dataclass directly, or you can provide the arguments in keyword form.

  For example::

    ckptr = Checkpointer(StandardCheckpointHandler())
    ckptr.save(path, state=pytree_of_arays)
    ckptr.restore(path, state=abstract_pytree_target)
  """

  def __init__(
      self,
      handler: checkpoint_handler.CheckpointHandler,
      *,
      multiprocessing_options: options_lib.MultiprocessingOptions = options_lib.MultiprocessingOptions(),
      file_options: options_lib.FileOptions = options_lib.FileOptions(),
      checkpoint_metadata_store: Optional[
          checkpoint.CheckpointMetadataStore
      ] = None,
      temporary_path_class: Optional[Type[atomicity.TemporaryPath]] = None,
  ):
    if not checkpoint_args.has_registered_args(handler):
      logging.warning(
          'No registered CheckpointArgs found for handler type: %s',
          type(handler),
      )
      handler = get_legacy_handler_wrapper(handler)
    self._handler = handler
    self._primary_host = multiprocessing_options.primary_host
    self._active_processes = multiprocessing_options.active_processes
    self._barrier_sync_key_prefix = (
        multiprocessing_options.barrier_sync_key_prefix
    )
    self._file_options = file_options
    self._temporary_path_class = temporary_path_class

    # If not provided then use checkpoint_metadata_store with blocking writes.
    self._checkpoint_metadata_store = (
        checkpoint_metadata_store
        or checkpoint.checkpoint_metadata_store(
            enable_write=True, blocking_write=True
        )
    )
    if not self._checkpoint_metadata_store.is_blocking_writer():
      raise ValueError('Checkpoint metadata store must be blocking writer.')

    jax.monitoring.record_event('/jax/orbax/checkpointer/init')

  async def create_temporary_path(
      self, directory: epath.Path
  ) -> atomicity.TemporaryPath:
    temporary_path_class = (
        self._temporary_path_class
        or atomicity.get_default_temporary_path_class(directory)
    )
    multiprocessing_options = options_lib.MultiprocessingOptions(
        primary_host=self._primary_host,
        active_processes=self._active_processes,
        barrier_sync_key_prefix=self._barrier_sync_key_prefix,
    )
    tmpdir = temporary_path_class.from_final(
        directory,
        checkpoint_metadata_store=self._checkpoint_metadata_store,
        multiprocessing_options=multiprocessing_options,
        file_options=self._file_options,
    )
    await atomicity.create_all(
        [tmpdir], multiprocessing_options=multiprocessing_options
    )
    return tmpdir

  def save(
      self, directory: epath.PathLike, *args, force: bool = False, **kwargs
  ):
    """Saves the given item to the provided directory.

    Delegates to the underlying CheckpointHandler. Ensures save operation
    atomicity.

    This method should be called by all hosts - process synchronization and
    actions that need to be performed on only one host are managed internally.

    Args:
      directory: a path to which to save.
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

    jax.monitoring.record_event('/jax/orbax/write/start')
    logging.info(
        '[process=%s] Started saving checkpoint to %s.',
        multihost.process_index(),
        directory,
    )

    if directory.exists():
      if force:
        if utils.is_primary_host(self._primary_host):
          logging.info('Specified `force`: removing existing directory.')
          directory.rmtree()  # Post-sync handled by create_tmp_directory.
      else:
        raise ValueError(f'Destination {directory} already exists.')
    ckpt_args = construct_checkpoint_args(self._handler, True, *args, **kwargs)
    tmpdir = asyncio.run(self.create_temporary_path(directory))
    self._handler.save(tmpdir.get(), args=ckpt_args)
    multihost.sync_global_processes(
        multihost.unique_barrier_key(
            'Checkpointer:save',
            prefix=self._barrier_sync_key_prefix,
            suffix=directory.name,
        ),
        processes=self._active_processes,
    )

    # Ensure save operation atomicity and record time saved by checkpoint.
    if utils.is_primary_host(self._primary_host):
      self._handler.finalize(tmpdir.get())
      atomicity.on_commit_callback(
          tmpdir,
          checkpoint_start_time=checkpoint_start_time,
      )
    multihost.sync_global_processes(
        multihost.unique_barrier_key(
            'Checkpointer:finalize',
            prefix=self._barrier_sync_key_prefix,
            suffix=directory.name,
        ),
        processes=self._active_processes,
    )

  def restore(self, directory: epath.PathLike, *args, **kwargs) -> Any:
    """See superclass documentation."""
    directory = epath.Path(directory)
    if not directory.exists():
      raise FileNotFoundError(f'Checkpoint at {directory} not found.')
    if not utils.is_checkpoint_finalized(directory):
      raise ValueError(f'Found incomplete checkpoint at {directory}.')
    logging.info('Restoring checkpoint from %s.', directory)
    ckpt_args = construct_checkpoint_args(self._handler, False, *args, **kwargs)
    restored = self._handler.restore(directory, args=ckpt_args)
    logging.info('Finished restoring checkpoint from %s.', directory)
    multihost.sync_global_processes(
        multihost.unique_barrier_key(
            'Checkpointer:restore',
            prefix=self._barrier_sync_key_prefix,
            suffix=directory.name,
        ),
        processes=self._active_processes,
    )
    return restored

  def metadata(self, directory: epath.PathLike) -> Optional[Any]:
    """See superclass documentation."""
    directory = epath.Path(directory)
    return self._handler.metadata(directory)

  def close(self):
    """Closes the underlying CheckpointHandler."""
    self._handler.close()
    self._checkpoint_metadata_store.close()

  @property
  def handler(self) -> checkpoint_handler.CheckpointHandler:
    return self._handler

  def __contextmanager__(
      self,
  ) -> Iterable[Self]:
    try:
      yield self
    finally:
      self.close()
