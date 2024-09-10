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

"""AsyncCheckpointer."""

import asyncio
import threading
import time
from typing import Any, Callable, Optional, Sequence, Type

from absl import logging
from etils import epath
import jax
from orbax.checkpoint import checkpoint_args
from orbax.checkpoint import checkpointer
from orbax.checkpoint import future as future_lib
from orbax.checkpoint import multihost
from orbax.checkpoint import options as options_lib
from orbax.checkpoint import utils
from orbax.checkpoint._src.handlers import async_checkpoint_handler
from orbax.checkpoint.metadata import checkpoint
from orbax.checkpoint.path import async_utils
from orbax.checkpoint.path import atomicity


BarrierSyncFn = multihost.BarrierSyncFn


def _on_commit_callback(
    tmpdir: atomicity.TemporaryPath,
    checkpoint_start_time: float,
):
  """Finalize atomic save and record checkpoint save metrics."""
  atomicity.on_commit_callback(
      tmpdir, checkpoint_start_time=checkpoint_start_time
  )
  jax.monitoring.record_event_duration_secs(
      '/jax/checkpoint/write/async/total_duration_secs',
      time.time() - checkpoint_start_time,
  )


class _AsyncManager:
  """Helper class for background checkpoint saving work orchestration."""

  def __init__(
      self,
      *,
      barrier_sync_fn: multihost.BarrierSyncFn,
      timeout_secs: int = 600,
      primary_host: Optional[int] = 0,
      barrier_sync_key_prefix: Optional[str] = None,
  ):
    logging.info(
        '[process=%s][thread=%s] Using barrier_sync_fn: %s timeout: %d secs and'
        ' primary_host=%s for async checkpoint writes',
        multihost.process_index(),
        threading.current_thread().name,
        barrier_sync_fn,
        timeout_secs,
        primary_host,
    )
    self._timeout_secs = timeout_secs
    self._primary_host = primary_host
    self._barrier_sync_key_prefix = barrier_sync_key_prefix

    self._thread = None
    self._exception = None

    timeout_in_ms = self._timeout_secs * 1000
    self._sync_fn: Callable[[str], None] = lambda key: barrier_sync_fn(
        key=key, timeout_ms=timeout_in_ms
    )

  def __del__(self):
    if self._thread is not None and self._thread.is_alive():
      logging.warning(
          'Please add `.wait_until_finished()` in the main thread '
          'before your program finishes because there is a '
          'possibility of losing errors raised if the '
          'this class is deleted before writing is completed.'
      )

  def _thread_func(
      self,
      directory: epath.Path,
      commit_futures: Sequence[future_lib.Future],
      on_commit_callback: Callable[[], None],
      unique_operation_id: str,
  ):
    """Awaits on commit futures and finalizes the checkpoint."""
    current_process = multihost.process_index()
    current_thread_id = threading.current_thread().name
    # The unique_operation_id allows pre-selecting an identifier to use for the
    # barriers in this background thread. If we have multiple background
    # threads running concurrently, relying on _module_unique_count can result
    # in deadlocks when threads on different processes arrive at the barriers
    # in a certain order.
    try:
      process_count = jax.process_count()
      logging.info(
          '[process=%s][thread=%s] Background save thread started.',
          current_process,
          current_thread_id,
      )
      thread_start_time = time.time()

      # Wait for commit operations to complete.
      for future in commit_futures:
        future.result()
      logging.info(
          '[process=%s][thread=%s] %d Handler Commit operations completed.',
          current_process,
          current_thread_id,
          len(commit_futures),
      )
      # Log the number of async writes that are in flight. Abuses a duration
      # metric as a counter since jax.monitoring only has events and durations.
      jax.monitoring.record_event_duration_secs(
          '/jax/checkpoint/write/async/commit_future_count',
          len(commit_futures),
      )

      # Log the per process storage commit latency excluding the barrier time.
      jax.monitoring.record_event_duration_secs(
          '/jax/checkpoint/write/async/commit_duration_sec',
          time.time() - thread_start_time,
      )

      if process_count > 1:
        # All processes will wait at the barrier. When all processes are at the
        # barrier, the barrier will be satisfied. If not, then it will timeout.
        self._sync_fn(
            multihost.unique_barrier_key(
                'async_write_complete',
                prefix=self._barrier_sync_key_prefix,
                suffix=f'{directory.name}.{unique_operation_id}',
            )
        )

      if utils.is_primary_host(self._primary_host):
        on_commit_callback()
      if process_count > 1:
        # Block until process 0 completes on_commit_callback.
        self._sync_fn(
            multihost.unique_barrier_key(
                'async_commit_complete',
                prefix=self._barrier_sync_key_prefix,
                suffix=f'{directory.name}.{unique_operation_id}',
            )
        )

      jax.monitoring.record_event_duration_secs(
          '/jax/checkpoint/write/async/thread_duration_sec',
          time.time() - thread_start_time,
      )
      logging.info(
          '[process=%s][thread=%s] Background save thread done.',
          current_process,
          current_thread_id,
      )

    except Exception as e:  # pylint: disable=broad-exception-caught
      msg = (
          f'[process={current_process}] Failed to run'
          f' {len(commit_futures)} Handler Commit operations or the Commit'
          f' callback in background save thread, directory: {directory}'
      )
      logging.error(msg, exc_info=True)
      self._exception = e

  def start_async_commit(
      self,
      directory: epath.Path,
      commit_futures: Sequence[future_lib.Future],
      on_commit_callback: Callable[[], None],
      unique_operation_id: str,
  ):
    """Completes checkpoint save in a background thread."""
    self._thread = threading.Thread(
        name=f'async_save_{unique_operation_id}',
        target=self._thread_func,
        args=(
            directory,
            commit_futures,
            on_commit_callback,
            unique_operation_id,
        ),
    )
    self._thread.start()

  def check_for_errors(self):
    """Surfaces any errors from the background commit operations."""
    if self._exception is not None:
      # Clears self._exception so it is only raised once.
      exception = self._exception
      self._exception = None
      raise exception  # pylint: disable=raising-bad-type

  def wait_until_finished(self):
    """Waits for any outstanding operations to complete."""
    current_thread_name = threading.current_thread().name
    background_thread_name = (
        self._thread.name if self._thread is not None else None
    )
    if self._thread is not None:
      logging.info(
          '[process=%s][thread=%s] Waiting for background save thread=%s.',
          multihost.process_index(),
          current_thread_name,
          background_thread_name,
      )
      self._thread.join()
      self._thread = None
      logging.info(
          '[process=%s][thread=%s] Done with waiting for background save'
          ' thread=%s.',
          multihost.process_index(),
          current_thread_name,
          background_thread_name,
      )

    self.check_for_errors()
    if background_thread_name is not None:
      logging.info(
          '[process=%s][thread=%s] No errors found in background save'
          ' thread=%s.',
          multihost.process_index(),
          current_thread_name,
          background_thread_name,
      )


class AsyncCheckpointer(checkpointer.Checkpointer):
  """An asynchronous implementation of Checkpointer.

  Save operations take place in a background thread (this functionality is
  provided by AsyncManager). Users should call `wait_until_finished` to block
  until a save operation running in the background is complete.

  Like its parent, AsyncCheckpointer also makes use of an underlying
  CheckpointHandler to deal with type-specific logic.

  Please see `Checkpointer` documentation for more generic usage instructions.
  """

  _handler: async_checkpoint_handler.AsyncCheckpointHandler

  # Options mirror checkpoint_manager.AsyncOptions.
  def __init__(
      self,
      handler: async_checkpoint_handler.AsyncCheckpointHandler,
      timeout_secs: Optional[int] = None,
      *,
      async_options: options_lib.AsyncOptions = options_lib.AsyncOptions(),
      multiprocessing_options: options_lib.MultiprocessingOptions = options_lib.MultiprocessingOptions(),
      file_options: options_lib.FileOptions = options_lib.FileOptions(),
      checkpoint_metadata_store: Optional[
          checkpoint.CheckpointMetadataStore
      ] = None,
      temporary_path_class: Optional[Type[atomicity.TemporaryPath]] = None,
  ):
    jax.monitoring.record_event('/jax/orbax/async_checkpointer/init')
    if not checkpoint_args.has_registered_args(handler):
      logging.warning(
          '[process=%s] No registered CheckpointArgs found for handler'
          ' type: %s',
          multihost.process_index(),
          type(handler),
      )
      handler = checkpointer.get_legacy_handler_wrapper(handler)
      assert isinstance(
          handler, async_checkpoint_handler.AsyncCheckpointHandler
      )
    self._handler = handler
    self._primary_host = multiprocessing_options.primary_host
    self._active_processes = multiprocessing_options.active_processes
    self._post_finalization_callback = async_options.post_finalization_callback
    unique_class_id = self._unique_operation_id()
    barrier_sync_key_prefix = (
        f'{unique_class_id}'
        if multiprocessing_options.barrier_sync_key_prefix is None
        else f'{multiprocessing_options.barrier_sync_key_prefix}.{unique_class_id}'
    )
    self._barrier_sync_key_prefix = barrier_sync_key_prefix
    self._file_options = file_options
    self._checkpoint_metadata_store = (
        checkpoint_metadata_store
        or checkpoint.checkpoint_metadata_store(enable_write=True)
    )
    self._temporary_path_class = temporary_path_class
    timeout_secs = timeout_secs or async_options.timeout_secs

    # TODO(dicentra): consider folding into AsyncCheckpointer directly.
    self._async_manager = _AsyncManager(
        barrier_sync_fn=(
            async_options.barrier_sync_fn
            or multihost.get_barrier_sync_fn(
                processes=multiprocessing_options.active_processes
            )
        ),
        timeout_secs=timeout_secs,
        primary_host=multiprocessing_options.primary_host,
        barrier_sync_key_prefix=barrier_sync_key_prefix,
    )

  def _unique_operation_id(self) -> str:
    return multihost.counters.async_save_counter()

  async def _save(
      self, directory: epath.PathLike, *args, force: bool = False, **kwargs
  ):
    checkpoint_start_time = time.time()
    directory = epath.Path(directory)
    self.wait_until_finished()

    jax.monitoring.record_event('/jax/orbax/write/async/start')
    logging.info(
        '[process=%s] Started async saving checkpoint to %s.',
        multihost.process_index(),
        directory,
    )

    if await async_utils.async_exists(directory):
      if force:
        if utils.is_primary_host(self._primary_host):
          logging.info(
              '[process=%s] Specified `force`: removing existing directory.',
              multihost.process_index(),
          )
          await async_utils.async_rmtree(
              directory
          )  # Post-sync handled by create_tmp_directory.
      else:
        raise ValueError(f'Destination {directory} already exists.')

    tmpdir = await self.create_temporary_path(directory)
    # Run copy ops.
    # Try to save using new CheckpointArgs API if supported by the handler.
    ckpt_args = checkpointer.construct_checkpoint_args(
        self._handler, True, *args, **kwargs
    )
    commit_ops = await self._handler.async_save(tmpdir.get(), args=ckpt_args)
    commit_ops, _ = jax.tree.flatten(commit_ops)
    commit_ops = [op for op in commit_ops if op is not None]

    # Directory is the final directory.
    def _callback() -> None:
      logging.info(
          '[process=%s][thread=%s] Async Save Callback [1/3]: Finalizing'
          ' Handler: %s on %s',
          multihost.process_index(),
          threading.current_thread().name,
          self._handler,
          tmpdir.get(),
      )
      self._handler.finalize(tmpdir.get())
      logging.info(
          '[process=%s][thread=%s] Async Save Callback [2/3]: Running'
          ' post_finalization_callback: %s on %s',
          multihost.process_index(),
          threading.current_thread().name,
          self._post_finalization_callback,
          tmpdir.get_final(),
      )
      if self._post_finalization_callback is not None:
        self._post_finalization_callback()
      logging.info(
          '[process=%s][thread=%s] Async Save Callback [3/3]: Finalizing'
          ' checkpoint directory: %s',
          multihost.process_index(),
          threading.current_thread().name,
          tmpdir.get(),
      )
      _on_commit_callback(tmpdir, checkpoint_start_time)

    self._async_manager.start_async_commit(
        directory,
        commit_futures=commit_ops,
        on_commit_callback=_callback,
        unique_operation_id=self._unique_operation_id(),
    )

    jax.monitoring.record_event_duration_secs(
        '/jax/checkpoint/write/async/blocking_duration_secs',
        time.time() - checkpoint_start_time,
    )

  def save(
      self, directory: epath.PathLike, *args, force: bool = False, **kwargs
  ):
    """Saves the given item to the provided directory.

    Delegates to the underlying CheckpointHandler. Ensures save operation
    atomicity. Must first block until any previous save operations running in
    the background are completed.

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
    asyncio.run(self._save(directory, *args, force=force, **kwargs))

  def restore(self, directory: epath.PathLike, *args, **kwargs) -> Any:
    """See superclass documentation."""
    self.wait_until_finished()
    return super().restore(directory, *args, **kwargs)

  def check_for_errors(self):
    """Surfaces any errors from the background commit operations."""
    self._async_manager.check_for_errors()
    self._checkpoint_metadata_store.wait_until_finished()

  def wait_until_finished(self):
    """Waits for any outstanding operations to finish."""
    self._async_manager.wait_until_finished()
    self._checkpoint_metadata_store.wait_until_finished()

  def close(self):
    """Waits to finish any outstanding operations before closing."""
    self.wait_until_finished()
    super().close()
    self._checkpoint_metadata_store.close()

  @property
  def handler(self) -> async_checkpoint_handler.AsyncCheckpointHandler:
    return self._handler
