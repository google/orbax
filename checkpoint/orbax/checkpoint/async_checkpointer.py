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
import contextlib
import itertools
import threading
import time
from typing import Any, Callable, Optional, Protocol, Sequence

from absl import logging
from etils import epath
import jax
from orbax.checkpoint import async_checkpoint_handler
from orbax.checkpoint import checkpoint_args
from orbax.checkpoint import checkpointer
from orbax.checkpoint import future as future_lib
from orbax.checkpoint import utils


_module_unique_count = itertools.count()


def _get_sync_key(suffix: str, count: int) -> str:
  return f'orbax_checkpoint_{suffix}_{count}'


def _on_commit_callback(
    temp_ckpt_dir: epath.Path,
    final_ckpt_dir: epath.Path,
    checkpoint_start_time: float,
):
  """Finalize atomic save and record checkpoint save metrics."""
  utils.on_commit_callback(temp_ckpt_dir, final_ckpt_dir, checkpoint_start_time)
  jax.monitoring.record_event_duration_secs(
      '/jax/checkpoint/write/async/total_duration_secs',
      time.time() - checkpoint_start_time,
  )


class BarrierSyncFn(Protocol):
  """Protocol for a barrier synchronization callable."""

  def __call__(self, *, key: str, timeout_ms: int) -> None:
    """Blocks on a barrier identified by key with the given timeout."""
    ...


# TODO(dicentra): move this to jax/experimental/multihost_utils.py
def _get_barrier_sync_fn() -> Optional[BarrierSyncFn]:
  """Provides a barrier synchronization function for JAX processes.

  Barriers with different sync keys are safe to use from independent background
  threads.

  Returns:
    None if there is a single JAX process.
    A barrier synchronization callable which accepts two arguments:
      - "key": [str] unique barrier id;
      - "timeout_ms": [int] timeout to use for waiting on the barrier.
    Should be called from all JAX processes with the same sync key and will
    block until either 1) all processes have reached the barrier or
    2) the timeout is exceeded.
  """
  if jax.process_count() == 1:
    return None

  client = jax._src.distributed.global_state.client  # pylint: disable=protected-access
  if client is None:
    raise ValueError(
        'Distributed system is not available; please initialize it via '
        '`jax.distributed.initialize()` at the start of your program.'
    )

  def _fn(*, key: str, timeout_ms: int) -> None:
    current_process = jax.process_index()
    logging.info(
        'Key used for barrier is %s for process %s', key, current_process
    )
    client.wait_at_barrier(key, timeout_ms)
    logging.info('Finished waiting at barrier for process %s', current_process)

  return _fn


class _AsyncManager:
  """Helper class for background checkpoint saving work orchestration."""

  def __init__(
      self,
      timeout_secs: int = 300,
      primary_host: int = 0,
      barrier_sync_fn: Optional[BarrierSyncFn] = None,
  ):
    self._timeout_secs = timeout_secs
    self._primary_host = primary_host

    self._thread = None
    self._exception = None

    timeout_in_ms = self._timeout_secs * 1000
    if barrier_sync_fn is None:
      default_fn = _get_barrier_sync_fn()
      if jax.process_count() > 1 and default_fn is None:
        raise ValueError(
            'Barrier sync function should be provided for multi-host setup!'
        )

      def _fn(*, key: str, timeout_ms: int) -> None:
        if default_fn is not None:
          default_fn(key=key, timeout_ms=timeout_ms)

      barrier_sync_fn = _fn

    sync_fn = lambda key: barrier_sync_fn(key=key, timeout_ms=timeout_in_ms)
    self._sync_fn: Callable[[str], None] = sync_fn

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
      commit_futures: Sequence[future_lib.Future],
      on_commit_callback: Callable[[], None],
      sync_count: int,
  ):
    """Awaits on commit futures and finalizes the checkpoint."""
    try:
      current_process = jax.process_index()
      process_count = jax.process_count()
      logging.info(
          'Starting commit to storage layer by process: %s', current_process
      )
      thread_start_time = time.time()

      # Wait for commit operations to complete.
      for future in commit_futures:
        future.result()
      logging.info(
          'Finished committing to storage layer by process: %s', current_process
      )

      if process_count > 1:
        # All processes will wait at the barrier. When all processes are at the
        # barrier, the barrier will be satisfied. If not, then it will timeout.
        self._sync_fn(_get_sync_key('write_complete', sync_count))

      if current_process == self._primary_host:
        on_commit_callback()
      if process_count > 1:
        # Block until process 0 completes on_commit_callback.
        self._sync_fn(_get_sync_key('commit_complete', sync_count))

      jax.monitoring.record_event_duration_secs(
          '/jax/checkpoint/write/async/thread_duration_sec',
          time.time() - thread_start_time,
      )

    except Exception as e:  # pylint: disable=broad-exception-caught
      self._exception = e

  def start_async_commit(
      self,
      commit_futures: Sequence[future_lib.Future],
      on_commit_callback: Callable[[], None],
  ):
    """Completes checkpoint save in a background thread."""
    sync_count = next(_module_unique_count)
    self._thread = threading.Thread(
        target=self._thread_func,
        args=(commit_futures, on_commit_callback, sync_count),
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
    if self._thread is not None:
      self._thread.join()
      self._thread = None
      logging.info('Commit thread joined successfully')

    self.check_for_errors()
    logging.info('Commit thread error check finished successfully')


class AsyncCheckpointer(checkpointer.Checkpointer):
  """An asynchronous implementation of Checkpointer.

  Save operations take place in a background thread (this functionality is
  provided by AsyncManager). Users should call `wait_until_finished` to block
  until a save operation running in the background is complete.

  Like its parent, AsyncCheckpointer also makes use of an underlying
  CheckpointHandler to deal with type-specific logic.

  Please see `Checkpointer` documentation for more generic usage instructions.
  """

  # Options mirror checkpoint_manager.AsyncOptions.
  def __init__(
      self,
      handler: async_checkpoint_handler.AsyncCheckpointHandler,
      timeout_secs: int = 300,
      primary_host: int = 0,
      *,
      barrier_sync_fn: Optional[BarrierSyncFn] = None,
  ):
    jax.monitoring.record_event('/jax/orbax/async_checkpointer/init')
    if not checkpoint_args.has_registered_args(handler):
      logging.warning(
          'No registered CheckpointArgs found for handler type: %s',
          type(handler),
      )
      handler = checkpointer.get_legacy_handler_wrapper(handler)
      assert isinstance(
          handler, async_checkpoint_handler.AsyncCheckpointHandler
      )
    self._handler = handler
    self._primary_host = primary_host

    # TODO(dicentra): consider folding into AsyncCheckpointer directly.
    self._async_manager = _AsyncManager(
        timeout_secs=timeout_secs,
        primary_host=primary_host,
        barrier_sync_fn=barrier_sync_fn or _get_barrier_sync_fn(),
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
    checkpoint_start_time = time.time()
    directory = epath.Path(directory)
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

    logging.info('Async saving item to %s.', directory)
    # Run copy ops.
    # Try to save using new CheckpointArgs API if supported by the handler.
    ckpt_args = checkpointer.construct_checkpoint_args(
        self._handler, True, *args, **kwargs
    )
    commit_ops = asyncio.run(self._handler.async_save(tmpdir, args=ckpt_args))
    commit_ops, _ = jax.tree_util.tree_flatten(commit_ops)
    commit_ops = [op for op in commit_ops if op is not None]

    # Directory is the final directory.
    def _callback() -> None:
      self._handler.finalize(tmpdir)
      _on_commit_callback(tmpdir, directory, checkpoint_start_time)

    self._async_manager.start_async_commit(
        commit_futures=commit_ops, on_commit_callback=_callback
    )

    jax.monitoring.record_event_duration_secs(
        '/jax/checkpoint/write/async/blocking_duration_secs',
        time.time() - checkpoint_start_time,
    )

  def restore(self, directory: epath.PathLike, *args, **kwargs) -> Any:
    """See superclass documentation."""
    self.wait_until_finished()
    return super().restore(directory, *args, **kwargs)

  def check_for_errors(self):
    """Surfaces any errors from the background commit operations."""
    self._async_manager.check_for_errors()

  def wait_until_finished(self):
    """Waits for any outstanding operations to finish."""
    self._async_manager.wait_until_finished()

  def close(self):
    """Waits to finish any outstanding operations before closing."""
    self.wait_until_finished()
    super().close()

  @property
  def handler(self) -> async_checkpoint_handler.AsyncCheckpointHandler:
    return self._handler


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
