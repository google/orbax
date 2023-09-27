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
import itertools
import threading
import time
from typing import Any, Callable, Sequence

from absl import logging
from etils import epath
import jax
from orbax.checkpoint import future as future_lib
from orbax.checkpoint import utils
from orbax.checkpoint.async_checkpoint_handler import AsyncCheckpointHandler
from orbax.checkpoint.checkpointer import Checkpointer


_module_unique_count = itertools.count()
_CHECKPOINT_SUCCESS = 'checkpoint_write_success'
_DISTRIBUTED_SYSTEM_MSG = (
    'Please initialize the distributed system via '
    '`jax.distributed.initialize()` at the start of your program.')


def _get_sync_key(count: int) -> str:
  return f'orbax_checkpoint_{count}'


def _on_commit_callback(temp_ckpt_dir: epath.Path, final_ckpt_dir: epath.Path,
                        checkpoint_start_time: float):
  """Finalize atomic save and record checkpoint save metrics."""
  utils.on_commit_callback(temp_ckpt_dir, final_ckpt_dir, checkpoint_start_time)
  jax.monitoring.record_event_duration_secs(
      '/jax/checkpoint/write/async/total_duration_secs',
      time.time() - checkpoint_start_time)


class _AsyncManager:
  """Helper class for background checkpoint saving work orchestration."""

  def __init__(self, timeout_secs: int = 300):
    self._timeout_in_ms = timeout_secs * 1000

    self._thread = None
    self._exception = None

    # TODO(dicentra): tidy this up.
    if (
        jax.process_count() > 1
        and jax._src.distributed.global_state.client is None  # pylint: disable=protected-access
    ):
      raise ValueError(_DISTRIBUTED_SYSTEM_MSG)
    if jax.process_count() > 1:
      self._client = jax._src.distributed.global_state.client  # pylint: disable=protected-access

    self._count = None

  def __del__(self):
    if self._thread is not None and self._thread.is_alive():
      logging.warning('Please add `.wait_until_finished()` in the main thread '
                      'before your program finishes because there is a '
                      'possibility of losing errors raised if the '
                      'this class is deleted before writing is completed.')

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
      logging.info('Starting commit to storage layer by process: %s',
                   current_process)
      thread_start_time = time.time()

      # Wait for commit operations to complete.
      for future in commit_futures:
        future.result()
      logging.info('Finished committing to storage layer by process: %s',
                   current_process)

      if process_count > 1:
        # All processes will wait at the barrier. When all processes are at the
        # barrier, the barrier will be satisfied. If not, then it will timeout.
        key_for_barrier = _get_sync_key(sync_count)
        logging.info('Key used for barrier is %s for process %s',
                     key_for_barrier, current_process)
        self._client.wait_at_barrier(key_for_barrier, self._timeout_in_ms)
        logging.info('Finished waiting at barrier for process %s',
                     current_process)

      if current_process == 0:
        on_commit_callback()
        logging.info('on_commit_callback successfully ran!')
        if process_count > 1:
          self._client.key_value_set(key_for_barrier, _CHECKPOINT_SUCCESS)
          logging.info('Process 0 successfully set key %s in the kv store',
                       key_for_barrier)

      jax.monitoring.record_event_duration_secs(
          '/jax/checkpoint/write/async/thread_duration_sec',
          time.time() - thread_start_time)

    except Exception as e:  # pylint: disable=broad-exception-caught
      self._exception = e

  def start_async_commit(
      self,
      commit_futures: Sequence[future_lib.Future],
      on_commit_callback: Callable[[], None],
  ):
    """Completes checkpoint save in a background thread."""
    self._count = next(_module_unique_count)
    self._thread = threading.Thread(
        target=self._thread_func,
        args=(commit_futures, on_commit_callback, self._count),
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
      logging.info('Thread joined successfully')

    self.check_for_errors()
    logging.info('Error check finished successfully')

    if jax.process_count() > 1 and self._count is not None:
      # Block until process 0 writes success value to the key value store.
      # If it fails to write it, then `blocking_key_value_get` will time out.
      get_key = _get_sync_key(self._count)
      self._client.blocking_key_value_get(get_key, self._timeout_in_ms)
      logging.info('blocking_key_value_get on key %s was successfully '
                   'completed.', get_key)


class AsyncCheckpointer(Checkpointer):
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
    # TODO(dicentra): consider folding into AsyncCheckpointer directly.
    self._async_manager = _AsyncManager(timeout_secs=timeout_secs)

  def save(self,
           directory: epath.PathLike,
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
        self._handler.async_save(tmpdir, *args, **kwargs))
    commit_ops, _ = jax.tree_util.tree_flatten(commit_ops)
    commit_ops = [op for op in commit_ops if op is not None]

    # Directory is the final directory
    self._async_manager.start_async_commit(
        commit_futures=commit_ops,
        on_commit_callback=functools.partial(
            _on_commit_callback, tmpdir, directory, checkpoint_start_time
        ),
    )
    jax.monitoring.record_event_duration_secs(
        '/jax/checkpoint/write/async/blocking_duration_secs',
        time.time() - checkpoint_start_time)

  def restore(self,
              directory: epath.PathLike,
              *args,
              **kwargs) -> Any:
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
