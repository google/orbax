# Copyright 2025 The Orbax Authors.
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

"""Utils for creating and finalizing temporary paths.

Note that the configurability provided by this feature does
not leave users free to define their own temporary path structure. The current
implementation is mainly a refactoring of old logic that separately created
temp directories and finalized them. It does not touch other logic that detects
temp checkpoints and cleans them up (primarily located in
orbax.checkpoint.path.step and :py:class:`.CheckpointManager`).

Ordinarily, atomic logic defaults to :py:class:`AtomicRenameTemporaryPath`,
which uses an atomic rename to indicate checkpoint completion. However, not all
filesystems support atomic rename, so :py:class:`CommitFileTemporaryPath` is
provided as an alternative, which uses a "commit_success" file to indicate
completion.

Ideally, we would standardize on a single behavior, but it is difficult, largely
for legacy reasons, to achieve this. Furthermore, there are many other
alternative ways of ensuring save atomicity. As such, we have opted to provide
a more flexible approach that allows users to configure the behavior they want.

Configuration can be done in the following way::

  AsyncCheckpointer(
      StandardCheckpointHandler(),
      temporary_path_class=CommitFileTemporaryPath,
  )

  # OR

  CheckpointManager(
      directory,
      item_names=('state', 'dataset',),
      options=CheckpointManagerOptions(
          temporary_path_class=atomicity.CommitFileTemporaryPath
      ),
  )
"""

from __future__ import annotations

import asyncio
import pickle
import threading
import time
from typing import Awaitable, Protocol, Sequence, TypeVar

from absl import logging
from etils import epath
import jax
from orbax.checkpoint import options as options_lib
from orbax.checkpoint._src.futures import future
from orbax.checkpoint._src.futures import synchronization
from orbax.checkpoint._src.logging import event_tracking
from orbax.checkpoint._src.metadata import checkpoint as checkpoint_metadata
from orbax.checkpoint._src.metadata import step_metadata_serialization
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.path import async_path
from orbax.checkpoint._src.path import atomicity_types
from orbax.checkpoint._src.path import utils
from orbax.checkpoint._src.path.snapshot import snapshot as snapshot_lib


ValidationError = atomicity_types.ValidationError

TMP_DIR_SUFFIX = atomicity_types.TMP_DIR_SUFFIX
COMMIT_SUCCESS_FILE = atomicity_types.COMMIT_SUCCESS_FILE  # pylint: disable=protected-access

_LAST_CHECKPOINT_WRITE_TIME = time.time()

_T = TypeVar('_T', bound='TemporaryPathBase')


class AsyncMakeDirFunc(Protocol):

  def __call__(
      self,
      path: epath.Path,
      parents: bool = False,
      exist_ok: bool = False,
      mode: int | None = None,
      **kwargs,
  ) -> Awaitable[None]:
    """Creates the directory at path."""
    pass


async def _create_tmp_directory(
    async_makedir_func: AsyncMakeDirFunc,
    tmp_dir: epath.Path,
    *,
    path_permission_mode: int | None = None,
    checkpoint_metadata_store: checkpoint_metadata.MetadataStore | None = None,
    snapshot: snapshot_lib.Snapshot | None = None,
    **kwargs,
) -> epath.Path:
  """Creates a non-deterministic tmp directory for saving for given `final_dir`.

  Also writes checkpoint metadata in the tmp directory.

  Args:
    async_makedir_func: An implementation of AsyncMakeDirFunc to call.
    tmp_dir: The temporary directory path.
    path_permission_mode: Path permission mode for the temp directory. e.g.
      0o750. Please check
      https://github.com/google/etils/blob/main/etils/epath/backend.py if your
        path is supported.
    checkpoint_metadata_store: optional `CheckpointMetadataStore` instance. If
      present then it is used to create `StepMetadata` with current timestamp.
    snapshot: optional `Snapshot` instance. If present then it is used to create
      a snapshot of the final path.
    **kwargs: Optional. Additional kwargs to pass to `async_makedir_func`

  Returns:
    The tmp directory.
  """
  if await async_path.exists(tmp_dir):
    logging.warning(
        'Attempted to create temporary directory %s which already exists.'
        ' Removing existing directory since it is not finalized.',
        tmp_dir,
    )
    await async_path.rmtree(tmp_dir)
  logging.info('Creating tmp directory %s', tmp_dir)
  if snapshot is not None:
    await snapshot.create_snapshot()
  else:
    await async_makedir_func(
        tmp_dir,
        parents=True,
        exist_ok=False,
        mode=path_permission_mode,
        **kwargs,
    )
  if checkpoint_metadata_store is not None:
    checkpoint_metadata_store.write(
        file_path=checkpoint_metadata.step_metadata_file_path(tmp_dir),
        metadata=step_metadata_serialization.serialize(
            checkpoint_metadata.StepMetadata(
                init_timestamp_nsecs=time.time_ns()
            )
        ),
    )

  return tmp_dir


def _get_tmp_directory(final_path: epath.Path) -> epath.Path:
  # Path may not be completely unique if a preemption occurs. We rely on the
  # existing tmp directory being deleted elsewhere.
  return epath.Path(final_path.parent) / (final_path.name + TMP_DIR_SUFFIX)


def _get_final_directory(tmp_path: epath.Path) -> epath.Path:
  if (suffix_idx := tmp_path.name.find(TMP_DIR_SUFFIX)) == -1:
    raise ValueError(f'Expected {tmp_path} to end with "{TMP_DIR_SUFFIX}".')
  return epath.Path(tmp_path.parent) / tmp_path.name[:suffix_idx]


class TemporaryPathBase(atomicity_types.TemporaryPath):
  """A base class for TemporaryPath implementations."""

  def __init__(
      self,
      temporary_path: epath.Path | None,
      final_path: epath.Path,
      *,
      checkpoint_metadata_store: (
          checkpoint_metadata.MetadataStore | None
      ) = None,
      file_options: options_lib.FileOptions | None = None,
      use_snapshot: bool | None = False,
  ):
    self._tmp_path = temporary_path
    self._final_path = final_path

    file_options = file_options or options_lib.FileOptions()
    self._checkpoint_metadata_store = checkpoint_metadata_store
    self._path_permission_mode = file_options.path_permission_mode
    self._snapshot = None
    if use_snapshot:
      self._snapshot = snapshot_lib.create_instance(
          source=final_path,
          snapshot=temporary_path,
          set_immutable=False,
      )

  def get(self) -> epath.Path:
    """Returns the temporary path."""
    if not self._tmp_path:
      raise ValueError(
          'Temporary path has not been created yet. Please call `create` first.'
      )
    return self._tmp_path


class ReadOnlyTemporaryPath(atomicity_types.TemporaryPath):
  """A read-only, serializable object providing path properties access.

  This implementation is not meant to be used for creating or finalizing
  checkpoints. Its purpose is to be serialized and sent to other processes that
  only need access to temporary and final checkpoint paths.
  """

  def __init__(self, *, temporary_path: epath.Path, final_path: epath.Path):
    """Initializes ReadOnlyTemporaryPath.

    Args:
      temporary_path: The temporary path.
      final_path: The final path.
    """
    self._tmp_path = temporary_path
    self._final_path = final_path

  def get(self) -> epath.Path:
    """Returns the temporary path."""
    return self._tmp_path

  def get_final(self) -> epath.Path:
    """Returns the final path."""
    return self._final_path

  @classmethod
  def from_paths(
      cls,
      *,
      temporary_path: epath.Path,
      final_path: epath.Path,
  ) -> ReadOnlyTemporaryPath:
    """Constructs a ReadOnlyTemporaryPath from a temporary and final path.

    Args:
      temporary_path: The temporary path.
      final_path: The final path.

    Returns:
      A ReadOnlyTemporaryPath instance.
    """
    return ReadOnlyTemporaryPath(
        temporary_path=temporary_path, final_path=final_path
    )

  def to_bytes(self) -> bytes:
    """Serializes the object to bytes.

    Returns:
      The serialized object.
    """
    return pickle.dumps({
        'tmp_path': self._tmp_path,
        'final_path': self._final_path,
    })

  @classmethod
  def from_bytes(
      cls: type['ReadOnlyTemporaryPath'],
      data: bytes,
  ) -> ReadOnlyTemporaryPath:
    """Deserializes the object from bytes.

    Args:
      data: The serialized object.

    Returns:
      A ReadOnlyTemporaryPath instance.
    """
    data = pickle.loads(data)
    return cls(
        temporary_path=data['tmp_path'],
        final_path=data['final_path'],
    )

  @classmethod
  def from_temporary(
      cls,
      temporary_path: epath.Path,
      *,
      file_options: options_lib.FileOptions | None = None,
      use_snapshot: bool | None = None,
  ) -> ReadOnlyTemporaryPath:
    """Not implemented for ReadOnlyTemporaryPath."""
    raise NotImplementedError(
        'ReadOnlyTemporaryPath is not constructible from a temporary path.'
    )

  @classmethod
  def from_final(
      cls,
      final_path: epath.Path,
      *,
      checkpoint_metadata_store: (
          checkpoint_metadata.MetadataStore | None
      ) = None,
      file_options: options_lib.FileOptions | None = None,
      use_snapshot: bool | None = None,
  ) -> ReadOnlyTemporaryPath:
    """Not implemented for ReadOnlyTemporaryPath."""
    raise NotImplementedError(
        'ReadOnlyTemporaryPath is not constructible from a final path.'
    )

  async def create(
      self,
      *,
      file_options: options_lib.FileOptions = options_lib.FileOptions(),
  ) -> epath.Path:
    """Not supported for ReadOnlyTemporaryPath."""
    raise NotImplementedError('`create` is not supported.')

  async def finalize(
      self,
  ) -> None:
    """Not supported for ReadOnlyTemporaryPath."""
    raise NotImplementedError('`finalize` is not supported.')

  @classmethod
  async def validate(
      cls,
      temporary_path: epath.Path,
  ):
    """Validates the temporary path or raises an error."""
    raise NotImplementedError('`validate` is not supported.')

  @classmethod
  async def validate_final(
      cls,
      final_path: epath.Path,
  ):
    """Validates the final path or raises an error."""
    raise NotImplementedError('`validate_final` is not supported.')


async def _shared_validate(class_name: str, path: epath.Path):
  if not await async_path.is_dir(path):
    raise ValidationError(
        f'Expected {class_name} ({path}) to be a directory.'
    )
  if not await async_path.exists(path):
    raise ValidationError(f'Expected {class_name} ({path}) to exist.')


async def validate_atomic_rename_temporary_path(
    class_name: str,
    temporary_path: epath.Path,
):
  """Validates the temporary path or raises an error."""
  if await async_path.is_link(temporary_path):
    raise ValidationError(
        f'Path {temporary_path} is a symbolic link and cannot be treated as a'
        ' temporary checkpoint.',
    )
  # Does not perform I/O.
  if TMP_DIR_SUFFIX not in temporary_path.name:
    raise ValidationError(
        f'Expected {class_name} ({temporary_path}) to end with'
        f' "{TMP_DIR_SUFFIX}".'
    )
  if await async_path.exists(temporary_path / COMMIT_SUCCESS_FILE):
    raise ValidationError(
        f'Expected {class_name} ({temporary_path}) not to'
        f' contain the "{COMMIT_SUCCESS_FILE}" file.'
    )
  await _shared_validate(class_name, temporary_path)


async def validate_atomic_rename_final_path(
    class_name: str,
    final_path: epath.Path,
):
  """Validates the final path or raises an error."""
  # Does not perform I/O.
  if TMP_DIR_SUFFIX in final_path.name:
    raise ValidationError(
        f'Expected final {class_name} ({final_path}) not to end'
        f' with "{TMP_DIR_SUFFIX}".'
    )
  await _shared_validate(class_name, final_path)


class AtomicRenameTemporaryPath(TemporaryPathBase):
  """TemporaryPath implementation that uses atomic rename."""

  @classmethod
  async def validate(
      cls,
      temporary_path: epath.Path,
  ):
    await validate_atomic_rename_temporary_path(
        cls.__name__, temporary_path
    )

  @classmethod
  async def validate_final(
      cls,
      final_path: epath.Path,
  ):
    await validate_atomic_rename_final_path(cls.__name__, final_path)

  @classmethod
  def from_temporary(
      cls,
      temporary_path: epath.Path,
      *,
      file_options: options_lib.FileOptions | None = None,
      use_snapshot: bool | None = None,
  ) -> AtomicRenameTemporaryPath:
    return cls(
        temporary_path,
        _get_final_directory(temporary_path),
        file_options=file_options,
        use_snapshot=use_snapshot,
    )

  @classmethod
  def from_final(
      cls,
      final_path: epath.Path,
      *,
      checkpoint_metadata_store: (
          checkpoint_metadata.MetadataStore | None
      ) = None,
      file_options: options_lib.FileOptions | None = None,
      use_snapshot: bool | None = None,
  ) -> AtomicRenameTemporaryPath:
    return cls(
        _get_tmp_directory(final_path),
        final_path,
        checkpoint_metadata_store=checkpoint_metadata_store,
        file_options=file_options,
        use_snapshot=use_snapshot,
    )

  def get_final(self) -> epath.Path:
    return self._final_path

  async def create(self) -> epath.Path:
    """Creates a non-deterministic tmp directory for saving for given `final_dir`.

    Also writes checkpoint metadata in the tmp directory.

    NOTE: This function does not include any barrier syncs, and calling it
    directly from multiprocess code can lead to race conditions. Prefer to
    use `atomicity.create_all` in such cases.

    Returns:
      The tmp directory.

    Raises:
      FileExistsError: if tmp directory already exists.
    """
    return await _create_tmp_directory(
        async_path.mkdir,
        self._tmp_path,
        path_permission_mode=self._path_permission_mode,
        checkpoint_metadata_store=self._checkpoint_metadata_store,
        snapshot=self._snapshot,
    )

  async def finalize(
      self,
  ):
    """Finalizes atomic save by renaming tmp_dir.

    Updates checkpoint metadata with commit_timestamp_nsecs.

    """
    logging.info('Renaming %s to %s', self._tmp_path, self._final_path)
    if self._checkpoint_metadata_store:
      await asyncio.to_thread(
          self._checkpoint_metadata_store.wait_until_finished
      )
      await asyncio.to_thread(
          self._checkpoint_metadata_store.update,
          file_path=checkpoint_metadata.step_metadata_file_path(self._tmp_path),
          commit_timestamp_nsecs=time.time_ns(),
      )
      await asyncio.to_thread(
          self._checkpoint_metadata_store.wait_until_finished
      )

    if self._snapshot is not None:
      await self._snapshot.replace_source()
    else:
      await async_path.rename(self._tmp_path, self._final_path)

  def __repr__(self) -> str:
    return (
        f'AtomicRenameTemporaryPath(tmp="{self.get().name}",'
        f' final="{self._final_path.name}",'
        f' directory="{self._final_path.parent}")'
    )


class CommitFileTemporaryPath(TemporaryPathBase):
  """TemporaryPath implementation that uses a commit file."""

  @classmethod
  async def validate(
      cls,
      temporary_path: epath.Path,
  ):
    if await async_path.is_link(temporary_path):
      raise ValidationError(
          f'Path {temporary_path} is a symbolic link and cannot be treated as a'
          ' temporary checkpoint.',
      )
    if await async_path.exists(temporary_path / COMMIT_SUCCESS_FILE):
      raise ValidationError(
          f'Expected {cls.__name__} ({temporary_path}) not to contain the'
          f' "{COMMIT_SUCCESS_FILE}" file.'
      )
    await _shared_validate(cls.__name__, temporary_path)

  @classmethod
  async def validate_final(
      cls,
      final_path: epath.Path,
  ):
    if not await async_path.exists(final_path / COMMIT_SUCCESS_FILE):
      raise ValidationError(
          f'Expected {cls.__name__} ({final_path}) to contain the'
          f' "{COMMIT_SUCCESS_FILE}" file.'
      )
    await _shared_validate(cls.__name__, final_path)

  @classmethod
  def from_temporary(
      cls,
      temporary_path: epath.Path,
      *,
      file_options: options_lib.FileOptions | None = None,
      use_snapshot: bool | None = None,
  ) -> CommitFileTemporaryPath:
    return cls(
        temporary_path,
        temporary_path,
        file_options=file_options,
        use_snapshot=use_snapshot,
    )

  @classmethod
  def from_final(
      cls,
      final_path: epath.Path,
      *,
      checkpoint_metadata_store: (
          checkpoint_metadata.MetadataStore | None
      ) = None,
      file_options: options_lib.FileOptions | None = None,
      use_snapshot: bool | None = None,
  ) -> CommitFileTemporaryPath:
    if use_snapshot:
      raise ValueError('Snapshot is not supported for CommitFileTemporaryPath.')

    return cls(
        final_path,
        final_path,
        checkpoint_metadata_store=checkpoint_metadata_store,
        file_options=file_options,
    )

  def get_final(self) -> epath.Path:
    return self._final_path

  async def create(self) -> epath.Path:
    """Creates a non-deterministic tmp directory for saving for given `final_dir`.

    Also writes checkpoint metadata in the tmp directory.

    NOTE: This function does not include any barrier syncs, and calling it
    directly from multiprocess code can lead to race conditions. Prefer to
    use `atomicity.create_all` in such cases.

    Returns:
      The tmp directory.

    Raises:
      FileExistsError: if tmp directory already exists.
    """
    return await _create_tmp_directory(
        async_path.mkdir,
        self._tmp_path,
        path_permission_mode=self._path_permission_mode,
        checkpoint_metadata_store=self._checkpoint_metadata_store,
    )

  async def finalize(
      self,
  ):
    """Finalizes atomic save by writing a success file.

    Updates checkpoint metadata with commit_timestamp_nsecs.

    """
    logging.info('Finalizing %s', self._tmp_path)
    if self._checkpoint_metadata_store:
      await asyncio.to_thread(
          self._checkpoint_metadata_store.wait_until_finished
      )
      await asyncio.to_thread(
          self._checkpoint_metadata_store.update,
          file_path=checkpoint_metadata.step_metadata_file_path(self._tmp_path),
          commit_timestamp_nsecs=time.time_ns(),
      )
      await asyncio.to_thread(
          self._checkpoint_metadata_store.wait_until_finished
      )

    commit_success_file = self._final_path / COMMIT_SUCCESS_FILE
    await async_path.write_text(
        commit_success_file,
        f'Checkpoint commit was successful to {self._final_path}',
    )


async def create_all(
    paths: Sequence[atomicity_types.TemporaryPath],
    *,
    multiprocessing_options: options_lib.MultiprocessingOptions | None = None,
):
  """Creates all temporary paths in parallel."""
  start = time.time()
  multiprocessing_options = (
      multiprocessing_options or options_lib.MultiprocessingOptions()
  )
  barrier_sync_key_prefix = multiprocessing_options.barrier_sync_key_prefix
  active_processes = multiprocessing_options.active_processes
  # Sync before existence is checked and directory is created because there are
  # additional existence checks happening in the callers of this function.
  multihost.sync_global_processes(
      multihost.unique_barrier_key(
          'create_tmp_directory:pre',
          prefix=barrier_sync_key_prefix,
      ),
      timeout=multihost.coordination_timeout(),
      processes=active_processes,
  )
  if multihost.is_primary_host(multiprocessing_options.primary_host):
    await asyncio.gather(*[path.create() for path in paths])
  multihost.sync_global_processes(
      multihost.unique_barrier_key(
          'create_tmp_directory:post',
          prefix=barrier_sync_key_prefix,
      ),
      timeout=multihost.coordination_timeout(),
      processes=active_processes,
  )
  directory_creation_secs = time.time() - start
  jax.monitoring.record_event_duration_secs(
      '/jax/orbax/write/directory_creation_secs', directory_creation_secs
  )
  logging.vlog(
      1,
      'Synchronous directory creation took %s seconds',
      directory_creation_secs,
  )


def create_all_async(
    paths: Sequence[atomicity_types.TemporaryPath],
    completion_signals: Sequence[synchronization.HandlerAwaitableSignal],
    *,
    multiprocessing_options: options_lib.MultiprocessingOptions | None = None,
    subdirectories: Sequence[str] | None = None,
    operation_id: str | None = None,
) -> future.Future:
  """Creates all temporary paths in parallel asynchronously.

  Args:
    paths: Sequence of temporary paths to create.
    completion_signals: Sequence of signals to send when all paths are created.
      Also adds them to the awaitable signals contract.
    multiprocessing_options: MultiprocessingOptions to use for barrier syncs and
      primary host.
    subdirectories: Sequence of subdirectories to create under `paths`. If not
      provided, no subdirectories will be created. The same set of
      subdirectories will be created under each path in `paths`.
    operation_id: The operation id to use for the barrier keys. If None, the
      current operation id is used.

  Returns:
    A future that which sends the completion signals when all paths are created.
  """
  multiprocessing_options = (
      multiprocessing_options or options_lib.MultiprocessingOptions()
  )
  barrier_sync_key_prefix = multiprocessing_options.barrier_sync_key_prefix
  active_processes = multiprocessing_options.active_processes
  primary_host = multiprocessing_options.primary_host
  # Sync for existence check to complete on all hosts before directory
  # creation starts.
  multihost.sync_global_processes(
      multihost.unique_barrier_key(
          'create_tmp_directory:post_existence_check',
          prefix=barrier_sync_key_prefix,
      ),
      timeout=multihost.coordination_timeout(),
      processes=active_processes,
  )

  commit_future = future.NoopFuture()
  if multihost.is_primary_host(primary_host):
    commit_future = future.CommitFutureAwaitingContractedSignals(
        _create_paths(
            paths,
            subdirectories=subdirectories,
        ),
        send_signals=completion_signals,
        timeout_secs=multihost.coordination_timeout(),
        operation_id=operation_id,
    )
    future.AwaitableSignalsContract.add_to_awaitable_signals_contract(
        completion_signals
    )

  # Sync to enusre that all hosts have the same awaitable signals contract.
  multihost.sync_global_processes(
      multihost.unique_barrier_key(
          'add_to_awaitable_signals_contract',
          prefix=barrier_sync_key_prefix,
      ),
      timeout=multihost.coordination_timeout(),
      processes=active_processes,
  )
  return commit_future


async def _create_paths(
    tmp_paths: Sequence[atomicity_types.TemporaryPath],
    subdirectories: Sequence[str] | None = None,
):
  """Creates all temporary paths in parallel."""
  start = time.time()
  paths = await asyncio.gather(*[path.create() for path in tmp_paths])
  if subdirectories:
    creation_ops = []
    for path in paths:
      creation_ops.extend([
          async_path.mkdir(path / name, parents=False, exist_ok=False)
          for name in subdirectories
      ])
    await asyncio.gather(*creation_ops)
  directory_creation_secs = time.time() - start
  jax.monitoring.record_event_duration_secs(
      '/jax/orbax/write/directory_creation_secs',
      directory_creation_secs,
  )
  # TODO(mridulsahu): Adding a new metric to track only async directory creation
  # time for savings. This can eventually be removed once we completely disable
  # sync directory creation.
  jax.monitoring.record_event_duration_secs(
      '/jax/orbax/write/async_directory_creation_secs',
      directory_creation_secs,
  )
  logging.vlog(
      1,
      'Asynchronous directory creation took %s seconds',
      directory_creation_secs,
  )


async def on_commit_callback(
    tmp_dir: atomicity_types.TemporaryPath,
    *,
    checkpoint_start_time: float,
):
  """To commit save operation, atomically finalizes step dir.

  Records save duration and lineage-logs step dir.

  Args:
    tmp_dir: A temporary checkpoint directory, where the checkpoint data is
      currently saved.
    checkpoint_start_time: The time at which checkpoint saving began. # BEGIN
    tree_verity_options: Options to configure checkpoint signing and integrity
  """
  await tmp_dir.finalize(
  )
  record_saved_duration(checkpoint_start_time)
  jax.monitoring.record_event('/jax/orbax/write/success')
  logging.info(
      '[process=%s][thread=%s] Finished saving checkpoint (finalized tmp dir)'
      ' to `%s`.',
      multihost.process_index(),
      threading.current_thread().name,
      tmp_dir.get_final(),
  )


def record_saved_duration(checkpoint_start_time: float):
  """Record program duration that is accounted for by this checkpoint.

  For the very first checkpoint, this is the interval between program init and
  current checkpoint start time.

  Note that we use the checkpoint start time instead of end time. The saved
  duration should not include parallel training duration while the async
  checkpoint is being written in the background.

  Args:
    checkpoint_start_time: Start time of current checkpoint.
  """
  global _LAST_CHECKPOINT_WRITE_TIME
  # Note: for the very first checkpoint, this is the interval between program
  # init and the current checkpoint start time.
  duration_since_last_checkpoint = (
      checkpoint_start_time - _LAST_CHECKPOINT_WRITE_TIME
  )
  # TODO(hanyangtay): Remove version guard.
  if jax.version.__version_info__ > (0, 3, 25):
    jax.monitoring.record_event_duration_secs(
        '/jax/checkpoint/write/duration_since_last_checkpoint_secs',
        duration_since_last_checkpoint,
    )
  _LAST_CHECKPOINT_WRITE_TIME = checkpoint_start_time
