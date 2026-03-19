# Copyright 2026 The Orbax Authors.
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

"""Checkpoint deleter."""

import datetime
import pathlib
import queue
import threading
import time
from typing import Optional, Protocol, Sequence
from urllib import parse

from absl import logging
from etils import epath
import jax
from orbax.checkpoint._src.logging import event_tracking
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.path import gcs_utils
from orbax.checkpoint._src.path import step as step_lib


urlparse = parse.urlparse
PurePosixPath = pathlib.PurePosixPath

_THREADED_DELETE_DURATION = (
    '/jax/orbax/checkpoint_manager/threaded_checkpoint_deleter/duration'
)

_STANDARD_DELETE_DURATION = (
    '/jax/orbax/checkpoint_manager/standard_checkpoint_deleter/duration'
)


class CheckpointDeleter(Protocol):
  """A protocol defined a CheckpointDeleter."""

  def delete(self, step: int) -> None:
    """Delete a step."""
    ...

  def delete_steps(self, steps: Sequence[int]) -> None:
    """Delete steps."""
    ...

  def close(self) -> None:
    """Perform any cleanup before closing this deleter."""
    ...


class StandardCheckpointDeleter:
  """A StandardCheckpointDeleter."""

  def __init__(
      self,
      directory: epath.Path,
      *,
      name_format: step_lib.NameFormat[step_lib.Metadata],
      primary_host: Optional[int] = 0,
      todelete_subdir: Optional[str] = None,
      todelete_full_path: Optional[str] = None,
      duration_metric: Optional[str] = _STANDARD_DELETE_DURATION,
  ):
    """StandardCheckpointDeleter constructor.

    Args:
      directory: refer to CheckpointManager.directory
      name_format: refer to CheckpointManager._name_format
      primary_host: refer to CheckpointManager.primary_host
      todelete_subdir: refer to CheckpointManagerOptions.todelete_subdir
      todelete_full_path: refer to CheckpointManagerOptions.todelete_full_path
      duration_metric: the name of the total delete duration metric
    """
    self._primary_host = primary_host
    self._directory = directory
    self._todelete_subdir = todelete_subdir
    self._todelete_full_path = todelete_full_path
    self._name_format = name_format
    self._duration_metric = duration_metric

  def _rmtree(self, path: epath.Path):
    """Recursively deletes a path.

    Args:
      path: the path to delete.
    """
    # TODO(b/493110683): Cleanup with refactoring of HNS GCS logic into
    # StorageBackend.
    if gcs_utils.is_gcs_path(path):
      gcs_utils.rmtree(path)
    else:
      path.rmtree()

  def delete(self, step: int) -> None:
    """Deletes step dir or renames it if options are set.

    See `CheckpointManagerOptions.todelete_subdir` for details.

    Args:
      step: checkpointing step number.
    """
    start = time.time()
    try:
      if not multihost.is_primary_host(self._primary_host):
        logging.info(
            'Not primary host(%s), skipping deletion of step %d.',
            self._primary_host,
            step,
        )
        return

      try:
        delete_target = step_lib.find_step_path(
            self._directory,
            self._name_format,
            step=step,
            include_uncommitted=True,
        )
      except ValueError as e:
        logging.warning(
            'Unable to find the step %d for deletion or renaming, err=%s',
            step,
            e,
        )
        return

      # Attempt to rename using GCS HNS API if configured.
      if self._todelete_full_path is not None:
        if gcs_utils.is_gcs_path(self._directory):
          # This is recommended for GCS buckets with HNS enabled and requires
          # `_todelete_full_path` to be specified.
          self._gcs_rename_step(step, delete_target)
        else:
          raise NotImplementedError()
      # Attempt to rename to local subdirectory using `todelete_subdir`
      # if configured.
      elif self._todelete_subdir is not None and not gcs_utils.is_gcs_path(
          self._directory
      ):
        self._rename_step_to_subdir(step, delete_target)
      # The final case: fall back to permanent deletion.
      else:
        self._delete_step_permanently(step, delete_target)

      event_tracking.record_delete_event(delete_target)

    finally:
      jax.monitoring.record_event_duration_secs(
          self._duration_metric,
          time.time() - start,
      )

  def _gcs_rename_step(
      self, step: int, delete_target: epath.Path
  ):
    """Renames a GCS directory to a temporary location for deletion.

    This method renames the directory using the
    underlying `tf.io.gfile.rename` method. This underlying
    implementation will automatically detect if the bucket is HNS-enabled
    and use a fast atomic rename, or fall back to a legacy
    copy/delete rename if it is not.

    Args:
      step: The checkpoint step number.
      delete_target: The path to the directory to be renamed.
    """
    try:
      # Get the bucket name from the source path
      bucket_name = urlparse(str(delete_target)).netloc
      if not bucket_name:
        raise ValueError(
            f'Could not parse bucket name from path: {delete_target}'
        )

      # Construct the destination path inside the `_todelete_full_path` dir.
      destination_parent_path = epath.Path(
          f'gs://{bucket_name}/{self._todelete_full_path}'
      )
      destination_parent_path.mkdir(parents=True, exist_ok=True)

      # Create a unique name for the destination to avoid collisions.
      now = datetime.datetime.now()
      timestamp_str = now.strftime('%Y%m%d-%H%M%S-%f')
      new_name_with_timestamp = f'{delete_target.name}-{timestamp_str}'
      dest_path = destination_parent_path / new_name_with_timestamp

      logging.info(
          'Executing filesystem-aware rename: Source=`%s`, Destination=`%s`',
          delete_target,
          dest_path,
      )

      # Call the high-level rename method.
      # This will be fast on HNS and slow (but functional) on non-HNS.
      delete_target.rename(dest_path)
      logging.info('Successfully renamed step %d to %s', step, dest_path)

    except Exception as e:
      message = f'Rename failed for step {step}. Error: {e}'
      logging.error(message)
      raise RuntimeError(message) from e

  def _rename_step_to_subdir(self, step: int, delete_target: epath.Path):
    """Renames a step directory to its corresponding todelete_subdir."""
    rename_dir = self._directory / self._todelete_subdir
    rename_dir.mkdir(parents=True, exist_ok=True)
    dst = step_lib.build_step_path(rename_dir, self._name_format, step)
    delete_target.replace(dst)
    logging.info('Renamed step %d to %s', step, dst)

  def _delete_step_permanently(self, step: int, delete_target: epath.Path):
    """Permanently deletes a step directory."""
    self._rmtree(delete_target)
    logging.info('Deleted step %d.', step)

  def delete_steps(self, steps: Sequence[int]) -> None:
    for step in steps:
      self.delete(step)

  def close(self) -> None:
    pass


class ThreadedCheckpointDeleter:
  """A threaded CheckpointDeleter."""

  def __init__(
      self,
      directory: epath.Path,
      *,
      name_format: step_lib.NameFormat[step_lib.Metadata],
      primary_host: Optional[int] = 0,
      todelete_subdir: Optional[str] = None,
      todelete_full_path: Optional[str] = None,
  ):
    """ThreadedCheckpointDeleter deletes checkpoints in a background thread."""
    self._standard_deleter = StandardCheckpointDeleter(
        primary_host=primary_host,
        directory=directory,
        todelete_subdir=todelete_subdir,
        todelete_full_path=todelete_full_path,
        name_format=name_format,
        duration_metric=_THREADED_DELETE_DURATION,
    )
    self._delete_queue = queue.Queue()
    # Turn on daemon=True so the thread won't block the main thread and die
    # when the program exits.
    self._delete_thread = threading.Thread(
        target=self._delete_thread_run, name='DeleterThread', daemon=True
    )
    self._delete_thread.start()

    jax.monitoring.record_event(
        '/jax/orbax/checkpoint_manager/threaded_checkpoint_deleter/init'
    )

  def _delete_thread_run(self) -> None:
    logging.info('Delete thread has started.')
    while True:
      step = self._delete_queue.get(block=True)
      if step < 0:
        break
      self._standard_deleter.delete(step)
    logging.info('Delete thread exited.')

  def delete(self, step: int) -> None:
    self._delete_queue.put(step)

  def delete_steps(self, steps: Sequence[int]) -> None:
    for step in steps:
      self.delete(step)

  def close(self) -> None:
    # make sure all steps are deleted before exit.
    if self._delete_thread and self._delete_thread.is_alive():
      self._delete_queue.put(-1)
      self._delete_thread.join()

    self._standard_deleter.close()


def create_checkpoint_deleter(
    directory: epath.Path,
    *,
    name_format: step_lib.NameFormat[step_lib.Metadata],
    primary_host: Optional[int] = 0,
    todelete_subdir: Optional[str] = None,
    todelete_full_path: Optional[str] = None,
    enable_background_delete: bool = False,
) -> CheckpointDeleter:
  """Creates a CheckpointDeleter."""

  if enable_background_delete:
    return ThreadedCheckpointDeleter(
        directory,
        name_format=name_format,
        primary_host=primary_host,
        todelete_subdir=todelete_subdir,
        todelete_full_path=todelete_full_path,
    )
  else:
    return StandardCheckpointDeleter(
        directory,
        name_format=name_format,
        primary_host=primary_host,
        todelete_subdir=todelete_subdir,
        todelete_full_path=todelete_full_path,
    )
