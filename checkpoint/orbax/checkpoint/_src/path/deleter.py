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

"""Checkpoint deleter."""

import functools
import os
import queue
import threading
import time
from typing import Optional, Protocol, Sequence
from urllib.parse import urlparse  # pylint: disable=g-importing-member
from absl import logging
from etils import epath
import jax
from orbax.checkpoint import utils
from orbax.checkpoint._src.path import step as step_lib

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
      primary_host: Optional[int],
      directory: epath.Path,
      todelete_subdir: Optional[str],
      name_format: step_lib.NameFormat[step_lib.Metadata],
      enable_hns_rmtree: bool = False,
      duration_metric: Optional[str] = _STANDARD_DELETE_DURATION,
  ):
    """StandardCheckpointDeleter constructor.

    Args:
      primary_host: refer to CheckpointManager.primary_host
      directory: refer to CheckpointManager.directory
      todelete_subdir: refer to CheckpointManagerOptions.todelete_subdir
      name_format: refer to CheckpointManager._name_format
      enable_hns_rmtree: refer to CheckpointManagerOptions.enable_hns_rmtree
      duration_metric: the name of the total delete duration metric
    """
    self._primary_host = primary_host
    self._directory = directory
    self._todelete_subdir = todelete_subdir
    self._name_format = name_format
    self._enable_hns_rmtree = enable_hns_rmtree
    self._duration_metric = duration_metric

  @functools.lru_cache(maxsize=32)
  def _is_hierarchical_namespace_enabled(self, path: epath.Path) -> bool:
    """Return whether hierarchical namespace is enabled."""
    # pylint: disable=g-import-not-at-top
    from google.cloud import storage  # pytype: disable=import-error

    parsed = urlparse(str(path))
    assert parsed.scheme == 'gs', f'Unsupported scheme for HNS: {parsed.scheme}'
    bucket_name = parsed.netloc

    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    return bucket.hierarchical_namespace_enabled

  def _rm_empty_folders(self, path: epath.Path) -> None:
    """For a hierarchical namespace bucket, delete empty folders recursively."""
    # pylint: disable=g-import-not-at-top
    from google.cloud import storage_control_v2  # pytype: disable=import-error

    parsed = urlparse(str(path))
    assert parsed.scheme == 'gs', f'Unsupported scheme for HNS: {parsed.scheme}'
    bucket = parsed.netloc
    prefix = parsed.path

    client = storage_control_v2.StorageControlClient()
    project_path = client.common_project_path('_')
    bucket_path = f'{project_path}/buckets/{bucket}'
    folders = set(
        # Format: "projects/{project}/buckets/{bucket}/folders/{folder}"
        folder.name
        for folder in client.list_folders(
            request=storage_control_v2.ListFoldersRequest(
                parent=bucket_path, prefix=prefix.strip('/') + '/'
            )
        )
    )

    while folders:
      parents = set(os.path.dirname(x.rstrip('/')) + '/' for x in folders)
      leaves = folders - parents
      requests = [
          storage_control_v2.DeleteFolderRequest(name=f) for f in leaves
      ]
      for req in requests:
        client.delete_folder(request=req)
      folders = folders - leaves
      logging.vlog(
          1,
          'Deleted %s folders, %s remaining. [%s][%s]',
          len(leaves),
          len(folders),
          bucket,
          prefix,
      )

  def _rmtree(self, path: epath.Path):
    """Recursively deletes a path.

    For a hierarchical namespace bucket, `path.rmtree()` only removes objects,
    leaving all the empty parent folders intact. Here we manually delete the
    empty folders recursively.

    Args:
      path: the path to delete.
    """
    # Step 1: Delete all files within the tree.
    path.rmtree()

    # Step 2: For HNS, clean up the remaining empty directory structure.
    if self._enable_hns_rmtree and self._is_hierarchical_namespace_enabled(
        path
    ):
      self._rm_empty_folders(path)

  def delete(self, step: int) -> None:
    """Deletes step dir or renames it if _todelete_subdir is set.

    See `CheckpointManagerOptions.todelete_subdir` for details.

    Args:
      step: checkpointing step number.
    """
    start = time.time()
    try:
      if not utils.is_primary_host(self._primary_host):
        logging.info(
            'Not primary host(%s), skipping deletion of step %d.',
            self._primary_host,
            step,
        )
        return

      # Delete if storage is on gcs or todelete_subdir is not set.
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

      if self._todelete_subdir is None or step_lib.is_gcs_path(self._directory):
        self._rmtree(delete_target)
        logging.info('Deleted step %d.', step)
        return

      # Rename step dir.
      rename_dir = self._directory / self._todelete_subdir
      rename_dir.mkdir(parents=True, exist_ok=True)

      dst = step_lib.build_step_path(rename_dir, self._name_format, step)

      delete_target.replace(dst)
      logging.info('Renamed step %d to %s', step, dst)
    finally:
      jax.monitoring.record_event_duration_secs(
          self._duration_metric,
          time.time() - start,
      )

  def delete_steps(self, steps: Sequence[int]) -> None:
    for step in steps:
      self.delete(step)

  def close(self) -> None:
    pass


class ThreadedCheckpointDeleter:
  """A threaded CheckpointDeleter."""

  def __init__(
      self,
      primary_host: Optional[int],
      directory: epath.Path,
      todelete_subdir: Optional[str],
      name_format: step_lib.NameFormat[step_lib.Metadata],
      enable_hns_rmtree: bool,
  ):
    """ThreadedCheckpointDeleter deletes checkpoints in a background thread."""
    self._standard_deleter = StandardCheckpointDeleter(
        primary_host=primary_host,
        directory=directory,
        todelete_subdir=todelete_subdir,
        name_format=name_format,
        enable_hns_rmtree=enable_hns_rmtree,
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
    primary_host: Optional[int],
    directory: epath.Path,
    todelete_subdir: Optional[str],
    name_format: step_lib.NameFormat[step_lib.Metadata],
    enable_hns_rmtree: bool,
    enable_background_delete: bool,
) -> CheckpointDeleter:
  """Creates a CheckpointDeleter."""

  if enable_background_delete:
    return ThreadedCheckpointDeleter(
        primary_host,
        directory,
        todelete_subdir,
        name_format,
        enable_hns_rmtree,
    )
  else:
    return StandardCheckpointDeleter(
        primary_host,
        directory,
        todelete_subdir,
        name_format,
        enable_hns_rmtree,
    )
