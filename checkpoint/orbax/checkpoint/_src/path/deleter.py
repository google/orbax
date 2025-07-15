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

import concurrent.futures
import functools
import os
import queue
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
      deleter_name: Optional[str] = None,
  ):
    """StandardCheckpointDeleter constructor.

    Args:
      primary_host: refer to CheckpointManager.primary_host
      directory: refer to CheckpointManager.directory
      todelete_subdir: refer to CheckpointManagerOptions.todelete_subdir
      name_format: refer to CheckpointManager._name_format
      enable_hns_rmtree: refer to CheckpointManagerOptions.enable_hns_rmtree
      duration_metric: the name of the total delete duration metric
      deleter_name: the name of the deleter. This is used for logging.
    """
    self._primary_host = primary_host
    self._directory = directory
    self._todelete_subdir = todelete_subdir
    self._name_format = name_format
    self._enable_hns_rmtree = enable_hns_rmtree
    self._duration_metric = duration_metric
    self._deleter_name = deleter_name

  @functools.lru_cache(maxsize=32)
  def _is_hierarchical_namespace_enabled(self, bucket_name: str) -> bool:
    """Return whether hierarchical namespace is enabled."""
    # pylint: disable=g-import-not-at-top
    from google.cloud import storage  # pytype: disable=import-error

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
    if self._enable_hns_rmtree:
      parsed = urlparse(str(path))
      assert (
          parsed.scheme == 'gs'
      ), f'Unsupported scheme for HNS: {parsed.scheme}'
      bucket_name = parsed.netloc
      if self._is_hierarchical_namespace_enabled(bucket_name):
        self._rm_empty_folders(path)

  def delete(self, step: int) -> None:
    """Deletes step dir or renames it if _todelete_subdir is set.

    See `CheckpointManagerOptions.todelete_subdir` for details.

    Args:
      step: checkpointing step number.
    """
    start = time.time()
    action_taken = 'SkippedDelete'
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
        action_taken = 'Deleted'
        return

      # Rename step dir.
      rename_dir = self._directory / self._todelete_subdir
      rename_dir.mkdir(parents=True, exist_ok=True)

      dst = step_lib.build_step_path(rename_dir, self._name_format, step)

      delete_target.replace(dst)
      action_taken = f'Renamed to {dst}'
    finally:
      elapsed_time = time.time() - start
      logging.info(
          '%s(step=%d)%s took %ss.',
          action_taken,
          step,
          f' by {self._deleter_name}' if self._deleter_name else '',
          elapsed_time,
      )
      jax.monitoring.record_event_duration_secs(
          self._duration_metric,
          elapsed_time,
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
      background_thread_count: int,
  ):
    """ThreadedCheckpointDeleter deletes checkpoints in a background thread."""
    self._primary_host = primary_host
    self._directory = directory
    self._todelete_subdir = todelete_subdir
    self._name_format = name_format
    self._enable_hns_rmtree = enable_hns_rmtree

    self._delete_queue = queue.Queue()
    self._executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=background_thread_count, thread_name_prefix='DeleterThread'
    )
    self._background_thread_count = background_thread_count
    for i in range(self._background_thread_count):
      self._executor.submit(self._delete_thread_run, i)

    jax.monitoring.record_event(
        '/jax/orbax/checkpoint_manager/threaded_checkpoint_deleter/init'
    )

  def _delete_thread_run(self, thread_id: int) -> None:
    """The actual function run by a DeleteThread."""

    logging.info('DeleteThread%d has started.', thread_id)
    standard_deleter = StandardCheckpointDeleter(
        primary_host=self._primary_host,
        directory=self._directory,
        todelete_subdir=self._todelete_subdir,
        name_format=self._name_format,
        enable_hns_rmtree=self._enable_hns_rmtree,
        duration_metric=_THREADED_DELETE_DURATION,
        deleter_name=f'DeleteThread{thread_id}',
    )
    while True:
      step = self._delete_queue.get(block=True)
      if step < 0:
        break
      standard_deleter.delete(step)

    standard_deleter.close()
    logging.info('Delete thread%d exited.', thread_id)

  def delete(self, step: int) -> None:
    self._delete_queue.put(step)

  def delete_steps(self, steps: Sequence[int]) -> None:
    for step in steps:
      self.delete(step)

  def close(self) -> None:
    # make sure all steps are deleted before exit.
    for _ in range(self._background_thread_count):
      self._delete_queue.put(-1)
    self._executor.shutdown(wait=True)


def create_checkpoint_deleter(
    primary_host: Optional[int],
    directory: epath.Path,
    todelete_subdir: Optional[str],
    name_format: step_lib.NameFormat[step_lib.Metadata],
    enable_hns_rmtree: bool,
    enable_background_delete: bool,
    background_thread_count: int = 1,
) -> CheckpointDeleter:
  """Creates a CheckpointDeleter.

  Args:
    primary_host: The primary_host which wil be reponsible for deleting the
      checkpoints.
    directory: The root directory of the checkpoints.
    todelete_subdir: If set, checkpoints to be deleted will be only renamed into
      a subdirectory with the provided string.
    name_format: NameFormat to build or find steps under input root directory.
    enable_hns_rmtree: If True, enables additional step of HNS bucket empty
      folder deletion.
    enable_background_delete: If True, old checkpoint deletions will be done in
      a background thread, otherwise, it will be done at the end of each save.
      When it's enabled, make sure to call CheckpointManager.close() or use
      context to make sure all old steps are deleted before exit.
    background_thread_count: If enable_background_delete, this is the number of
      background threads to use in the thread pool.

  Returns:
    A CheckpointDeleter.
  """

  if enable_background_delete:
    return ThreadedCheckpointDeleter(
        primary_host,
        directory,
        todelete_subdir,
        name_format,
        enable_hns_rmtree,
        background_thread_count,
    )
  else:
    return StandardCheckpointDeleter(
        primary_host,
        directory,
        todelete_subdir,
        name_format,
        enable_hns_rmtree,
    )
