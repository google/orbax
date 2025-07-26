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

import datetime
import os
import pathlib
import queue
import threading
import time
from typing import Optional, Protocol, Sequence
from absl import logging
from etils import epath
import jax
from orbax.checkpoint import utils
from orbax.checkpoint._src.path import gcs_utils
from orbax.checkpoint._src.path import step as step_lib
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
      enable_hns: bool = False,
      duration_metric: Optional[str] = _STANDARD_DELETE_DURATION,
  ):
    """StandardCheckpointDeleter constructor.

    Args:
      directory: refer to CheckpointManager.directory
      name_format: refer to CheckpointManager._name_format
      primary_host: refer to CheckpointManager.primary_host
      todelete_subdir: refer to CheckpointManagerOptions.todelete_subdir
      todelete_full_path: refer to CheckpointManagerOptions.todelete_full_path
      enable_hns: refer to CheckpointManagerOptions.enable_hns
      duration_metric: the name of the total delete duration metric
    """
    self._primary_host = primary_host
    self._directory = directory
    self._todelete_subdir = todelete_subdir
    self._todelete_full_path = todelete_full_path
    self._name_format = name_format
    self._enable_hns = enable_hns
    self._duration_metric = duration_metric

  def _rm_empty_folders(self, path: epath.Path) -> None:
    """For a hierarchical namespace bucket, delete empty folders recursively."""
    # pylint: disable=g-import-not-at-top
    from google.cloud import storage_control_v2  # pytype: disable=import-error

    bucket, prefix = gcs_utils.parse_gcs_path(path)

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
    if self._enable_hns:
      if gcs_utils.is_hierarchical_namespace_enabled(path):
        self._rm_empty_folders(path)

  def delete(self, step: int) -> None:
    """Deletes step dir or renames it if options are set.

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
        if step_lib.is_gcs_path(self._directory):
          self._rename_gcs_step_with_hns(step, delete_target)
          return
        else:
          raise NotImplementedError()

      # Attempt to rename to local subdirectory using `todelete_subdir`
      # if configured.
      if self._todelete_subdir is not None and not step_lib.is_gcs_path(
          self._directory
      ):
        self._rename_step_to_subdir(step, delete_target)
        return

      # The final case: fall back to permanent deletion.
      self._delete_step_permanently(step, delete_target)

    finally:
      jax.monitoring.record_event_duration_secs(
          self._duration_metric,
          time.time() - start,
      )

  def _rename_gcs_step_with_hns(
      self, step: int, delete_target: epath.Path
  ):
    """Renames a GCS directory using the Storage Control API.

    Args:
      step: The checkpoint step number.
      delete_target: The path to the directory to be renamed.

    Raises:
      ValueError: If the GCS bucket is not HNS-enabled, as this is a
        hard requirement for this operation.
    """
    logging.info(
        'Condition: GCS path with `todelete_full_path` set. Checking for HNS.'
    )
    bucket_name, _ = gcs_utils.parse_gcs_path(self._directory)
    if not gcs_utils.is_hierarchical_namespace_enabled(self._directory):
      raise ValueError(
          f'Bucket "{bucket_name}" does not have Hierarchical Namespace'
          ' enabled, which is required when _todelete_full_path is set.'
      )

    logging.info('HNS bucket detected. Attempting to rename step %d.', step)
    # pylint: disable=g-import-not-at-top
    from google.api_core import exceptions as google_exceptions  # pytype: disable=import-error
    try:
      from google.cloud import storage_control_v2  # pytype: disable=import-error
      import google.auth  # pytype: disable=import-error

      # Use default credentials, but without a quota project to avoid
      # quota issues with this API.
      credentials, _ = google.auth.default()
      creds_without_quota_project = credentials.with_quota_project(None)
      client = storage_control_v2.StorageControlClient(
          credentials=creds_without_quota_project
      )
      # Destination parent is the absolute path to the bucket.
      destination_parent_dir_str = (
          f'gs://{bucket_name}/{self._todelete_full_path}'
      )
      destination_parent_path = PurePosixPath(destination_parent_dir_str)
      logging.info(
          'Ensuring destination parent folder exists via HNS API: %s',
          destination_parent_dir_str,
      )
      try:
        parent_folder_id = str(
            destination_parent_path.relative_to(f'gs://{bucket_name}')
        )
        bucket_resource_name = f'projects/_/buckets/{bucket_name}'
        client.create_folder(
            request=storage_control_v2.CreateFolderRequest(
                parent=bucket_resource_name,
                folder_id=parent_folder_id,
                recursive=True,
            )
        )
        logging.info('HNS parent folder creation request sent.')
      except google_exceptions.AlreadyExists:
        logging.info('HNS parent folder already exists, proceeding.')

      now = datetime.datetime.now()
      timestamp_str = now.strftime('%Y%m%d-%H%M%S-%f')
      new_name_with_timestamp = f'{delete_target.name}-{timestamp_str}'
      dest_path = destination_parent_path / new_name_with_timestamp
      source_folder_id = str(delete_target.relative_to(f'gs://{bucket_name}'))
      destination_folder_id = str(dest_path.relative_to(f'gs://{bucket_name}'))
      source_resource_name = (
          f'projects/_/buckets/{bucket_name}/folders/{source_folder_id}'
      )
      logging.info('Rename API call: Source: %s', source_resource_name)
      logging.info('Rename API call: Destination ID: %s', destination_folder_id)
      request = storage_control_v2.RenameFolderRequest(
          name=source_resource_name,
          destination_folder_id=destination_folder_id,
      )
      op = client.rename_folder(request=request)
      op.result()
      logging.info('Successfully renamed step %d to %s', step, dest_path)
    except google_exceptions.GoogleAPIError as e:
      logging.error('HNS rename failed for step %d. Error: %s', step, e)

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
      enable_hns: bool = False,
  ):
    """ThreadedCheckpointDeleter deletes checkpoints in a background thread."""
    self._standard_deleter = StandardCheckpointDeleter(
        primary_host=primary_host,
        directory=directory,
        todelete_subdir=todelete_subdir,
        todelete_full_path=todelete_full_path,
        name_format=name_format,
        enable_hns=enable_hns,
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
    enable_hns: bool = False,
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
        enable_hns=enable_hns,
    )
  else:
    return StandardCheckpointDeleter(
        directory,
        name_format=name_format,
        primary_host=primary_host,
        todelete_subdir=todelete_subdir,
        todelete_full_path=todelete_full_path,
        enable_hns=enable_hns,
    )
