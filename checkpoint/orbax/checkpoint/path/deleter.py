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

"""Checkpoint deleter."""

import queue
import threading
from typing import Optional, Protocol, Sequence
from absl import logging
from etils import epath
import jax
from orbax.checkpoint import utils
from orbax.checkpoint.path import step as step_lib


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
      name_format: step_lib.NameFormat,
  ):
    """StandardCheckpointDeleter constructor.

    Args:
      primary_host: refer to CheckpointManager.primary_host
      directory: refer to CheckpointManager.directory
      todelete_subdir: refer to CheckpointManagerOptions.todelete_subdir
      name_format: refer to CheckpointManager._name_format
    """
    self._primary_host = primary_host
    self._directory = directory
    self._todelete_subdir = todelete_subdir
    self._name_format = name_format

  def delete(self, step: int) -> None:
    """Deletes step dir or renames it if _todelete_subdir is set.

    See `CheckpointManagerOptions.todelete_subdir` for details.

    Args:
      step: checkpointing step number.
    """
    if not utils.is_primary_host(self._primary_host):
      return

    # Delete if storage is on gcs or todelete_subdir is not set.
    try:
      delete_target = self._name_format.find_step(
          self._directory,
          step,
      ).path
    except ValueError as e:
      logging.warning(
          'Unable to find the step %d for deletion or renaming, err=%s', step, e
      )
      return

    if self._todelete_subdir is None or utils.is_gcs_path(self._directory):
      delete_target.rmtree()
      logging.info('Deleted step %d.', step)
      return

    # Rename step dir.
    rename_dir = self._directory / self._todelete_subdir
    rename_dir.mkdir(parents=True, exist_ok=True)

    dst = step_lib.build_step_path(rename_dir, self._name_format, step)

    delete_target.replace(dst)
    logging.info('Renamed step %d to %s', step, dst)

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
      name_format: step_lib.NameFormat,
  ):
    """ThreadedCheckpointDeleter deletes checkpoints in a background thread."""
    self._standard_deleter = StandardCheckpointDeleter(
        primary_host=primary_host,
        directory=directory,
        todelete_subdir=todelete_subdir,
        name_format=name_format,
    )
    self._delete_queue = queue.Queue()
    self._delete_thread = threading.Thread(
        target=self._delete_thread_run, name='DeleterThread'
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
        logging.info('Delete thread exited.')
        break
      self._standard_deleter.delete(step)

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
    name_format: step_lib.NameFormat,
    enable_background_delete: bool,
) -> CheckpointDeleter:
  """Creates a CheckpointDeleter."""

  if enable_background_delete:
    return ThreadedCheckpointDeleter(
        primary_host,
        directory,
        todelete_subdir,
        name_format,
    )
  else:
    return StandardCheckpointDeleter(
        primary_host,
        directory,
        todelete_subdir,
        name_format,
    )
