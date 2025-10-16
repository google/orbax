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

"""Logging utilities for tracking checkpoint events."""

from absl import logging
from etils import epath
import jax
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.path import utils as path_utils



def record_read_event(directory: epath.Path):
  """Records a dataread event for the checkpoint."""
  return None


def record_read_metadata_event(directory: epath.Path):
  """Records a meatadataread event for the checkpoint."""
  return None


def record_write_event(directory: epath.Path):
  """Records a write event for the checkpoint."""
  return None


def record_delete_event(directory: epath.Path):
  """Records a deletion event for the checkpoint."""
  return None


def record_save_start(path: epath.Path, *, async_origin: bool):
  """Records the start of a save operation."""
  logging.info(
      '[process=%s] Started %s checkpoint to %s.',
      multihost.process_index(),
      'async saving' if async_origin else 'saving',
      path,
  )
  if async_origin:
    event_name = '/jax/orbax/write/async/start'
  else:
    event_name = '/jax/orbax/write/start'
  jax.monitoring.record_event(event_name)
  jax.monitoring.record_event(
      '/jax/orbax/write/storage_type',
      storage_type=path_utils.get_storage_type(path),
  )


def record_save_completion(
    path: epath.Path,
    *,
    total_duration_secs: float,
    async_origin: bool,
):
  """Records the completion of a save operation."""
  logging.info(
      'Finished asynchronous save (blocking + background) in %.2f seconds'
      ' to %s',
      total_duration_secs,
      path,
  )
  # TODO(cpgaffney): No event is currently being recorded for synchronous saves.
  # Consider collecting this information
  if async_origin:
    jax.monitoring.record_event_duration_secs(
        '/jax/checkpoint/write/async/total_duration_secs',
        total_duration_secs,
    )
