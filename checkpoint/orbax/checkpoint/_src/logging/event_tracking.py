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

"""Logging utilities for tracking checkpoint events."""

import enum

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


class OperationType(enum.Enum):
  SAVE = 'save'
  LOAD = 'load'


class OperationRecorder:
  """Records durations and events for checkpointing (save/load) operations."""

  def __init__(
      self,
      path: epath.Path,
      operation_type: OperationType,
      *,
      async_origin: bool,
      primary_host: int = 0,
  ):
    self._path = path
    self._operation_type = operation_type
    self._async_origin = async_origin
    self._primary_host = primary_host

  def record_start(self):
    """Records the start of an operation."""
    logging.info(
        '[process=%s] [%s] Started %s checkpoint @ %s.',
        multihost.process_index(),
        'async' if self._async_origin else 'sync',
        self._operation_type.value,
        self._path,
    )

    match self._operation_type:
      case OperationType.SAVE:
        event_name = (
            '/jax/orbax/write/async/start'
            if self._async_origin
            else '/jax/orbax/write/start'
        )
      case OperationType.LOAD:
        event_name = (
            '/jax/orbax/read/async/start'
            if self._async_origin
            else '/jax/orbax/read/start'
        )

    if multihost.is_primary_host(self._primary_host):
      jax.monitoring.record_event(event_name)

    if self._operation_type == OperationType.SAVE:
      jax.monitoring.record_event(
          '/jax/orbax/write/storage_type',
          storage_type=path_utils.get_storage_type(self._path),
      )

  def record_blocking_completion(self, duration_secs: float):
    """Records the completion of the blocking part of an operation."""
    match self._operation_type:
      case OperationType.SAVE:
        event_name = (
            '/jax/checkpoint/write/async/blocking_duration_secs'
            if self._async_origin
            else '/jax/orbax/write/blocking_duration_secs'
        )
        record_write_event(self._path)
      case OperationType.LOAD:
        event_name = (
            '/jax/orbax/read/async/blocking_duration_secs'
            if self._async_origin
            else '/jax/orbax/read/blocking_duration_secs'
        )
        record_read_event(self._path)

    if multihost.is_primary_host(self._primary_host):
      jax.monitoring.record_event_duration_secs(
          event_name,
          duration_secs,
      )

    logging.info(
        '[process=%s] [%s] Finished blocking %s in %.2f seconds. Continuing %s'
        ' @ %s.',
        multihost.process_index(),
        'async' if self._async_origin else 'sync',
        self._operation_type.value,
        duration_secs,
        self._operation_type.value,
        self._path,
    )

  def record_completion(self, duration_secs: float):
    """Records the completion of an entire operation."""
    logging.info(
        '[process=%s] [%s] Finished %s%s in %.2f seconds @ %s',
        multihost.process_index(),
        'async' if self._async_origin else 'sync',
        self._operation_type.value,
        ' (blocking + background)' if self._async_origin else '',
        duration_secs,
        self._path,
    )
    match self._operation_type:
      case OperationType.SAVE:
        duration_event_name = (
            '/jax/checkpoint/write/async/total_duration_secs'
            if self._async_origin
            else '/jax/orbax/write/total_duration_secs'
        )
        success_event_name = '/jax/orbax/write/success'
      case OperationType.LOAD:
        duration_event_name = (
            '/jax/orbax/read/async/total_duration_secs'
            if self._async_origin
            else '/jax/orbax/read/total_duration_secs'
        )
        success_event_name = '/jax/orbax/read/success'
    if multihost.is_primary_host(self._primary_host):
      jax.monitoring.record_event(success_event_name)
      jax.monitoring.record_event_duration_secs(
          duration_event_name,
          duration_secs,
      )
