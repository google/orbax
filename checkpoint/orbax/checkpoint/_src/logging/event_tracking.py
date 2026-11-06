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
import re
import time

from absl import logging
from etils import epath
import jax
from orbax.checkpoint._src.logging import monitoring
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.path import utils as path_utils



def _seconds_to_milliseconds(seconds: float) -> float:
  """Converts seconds to milliseconds."""
  return seconds * 1000.0


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


_MONITORING_OPERATION_TYPES = {
    OperationType.SAVE: monitoring.OperationType.OPERATION_TYPE_SAVE,
    OperationType.LOAD: monitoring.OperationType.OPERATION_TYPE_LOAD,
}


_STEP_FROM_PATH_REGEX = re.compile(r'.+_(\d+)')


def _parse_step_from_path(path: epath.PathLike) -> int | None:
  """Parses a numeric step from bare ('100') or prefixed ('step_100') dirs.

  Args:
    path: Checkpoint directory path.

  Returns:
    The integer step parsed from the path name, or None if the directory name
    does not end in a numeric step.
  """
  name = epath.Path(path).name
  if name.isdigit():
    return int(name)
  match = _STEP_FROM_PATH_REGEX.fullmatch(name)
  return int(match.group(1)) if match else None


class OperationRecorder:
  """Records durations and events for checkpointing (save/load) operations."""

  def __init__(
      self,
      path: epath.PathLike,
      operation_type: OperationType,
      *,
      async_origin: bool,
      primary_host: int = 0,
      step: int | None = None,
      operation_id: str | None = None,
  ):
    """Initializes the instance.

    Args:
      path: Target checkpoint directory path.
      operation_type: Checkpointing operation type (`SAVE` or `LOAD`).
      async_origin: Whether the operation runs asynchronously in the background.
      primary_host: Host index designated as primary for coordination metrics.
      step: Checkpoint step number, or None to parse it from `path.name`.
      operation_id: Shared operation identifier, or None to format it from
        `operation_type` and the resolved step.
    """
    self._path = epath.Path(path)
    self._operation_type = operation_type
    self._monitoring_operation_type = _MONITORING_OPERATION_TYPES[
        operation_type
    ]
    self._async_origin = async_origin
    self._primary_host = primary_host
    self._step = _parse_step_from_path(self._path) if step is None else step
    if operation_id is None:
      self._operation_id = monitoring.format_operation_id(
          self._monitoring_operation_type, self._step or 0
      )
    else:
      self._operation_id = operation_id
    self._storage_type = path_utils.get_storage_type(self._path)

  def record_start(self, start_time: float):
    """Records the start of an operation."""
    logging.info(
        '[process=%s] [%s] Started %s checkpoint @ %s.',
        multihost.process_index(),
        'async' if self._async_origin else 'sync',
        self._operation_type.value,
        self._path,
    )

    record_start_time_name = None
    match self._operation_type:
      case OperationType.SAVE:
        record_event_name = (
            '/jax/orbax/write/async/start'
            if self._async_origin
            else '/jax/orbax/write/start'
        )
        record_start_time_name = (
            '/jax/orbax/write/async/start_time'
            if self._async_origin
            else '/jax/orbax/write/start_time'
        )
      case OperationType.LOAD:
        record_event_name = (
            '/jax/orbax/read/async/start'
            if self._async_origin
            else '/jax/orbax/read/start'
        )

    if multihost.is_primary_host(self._primary_host):
      jax.monitoring.record_event(record_event_name)
      if record_start_time_name is not None:
        jax.monitoring.record_scalar(
            record_start_time_name,
            _seconds_to_milliseconds(start_time),
            storage_type=self._storage_type,
        )

    if self._operation_type == OperationType.SAVE:
      jax.monitoring.record_event(
          '/jax/orbax/write/storage_type',
          storage_type=path_utils.get_storage_type(self._path),
      )
    elif self._operation_type == OperationType.LOAD:
      jax.monitoring.record_event(
          '/jax/orbax/read/storage_type',
          storage_type=path_utils.get_storage_type(self._path),
      )

    monitoring.record_operation(
        'begin',
        self._operation_id,
        self._monitoring_operation_type,
        timestamp=start_time,
        directory=str(self._path),
        step=self._step,
    )

  def record_blocking_completion(self, duration_secs: float, end_time: float):
    """Records the completion of the blocking part of an operation."""
    match self._operation_type:
      case OperationType.SAVE:
        if multihost.is_primary_host(self._primary_host):
          if self._async_origin:
            # Keep both /jax/checkpoint/write/... and /jax/orbax/write/...
            # as the /jax/checkpoint/write/... event is used for alerting.
            jax.monitoring.record_event_duration_secs(
                '/jax/checkpoint/write/async/blocking_duration_secs',
                duration_secs,
            )
            jax.monitoring.record_event_duration_secs(
                '/jax/orbax/write/async/blocking_duration_secs',
                duration_secs,
                storage_type=self._storage_type,
            )
            jax.monitoring.record_scalar(
                '/jax/orbax/write/async/blocking_end_time',
                _seconds_to_milliseconds(end_time),
                storage_type=self._storage_type,
            )
          else:
            jax.monitoring.record_event_duration_secs(
                '/jax/orbax/write/blocking_duration_secs',
                duration_secs,
                storage_type=self._storage_type,
            )
            jax.monitoring.record_scalar(
                '/jax/orbax/write/blocking_end_time',
                _seconds_to_milliseconds(end_time),
                storage_type=self._storage_type,
            )
      case OperationType.LOAD:
        record_read_event(self._path)
        if multihost.is_primary_host(self._primary_host):
          event_name = (
              '/jax/orbax/read/async/blocking_duration_secs'
              if self._async_origin
              else '/jax/orbax/read/blocking_duration_secs'
          )
          jax.monitoring.record_event_duration_secs(
              event_name,
              duration_secs,
              storage_type=self._storage_type,
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
        success_event_name = '/jax/orbax/write/success'
        if multihost.is_primary_host(self._primary_host):
          # Keep both /jax/checkpoint/write/... and /jax/orbax/write/...
          # as the /jax/checkpoint/write/... event is used for alerting.
          jax.monitoring.record_event(success_event_name)
          if self._async_origin:
            jax.monitoring.record_event_duration_secs(
                '/jax/checkpoint/write/async/total_duration_secs',
                duration_secs,
                storage_type=self._storage_type,
            )
            jax.monitoring.record_event_duration_secs(
                '/jax/orbax/write/async/total_duration_secs',
                duration_secs,
                storage_type=self._storage_type,
            )
          else:
            jax.monitoring.record_event_duration_secs(
                '/jax/orbax/write/total_duration_secs',
                duration_secs,
                storage_type=self._storage_type,
            )
      case OperationType.LOAD:
        success_event_name = '/jax/orbax/read/success'
        if multihost.is_primary_host(self._primary_host):
          jax.monitoring.record_event(success_event_name)
          duration_event_name = (
              '/jax/orbax/read/async/total_duration_secs'
              if self._async_origin
              else '/jax/orbax/read/total_duration_secs'
          )
          jax.monitoring.record_event_duration_secs(
              duration_event_name,
              duration_secs,
              storage_type=self._storage_type,
          )
    monitoring.record_operation(
        'end',
        self._operation_id,
        self._monitoring_operation_type,
        timestamp=time.time(),
        status='SUCCESS',
    )

  def record_failure(
      self,
      error: BaseException | str | None = None,
      *,
      status: str | None = None,
  ) -> None:
    """Records a failed or cancelled checkpointing operation end event.

    Args:
      error: Exception or error message that caused the failure, or None when
        cancelled without an exception.
      status: Explicit status override (e.g. 'CANCELLED'), or None to infer
        'TIMED_OUT' for TimeoutError and 'FAILURE' otherwise.
    """
    if status is None:
      status = 'TIMED_OUT' if isinstance(error, TimeoutError) else 'FAILURE'
    monitoring.record_operation(
        'end',
        self._operation_id,
        self._monitoring_operation_type,
        timestamp=time.time(),
        status=status,
        error=error,
    )


