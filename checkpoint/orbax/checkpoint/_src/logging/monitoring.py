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

"""Orbax metrics telemetry base."""

from __future__ import annotations

import abc
import enum
import logging
import os
import threading
import time
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
  from orbax.checkpoint.proto import checkpoint_operation_pb2  # pylint: disable=g-import-not-at-top

  OperationType = checkpoint_operation_pb2.CheckpointOperation.OperationType
else:
  try:
    from orbax.checkpoint.proto import checkpoint_operation_pb2  # pylint: disable=g-import-not-at-top

    OperationType = checkpoint_operation_pb2.CheckpointOperation.OperationType
  except ImportError:

    class OperationType(enum.IntEnum):
      """Fallback enum when compiled protobuf bindings are unavailable."""

      OPERATION_TYPE_UNSPECIFIED = 0
      OPERATION_TYPE_SAVE = 1
      OPERATION_TYPE_PARTIAL_SAVE = 2
      OPERATION_TYPE_LOAD = 3

      @classmethod
      def Name(cls, value: int) -> str:  # pylint: disable=invalid-name
        return cls(value).name


_DEFAULT_PROMETHEUS_PORT = 9431


class MetricRecorder(abc.ABC):
  """Abstract base class for Orbax metric recorders."""

  @abc.abstractmethod
  def record_event(self, metric_name: str, **kwargs: Any) -> None:
    """Records a named event with optional metadata."""
    pass

  @abc.abstractmethod
  def record_scalar(
      self, metric_name: str, value: float | int, **kwargs: Any
  ) -> None:
    """Records a scalar summary value with optional metadata."""
    pass

  @abc.abstractmethod
  def record_duration(
      self, metric_name: str, duration: float | int, **kwargs: Any
  ) -> None:
    """Records an event duration in seconds with optional metadata."""
    pass

  def record_operation(
      self,
      event: str,
      operation_id: str,
      operation_type: OperationType,
      timestamp: float,
      **kwargs: Any,
  ) -> None:
    """Records an operation lifecycle event ('begin' or 'end').

    Args:
      event: Lifecycle event ('begin' or 'end').
      operation_id: Operation identifier.
      operation_type: Protobuf OperationType enum value.
      timestamp: Event timestamp in epoch seconds.
      **kwargs: Optional metadata.
    """
    del event, operation_id, operation_type, timestamp, kwargs


_recorders: list[MetricRecorder] = []  # Protected by _init_lock
_initialized = False  # Protected by _init_lock
_init_lock = threading.Lock()


def initialize(recorder: MetricRecorder) -> None:
  """Registers a recorder and binds its methods to JAX monitoring listeners."""
  global _initialized

  with _init_lock:
    _recorders.append(recorder)

    if _initialized:
      return

    from jax import monitoring as jax_monitoring  # pylint: disable=g-import-not-at-top

    def _proxy_record_event(metric_name: str, **kwargs: Any) -> None:
      with _init_lock:
        recorders_snapshot = _recorders[:]
      for r in recorders_snapshot:
        r.record_event(metric_name, **kwargs)

    def _proxy_record_scalar(
        metric_name: str, value: float | int, **kwargs: Any
    ) -> None:
      with _init_lock:
        recorders_snapshot = _recorders[:]
      for r in recorders_snapshot:
        r.record_scalar(metric_name, value, **kwargs)

    def _proxy_record_duration(
        metric_name: str, duration: float | int, **kwargs: Any
    ) -> None:
      with _init_lock:
        recorders_snapshot = _recorders[:]
      for r in recorders_snapshot:
        r.record_duration(metric_name, duration, **kwargs)

    jax_monitoring.register_event_listener(_proxy_record_event)  # pyrefly: ignore[bad-argument-type]
    jax_monitoring.register_scalar_listener(_proxy_record_scalar)  # pyrefly: ignore[bad-argument-type]
    jax_monitoring.register_event_duration_secs_listener(_proxy_record_duration)  # pyrefly: ignore[bad-argument-type]

    _initialized = True
    logging.info('Installed JAX monitoring proxy listeners for Orbax.')


def unregister_recorders_by_type(recorder_type: type[MetricRecorder]) -> None:
  """Removes all registered recorders matching the given recorder type.

  Args:
    recorder_type: The MetricRecorder subclass to unregister.
  """
  global _recorders
  with _init_lock:
    _recorders = [r for r in _recorders if not isinstance(r, recorder_type)]


def record_operation(
    event: str,
    operation_id: str,
    operation_type: OperationType,
    timestamp: float | None = None,
    **kwargs: Any,
) -> None:
  """Records an operation lifecycle event ('begin' or 'end').

  Operation ids go only to recorders, never as jax.monitoring fields, to keep
  metric cardinality bounded.

  Args:
    event: Lifecycle event name, such as 'begin' or 'end'.
    operation_id: Unique identifier for the operation.
    operation_type: Protobuf OperationType enum value.
    timestamp: Optional event timestamp in seconds; defaults to current time if
      None.
    **kwargs: Additional metadata forwarded to registered recorders.
  """
  if timestamp is None:
    timestamp = time.time()
  with _init_lock:
    recorders_snapshot = _recorders[:]
  for recorder in recorders_snapshot:
    recorder.record_operation(
        event, operation_id, operation_type, timestamp, **kwargs
    )


def format_operation_id(
    operation_type: OperationType,
    step: int,
    job_id: int | str | None = None,
    sub_operation: str | None = None,
) -> str:
  """Returns an operation id shared across workers.

  Args:
    operation_type: Protobuf OperationType enum value.
    step: Checkpoint step number.
    job_id: Optional job UID prefix, or None to omit it.
    sub_operation: Optional id appended as `#<id>`, or None to omit it.

  Returns:
    Formatted operation ID string.
  """
  op_name = (
      OperationType.Name(operation_type).removeprefix('OPERATION_TYPE_').lower()
  )
  base_id = (
      f'{job_id}.{op_name}.{step}'
      if job_id is not None
      else f'{op_name}.{step}'
  )
  return f'{base_id}#{sub_operation}' if sub_operation else base_id


def initialize_from_env() -> None:
  """Initializes monitoring based on environment and build type."""

  enable_telemetry = os.environ.get(
      'ENABLE_ORBAX_PROMETHEUS_TELEMETRY', 'false'
  )
  if enable_telemetry.lower() == 'true':
    try:
      from orbax.checkpoint._src.logging import prometheus_monitoring  # pylint: disable=g-import-not-at-top
      import multiprocessing  # pylint: disable=g-import-not-at-top

      env_port = os.environ.get('ORBAX_PROMETHEUS_PORT')
      default_port = _DEFAULT_PROMETHEUS_PORT
      if env_port:
        try:
          default_port = int(env_port)
        except ValueError:
          logging.warning(
              'Invalid ORBAX_PROMETHEUS_PORT "%s". Falling back to default %d.',
              env_port,
              _DEFAULT_PROMETHEUS_PORT,
          )
      port = (
          default_port
          if multiprocessing.current_process().name == 'MainProcess'
          else 0
      )
      initialize(prometheus_monitoring.PrometheusMonitoring(port=port))
    except ImportError as e:
      logging.warning('Failed to import PrometheusMonitoring: %s', e)
