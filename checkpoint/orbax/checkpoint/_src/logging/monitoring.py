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
from collections.abc import Iterator
import contextlib
import enum
import logging
import os
import threading
import time
from typing import Any


class OperationEvent(enum.Enum):
  """Lifecycle event for a checkpoint operation."""

  BEGIN = 'begin'
  END = 'end'


class OperationType(enum.IntEnum):
  """Type of checkpoint operation."""

  OPERATION_TYPE_UNSPECIFIED = 0
  OPERATION_TYPE_SAVE = 1
  OPERATION_TYPE_PARTIAL_SAVE = 2
  OPERATION_TYPE_LOAD = 3


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
      event: OperationEvent,
      operation_id: str,
      operation_type: OperationType,
      timestamp: float,
      **kwargs: Any,
  ) -> None:
    """Records an operation lifecycle event (BEGIN or END)."""
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


def record_operation(
    event: OperationEvent,
    operation_id: str,
    operation_type: OperationType,
    timestamp: float | None = None,
    **kwargs: Any,
) -> None:
  """Forwards an operation BEGIN or END event to registered recorders.

  Prefer `track_operation()` for synchronous blocks. Use this two-call API when
  begin and end span threads/callbacks (e.g. async saves or Streamz
  `OperationRecorder` compatibility).

  Args:
    event: Lifecycle event (`OperationEvent.BEGIN` or `OperationEvent.END`).
    operation_id: Unique identifier for the operation.
    operation_type: Checkpoint OperationType enum value.
    timestamp: Event timestamp in seconds, or None for current time.
    **kwargs: Additional metadata forwarded to registered recorders.
  """
  # Kept out of jax.monitoring fields to keep metric cardinality bounded.
  event_timestamp = time.time() if timestamp is None else timestamp
  with _init_lock:
    recorders_snapshot = _recorders[:]
  for recorder in recorders_snapshot:
    recorder.record_operation(
        event, operation_id, operation_type, event_timestamp, **kwargs
    )


@contextlib.contextmanager
def track_operation(
    operation_id: str,
    operation_type: OperationType,
    **kwargs: Any,
) -> Iterator[None]:
  """Records operation BEGIN/END automatically around a synchronous block.

  Args:
    operation_id: Unique identifier for the operation.
    operation_type: Checkpoint OperationType enum value.
    **kwargs: Additional begin metadata (e.g. `step`, `directory`).

  Returns:
    Context manager that emits BEGIN on entry and END on exit with timestamps
    and exception status populated automatically.
  """
  record_operation(OperationEvent.BEGIN, operation_id, operation_type, **kwargs)
  try:
    yield
  # Catch BaseException so KeyboardInterrupt/SystemExit also emit END.
  except BaseException as exc:
    status = 'TIMED_OUT' if isinstance(exc, TimeoutError) else 'FAILURE'
    record_operation(
        OperationEvent.END,
        operation_id,
        operation_type,
        status=status,
        error=exc,
    )
    raise
  else:
    record_operation(
        OperationEvent.END, operation_id, operation_type, status='SUCCESS'
    )


def initialize_from_env() -> None:
  """Initializes monitoring based on environment variables."""

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
