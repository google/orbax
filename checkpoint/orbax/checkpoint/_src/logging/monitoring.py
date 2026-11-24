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
import asyncio
from collections.abc import Callable, Iterator, Mapping
import concurrent.futures
import contextlib
import enum
import logging
import os
import threading
import time
from typing import Any

from orbax.checkpoint.proto import checkpoint_operation_pb2

OperationType = checkpoint_operation_pb2.CheckpointOperation.OperationType
OperationStatus = checkpoint_operation_pb2.CheckpointOperation.Status


class OperationEvent(enum.Enum):
  """Lifecycle event for a checkpoint operation."""

  BEGIN = 'begin'
  END = 'end'


def status_from_error(error: BaseException | str | None) -> OperationStatus:
  """Returns the failure status corresponding to an error.

  Args:
    error: Exception, error string, or None.

  Returns:
    STATUS_CANCELLED for CancelledError, STATUS_TIMED_OUT for TimeoutError,
    otherwise STATUS_FAILURE.
  """
  if isinstance(
      error, (asyncio.CancelledError, concurrent.futures.CancelledError)
  ):
    return OperationStatus.STATUS_CANCELLED
  if isinstance(error, TimeoutError):
    return OperationStatus.STATUS_TIMED_OUT
  return OperationStatus.STATUS_FAILURE


@contextlib.contextmanager
def _span_impl(
    record_fn: Callable[..., None],
    metric_name: str,
    kwargs: Mapping[str, Any],
) -> Iterator[None]:
  """Records a timed span around a block, isolating recording errors."""
  span_kwargs = {
      k: v for k, v in kwargs.items() if k not in ('status', 'error')
  }
  start_time = time.time()
  outcome: dict[str, Any] = {'status': OperationStatus.STATUS_SUCCESS}
  try:
    yield
  except BaseException as exc:
    outcome['status'] = status_from_error(exc)
    outcome['error'] = exc
    raise
  finally:
    try:
      record_fn(
          metric_name,
          start_time,
          time.time(),
          **outcome,
          **span_kwargs,
      )
    except Exception:  # pylint: disable=broad-exception-caught
      logging.warning('Failed to record span %s.', metric_name, exc_info=True)


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

  def record_span(
      self,
      metric_name: str,
      start_time: float,
      end_time: float,
      **kwargs: Any,
  ) -> None:
    """Records a timed execution span.

    By default, forwards clean non-negative spans to `record_duration` without
    span-specific metadata so label cardinality stays bounded.

    Args:
      metric_name: Name of the execution span metric.
      start_time: Start timestamp in epoch seconds.
      end_time: End timestamp in epoch seconds.
      **kwargs: Optional span metadata (e.g. `operation_id`, `status`, `error`,
        `span_id`, `parent_span_id`).
    """
    if kwargs.get('error') is None and end_time >= start_time:
      self.record_duration(metric_name, end_time - start_time)

  @contextlib.contextmanager
  def span(self, metric_name: str, **kwargs: Any) -> Iterator[None]:
    """Context manager and decorator that records an execution span.

    Args:
      metric_name: Name of the execution span metric.
      **kwargs: Additional metadata forwarded to `record_span`.

    Returns:
      Context manager that records the span duration and outcome on exit.
    """
    with _span_impl(self.record_span, metric_name, kwargs):
      yield

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
    try:
      recorder.record_operation(
          event, operation_id, operation_type, event_timestamp, **kwargs
      )
    except Exception:  # pylint: disable=broad-exception-caught
      logging.warning(
          'Recorder %s failed to record operation %s (%s).',
          type(recorder).__name__,
          operation_id,
          event,
          exc_info=True,
      )


def record_span(
    metric_name: str,
    start_time: float,
    end_time: float,
    **kwargs: Any,
) -> None:
  """Forwards an execution span to registered recorders.

  Args:
    metric_name: Name of the execution span metric.
    start_time: Start timestamp in epoch seconds.
    end_time: End timestamp in epoch seconds.
    **kwargs: Additional metadata forwarded to registered recorders.
  """
  with _init_lock:
    recorders_snapshot = _recorders[:]
  for recorder in recorders_snapshot:
    try:
      recorder.record_span(metric_name, start_time, end_time, **kwargs)
    except Exception:  # pylint: disable=broad-exception-caught
      logging.warning(
          'Recorder %s failed to record span %s.',
          type(recorder).__name__,
          metric_name,
          exc_info=True,
      )


@contextlib.contextmanager
def span(metric_name: str, **kwargs: Any) -> Iterator[None]:
  """Context manager and decorator that records a span across recorders.

  Args:
    metric_name: Name of the execution span metric.
    **kwargs: Additional metadata forwarded to `record_span`.

  Returns:
    Context manager that records the span duration and outcome on exit.
  """
  with _span_impl(record_span, metric_name, kwargs):
    yield


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
