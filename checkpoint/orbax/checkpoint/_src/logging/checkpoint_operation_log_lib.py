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

"""Library for creating and populating CheckpointOperation protos."""

from __future__ import annotations

from collections.abc import Callable
import threading
import time
import traceback
from typing import Any

from absl import logging
from orbax.checkpoint._src.logging import monitoring
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.multihost import pathways
from orbax.checkpoint.proto import checkpoint_operation_pb2

from google.protobuf import timestamp_pb2

_MAX_BACKTRACE_FRAMES = 5
# Keeps serialized log record < 1 MiB even with backtrace info.
_MAX_SPANS = 1024
_NS_PER_SEC = 1_000_000_000
_ORBAX_METRIC_PREFIXES = ("/jax/orbax/", "/jax/checkpoint/")

# Legacy alerting aliases of /jax/orbax/ metrics (see event_tracking.py);
# skipped to avoid duplicate spans.
_LEGACY_ALIASES = frozenset({
    "/jax/checkpoint/write/async/blocking_duration_secs",
    "/jax/checkpoint/write/async/commit_duration_sec",
    "/jax/checkpoint/write/async/metadata_write_duration_secs",
    "/jax/checkpoint/write/async/ocdbt_merge_duration_secs",
    "/jax/checkpoint/write/async/thread_duration_sec",
    "/jax/checkpoint/write/async/total_duration_secs",
})

_LogProto = checkpoint_operation_pb2.CheckpointOperation


def _set_timestamp(
    timestamp_proto: timestamp_pb2.Timestamp, seconds: float
) -> None:
  """Populates a google.protobuf.Timestamp from epoch seconds."""
  timestamp_proto.FromNanoseconds(int(seconds * _NS_PER_SEC))


def _to_enum(enum_type: Any, prefix: str, name: str | int | None) -> Any:
  """Converts string name or integer to enum value, returning 0 on failure."""
  if isinstance(name, int):
    return name
  if not name:
    return 0
  clean_name = str(name).upper().removeprefix(f"{prefix}_")
  try:
    return enum_type.Value(f"{prefix}_{clean_name}")
  except ValueError:
    return 0


def _resolve_status(
    status: str | int | None, error: BaseException | str | None
) -> _LogProto.Status:
  """Returns the status enum; defaults to FAILURE on error, else SUCCESS."""
  default = "FAILURE" if error else "SUCCESS"
  resolved_status = default if status is None else status
  return _to_enum(_LogProto.Status, "STATUS", resolved_status)


def _detect_runtime() -> tuple[_LogProto.Runtime, bool]:
  """Returns (Runtime enum, is_controller) for the active JAX runtime."""
  try:
    if multihost.is_pathways_colocated_runtime_active():
      return _LogProto.Runtime.RUNTIME_PATHWAYS, False
    if multihost.is_pathways_backend():
      return _LogProto.Runtime.RUNTIME_PATHWAYS, True
    return _LogProto.Runtime.RUNTIME_MCJAX, False
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.warning("Failed to detect active runtime: %s", e)
    return _LogProto.Runtime.RUNTIME_UNSPECIFIED, False


def _process_count(is_controller: bool) -> int:
  """Returns the process count, or the Pathways worker count if controller."""
  # The Pathways controller is a single process (jax.process_count() == 1),
  # so report the number of Pathways workers instead.
  if is_controller:
    try:
      return pathways.worker_count(None)
    except Exception:  # pylint: disable=broad-exception-caught
      logging.warning("Failed to get Pathways worker count.", exc_info=True)
  return multihost.process_count()


def create_error_info(
    error: BaseException | str | None,
) -> _LogProto.ErrorInfo | None:
  """Creates ErrorInfo for an exception or error message, or None if empty.

  Args:
    error: Exception instance or error message string, or None.

  Returns:
    Populated ErrorInfo proto, or None if error is empty.
  """
  if not error:
    return None
  info = _LogProto.ErrorInfo()
  if isinstance(error, BaseException):
    info.error_type = type(error).__name__
    info.error_message = str(error)
    if error.__traceback__ is not None:
      frames = traceback.extract_tb(error.__traceback__)[
          -_MAX_BACKTRACE_FRAMES:
      ]
      for frame in frames:
        info.stack_frames.add(
            file_name=frame.filename,
            line_number=frame.lineno or 0,
            function_name=frame.name,
        )
  else:
    info.error_message = str(error)
  return info


def create_operation_log(
    operation_id: str,
    operation_type: str | int,
    start_time: float,
    *,
    step: int | None = None,
    directory: str | None = None,
    job_metadata: _LogProto.JobMetadata | None = None,
) -> _LogProto:
  """Creates and initializes a CheckpointOperation proto.

  Args:
    operation_id: Operation identifier shared across workers.
    operation_type: Operation type ('save', 'partial_save', 'load', or enum).
    start_time: Operation start timestamp in epoch seconds.
    step: Checkpoint step number, or None if unavailable or step-less.
    directory: Checkpoint directory path, or None to leave unset.
    job_metadata: JobMetadata message to attach, or None to omit it.

  Returns:
    Initialized CheckpointOperation with STATUS_IN_PROGRESS.
  """
  runtime, is_controller = _detect_runtime()
  proto = _LogProto(
      operation_id=operation_id,
      operation_type=_to_enum(
          _LogProto.OperationType, "OPERATION_TYPE", operation_type
      ),
      runtime_metadata=_LogProto.RuntimeMetadata(
          runtime=runtime,
          process_index=multihost.process_index(),
          process_count=_process_count(is_controller),
          is_controller=is_controller,
      ),
      status=_LogProto.Status.STATUS_IN_PROGRESS,
  )
  _set_timestamp(proto.start_time, start_time)
  if step is not None:
    proto.step = int(step)
  if directory is not None:
    proto.storage_metadata.directory = str(directory)
  if job_metadata is not None:
    proto.job_metadata.CopyFrom(job_metadata)
  return proto


def add_span(
    log_proto: _LogProto,
    metric_name: str,
    start_time: float,
    end_time: float,
    *,
    status: str | int | None = None,
    error: BaseException | str | None = None,
    span_id: int | None = None,
    parent_span_id: int | None = None,
    **unused_kwargs: Any,
) -> _LogProto.Span | None:
  """Appends an execution span to `log_proto` if eligible.

  Args:
    log_proto: Target CheckpointOperation message.
    metric_name: Name of the execution span metric.
    start_time: Start timestamp of the span in epoch seconds.
    end_time: End timestamp of the span in epoch seconds.
    status: Span status name or enum value, or None to default to SUCCESS (or
      FAILURE if error is set).
    error: Exception or error message string, or None when no error occurred.
    span_id: Optional span ID, or None to leave unset.
    parent_span_id: Optional parent span ID, or None to leave unset.
    **unused_kwargs: Extra forwarding metadata (ignored).

  Returns:
    The appended Span message, or None if filtered out or span cap reached.
  """
  if (
      not metric_name.startswith(_ORBAX_METRIC_PREFIXES)
      or metric_name in _LEGACY_ALIASES
      or len(log_proto.spans) >= _MAX_SPANS
      or start_time > end_time
  ):
    return None
  span = log_proto.spans.add(
      name=metric_name,
      thread_name=threading.current_thread().name,
      thread_id=threading.get_native_id(),
      status=_resolve_status(status, error),
  )
  _set_timestamp(span.start_time, start_time)
  _set_timestamp(span.end_time, end_time)
  if span_id is not None:
    span.span_id = int(span_id)
  if parent_span_id is not None:
    span.parent_span_id = int(parent_span_id)
  if (err := create_error_info(error)) is not None:
    span.error_info.CopyFrom(err)
  return span


def finalize_operation_log(
    log_proto: _LogProto,
    end_time: float,
    *,
    status: str | int | None = None,
    error: BaseException | str | None = None,
) -> _LogProto:
  """Populates completion fields on `log_proto`.

  Args:
    log_proto: Active CheckpointOperation message to finalize.
    end_time: Operation end timestamp in epoch seconds.
    status: Completion status name or enum value, or None to default to SUCCESS
      (or FAILURE if error is set).
    error: Exception or error message string, or None when no error occurred.

  Returns:
    The finalized `log_proto`.
  """
  _set_timestamp(log_proto.end_time, end_time)
  log_proto.status = _resolve_status(status, error)
  if (err := create_error_info(error)) is not None:
    log_proto.error_info.CopyFrom(err)
  return log_proto


class OperationLogRecorder(monitoring.MetricRecorder):
  """MetricRecorder that assembles CheckpointOperation protos."""

  def __init__(
      self,
      sink: Callable[[_LogProto], Any] | None = None,
      job_metadata: _LogProto.JobMetadata | None = None,
  ):
    """Initializes the instance.

    Args:
      sink: Callback invoked with completed operation log proto, or None to drop
        completed logs.
      job_metadata: JobMetadata attached to each operation log, or None to omit
        it.
    """
    self._lock = threading.Lock()
    self._sink = sink
    self._job_metadata = job_metadata
    self._operations: dict[str, _LogProto] = {}

  def record_event(self, metric_name: str, **kwargs: Any) -> None:
    """Ignored: operation log records spans only."""
    del metric_name, kwargs

  def record_scalar(
      self, metric_name: str, value: float | int, **kwargs: Any
  ) -> None:
    """Ignored: operation log records spans only."""
    del metric_name, value, kwargs

  def record_duration(
      self, metric_name: str, duration: float | int, **kwargs: Any
  ) -> None:
    """Records a duration metric as a span within the active operation.

    Args:
      metric_name: Name of the duration metric.
      duration: Duration in seconds.
      **kwargs: Additional metadata forwarded to record_span.
    """
    now = time.time()
    self.record_span(metric_name, now - float(duration), now, **kwargs)

  def record_span(
      self,
      metric_name: str,
      start_time: float,
      end_time: float,
      **kwargs: Any,
  ) -> None:
    """Adds a span to an open operation.

    Args:
      metric_name: Name of the execution span metric.
      start_time: Start timestamp of the span in seconds.
      end_time: End timestamp of the span in seconds.
      **kwargs: Accepted kwargs include 'operation_id', 'status', 'error',
        'span_id', and 'parent_span_id'.
    """
    with self._lock:
      # Fall back to the latest open op, not a per-thread one: async saves
      # record spans from background threads.
      op_id = kwargs.get("operation_id") or next(
          reversed(self._operations), None
      )
      op_proto = self._operations.get(op_id) if op_id else None
      if op_proto is None:
        return
      add_span(op_proto, metric_name, start_time, end_time, **kwargs)

  def record_operation(
      self,
      event: str,
      operation_id: str,
      operation_type: monitoring.OperationType,
      timestamp: float,
      **kwargs: Any,
  ) -> None:
    """Records an operation begin or end event.

    Args:
      event: Lifecycle event ('begin' or 'end').
      operation_id: Operation identifier shared across workers.
      operation_type: Protobuf OperationType enum value.
      timestamp: Event timestamp in epoch seconds.
      **kwargs: For begin, accepted kwargs include 'step', 'directory'. For end,
        accepted kwargs include 'status', 'error'.
    """
    if event == "begin":
      proto = create_operation_log(
          operation_id,
          operation_type,
          timestamp,
          step=kwargs.get("step"),
          directory=kwargs.get("directory"),
          job_metadata=self._job_metadata,
      )
      with self._lock:
        self._operations.pop(operation_id, None)
        self._operations[operation_id] = proto
    elif event == "end":
      with self._lock:
        proto = self._operations.pop(operation_id, None)
      if proto is None:
        return
      finalize_operation_log(
          proto,
          timestamp,
          status=kwargs.get("status"),
          error=kwargs.get("error"),
      )
      self._emit(proto)

  def _emit(self, proto: _LogProto) -> None:
    """Invokes the configured sink callback with the completed proto."""
    if self._sink is not None:
      try:
        self._sink(proto)
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.warning("Operation log sink failed: %s", e)
