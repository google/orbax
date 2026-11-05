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

"""CheckpointOperation telemetry log recorder for Orbax."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import threading
import time
import traceback
import types
from typing import Any, TypeVar

from absl import logging
from orbax.checkpoint._src.logging import monitoring
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.multihost import pathways
from orbax.checkpoint.proto import checkpoint_operation_pb2

_MAX_BACKTRACE_FRAMES = 5
_MAX_SPANS = 1024  # Keeps serialized log record < 1 MiB.
_NS_PER_SEC = 1_000_000_000
_T = TypeVar("_T")
# Verified real-time duration metrics emitted immediately at phase end, where
# [now - duration, now] accurately reflects the span window. Excludes deferred,
# summed, inter-checkpoint, and duplicate alias metrics.
# TODO(b/582858125): Add more spans (including cached metrics such as
# blocking_tree_map and blocking_d2h once instrumented with explicit start/end).
_ALLOWED_DURATION_SPANS = frozenset({
    "/jax/orbax/write/blocking_prepare_destination_secs",
    "/jax/orbax/write/blocking_directory_creation_secs",
    "/jax/orbax/write/finalize_duration_secs",
    "/jax/orbax/write/background_prepare_destination_secs",
    "/jax/orbax/write/background_directory_creation_secs",
    "/jax/orbax/write/background_ts_duration_secs",
    "/jax/orbax/write/background_meta_write_secs",
    "/jax/orbax/write/background_ocdbt_merge_secs",
    "/jax/orbax/write/background_item_finalize_secs",
    "/jax/orbax/write/background_finalize_secs",
    "/jax/orbax/write/background_duration_secs",
    "/jax/checkpoint/read/primary_replica_deserialization_duration_secs",
    "/jax/checkpoint/read/broadcast_duration_secs",
})
_LogProto = checkpoint_operation_pb2.CheckpointOperation
_PROTO_STORAGE_TYPES = types.MappingProxyType({
    "gcs": _LogProto.StorageType.STORAGE_TYPE_GCS,
    "other": _LogProto.StorageType.STORAGE_TYPE_OTHER,
})
_PROTO_STORAGE_LOCALITIES = types.MappingProxyType({
    "same_cluster": _LogProto.StorageLocality.STORAGE_LOCALITY_SAME_CLUSTER,
    "same_metro": _LogProto.StorageLocality.STORAGE_LOCALITY_SAME_METRO,
    "different_metro": (
        _LogProto.StorageLocality.STORAGE_LOCALITY_DIFFERENT_METRO
    ),
    "different_continent": (
        _LogProto.StorageLocality.STORAGE_LOCALITY_DIFFERENT_CONTINENT
    ),
})


def _safe_call(fn: Callable[[], _T], default: _T, description: str) -> _T:
  try:
    return fn()
  except Exception:  # pylint: disable=broad-exception-caught
    logging.warning("Failed to get %s.", description, exc_info=True)
    return default


def _detect_runtime() -> tuple[_LogProto.Runtime, bool]:
  if multihost.is_pathways_colocated_runtime_active():
    return _LogProto.Runtime.RUNTIME_PATHWAYS, False
  if multihost.is_pathways_backend():
    return _LogProto.Runtime.RUNTIME_PATHWAYS, True
  return _LogProto.Runtime.RUNTIME_MCJAX, False


def _get_runtime_metadata() -> _LogProto.RuntimeMetadata:
  """Returns RuntimeMetadata for the active JAX or Pathways runtime."""
  runtime, is_controller = _safe_call(
      _detect_runtime,
      (_LogProto.Runtime.RUNTIME_UNSPECIFIED, False),
      "active runtime",
  )
  process_index = _safe_call(multihost.process_index, 0, "process index")
  process_count = None
  if is_controller:
    # Pathways controller has jax.process_count() == 1; report worker count.
    process_count = _safe_call(
        lambda: pathways.worker_count(None), None, "Pathways worker count"
    )
  if not process_count:
    process_count = _safe_call(multihost.process_count, 0, "process count")
  return _LogProto.RuntimeMetadata(
      runtime=runtime,
      process_index=process_index,
      process_count=process_count,
      is_controller=is_controller,
  )


def _populate_storage_metadata(
    proto: _LogProto, kwargs: Mapping[str, Any]
) -> None:
  """Populates storage_metadata fields from optional BEGIN kwargs."""
  if (directory := kwargs.get("directory")) is not None:
    proto.storage_metadata.directory = str(directory)
  if (storage_type := kwargs.get("storage_type")) is not None:
    proto.storage_metadata.type = _PROTO_STORAGE_TYPES.get(
        str(storage_type).lower(),
        _LogProto.StorageType.STORAGE_TYPE_UNSPECIFIED,
    )
  if (locality := kwargs.get("locality")) is not None:
    proto.storage_metadata.locality = _PROTO_STORAGE_LOCALITIES.get(
        str(locality).lower(),
        _LogProto.StorageLocality.STORAGE_LOCALITY_UNSPECIFIED,
    )


def _set_completion(
    target: _LogProto | _LogProto.Span,
    end_time: float,
    status: monitoring.OperationStatus | None,
    error: BaseException | str | None,
) -> None:
  """Populates end_time, status, and error_info on an operation or span.

  Args:
    target: CheckpointOperation or Span message to update.
    end_time: Completion timestamp in epoch seconds.
    status: OperationStatus enum value, or None to default to
      `monitoring.status_from_error(error)` on error else STATUS_SUCCESS.
    error: Exception or error message string, or None when no error occurred.
  """
  target.end_time.FromNanoseconds(int(end_time * _NS_PER_SEC))
  if status is None:
    target.status = (
        monitoring.status_from_error(error)
        if error
        else _LogProto.Status.STATUS_SUCCESS
    )
  else:
    target.status = status
  if not error:
    return
  target.error_info.error_message = str(error)
  if isinstance(error, BaseException):
    target.error_info.error_type = type(error).__name__
    frames = traceback.extract_tb(error.__traceback__)[-_MAX_BACKTRACE_FRAMES:]
    for frame in frames:
      target.error_info.stack_frames.add(
          file_name=frame.filename,
          line_number=frame.lineno or 0,
          function_name=frame.name,
      )


class OperationMetricRecorder(monitoring.MetricRecorder):
  """MetricRecorder that assembles CheckpointOperation protos."""

  def __init__(
      self,
      sink: Callable[[_LogProto], Any] | None = None,
      job_metadata: _LogProto.JobMetadata | None = None,
  ):
    """Initializes the instance.

    Args:
      sink: Callback invoked with each completed proto, or None to drop logs.
      job_metadata: JobMetadata attached to each operation, or None to omit.
    """
    self._lock = threading.Lock()
    self._sink = sink
    self._job_metadata = job_metadata
    self._operations: dict[str, _LogProto] = {}

  def record_event(self, metric_name: str, **kwargs: Any) -> None:
    """Records discrete events (currently no-op)."""
    # TODO(b/582858125): Select useful Orbax events to log on the operation.
    del metric_name, kwargs

  def record_scalar(
      self, metric_name: str, value: float | int, **kwargs: Any
  ) -> None:
    """Records scalar metrics (currently no-op)."""
    # TODO(b/582858125): Record useful scalars (e.g., raw_bytes,
    # logical_bytes) onto the active CheckpointOperation.
    del metric_name, value, kwargs

  def record_duration(
      self, metric_name: str, duration: float | int, **kwargs: Any
  ) -> None:
    """Records an allowlisted real-time duration metric as a span."""
    if metric_name not in _ALLOWED_DURATION_SPANS:
      return
    now = time.time()
    self.record_span(metric_name, now - float(duration), now, **kwargs)

  def record_span(
      self, metric_name: str, start_time: float, end_time: float, **kwargs: Any
  ) -> None:
    """Adds an execution span to an open operation if eligible.

    Args:
      metric_name: Name of the execution span metric.
      start_time: Start timestamp of the span in epoch seconds.
      end_time: End timestamp of the span in epoch seconds.
      **kwargs: Optional 'operation_id', 'status', 'error', 'span_id', and
        'parent_span_id'.
    """
    try:
      self._record_span(metric_name, start_time, end_time, **kwargs)
    except Exception:  # pylint: disable=broad-exception-caught
      logging.warning("Failed to record span %r.", metric_name, exc_info=True)

  def _record_span(
      self, metric_name: str, start_time: float, end_time: float, **kwargs: Any
  ) -> None:
    if not metric_name or start_time > end_time:
      return
    with self._lock:
      # Fall back to the sole open op when operation_id is omitted; skip when
      # multiple ops overlap to avoid cross-operation span mis-attribution.
      # TODO(b/582858125): Pass explicit operation_id via an Orbax monitoring
      # wrapper to differentiate overlapping operations.
      op_id = kwargs.get("operation_id") or (
          next(iter(self._operations)) if len(self._operations) == 1 else ""
      )
      op_proto = self._operations.get(op_id)
      if op_proto is None or len(op_proto.spans) >= _MAX_SPANS:
        return
      span = op_proto.spans.add(
          name=metric_name,
          thread_name=threading.current_thread().name,
          thread_id=threading.get_native_id(),
          span_id=kwargs.get("span_id"),
          parent_span_id=kwargs.get("parent_span_id"),
      )
      span.start_time.FromNanoseconds(int(start_time * _NS_PER_SEC))
      _set_completion(span, end_time, kwargs.get("status"), kwargs.get("error"))

  def record_operation(
      self,
      event: monitoring.OperationEvent,
      operation_id: str,
      operation_type: monitoring.OperationType,
      timestamp: float,
      **kwargs: Any,
  ) -> None:
    """Records the start or end of a checkpoint save or load operation.

    On `OperationEvent.BEGIN`, creates an in-progress `CheckpointOperation`
    proto that subsequent `record_*` calls attach to. On `OperationEvent.END`,
    finalizes the proto and emits it to the configured sink via `_emit()`.

    Args:
      event: Lifecycle event (`OperationEvent.BEGIN` or `OperationEvent.END`).
      operation_id: Operation identifier shared across workers.
      operation_type: CheckpointOperation.OperationType enum value.
      timestamp: Event timestamp in epoch seconds.
      **kwargs: Optional 'step', 'directory', 'storage_type', and 'locality' for
        BEGIN; optional 'status' and 'error' for END.
    """
    try:
      self._record_operation(
          event, operation_id, operation_type, timestamp, **kwargs
      )
    except Exception:  # pylint: disable=broad-exception-caught
      logging.warning(
          "Failed to record operation %r (%s).",
          operation_id,
          event,
          exc_info=True,
      )

  def _record_operation(
      self,
      event: monitoring.OperationEvent,
      operation_id: str,
      operation_type: monitoring.OperationType,
      timestamp: float,
      **kwargs: Any,
  ) -> None:
    if event == monitoring.OperationEvent.BEGIN:
      proto = _LogProto(
          operation_id=operation_id,
          step=kwargs.get("step"),
          operation_type=operation_type,
          runtime_metadata=_get_runtime_metadata(),
          status=_LogProto.Status.STATUS_IN_PROGRESS,
          job_metadata=self._job_metadata,
      )
      proto.start_time.FromNanoseconds(int(timestamp * _NS_PER_SEC))
      _populate_storage_metadata(proto, kwargs)
      with self._lock:
        if operation_id in self._operations:
          logging.warning(
              "Duplicate BEGIN for operation_id=%r; ignoring duplicate.",
              operation_id,
          )
          return
        self._operations[operation_id] = proto
    elif event == monitoring.OperationEvent.END:
      with self._lock:
        proto = self._operations.pop(operation_id, None)
      if proto is None:
        logging.warning(
            "Ignoring END for unknown or already-ended operation_id=%r.",
            operation_id,
        )
        return
      status, error = kwargs.get("status"), kwargs.get("error")
      _set_completion(proto, timestamp, status, error)
      self._emit(proto)
    else:
      logging.warning(
          "Ignoring unknown operation event %r for operation_id=%r.",
          event,
          operation_id,
      )

  def _emit(self, proto: _LogProto) -> None:
    """Invokes the configured sink callback with the completed proto."""
    if self._sink is not None:
      try:
        self._sink(proto)
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.warning("Operation log sink failed: %s", e)
