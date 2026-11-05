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

"""Tests for CheckpointOperation recording."""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from typing import Any, cast
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from jax import monitoring as jax_monitoring
from orbax.checkpoint._src.logging import checkpoint_operation_log
from orbax.checkpoint._src.logging import monitoring
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.multihost import pathways
from orbax.checkpoint.proto import checkpoint_operation_pb2

_LogProto = checkpoint_operation_pb2.CheckpointOperation
_OP = monitoring.OperationType
_RT = _LogProto.Runtime
_STATUS = monitoring.OperationStatus
_STORAGE = _LogProto.StorageType
_LOCALITY = _LogProto.StorageLocality
_SAVE = _OP.OPERATION_TYPE_SAVE
_BEGIN = monitoring.OperationEvent.BEGIN
_END = monitoring.OperationEvent.END


def _raise_deep_os_error(depth: int = 7) -> None:
  if depth <= 1:
    raise OSError("write err")
  _raise_deep_os_error(depth - 1)


class _FailingOperationRecorder(
    checkpoint_operation_log.OperationMetricRecorder
):

  def record_operation(
      self,
      event: monitoring.OperationEvent,
      operation_id: str,
      operation_type: monitoring.OperationType,
      timestamp: float,
      **kwargs: Any,
  ) -> None:
    del event, operation_id, operation_type, timestamp, kwargs
    raise RuntimeError("recorder boom")

  def record_span(
      self, metric_name: str, start_time: float, end_time: float, **kwargs: Any
  ) -> None:
    del metric_name, start_time, end_time, kwargs
    raise RuntimeError("span boom")


class _DurationRecordingRecorder(monitoring.MetricRecorder):

  def __init__(self):
    self.durations: list[tuple[str, float | int, dict[str, Any]]] = []

  def record_event(self, metric_name: str, **kwargs: Any) -> None:
    del metric_name, kwargs

  def record_scalar(
      self, metric_name: str, value: float | int, **kwargs: Any
  ) -> None:
    del metric_name, value, kwargs

  def record_duration(
      self, metric_name: str, duration: float | int, **kwargs: Any
  ) -> None:
    self.durations.append((metric_name, duration, kwargs))


class OperationMetricRecorderTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.records: list[_LogProto] = []
    self.recorder = checkpoint_operation_log.OperationMetricRecorder(
        sink=self.records.append
    )

  def test_lifecycle_and_filtering(self):
    recorder = checkpoint_operation_log.OperationMetricRecorder(
        sink=self.records.append, job_metadata=_LogProto.JobMetadata()
    )
    span_name = "/jax/orbax/write/blocking_d2h_duration_secs"
    recorder.record_span(span_name, 99.0, 99.5)  # No open op -> ignored.
    recorder.record_operation(
        _BEGIN, "op-lc", _SAVE, 100.0, step=10, directory="/tmp/ckpt"
    )
    recorder.record_span(span_name, 100.1, 101.1, span_id=7, parent_span_id=3)
    recorder.record_span("some_op", 101.2, 101.5)
    recorder.record_span("", 100.0, 102.0)
    recorder.record_span("/jax/orbax/inverted", 100.0, 99.0)
    for ignored_duration in (
        "/jax/compile_duration",
        "/jax/orbax/write/blocking_d2h_duration_secs",
        "/jax/checkpoint/write/duration_since_last_checkpoint_secs",
        "/jax/checkpoint/write/async/blocking_duration_secs",
    ):
      recorder.record_duration(ignored_duration, 1.0)
    recorder.record_operation(
        _END, "op-lc", _SAVE, 105.0, status=_STATUS.STATUS_SUCCESS
    )

    self.assertLen(self.records, 1)
    record = self.records[0]
    self.assertEqual(
        (
            record.operation_id,
            record.operation_type,
            record.step,
            record.status,
        ),
        ("op-lc", _OP.OPERATION_TYPE_SAVE, 10, _STATUS.STATUS_SUCCESS),
    )
    self.assertEqual(
        (
            record.storage_metadata.directory,
            record.HasField("job_metadata"),
            record.start_time.ToNanoseconds(),
            record.end_time.ToNanoseconds(),
        ),
        ("/tmp/ckpt", True, 100_000_000_000, 105_000_000_000),
    )
    self.assertLen(record.spans, 2)
    span = record.spans[0]
    self.assertEqual(
        (span.name, span.thread_id, span.span_id, span.parent_span_id),
        (span_name, threading.get_native_id(), 7, 3),
    )
    self.assertEqual(
        (span.start_time.ToNanoseconds(), span.end_time.ToNanoseconds()),
        (100_100_000_000, 101_100_000_000),
    )
    self.assertEqual(record.spans[1].name, "some_op")

  @parameterized.parameters(
      (True, False, None, None, None, None, (_RT.RUNTIME_PATHWAYS, True, 2, 8)),
      (True, True, None, None, None, None, (_RT.RUNTIME_PATHWAYS, False, 2, 4)),
      (
          False,
          True,
          None,
          None,
          None,
          None,
          (_RT.RUNTIME_PATHWAYS, False, 2, 4),
      ),
      (False, False, None, None, None, None, (_RT.RUNTIME_MCJAX, False, 2, 4)),
      (
          False,
          False,
          OSError("x"),
          None,
          None,
          None,
          (_RT.RUNTIME_UNSPECIFIED, False, 2, 4),
      ),
      (
          True,
          False,
          None,
          OSError("x"),
          None,
          None,
          (_RT.RUNTIME_PATHWAYS, True, 2, 4),
      ),
      (
          False,
          False,
          None,
          None,
          RuntimeError("idx"),
          None,
          (_RT.RUNTIME_MCJAX, False, 0, 4),
      ),
      (
          False,
          False,
          None,
          None,
          None,
          RuntimeError("cnt"),
          (_RT.RUNTIME_MCJAX, False, 2, 0),
      ),
  )
  def test_runtime_detection(
      self,
      is_pathways,
      is_colocated,
      runtime_err,
      worker_count_err,
      index_err,
      count_err,
      expected,
  ):
    with (
        mock.patch.object(
            multihost,
            "is_pathways_backend",
            autospec=True,
            return_value=is_pathways,
        ),
        mock.patch.object(
            multihost,
            "is_pathways_colocated_runtime_active",
            autospec=True,
            side_effect=runtime_err,
            return_value=is_colocated,
        ),
        mock.patch.object(
            multihost,
            "process_index",
            autospec=True,
            side_effect=index_err,
            return_value=2,
        ),
        mock.patch.object(
            multihost,
            "process_count",
            autospec=True,
            side_effect=count_err,
            return_value=4,
        ),
        mock.patch.object(
            pathways,
            "worker_count",
            autospec=True,
            side_effect=worker_count_err,
            return_value=8,
        ),
    ):
      self.recorder.record_operation(_BEGIN, "op-rt", _SAVE, 100.0)
      self.recorder.record_operation(_END, "op-rt", _SAVE, 105.0)
    metadata = self.records[-1].runtime_metadata
    self.assertEqual(
        (
            metadata.runtime,
            metadata.is_controller,
            metadata.process_index,
            metadata.process_count,
        ),
        expected,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="partial_save_deep_exception",
          operation_type=_OP.OPERATION_TYPE_PARTIAL_SAVE,
          status=None,
          error_mode="deep_exc",
          expected=(_OP.OPERATION_TYPE_PARTIAL_SAVE, _STATUS.STATUS_FAILURE),
          expected_error=("OSError", "write err", 5),
      ),
      dict(
          testcase_name="timeout_exception_inferred",
          operation_type=_SAVE,
          status=None,
          error_mode="timeout",
          expected=(_OP.OPERATION_TYPE_SAVE, _STATUS.STATUS_TIMED_OUT),
          expected_error=("TimeoutError", "slow", 0),
      ),
      dict(
          testcase_name="explicit_status_and_message_error",
          operation_type=_OP.OPERATION_TYPE_LOAD,
          status=_STATUS.STATUS_TIMED_OUT,
          error_mode="str",
          expected=(_OP.OPERATION_TYPE_LOAD, _STATUS.STATUS_TIMED_OUT),
          expected_error=("", "write err", 0),
      ),
      dict(
          testcase_name="default_success",
          operation_type=_SAVE,
          status=None,
          error_mode=None,
          expected=(_OP.OPERATION_TYPE_SAVE, _STATUS.STATUS_SUCCESS),
          expected_error=("", "", 0),
      ),
  )
  def test_status_and_error_resolution(
      self, operation_type, status, error_mode, expected, expected_error
  ):
    error: BaseException | str | None = None
    if error_mode == "deep_exc":
      try:
        _raise_deep_os_error()
      except OSError as exc:
        error = exc
    elif error_mode == "timeout":
      error = TimeoutError("slow")
    elif error_mode == "str":
      error = "write err"
    self.recorder.record_operation(_BEGIN, "op-e", operation_type, 10.0)
    self.recorder.record_span(
        "/jax/orbax/span", 10.1, 10.2, status=status, error=error
    )
    self.recorder.record_operation(
        _END, "op-e", operation_type, 12.0, status=status, error=error
    )
    record = self.records[-1]
    span = record.spans[0]
    self.assertEqual((record.operation_type, record.status), expected)
    self.assertEqual(span.status, expected[1])
    self.assertEqual(
        (
            record.error_info.error_type,
            record.error_info.error_message,
            len(record.error_info.stack_frames),
        ),
        expected_error,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="gcs_same_cluster",
          storage_type="gcs",
          locality="same_cluster",
          expected=(
              _STORAGE.STORAGE_TYPE_GCS,
              _LOCALITY.STORAGE_LOCALITY_SAME_CLUSTER,
          ),
      ),
      dict(
          testcase_name="other_same_metro",
          storage_type="other",
          locality="same_metro",
          expected=(
              _STORAGE.STORAGE_TYPE_OTHER,
              _LOCALITY.STORAGE_LOCALITY_SAME_METRO,
          ),
      ),
      dict(
          testcase_name="different_metro",
          storage_type=None,
          locality="different_metro",
          expected=(
              _STORAGE.STORAGE_TYPE_UNSPECIFIED,
              _LOCALITY.STORAGE_LOCALITY_DIFFERENT_METRO,
          ),
      ),
      dict(
          testcase_name="different_continent",
          storage_type="unknown_fs",
          locality="different_continent",
          expected=(
              _STORAGE.STORAGE_TYPE_UNSPECIFIED,
              _LOCALITY.STORAGE_LOCALITY_DIFFERENT_CONTINENT,
          ),
      ),
      dict(
          testcase_name="unknown_locality",
          storage_type=None,
          locality="unknown",
          expected=(
              _STORAGE.STORAGE_TYPE_UNSPECIFIED,
              _LOCALITY.STORAGE_LOCALITY_UNSPECIFIED,
          ),
      ),
  )
  def test_storage_metadata_mapping(self, storage_type, locality, expected):
    kwargs: dict[str, Any] = {}
    if storage_type is not None:
      kwargs["storage_type"] = storage_type
    if locality is not None:
      kwargs["locality"] = locality
    self.recorder.record_operation(_BEGIN, "op-st", _SAVE, 10.0, **kwargs)
    self.recorder.record_operation(_END, "op-st", _SAVE, 11.0)
    meta = self.records[-1].storage_metadata
    self.assertEqual((meta.type, meta.locality), expected)

  def test_routing_edge_cases_span_cap_and_sink_errors(self):
    self.recorder.record_operation(_BEGIN, "op-1", _SAVE, 100.0)
    self.recorder.record_operation(_BEGIN, "op-2", _SAVE, 101.0, step=1)
    with self.assertLogs(level="WARNING") as logs:
      self.recorder.record_operation(_BEGIN, "op-2", _SAVE, 101.5, step=2)
      self.recorder.record_operation(_END, "never-begun", _SAVE, 101.6)
      self.recorder.record_operation(cast(Any, "bogus"), "op-2", _SAVE, 101.6)
    log_text = "\n".join(logs.output)
    self.assertIn("Duplicate BEGIN", log_text)
    self.assertIn("unknown or already-ended", log_text)
    self.assertIn("unknown operation event", log_text)
    # With 2 open ops, un-keyed span is skipped; explicit operation_id routes.
    self.recorder.record_span("/jax/orbax/ambiguous", 101.6, 101.7)
    self.recorder.record_span(
        "/jax/orbax/keyed", 101.6, 101.8, operation_id="op-1"
    )
    self.recorder.record_operation(_END, "op-1", _SAVE, 102.0)
    # Now only op-2 is open; un-keyed spans attach up to _MAX_SPANS (1024).
    for _ in range(1030):
      self.recorder.record_span("/jax/orbax/s", 101.6, 101.7)
    self.recorder.record_operation(_END, "op-2", _SAVE, 103.0)
    with self.assertLogs(level="WARNING") as dup_end_logs:
      self.recorder.record_operation(_END, "op-2", _SAVE, 104.0)
    self.assertIn("unknown or already-ended", "\n".join(dup_end_logs.output))
    self.assertLen(self.records, 2)
    self.assertEqual(
        (
            len(self.records[0].spans),
            self.records[1].step,
            self.records[1].start_time.ToNanoseconds(),
            len(self.records[1].spans),
        ),
        (1, 1, 101_000_000_000, 1024),
    )

    # Verify sink=None no-op and sink exception handling.
    noop_recorder = checkpoint_operation_log.OperationMetricRecorder(sink=None)
    noop_recorder.record_operation(_BEGIN, "op-n", _SAVE, 1.0)
    noop_recorder.record_operation(_END, "op-n", _SAVE, 2.0)

    def _bad_sink(unused_proto: _LogProto) -> None:
      raise RuntimeError("sink boom")

    bad_recorder = checkpoint_operation_log.OperationMetricRecorder(
        sink=_bad_sink
    )
    bad_recorder.record_operation(_BEGIN, "op-b", _SAVE, 1.0)
    with self.assertLogs(level="WARNING") as sink_logs:
      bad_recorder.record_operation(_END, "op-b", _SAVE, 2.0)
    self.assertIn("Operation log sink failed", "\n".join(sink_logs.output))

  def test_internal_exception_isolation(self):
    with (
        mock.patch.object(
            self.recorder,
            "_record_operation",
            autospec=True,
            side_effect=RuntimeError("op internal err"),
        ),
        mock.patch.object(
            self.recorder,
            "_record_span",
            autospec=True,
            side_effect=RuntimeError("span internal err"),
        ),
        self.assertLogs(level="WARNING") as logs,
    ):
      self.recorder.record_operation(_BEGIN, "op-x", _SAVE, 1.0)
      self.recorder.record_span("some_span", 1.0, 2.0)
    log_text = "\n".join(logs.output)
    self.assertIn("Failed to record operation 'op-x'", log_text)
    self.assertIn("Failed to record span 'some_span'", log_text)

  def test_monitoring_integration_and_span_api(self):
    self.enter_context(mock.patch.object(monitoring, "_recorders", []))
    monitoring.initialize(self.recorder)
    span_name = "/jax/orbax/write/background_ts_duration_secs"

    @monitoring.span("decorated_mod_span", span_id=2)
    def _mod_work() -> str:
      return "ok"

    @self.recorder.span("decorated_rec_span")
    def _rec_work() -> str:
      return "ok"

    def _fail_span(exc: BaseException) -> None:
      with monitoring.span("failing_span"):
        raise exc

    def _fail_rec_span(exc: BaseException) -> None:
      with self.recorder.span("failing_rec_span"):
        raise exc

    monitoring.record_operation(_BEGIN, "ctx-ok", _SAVE, step=1, directory="/d")
    jax_monitoring.record_event("/jax/orbax/write/start")
    jax_monitoring.record_scalar("/jax/orbax/write/gbytes", 1.0)
    jax_monitoring.record_event_duration_secs(span_name, 2.5)
    jax_monitoring.record_event_duration_secs(
        "/jax/orbax/write/blocking_d2h_duration_secs", 0.5
    )
    with monitoring.span("some_op", span_id=1):
      pass
    self.assertEqual(_mod_work(), "ok")
    with self.recorder.span("rec_with_span"):
      pass
    self.assertEqual(_rec_work(), "ok")
    self.assertRaisesRegex(
        RuntimeError, "boom", _fail_span, RuntimeError("boom")
    )
    self.assertRaisesRegex(
        asyncio.CancelledError,
        "cancel",
        _fail_span,
        asyncio.CancelledError("cancel"),
    )
    self.assertRaisesRegex(
        TimeoutError, "slow", _fail_rec_span, TimeoutError("slow")
    )
    monitoring.record_operation(
        _END, "ctx-ok", _SAVE, status=_STATUS.STATUS_SUCCESS
    )

    self.assertLen(self.records, 1)
    ok_record = self.records[0]
    self.assertEqual(
        (ok_record.operation_id, ok_record.step, ok_record.status),
        ("ctx-ok", 1, _STATUS.STATUS_SUCCESS),
    )
    self.assertEqual(ok_record.storage_metadata.directory, "/d")
    self.assertFalse(ok_record.HasField("job_metadata"))
    self.assertLessEqual(
        ok_record.start_time.ToNanoseconds(), ok_record.end_time.ToNanoseconds()
    )
    self.assertEqual(
        [(s.name, s.status, s.error_info.error_type) for s in ok_record.spans],
        [
            (span_name, _STATUS.STATUS_SUCCESS, ""),
            ("some_op", _STATUS.STATUS_SUCCESS, ""),
            ("decorated_mod_span", _STATUS.STATUS_SUCCESS, ""),
            ("rec_with_span", _STATUS.STATUS_SUCCESS, ""),
            ("decorated_rec_span", _STATUS.STATUS_SUCCESS, ""),
            ("failing_span", _STATUS.STATUS_FAILURE, "RuntimeError"),
            ("failing_span", _STATUS.STATUS_CANCELLED, "CancelledError"),
            ("failing_rec_span", _STATUS.STATUS_TIMED_OUT, "TimeoutError"),
        ],
    )
    jax_span = ok_record.spans[0]
    self.assertFalse(jax_span.HasField("span_id"))
    self.assertEqual(
        jax_span.end_time.ToNanoseconds() - jax_span.start_time.ToNanoseconds(),
        2_500_000_000,
    )
    self.assertEqual(
        (ok_record.spans[1].span_id, ok_record.spans[2].span_id), (1, 2)
    )

  def test_default_metric_recorder_record_span(self):
    recorder = _DurationRecordingRecorder()
    recorder.record_span(
        "clean_span",
        10.0,
        12.5,
        operation_id="op-1",
        span_id=5,
        status=_STATUS.STATUS_SUCCESS,
    )
    recorder.record_span("inverted", 12.0, 10.0)
    recorder.record_span("errored", 10.0, 11.0, error=RuntimeError("bad"))
    with recorder.span("ctx_span", operation_id="op-1"):
      pass

    def _raise_in_span() -> None:
      with recorder.span("ctx_err"):
        raise ValueError("fail")

    self.assertRaisesRegex(ValueError, "fail", _raise_in_span)
    self.assertLen(recorder.durations, 2)
    self.assertEqual(recorder.durations[0], ("clean_span", 2.5, {}))
    self.assertEqual(recorder.durations[1][0], "ctx_span")
    self.assertGreaterEqual(recorder.durations[1][1], 0.0)
    self.assertEqual(recorder.durations[1][2], {})

  def test_dispatch_isolates_failing_recorder(self):
    self.enter_context(mock.patch.object(monitoring, "_recorders", []))
    monitoring.initialize(_FailingOperationRecorder())
    monitoring.initialize(self.recorder)

    with self.assertLogs(level="WARNING") as logs:
      monitoring.record_operation(_BEGIN, "op-ok", _SAVE)
      with monitoring.span("isolated_span", status="ignored", error="ignored"):
        pass
      monitoring.record_operation(
          _END, "op-ok", _SAVE, status=_STATUS.STATUS_SUCCESS
      )

    log_text = "\n".join(logs.output)
    self.assertIn(
        "_FailingOperationRecorder failed to record operation op-ok", log_text
    )
    self.assertIn(
        "_FailingOperationRecorder failed to record span isolated_span",
        log_text,
    )
    self.assertLen(self.records, 1)
    self.assertEqual(
        (self.records[0].operation_id, self.records[0].status),
        ("op-ok", _STATUS.STATUS_SUCCESS),
    )
    self.assertEqual(
        [(s.name, s.status) for s in self.records[0].spans],
        [("isolated_span", _STATUS.STATUS_SUCCESS)],
    )

  def test_recorder_span_isolates_recording_errors_and_strips_collisions(self):
    failing = _FailingOperationRecorder()

    def _raise_in_span() -> None:
      with failing.span("failing_err_span", status="bad", error="bad"):
        raise ValueError("original error")

    with self.assertLogs(level="WARNING") as logs:
      with failing.span("failing_ok_span", status="bad", error="bad"):
        pass
      self.assertRaisesRegex(ValueError, "original error", _raise_in_span)
    log_text = "\n".join(logs.output)
    self.assertIn("Failed to record span failing_ok_span", log_text)
    self.assertIn("Failed to record span failing_err_span", log_text)

  @parameterized.named_parameters(
      dict(
          testcase_name="asyncio_cancelled_error",
          error=asyncio.CancelledError(),
          expected=_STATUS.STATUS_CANCELLED,
      ),
      dict(
          testcase_name="futures_cancelled_error",
          error=concurrent.futures.CancelledError(),
          expected=_STATUS.STATUS_CANCELLED,
      ),
      dict(
          testcase_name="timeout_error",
          error=TimeoutError("slow"),
          expected=_STATUS.STATUS_TIMED_OUT,
      ),
      dict(
          testcase_name="value_error",
          error=ValueError("bad"),
          expected=_STATUS.STATUS_FAILURE,
      ),
      dict(
          testcase_name="error_string",
          error="something failed",
          expected=_STATUS.STATUS_FAILURE,
      ),
      dict(
          testcase_name="none",
          error=None,
          expected=_STATUS.STATUS_FAILURE,
      ),
  )
  def test_status_from_error(self, error, expected):
    self.assertEqual(monitoring.status_from_error(error), expected)


if __name__ == "__main__":
  absltest.main()
