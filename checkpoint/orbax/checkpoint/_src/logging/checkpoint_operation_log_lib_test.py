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

import threading
from typing import Any, cast
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from jax import monitoring as jax_monitoring
from orbax.checkpoint._src.logging import checkpoint_operation_log_lib
from orbax.checkpoint._src.logging import monitoring
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.multihost import pathways
from orbax.checkpoint.proto import checkpoint_operation_pb2

_LogProto = checkpoint_operation_pb2.CheckpointOperation
_OP = _LogProto.OperationType
_RT = _LogProto.Runtime
_STATUS = _LogProto.Status
_MOP = monitoring.OperationType
_SAVE = _MOP.OPERATION_TYPE_SAVE
_BEGIN = monitoring.OperationEvent.BEGIN
_END = monitoring.OperationEvent.END


def _raise_deep_os_error(depth: int = 7) -> None:
  if depth <= 1:
    raise OSError("write err")
  _raise_deep_os_error(depth - 1)


class OperationLogRecorderTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.records: list[_LogProto] = []
    self.recorder = checkpoint_operation_log_lib.OperationLogRecorder(
        sink=self.records.append
    )

  def test_lifecycle_and_filtering(self):
    recorder = checkpoint_operation_log_lib.OperationLogRecorder(
        sink=self.records.append, job_metadata=_LogProto.JobMetadata()
    )
    span_name = "/jax/orbax/write/blocking_d2h_duration_secs"
    recorder.record_span(span_name, 99.0, 99.5)  # No open op -> ignored.
    recorder.record_operation(
        _BEGIN, "op-lc", _SAVE, 100.0, step=10, directory="/tmp/ckpt"
    )
    recorder.record_span(span_name, 100.1, 101.1, span_id=7, parent_span_id=3)
    recorder.record_span("/jax/compile_duration", 100.0, 102.0)
    recorder.record_span("/jax/orbax/inverted", 100.0, 99.0)
    for ignored_duration in (
        "/jax/orbax/write/blocking_d2h_duration_secs",
        "/jax/checkpoint/write/duration_since_last_checkpoint_secs",
        "/jax/checkpoint/write/async/blocking_duration_secs",
    ):
      recorder.record_duration(ignored_duration, 1.0)
    recorder.record_operation(_END, "op-lc", _SAVE, 105.0, status="SUCCESS")

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
    self.assertLen(record.spans, 1)
    span = record.spans[0]
    self.assertEqual(
        (span.name, span.thread_id, span.span_id, span.parent_span_id),
        (span_name, threading.get_native_id(), 7, 3),
    )
    self.assertEqual(
        (span.start_time.ToNanoseconds(), span.end_time.ToNanoseconds()),
        (100_100_000_000, 101_100_000_000),
    )

  @parameterized.parameters(
      (True, False, None, None, (_RT.RUNTIME_PATHWAYS, True, 8)),
      (True, True, None, None, (_RT.RUNTIME_PATHWAYS, False, 4)),
      (False, True, None, None, (_RT.RUNTIME_PATHWAYS, False, 4)),
      (False, False, None, None, (_RT.RUNTIME_MCJAX, False, 4)),
      (False, False, OSError("x"), None, (_RT.RUNTIME_UNSPECIFIED, False, 4)),
      (True, False, None, OSError("x"), (_RT.RUNTIME_PATHWAYS, True, 4)),
  )
  def test_runtime_detection(
      self, is_pathways, is_colocated, runtime_err, count_err, expected
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
            multihost, "process_count", autospec=True, return_value=4
        ),
        mock.patch.object(
            pathways,
            "worker_count",
            autospec=True,
            side_effect=count_err,
            return_value=8,
        ),
    ):
      self.recorder.record_operation(_BEGIN, "op-rt", _SAVE, 100.0)
      self.recorder.record_operation(_END, "op-rt", _SAVE, 105.0)
    metadata = self.records[-1].runtime_metadata
    self.assertEqual(
        (metadata.runtime, metadata.is_controller, metadata.process_count),
        expected,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="enum_and_deep_exception",
          operation_type=_MOP.OPERATION_TYPE_PARTIAL_SAVE,
          status=None,
          error_mode="deep_exc",
          expected=(_OP.OPERATION_TYPE_PARTIAL_SAVE, _STATUS.STATUS_FAILURE),
          expected_error=("OSError", "write err", 5),
      ),
      dict(
          testcase_name="prefixed_str_and_message_error",
          operation_type="OPERATION_TYPE_LOAD",
          status="TIMED_OUT",
          error_mode="str",
          expected=(_OP.OPERATION_TYPE_LOAD, _STATUS.STATUS_TIMED_OUT),
          expected_error=("", "write err", 0),
      ),
      dict(
          testcase_name="short_str_and_default_success",
          operation_type="save",
          status=None,
          error_mode=None,
          expected=(_OP.OPERATION_TYPE_SAVE, _STATUS.STATUS_SUCCESS),
          expected_error=("", "", 0),
      ),
      dict(
          testcase_name="empty_and_unknown_enum",
          operation_type="",
          status="bogus",
          error_mode=None,
          expected=(_OP.OPERATION_TYPE_UNSPECIFIED, _STATUS.STATUS_UNSPECIFIED),
          expected_error=("", "", 0),
      ),
  )
  def test_enum_and_error_resolution(
      self, operation_type, status, error_mode, expected, expected_error
  ):
    error: BaseException | str | None = None
    if error_mode == "deep_exc":
      try:
        _raise_deep_os_error()
      except OSError as exc:
        error = exc
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
    noop_rec = checkpoint_operation_log_lib.OperationLogRecorder(sink=None)
    noop_rec.record_operation(_BEGIN, "op-n", _SAVE, 1.0)
    noop_rec.record_operation(_END, "op-n", _SAVE, 2.0)

    def _bad_sink(unused_proto: _LogProto) -> None:
      raise RuntimeError("sink boom")

    bad_rec = checkpoint_operation_log_lib.OperationLogRecorder(sink=_bad_sink)
    bad_rec.record_operation(_BEGIN, "op-b", _SAVE, 1.0)
    with self.assertLogs(level="WARNING") as sink_logs:
      bad_rec.record_operation(_END, "op-b", _SAVE, 2.0)
    self.assertIn("Operation log sink failed", "\n".join(sink_logs.output))

  def test_monitoring_integration_and_track_operation(self):
    self.enter_context(mock.patch.object(monitoring, "_recorders", []))
    monitoring.initialize(self.recorder)
    span_name = "/jax/orbax/write/background_ts_duration_secs"
    with monitoring.track_operation("ctx-ok", _SAVE, step=1, directory="/d"):
      jax_monitoring.record_event("/jax/orbax/write/start")
      jax_monitoring.record_scalar("/jax/orbax/write/gbytes", 1.0)
      jax_monitoring.record_event_duration_secs(span_name, 2.5)
      jax_monitoring.record_event_duration_secs(
          "/jax/orbax/write/blocking_d2h_duration_secs", 0.5
      )

    def _run_failing(op_id: str, exc: BaseException) -> None:
      with monitoring.track_operation(op_id, _SAVE):
        raise exc

    self.assertRaisesRegex(
        RuntimeError, "boom", _run_failing, "ctx-fail", RuntimeError("boom")
    )
    self.assertRaisesRegex(
        TimeoutError, "slow", _run_failing, "ctx-to", TimeoutError("slow")
    )
    self.assertLen(self.records, 3)
    ok_record = self.records[0]
    fail_record = self.records[1]
    timeout_record = self.records[2]
    self.assertEqual(
        (ok_record.operation_id, ok_record.step, ok_record.status),
        ("ctx-ok", 1, _STATUS.STATUS_SUCCESS),
    )
    self.assertEqual(ok_record.storage_metadata.directory, "/d")
    self.assertFalse(ok_record.HasField("job_metadata"))
    self.assertLessEqual(
        ok_record.start_time.ToNanoseconds(), ok_record.end_time.ToNanoseconds()
    )
    self.assertLen(ok_record.spans, 1)
    span = ok_record.spans[0]
    self.assertEqual((span.name, span.HasField("span_id")), (span_name, False))
    self.assertEqual(
        span.end_time.ToNanoseconds() - span.start_time.ToNanoseconds(),
        2_500_000_000,
    )
    self.assertEqual(
        (fail_record.status, fail_record.error_info.error_type),
        (_STATUS.STATUS_FAILURE, "RuntimeError"),
    )
    self.assertEqual(
        (timeout_record.status, timeout_record.error_info.error_type),
        (_STATUS.STATUS_TIMED_OUT, "TimeoutError"),
    )


if __name__ == "__main__":
  absltest.main()
