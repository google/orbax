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

"""Tests for CheckpointOperation construction and recording."""

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


_SAVE = monitoring.OperationType.OPERATION_TYPE_SAVE
_PARTIAL_SAVE = monitoring.OperationType.OPERATION_TYPE_PARTIAL_SAVE
_LOAD = monitoring.OperationType.OPERATION_TYPE_LOAD
_UNSPECIFIED = monitoring.OperationType.OPERATION_TYPE_UNSPECIFIED


class FormatOperationIdTest(parameterized.TestCase):

  @parameterized.parameters(
      (_SAVE, 42, 123456, None, "123456.save.42"),
      (_SAVE, 42, "job-789", None, "job-789.save.42"),
      (_SAVE, 42, 123456, "params", "123456.save.42#params"),
      (_SAVE, 100, None, None, "save.100"),
      (_PARTIAL_SAVE, 7, None, None, "partial_save.7"),
      (_LOAD, 100, None, "state", "load.100#state"),
  )
  def test_format_operation_id(
      self, operation_type, step, job_id, sub_operation, expected
  ):
    self.assertEqual(
        monitoring.format_operation_id(
            operation_type, step, job_id, sub_operation
        ),
        expected,
    )

  def test_rejects_invalid_operation_type(self):
    with self.assertRaisesRegex(TypeError, r"must be an int"):
      monitoring.format_operation_id(cast(Any, "save"), 10)
    with self.assertRaisesRegex(ValueError, r"999"):
      monitoring.format_operation_id(cast(Any, 999), 10)


class ProtoBuilderApiTest(parameterized.TestCase):

  def test_create_add_span_and_finalize(self):
    job_meta = _LogProto.JobMetadata()
    proto = checkpoint_operation_log_lib.create_operation_log(
        "op-1",
        "save",
        10.0,
        step=5,
        directory="/tmp/ckpt",
        job_metadata=job_meta,
    )
    self.assertEqual(proto.operation_id, "op-1")
    self.assertEqual(
        proto.operation_type, _LogProto.OperationType.OPERATION_TYPE_SAVE
    )
    self.assertEqual(proto.step, 5)
    self.assertEqual(proto.storage_metadata.directory, "/tmp/ckpt")
    self.assertEqual(proto.start_time.ToNanoseconds(), 10_000_000_000)
    self.assertEqual(proto.status, _LogProto.Status.STATUS_IN_PROGRESS)
    self.assertTrue(proto.HasField("job_metadata"))

    span = checkpoint_operation_log_lib.add_span(
        proto,
        "/jax/orbax/write/span",
        10.5,
        11.5,
        span_id=7,
        parent_span_id=3,
    )
    self.assertIsNotNone(span)
    self.assertEqual(span.span_id, 7)
    self.assertEqual(span.parent_span_id, 3)

    checkpoint_operation_log_lib.finalize_operation_log(
        proto, 12.0, status="SUCCESS"
    )
    self.assertEqual(proto.end_time.ToNanoseconds(), 12_000_000_000)
    self.assertEqual(proto.status, _LogProto.Status.STATUS_SUCCESS)

  def test_preserves_unspecified_status(self):
    proto = checkpoint_operation_log_lib.create_operation_log(
        "op-u", "save", 1.0
    )
    checkpoint_operation_log_lib.finalize_operation_log(
        proto, 2.0, status=_LogProto.Status.STATUS_UNSPECIFIED
    )
    self.assertEqual(proto.status, _LogProto.Status.STATUS_UNSPECIFIED)

  @parameterized.parameters(
      (True, False, None, None, _LogProto.Runtime.RUNTIME_PATHWAYS, True, 8),
      (True, True, None, None, _LogProto.Runtime.RUNTIME_PATHWAYS, False, 4),
      (
          False,
          True,
          None,
          None,
          _LogProto.Runtime.RUNTIME_PATHWAYS,
          False,
          4,
      ),
      (False, False, None, None, _LogProto.Runtime.RUNTIME_MCJAX, False, 4),
      (
          False,
          False,
          RuntimeError("no jax backend"),
          None,
          _LogProto.Runtime.RUNTIME_UNSPECIFIED,
          False,
          4,
      ),
      (
          True,
          False,
          None,
          RuntimeError("no mesh"),
          _LogProto.Runtime.RUNTIME_PATHWAYS,
          True,
          4,
      ),
  )
  def test_runtime_detection(
      self,
      pathways_backend: bool,
      colocated_active: bool,
      backend_error: BaseException | None,
      worker_count_error: BaseException | None,
      expected_runtime: Any,
      expected_controller: bool,
      expected_process_count: int,
  ):
    with (
        mock.patch.object(
            multihost,
            "is_pathways_backend",
            return_value=pathways_backend,
        ),
        mock.patch.object(
            multihost,
            "is_pathways_colocated_runtime_active",
            side_effect=backend_error,
            return_value=colocated_active,
        ),
        mock.patch.object(multihost, "process_count", return_value=4),
        mock.patch.object(
            pathways,
            "worker_count",
            side_effect=worker_count_error,
            return_value=8,
        ),
    ):
      proto = checkpoint_operation_log_lib.create_operation_log(
          "op-rt", "save", 10.0
      )
    self.assertEqual(proto.runtime_metadata.runtime, expected_runtime)
    self.assertEqual(proto.runtime_metadata.is_controller, expected_controller)
    self.assertEqual(
        proto.runtime_metadata.process_count, expected_process_count
    )


class OperationLogRecorderTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.records: list[_LogProto] = []
    self.recorder = checkpoint_operation_log_lib.OperationLogRecorder(
        sink=self.records.append
    )

  def _run_operation(
      self,
      op_id: str = "op-1",
      op_type: monitoring.OperationType = _SAVE,
      start: float = 100.0,
      end: float = 105.0,
      **kwargs: Any,
  ) -> _LogProto:
    self.recorder.record_operation("begin", op_id, op_type, start, **kwargs)
    self.recorder.record_operation("end", op_id, op_type, end, **kwargs)
    return self.records[-1]

  def test_lifecycle_and_filtering(self):
    self.recorder.record_operation(
        "begin", "op-lc", _SAVE, 100.0, step=10, directory="/tmp/ckpt"
    )
    self.recorder.record_span(
        "/jax/orbax/write/blocking_d2h_duration_secs", 100.1, 101.1
    )
    # Non-Orbax and legacy alias metrics must be ignored.
    self.recorder.record_span("/jax/compile_duration", 100.0, 102.0)
    self.recorder.record_span(
        "/jax/checkpoint/write/async/blocking_duration_secs", 100.0, 102.0
    )
    self.recorder.record_span(
        "/jax/checkpoint/write/async/commit_duration_sec", 100.0, 102.0
    )
    self.recorder.record_span(
        "/jax/checkpoint/write/async/thread_duration_sec", 100.0, 102.0
    )
    self.recorder.record_span("/jax/orbax/inverted", 105.0, 100.0)
    self.recorder.record_operation(
        "end", "op-lc", _SAVE, 105.0, status="SUCCESS"
    )
    self.assertLen(self.records, 1)
    record = self.records[0]
    self.assertEqual(record.operation_id, "op-lc")
    self.assertEqual(
        record.operation_type, _LogProto.OperationType.OPERATION_TYPE_SAVE
    )
    self.assertEqual(record.step, 10)
    self.assertEqual(record.storage_metadata.directory, "/tmp/ckpt")
    self.assertEqual(record.start_time.ToNanoseconds(), 100_000_000_000)
    self.assertEqual(record.end_time.ToNanoseconds(), 105_000_000_000)
    self.assertEqual(record.status, _LogProto.Status.STATUS_SUCCESS)
    self.assertLen(record.spans, 1)
    span = record.spans[0]
    self.assertEqual(span.name, "/jax/orbax/write/blocking_d2h_duration_secs")
    self.assertEqual(span.thread_id, threading.get_native_id())
    self.assertEqual(span.start_time.ToNanoseconds(), 100_100_000_000)
    self.assertEqual(span.end_time.ToNanoseconds(), 101_100_000_000)

  @parameterized.parameters(
      (_SAVE, _LogProto.OperationType.OPERATION_TYPE_SAVE),
      (_PARTIAL_SAVE, _LogProto.OperationType.OPERATION_TYPE_PARTIAL_SAVE),
      (_LOAD, _LogProto.OperationType.OPERATION_TYPE_LOAD),
      (_UNSPECIFIED, _LogProto.OperationType.OPERATION_TYPE_UNSPECIFIED),
  )
  def test_operation_types(self, op_type, expected_enum):
    record = self._run_operation(op_type=op_type)
    self.assertEqual(record.operation_type, expected_enum)

  @parameterized.parameters(
      ("save", _LogProto.OperationType.OPERATION_TYPE_SAVE),
      ("OPERATION_TYPE_SAVE", _LogProto.OperationType.OPERATION_TYPE_SAVE),
      ("partial_save", _LogProto.OperationType.OPERATION_TYPE_PARTIAL_SAVE),
      ("load", _LogProto.OperationType.OPERATION_TYPE_LOAD),
      ("unknown", _LogProto.OperationType.OPERATION_TYPE_UNSPECIFIED),
  )
  def test_create_operation_log_string_types(self, op_type_str, expected_enum):
    proto = checkpoint_operation_log_lib.create_operation_log(
        "op-str", op_type_str, 100.0
    )
    self.assertEqual(proto.operation_type, expected_enum)

  @parameterized.parameters(
      (True, "OSError", "write err"),
      (False, "", "string error message"),
  )
  def test_errors(self, is_exception, exp_type, exp_msg):
    if is_exception:
      try:
        raise OSError(exp_msg)
      except OSError as exc:
        err_input = exc
    else:
      err_input = exp_msg
    self.recorder.record_operation("begin", "op-err", _SAVE, 10.0)
    self.recorder.record_span(
        "/jax/orbax/write/span", 10.1, 10.2, error=err_input
    )
    self.recorder.record_operation(
        "end", "op-err", _SAVE, 12.0, error=err_input
    )
    record = self.records[-1]
    self.assertEqual(record.status, _LogProto.Status.STATUS_FAILURE)
    self.assertEqual(record.error_info.error_type, exp_type)
    self.assertEqual(record.error_info.error_message, exp_msg)
    self.assertEqual(bool(record.error_info.stack_frames), is_exception)
    self.assertEqual(record.spans[0].status, _LogProto.Status.STATUS_FAILURE)
    self.assertEqual(record.spans[0].error_info.error_message, exp_msg)

  def test_record_duration(self):
    self.recorder.record_operation("begin", "op-dur", _SAVE, 100.0)
    self.recorder.record_duration("/jax/orbax/span", duration=2.5)
    self.recorder.record_operation("end", "op-dur", _SAVE, 105.0)
    span = self.records[-1].spans[0]
    self.assertEqual(span.name, "/jax/orbax/span")
    self.assertEqual(
        span.end_time.ToNanoseconds() - span.start_time.ToNanoseconds(),
        2_500_000_000,
    )

  def test_explicit_operation_id_routing(self):
    self.recorder.record_operation("begin", "op-1", _SAVE, 100.0)
    self.recorder.record_operation("begin", "op-2", _SAVE, 101.0)
    self.recorder.record_span(
        "/jax/orbax/span", 101.1, 101.5, operation_id="op-1"
    )
    self.recorder.record_operation("end", "op-1", _SAVE, 102.0)
    self.recorder.record_operation("end", "op-2", _SAVE, 103.0)
    self.assertLen(self.records[0].spans, 1)
    self.assertEmpty(self.records[1].spans)

  def test_duplicate_begin(self):
    self.recorder.record_operation("begin", "dup-op", _SAVE, 100.0, step=1)
    self.recorder.record_operation("begin", "dup-op", _SAVE, 101.0, step=2)
    self.recorder.record_operation("end", "dup-op", _SAVE, 102.0)
    self.assertLen(self.records, 1)
    self.assertEqual(self.records[0].step, 2)

  def test_span_cap(self):
    self.recorder.record_operation("begin", "cap-op", _SAVE, 100.0)
    for _ in range(1030):
      self.recorder.record_span("/jax/orbax/span", 100.1, 100.2)
    self.recorder.record_operation("end", "cap-op", _SAVE, 105.0)
    self.assertLen(self.records[-1].spans, 1024)

  def test_status_none_defaults_to_success(self):
    self.recorder.record_operation("begin", "op-none", _SAVE, 100.0)
    self.recorder.record_span("/jax/orbax/span", 100.1, 100.2, status=None)
    self.recorder.record_operation("end", "op-none", _SAVE, 105.0, status=None)
    record = self.records[-1]
    self.assertEqual(record.status, _LogProto.Status.STATUS_SUCCESS)
    self.assertEqual(record.spans[0].status, _LogProto.Status.STATUS_SUCCESS)

  def test_monitoring_integration(self):
    monitoring.initialize(self.recorder)
    self.addCleanup(
        monitoring.unregister_recorders_by_type,
        checkpoint_operation_log_lib.OperationLogRecorder,
    )
    monitoring.record_operation("begin", "e2e-op", _SAVE, 10.0, step=1)
    span_name = "/jax/orbax/write/blocking_d2h_duration_secs"
    jax_monitoring.record_event_duration_secs(span_name, 1.0)
    monitoring.record_operation("end", "e2e-op", _SAVE, 12.0)

    self.assertLen(self.records, 1)
    self.assertEqual(self.records[0].operation_id, "e2e-op")
    self.assertEqual(self.records[0].step, 1)
    self.assertLen(self.records[0].spans, 1)
    self.assertEqual(self.records[0].spans[0].name, span_name)


if __name__ == "__main__":
  absltest.main()
