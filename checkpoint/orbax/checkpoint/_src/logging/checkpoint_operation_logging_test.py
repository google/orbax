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
from orbax.checkpoint._src.logging import checkpoint_operation_logging
from orbax.checkpoint._src.logging import monitoring
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.multihost import pathways
from orbax.checkpoint.proto import checkpoint_operation_pb2

_LogProto = checkpoint_operation_pb2.CheckpointOperation
_RT = _LogProto.Runtime
_STATUS = monitoring.OperationStatus
_STORAGE = _LogProto.StorageType
_LOCALITY = _LogProto.StorageLocality
_SAVE = monitoring.OperationType.OPERATION_TYPE_SAVE
_LOAD = monitoring.OperationType.OPERATION_TYPE_LOAD
_BEGIN = monitoring.OperationEvent.BEGIN
_END = monitoring.OperationEvent.END
_SPAN = "/jax/orbax/write/background_ts_duration_secs"  # Allowlisted duration.


def _raise_deep_os_error(depth: int = 7) -> None:
  if depth <= 1:
    raise OSError("write err")
  _raise_deep_os_error(depth - 1)


class _FailingRecorder(checkpoint_operation_logging.OperationMetricRecorder):

  def record_operation(self, *unused_args: Any, **unused_kwargs: Any) -> None:
    raise RuntimeError("recorder boom")


class OperationMetricRecorderTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.records: list[_LogProto] = []
    self.recorder = checkpoint_operation_logging.OperationMetricRecorder(
        sink=self.records.append
    )

  def _record(self, **begin_kwargs: Any) -> _LogProto:
    """Records one SAVE operation and returns the emitted proto."""
    self.recorder.record_operation(_BEGIN, "op", _SAVE, 10.0, **begin_kwargs)
    self.recorder.record_operation(_END, "op", _SAVE, 12.0)
    return self.records[-1]

  def test_lifecycle_and_filtering(self):
    recorder = checkpoint_operation_logging.OperationMetricRecorder(
        sink=self.records.append, job_metadata=_LogProto.JobMetadata()
    )
    self.enter_context(mock.patch.object(monitoring, "_recorders", []))
    monitoring.initialize(recorder)
    jax_monitoring.record_event_duration_secs(_SPAN, 0.5)  # No open op.
    monitoring.record_operation(
        _BEGIN, "op-lc", _LOAD, 100.0, step=10, directory="/tmp/ckpt"
    )
    jax_monitoring.record_event("/jax/orbax/write/start")
    jax_monitoring.record_scalar("/jax/orbax/write/gbytes", 1.0)
    jax_monitoring.record_event_duration_secs(_SPAN, 1.0)
    # Ignored: negative duration, and a duration that is not allowlisted.
    recorder.record_duration(_SPAN, -1.0)
    recorder.record_duration("/jax/orbax/write/blocking_d2h_duration_secs", 1.0)
    monitoring.record_operation(_END, "op-lc", _LOAD, 105.0)

    self.assertLen(self.records, 1)
    record = self.records[0]
    self.assertEqual(
        (
            record.operation_id,
            record.operation_type,
            record.step,
            record.status,
        ),
        ("op-lc", _LOAD, 10, _STATUS.STATUS_SUCCESS),
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
        (span.name, span.thread_id, span.status),
        (_SPAN, threading.get_native_id(), _STATUS.STATUS_SUCCESS),
    )
    self.assertEqual(
        span.end_time.ToNanoseconds() - span.start_time.ToNanoseconds(),
        1_000_000_000,
    )

  @parameterized.named_parameters(
      dict(testcase_name="mcjax", expected=(_RT.RUNTIME_MCJAX, False, 2, 4)),
      dict(
          testcase_name="pathways_colocated",
          is_pathways_colocated_runtime_active=True,
          expected=(_RT.RUNTIME_PATHWAYS, False, 2, 4),
      ),
      dict(
          testcase_name="pathways_colocated_controller",
          is_pathways_colocated_runtime_active=True,
          is_pathways_controller=True,
          expected=(_RT.RUNTIME_PATHWAYS, True, 2, 8),
      ),
      dict(
          testcase_name="pathways_proxy",
          is_pathways_backend=True,
          expected=(_RT.RUNTIME_PATHWAYS, False, 2, 4),
      ),
      dict(
          testcase_name="pathways_controller",
          is_pathways_backend=True,
          is_pathways_controller=True,
          expected=(_RT.RUNTIME_PATHWAYS, True, 2, 8),
      ),
      dict(
          testcase_name="worker_count_error",
          is_pathways_backend=True,
          is_pathways_controller=True,
          worker_count=OSError("x"),
          expected=(_RT.RUNTIME_PATHWAYS, True, 2, 4),
      ),
      dict(
          testcase_name="probe_errors",
          is_pathways_backend=OSError("x"),
          process_index=OSError("x"),
          process_count=OSError("x"),
          expected=(_RT.RUNTIME_UNSPECIFIED, False, 0, 0),
      ),
  )
  def test_runtime_detection(self, expected, **probes):
    defaults = dict(
        is_pathways_backend=False,
        is_pathways_controller=False,
        is_pathways_colocated_runtime_active=False,
        process_index=2,
        process_count=4,
        worker_count=8,
    )
    for name, value in (defaults | probes).items():
      self.enter_context(
          mock.patch.object(
              pathways if name == "worker_count" else multihost,
              name,
              autospec=True,
              side_effect=value if isinstance(value, Exception) else None,
              return_value=value,
          )
      )
    metadata = self._record().runtime_metadata
    self.assertEqual(
        (
            metadata.runtime,
            metadata.is_controller,
            metadata.process_index,
            metadata.process_count,
        ),
        expected,
    )

  def test_status_and_error_resolution(self):
    for op_id in ("ok", "raised", "msg"):
      self.recorder.record_operation(_BEGIN, op_id, _SAVE, 10.0)
    self.recorder.record_operation(_END, "ok", _SAVE, 12.0)
    try:
      _raise_deep_os_error()
    except OSError as e:
      self.recorder.record_operation(_END, "raised", _SAVE, 12.0, error=e)
    self.recorder.record_operation(
        _END, "msg", _SAVE, 12.0, status=_STATUS.STATUS_TIMED_OUT, error="io"
    )
    self.assertEqual(
        [
            (
                r.status,
                r.error_info.error_type,
                r.error_info.error_message,
                len(r.error_info.stack_frames),
            )
            for r in self.records
        ],
        [
            (_STATUS.STATUS_SUCCESS, "", "", 0),
            (_STATUS.STATUS_FAILURE, "OSError", "write err", 5),
            (_STATUS.STATUS_TIMED_OUT, "", "io", 0),
        ],
    )

  @parameterized.parameters(
      ("GCS", _STORAGE.STORAGE_TYPE_GCS),
      ("other", _STORAGE.STORAGE_TYPE_OTHER),
      ("unknown_fs", _STORAGE.STORAGE_TYPE_UNSPECIFIED),
  )
  def test_storage_type_mapping(self, storage_type, expected):
    record = self._record(storage_type=storage_type)
    self.assertEqual(record.storage_metadata.type, expected)

  @parameterized.parameters(
      ("same_cluster", _LOCALITY.STORAGE_LOCALITY_SAME_CLUSTER),
      ("same_metro", _LOCALITY.STORAGE_LOCALITY_SAME_METRO),
      ("different_metro", _LOCALITY.STORAGE_LOCALITY_DIFFERENT_METRO),
      ("different_continent", _LOCALITY.STORAGE_LOCALITY_DIFFERENT_CONTINENT),
      ("unknown", _LOCALITY.STORAGE_LOCALITY_UNSPECIFIED),
  )
  def test_storage_locality_mapping(self, locality, expected):
    record = self._record(locality=locality)
    self.assertEqual(record.storage_metadata.locality, expected)

  def test_routing_edge_cases_and_span_cap(self):
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
    # With 2 open ops, un-keyed durations skip; explicit operation_id routes.
    self.recorder.record_duration(_SPAN, 0.1)
    self.recorder.record_duration(_SPAN, 0.2, operation_id="op-1")
    self.recorder.record_operation(_END, "op-1", _SAVE, 102.0)
    # Now only op-2 is open; un-keyed spans attach up to _MAX_SPANS (1024).
    for _ in range(1030):
      self.recorder.record_duration(_SPAN, 0.1)
    self.recorder.record_operation(_END, "op-2", _SAVE, 103.0)
    self.assertEqual([len(r.spans) for r in self.records], [1, 1024])
    self.assertEqual(self.records[1].step, 1)  # Duplicate BEGIN was ignored.

  def test_missing_or_failing_sink_is_tolerated(self):
    def _bad_sink(unused_proto: _LogProto) -> None:
      raise RuntimeError("sink boom")

    with self.assertLogs(level="WARNING") as logs:
      for sink in (None, _bad_sink):
        recorder = checkpoint_operation_logging.OperationMetricRecorder(
            sink=sink
        )
        recorder.record_operation(_BEGIN, "op", _SAVE, 1.0)
        recorder.record_operation(_END, "op", _SAVE, 2.0)
    self.assertIn("Operation log sink failed", "\n".join(logs.output))

  def test_internal_exception_isolation(self):
    with self.assertLogs(level="WARNING") as logs:
      self.recorder.record_operation(_BEGIN, "op-x", _SAVE, 1.0, step="bad")
      self.recorder.record_duration(_SPAN, cast(Any, "bad"))
    log_text = "\n".join(logs.output)
    self.assertIn("Failed to record operation 'op-x'", log_text)
    self.assertIn(f"Failed to record duration {_SPAN!r}", log_text)

  def test_dispatch_isolates_failing_recorder(self):
    self.enter_context(mock.patch.object(monitoring, "_recorders", []))
    monitoring.initialize(_FailingRecorder())
    monitoring.initialize(self.recorder)
    with self.assertLogs(level="WARNING") as logs:
      monitoring.record_operation(_BEGIN, "op-ok", _SAVE)
      monitoring.record_operation(_END, "op-ok", _SAVE)
    log_text = "\n".join(logs.output)
    self.assertIn("_FailingRecorder failed to record operation op-ok", log_text)
    self.assertLen(self.records, 1)
    record = self.records[0]
    self.assertEqual(
        (record.operation_id, record.status, record.HasField("job_metadata")),
        ("op-ok", _STATUS.STATUS_SUCCESS, False),
    )
    self.assertGreater(record.start_time.ToNanoseconds(), 0)  # Defaults to now.

  @parameterized.parameters(
      (asyncio.CancelledError(), _STATUS.STATUS_CANCELLED),
      (concurrent.futures.CancelledError(), _STATUS.STATUS_CANCELLED),
      (TimeoutError("slow"), _STATUS.STATUS_TIMED_OUT),
      (ValueError("bad"), _STATUS.STATUS_FAILURE),
      ("something failed", _STATUS.STATUS_FAILURE),
      (None, _STATUS.STATUS_FAILURE),
  )
  def test_status_from_error(self, error, expected):
    self.assertEqual(monitoring.status_from_error(error), expected)


if __name__ == "__main__":
  absltest.main()
