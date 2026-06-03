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

import time
import tracemalloc
from unittest import mock

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from orbax.checkpoint._src.testing.benchmarks.core import metric as metric_lib
import tensorstore as ts


class MetricsTest(parameterized.TestCase):

  def test_time(self):
    metrics = metric_lib.Metrics()

    with mock.patch.object(time, 'perf_counter', side_effect=[1.0, 3.0, 3.0]):
      with metrics.measure('test_metric', ['time']):
        pass

    self.assertEqual(
        metrics.results, {'test_metric_0_basics/time_s': (2.0, 's')}
    )

  @mock.patch(
      'orbax.checkpoint._src.testing.benchmarks.core.metric.psutil.Process'
  )
  def test_process_rss(self, mock_process):
    mock_process.return_value.memory_info.side_effect = [
        mock.Mock(rss=1 * 1024 * 1024),
        mock.Mock(rss=3 * 1024 * 1024),
    ]
    metrics = metric_lib.Metrics()
    with metrics.measure('test_metric', ['rss']):
      pass
    self.assertEqual(
        metrics.results, {'test_metric_0_basics/host_rss_diff_mb': (2.0, 'MB')}
    )

  @mock.patch(
      'orbax.checkpoint._src.testing.benchmarks.core.metric.tracemalloc'
  )
  def test_tracemalloc(self, mock_tracemalloc):
    mock_tracemalloc.Snapshot = tracemalloc.Snapshot
    mock_tracemalloc.get_traced_memory.side_effect = [
        (0, 1 * 1024 * 1024),
        (0, 3 * 1024 * 1024),
    ]
    mock_snapshot = mock.Mock()
    mock_snapshot.__class__ = tracemalloc.Snapshot
    mock_snapshot.compare_to.return_value = []
    mock_tracemalloc.take_snapshot.return_value = mock_snapshot
    metric_lib.TracemallocMetric._active_count = 0

    metrics = metric_lib.Metrics()
    with metrics.measure('test_metric', ['tracemalloc']):
      self.assertEqual(metric_lib.TracemallocMetric._active_count, 1)
    self.assertEqual(
        metrics.results,
        {'test_metric_7_memory/tracemalloc_peak_diff_mb': (2.0, 'MB')},
    )
    self.assertEqual(metric_lib.TracemallocMetric._active_count, 0)
    mock_tracemalloc.start.assert_called_once()
    mock_tracemalloc.stop.assert_called_once()

  def test_generate_report(self):
    metrics = metric_lib.Metrics(name='TestMetrics')
    metrics.results = {'metric1': (1.23, 's'), 'metric2': (4.56, 's')}
    expected_report = """---[process_id=0] TestMetrics Metrics Report ---
metric1: 1.2300 s
metric2: 4.5600 s
----------------------"""

    with mock.patch.object(logging, 'info') as mock_log:
      metrics.report()

      mock_log.assert_any_call(expected_report)

  def test_tensorstore_metrics(self):
    metrics = metric_lib.Metrics()
    with metrics.measure('test_metric', ['tensorstore']):
      ts_spec = ts.Spec(
          {
              'driver': 'zarr',
              'kvstore': {
                  'driver': 'file',
                  'path': self.create_tempdir().full_path,
              },
          },
      )
      ts_store = ts.open(
          ts_spec,
          create=True,
          delete_existing=True,
          dtype=np.int32,
          shape=(10,),
      ).result()
      ts_store.write(np.asarray(range(10)).astype(np.int32)).result()

    self.assertIn(
        'test_metric_6_tensorstore/changed_metric_count', metrics.results
    )

  @mock.patch.object(metric_lib.TensorstoreMetric, '_collect_metrics')
  def test_tensorstore_whitelisted_metrics_emit_per_bucket_scalars(
      self, mock_collect
  ):
    # Simulate two collections — start (baseline) then stop. The kvstore
    # metric grew by 5 (from 10→15) so the diff should be 5; the cache
    # metric grew by 100 bytes. The internal/thread metric is OUTSIDE the
    # whitelist and gets filtered out.
    start = {
        '/tensorstore/kvstore/file/read': {'count': 10},
        '/tensorstore/cache/bytes': {'value': 100},
        '/tensorstore/internal/thread/foo': {'value': 1},
    }
    stop = {
        '/tensorstore/kvstore/file/read': {'count': 15},
        '/tensorstore/cache/bytes': {'value': 200},
        '/tensorstore/internal/thread/foo': {'value': 99},  # filtered out
    }
    mock_collect.side_effect = [start, stop]
    metrics = metric_lib.Metrics()
    with metrics.measure('op', ['tensorstore']):
      pass
    self.assertEqual(
        metrics.results[
            'op_6_tensorstore/tensorstore_kvstore_file_read_count_diff'
        ][0],
        5,
    )
    self.assertEqual(
        metrics.results['op_6_tensorstore/tensorstore_cache_bytes_value_diff'][
            0
        ],
        100,
    )
    # Internal metric outside the whitelist not exported.
    self.assertFalse(any('internal_thread_foo' in k for k in metrics.results))

  def test_all_metrics(self):
    metric_lib.TracemallocMetric._active_count = 0
    metrics = metric_lib.Metrics()
    all_metrics = list(metric_lib.METRIC_REGISTRY.keys())
    with metrics.measure('test_metric', all_metrics):
      ts_spec = ts.Spec(
          {
              'driver': 'zarr',
              'kvstore': {
                  'driver': 'file',
                  'path': self.create_tempdir().full_path,
              },
          },
      )
      ts_store = ts.open(
          ts_spec,
          create=True,
          delete_existing=True,
          dtype=np.int32,
          shape=(10,),
      ).result()
      ts_store.write(np.asarray(range(10)).astype(np.int32)).result()

    self.assertIn('test_metric_0_basics/time_s', metrics.results)
    self.assertIn('test_metric_0_basics/host_rss_diff_mb', metrics.results)
    self.assertIn(
        'test_metric_7_memory/tracemalloc_peak_diff_mb', metrics.results
    )
    self.assertIn(
        'test_metric_6_tensorstore/changed_metric_count', metrics.results
    )


def _fake_device(peak_bytes):
  d = mock.Mock()
  if peak_bytes is None:
    d.memory_stats.return_value = None
  else:
    d.memory_stats.return_value = {
        'peak_bytes_in_use': peak_bytes,
        'bytes_in_use': peak_bytes // 2,
    }
  return d


class DeviceMemoryMetricTest(parameterized.TestCase):

  @mock.patch('jax.live_arrays')
  @mock.patch('jax.local_devices')
  def test_live_arrays_delta_captured(self, mock_devices, mock_live):
    mock_devices.return_value = [_fake_device(None)]
    mock_live.side_effect = [
        [mock.Mock()] * 3,
        [mock.Mock()] * 5,
    ]
    metrics = metric_lib.Metrics()
    with metrics.measure('op', ['device_memory']):
      pass
    self.assertIn('op_7_memory/jax_live_arrays_count_delta', metrics.results)
    val, _ = metrics.results['op_7_memory/jax_live_arrays_count_delta']
    self.assertEqual(val, 2)

  @mock.patch('jax.live_arrays', return_value=[])
  @mock.patch('jax.local_devices')
  def test_hbm_peak_diff_when_memory_stats_available(
      self, mock_devices, unused_mock_live
  ):
    gb = 1024**3
    d0 = mock.Mock()
    d0.memory_stats.side_effect = [
        {'peak_bytes_in_use': 4 * gb},
        {'peak_bytes_in_use': 5 * gb},
    ]
    d1 = mock.Mock()
    d1.memory_stats.side_effect = [
        {'peak_bytes_in_use': 4 * gb},
        {'peak_bytes_in_use': 6 * gb},
    ]
    mock_devices.return_value = [d0, d1]
    metrics = metric_lib.Metrics()
    with metrics.measure('op', ['device_memory']):
      pass
    self.assertIn('op_7_memory/device_hbm_peak_diff_gb', metrics.results)
    val, unit = metrics.results['op_7_memory/device_hbm_peak_diff_gb']
    self.assertAlmostEqual(val, 2.0)
    self.assertEqual(unit, 'GiB')

  @mock.patch('jax.live_arrays', return_value=[])
  @mock.patch('jax.local_devices')
  def test_cpu_no_memory_stats_skips_hbm_tag(
      self, mock_devices, unused_mock_live
  ):
    mock_devices.return_value = [_fake_device(None), _fake_device(None)]
    metrics = metric_lib.Metrics()
    with metrics.measure('op', ['device_memory']):
      pass
    self.assertNotIn('op_7_memory/device_hbm_peak_diff_gb', metrics.results)
    self.assertIn('op_7_memory/jax_live_arrays_count_delta', metrics.results)


if __name__ == '__main__':
  absltest.main()
