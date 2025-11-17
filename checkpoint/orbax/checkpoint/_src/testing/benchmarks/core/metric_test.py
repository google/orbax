# Copyright 2025 The Orbax Authors.
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

    self.assertEqual(metrics.results, {'test_metric_time_duration': (2.0, 's')})

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
    self.assertEqual(metrics.results, {'test_metric_rss_diff': (2.0, 'MB')})

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
        metrics.results, {'test_metric_tracemalloc_peak_diff': (2.0, 'MB')}
    )
    self.assertEqual(metric_lib.TracemallocMetric._active_count, 0)
    mock_tracemalloc.start.assert_called_once()
    mock_tracemalloc.stop.assert_called_once()

  def test_io_metrics(self):
    metrics = metric_lib.Metrics()
    with metrics.measure('test_metric', ['io']):
      pass
    self.assertIn('test_metric_io_read_bytes', metrics.results)
    self.assertIn('test_metric_io_write_bytes', metrics.results)
    self.assertIn('test_metric_io_read_throughput', metrics.results)
    self.assertIn('test_metric_io_write_throughput', metrics.results)

  def test_report(self):
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

    self.assertIn('test_metric_tensorstore_diff_count', metrics.results)

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

    self.assertIn('test_metric_time_duration', metrics.results)
    self.assertIn('test_metric_rss_diff', metrics.results)
    self.assertIn('test_metric_tracemalloc_peak_diff', metrics.results)
    self.assertIn('test_metric_io_read_bytes', metrics.results)
    self.assertIn('test_metric_io_write_bytes', metrics.results)
    self.assertIn('test_metric_io_read_throughput', metrics.results)
    self.assertIn('test_metric_io_write_throughput', metrics.results)
    self.assertIn('test_metric_tensorstore_diff_count', metrics.results)


class MetricsManagerTest(parameterized.TestCase):

  def test_add_result_and_generate_report_no_repeats(self):
    manager = metric_lib.MetricsManager(name='Suite', num_repeats=1)
    metrics1 = metric_lib.Metrics()
    metrics1.results['op1_time_duration'] = (1.0, 's')
    manager.add_result('bench1', metrics1, None)

    metrics2 = metric_lib.Metrics()
    metrics2.results['op1_time_duration'] = (2.0, 's')
    manager.add_result('bench2', metrics2, ValueError('failure'))

    report = manager.generate_report()
    self.assertIn('Suite', report)
    self.assertIn('Total benchmark configurations: 2', report)
    self.assertIn('Total runs (1 repeats): 2, Passed: 1, Failed: 1', report)
    self.assertNotIn('Aggregated Metrics', report)
    self.assertIn('Failed Runs', report)
    self.assertIn("Error: ValueError('failure')", report)

  def test_generate_report_with_repeats_and_aggregation(self):
    manager = metric_lib.MetricsManager(name='Suite', num_repeats=3)

    # Benchmark 1, Run 1
    m1r1 = metric_lib.Metrics()
    m1r1.results['op_time_duration'] = (1.0, 's')
    m1r1.results['op_rss_diff'] = (10.0, 'MB')
    manager.add_result('bench1', m1r1, None)
    # Benchmark 1, Run 2
    m1r2 = metric_lib.Metrics()
    m1r2.results['op_time_duration'] = (1.2, 's')
    m1r2.results['op_rss_diff'] = (12.0, 'MB')
    manager.add_result('bench1', m1r2, None)
    # Benchmark 1, Run 3 (Failed)
    m1r3 = metric_lib.Metrics()
    manager.add_result('bench1', m1r3, RuntimeError('Run 3 failed'))

    report = manager.generate_report()

    self.assertIn('Suite', report)
    self.assertIn('Total benchmark configurations: 1', report)
    self.assertIn('Total runs (3 repeats): 3, Passed: 2, Failed: 1', report)
    self.assertIn('Aggregated Metrics', report)
    self.assertIn('Benchmark: bench1', report)
    # mean=1.1, std=0.1, min=1.0, max=1.2
    self.assertIn(
        'op_time_duration: 1.1000 +/- 0.1000 s (min: 1.0000, max: 1.2000, n=2)',
        report,
    )
    # mean=11.0, std=1.0, min=10.0, max=12.0
    self.assertIn(
        'op_rss_diff: 11.0000 +/- 1.0000 MB (min: 10.0000, max: 12.0000, n=2)',
        report,
    )
    self.assertIn('Failed Runs', report)
    self.assertIn("Error: RuntimeError('Run 3 failed')", report)

  def test_generate_report_no_successful_runs_for_aggregation(self):
    manager = metric_lib.MetricsManager(name='Suite', num_repeats=2)
    manager.add_result('bench1', metric_lib.Metrics(), ValueError('1'))
    manager.add_result('bench1', metric_lib.Metrics(), ValueError('2'))
    report = manager.generate_report()
    self.assertIn('No successful runs to aggregate', report)


if __name__ == '__main__':
  absltest.main()
