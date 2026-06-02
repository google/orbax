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

"""Tests for MetricsManager — aggregation, per-host writers, and HParams."""

import dataclasses
from unittest import mock

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import numpy as np
from orbax.checkpoint._src.testing.benchmarks.core import metric as metric_lib
from orbax.checkpoint._src.testing.benchmarks.core import metrics_manager


class MetricsManagerTest(parameterized.TestCase):

  def test_add_result_and_generate_report_no_repeats(self):
    manager = metrics_manager.MetricsManager(name='Suite', num_repeats=1)
    metrics1 = metric_lib.Metrics()
    metrics1.results['op1_time_duration'] = (1.0, 's')
    manager.add_result('bench1', metrics1)

    metrics2 = metric_lib.Metrics()
    metrics2.results['op1_time_duration'] = (2.0, 's')
    manager.add_result('bench2', metrics2, error=ValueError('failure'))

    with mock.patch.object(logging, 'info') as mock_log:
      manager.generate_report()
      # Combine all log calls into one string for easier assertion
      report = '\n'.join([call.args[0] for call in mock_log.call_args_list])

      self.assertIn('Suite', report)
      self.assertIn('Total benchmark configurations: 2', report)
      self.assertIn('Total runs (1 repeats): 2, Passed: 1, Failed: 1', report)
      self.assertNotIn('Aggregated Metrics', report)
      self.assertIn('Failed Runs', report)
      self.assertIn("Error: ValueError('failure')", report)

  def test_generate_report_with_repeats_and_aggregation(self):
    manager = metrics_manager.MetricsManager(name='Suite', num_repeats=3)

    # Benchmark 1, Run 1
    m1r1 = metric_lib.Metrics()
    m1r1.results['op_time_duration'] = (1.0, 's')
    m1r1.results['op_rss_diff'] = (10.0, 'MB')
    manager.add_result('bench1', m1r1)
    # Benchmark 1, Run 2
    m1r2 = metric_lib.Metrics()
    m1r2.results['op_time_duration'] = (1.2, 's')
    m1r2.results['op_rss_diff'] = (12.0, 'MB')
    manager.add_result('bench1', m1r2)
    # Benchmark 1, Run 3 (Failed)
    m1r3 = metric_lib.Metrics()
    manager.add_result('bench1', m1r3, error=RuntimeError('Run 3 failed'))

    with mock.patch.object(logging, 'info') as mock_log:
      manager.generate_report()
      report = '\n'.join([call.args[0] for call in mock_log.call_args_list])

      self.assertIn('Suite', report)
      self.assertIn('Total benchmark configurations: 1', report)
      self.assertIn('Total runs (3 repeats): 3, Passed: 2, Failed: 1', report)
      self.assertIn('Aggregated Metrics', report)
      self.assertIn('Benchmark: bench1', report)
      # mean=1.1, std=0.1, min=1.0, max=1.2
      self.assertIn(
          'op_time_duration: 1.1000 +/- 0.1000 s (min: 1.0000, max: 1.2000,'
          ' n=2)',
          report,
      )
      # mean=11.0, std=1.0, min=10.0, max=12.0
      self.assertIn(
          'op_rss_diff: 11.0000 +/- 1.0000 MB (min: 10.0000, max: 12.0000,'
          ' n=2)',
          report,
      )
      self.assertIn('Failed Runs', report)
      self.assertIn("Error: RuntimeError('Run 3 failed')", report)

  @mock.patch('clu.metric_writers.create_default_writer')
  def test_generate_report_exports_to_tensorboard(self, mock_create_writer):
    mock_writer = mock.Mock()
    mock_create_writer.return_value = mock_writer
    temp_dir = epath.Path(self.create_tempdir().full_path)
    # Legacy single-writer path simplifies the call-list assertions below;
    # per-host fan-out is exercised separately in
    # MetricsManagerPerHostWriterTest.
    manager = metrics_manager.MetricsManager(
        name='TBSuite',
        num_repeats=2,
        tensorboard_dir=temp_dir,
        enable_per_host_metrics=False,
    )

    bench1_name = 'bench1'
    bench1_opts = {'opt1': 1}
    bench1_ckpt_config = {'ckpt1': 'path1'}
    bench2_name = 'bench2'
    bench2_opts = {'opt2': 2}
    bench2_ckpt_config = None

    # Benchmark 'bench1', Rep 1: success
    m1 = metric_lib.Metrics()
    m1.results['op_time_duration'] = (1.0, 's')
    m1.results['op_other_metric'] = ('some_string', 'text')
    manager.add_result(
        bench1_name,
        m1,
        benchmark_options=bench1_opts,
        checkpoint_config=bench1_ckpt_config,
    )

    # Benchmark 'bench1', Rep 2: failure
    m2 = metric_lib.Metrics()
    manager.add_result(
        bench1_name,
        m2,
        benchmark_options=bench1_opts,
        error=ValueError('failure'),
    )

    # Add a second benchmark to ensure writers are created per benchmark
    m3 = metric_lib.Metrics()
    m3.results['loss'] = (0.5, 'none')
    manager.add_result(
        bench2_name,
        m3,
        benchmark_options=bench2_opts,
        checkpoint_config=bench2_ckpt_config,
    )

    manager.generate_report()

    # Check that SummaryWriter was called for each benchmark with correct path
    self.assertEqual(mock_create_writer.call_count, 2)
    mock_create_writer.assert_has_calls(
        [
            mock.call(
                temp_dir,
                just_logging=False,
                collection=bench1_name,
            ),
            mock.call(
                temp_dir,
                just_logging=False,
                collection=bench2_name,
            ),
        ],
        any_order=True,
    )

    # Since the mock writer instance is reused, we check all calls on it.
    # Calls for 'bench1'
    mock_writer.write_scalars.assert_any_call(
        step=0, scalars={'op_time_duration_s': 1.0}
    )
    mock_writer.write_texts.assert_any_call(
        step=0, texts={'op_other_metric_text': 'some_string'}
    )
    mock_writer.write_texts.assert_any_call(
        step=1, texts={'error': "<pre>ValueError('failure')</pre>"}
    )
    # Calls for 'bench2'
    mock_writer.write_scalars.assert_any_call(
        step=0, scalars={'loss_none': 0.5}
    )

    # Configuration is now rendered as markdown; verify the benchmark name
    # header and each field/value row appears.
    configuration_blobs = [
        kwargs['texts']['configuration']
        for _, kwargs in mock_writer.write_texts.call_args_list
        if 'texts' in kwargs and 'configuration' in kwargs['texts']
    ]
    self.assertLen(configuration_blobs, 2)

    by_name = {}
    for blob in configuration_blobs:
      header_line = blob.splitlines()[0]
      name = header_line.removeprefix('## ').strip()
      by_name[name] = blob

    self.assertIn(bench1_name, by_name)
    self.assertIn('| `opt1` | `1` |', by_name[bench1_name])
    self.assertIn('| `ckpt1` | `path1` |', by_name[bench1_name])

    self.assertIn(bench2_name, by_name)
    self.assertIn('| `opt2` | `2` |', by_name[bench2_name])
    # bench2 has no checkpoint_config — the "Checkpoint config" section
    # is omitted entirely.
    self.assertNotIn('Checkpoint config', by_name[bench2_name])

    # Each benchmark's writer gets closed exactly once at end of
    # generate_report. Flush count varies with the number of writes; assert
    # ≥ close count as a sanity check rather than pinning a brittle exact
    # number.
    self.assertEqual(mock_writer.close.call_count, 2)
    self.assertGreaterEqual(mock_writer.flush.call_count, 2)

  def test_generate_report_no_successful_runs_for_aggregation(self):
    manager = metrics_manager.MetricsManager(name='Suite', num_repeats=2)
    manager.add_result('bench1', metric_lib.Metrics(), error=ValueError('1'))
    manager.add_result('bench1', metric_lib.Metrics(), error=ValueError('2'))
    with mock.patch.object(logging, 'info') as mock_log:
      manager.generate_report()
      report = '\n'.join([call.args[0] for call in mock_log.call_args_list])
      self.assertIn('No successful runs to aggregate', report)

  @mock.patch('clu.metric_writers.create_default_writer')
  def test_add_result_writes_incrementally(self, mock_create_writer):
    mock_writer = mock.Mock()
    mock_create_writer.return_value = mock_writer
    temp_dir = epath.Path(self.create_tempdir().full_path)
    # Legacy path keeps the contract simple to assert against.
    manager = metrics_manager.MetricsManager(
        name='IncSuite',
        num_repeats=1,
        tensorboard_dir=temp_dir,
        enable_per_host_metrics=False,
    )

    m1 = metric_lib.Metrics()
    m1.results['acc'] = (0.9, '')
    manager.add_result('bench1', m1)

    mock_create_writer.assert_called_once_with(
        temp_dir,
        just_logging=False,
        collection='bench1',
    )
    mock_writer.write_scalars.assert_called_once_with(
        step=0, scalars={'acc_': 0.9}
    )
    mock_writer.flush.assert_called_once()


class MlperfAggregatesTest(parameterized.TestCase):
  """Cross-host aggregation math: max wins, p50/p99 from the host distribution."""

  def test_aggregates_compute_max_min_mean_p50_p99(self):
    keys = ['op_4_throughput/save_total_gbps', 'op_3_load_breakdown/blocking_s']
    per_host_matrix = np.array([
        [22.4, 4.21],
        [21.8, 4.50],
        [18.0, 5.10],
        [7.8, 8.20],  # straggler
    ])
    out = metrics_manager._summary_aggregates(per_host_matrix, keys)
    self.assertEqual(set(out), set(keys))
    save = out['op_4_throughput/save_total_gbps']
    self.assertAlmostEqual(save['max'], 22.4)
    self.assertAlmostEqual(save['min'], 7.8)
    self.assertAlmostEqual(save['mean'], 17.5)
    self.assertAlmostEqual(save['p50'], 19.9)
    self.assertAlmostEqual(save['p99'], 22.382)
    load = out['op_3_load_breakdown/blocking_s']
    self.assertAlmostEqual(load['max'], 8.20)
    self.assertAlmostEqual(load['min'], 4.21)

  def test_aggregates_single_host_skips_percentiles(self):
    out = metrics_manager._summary_aggregates(np.array([[1.5]]), ['m'])
    self.assertEqual(out['m']['max'], 1.5)
    self.assertEqual(out['m']['min'], 1.5)
    self.assertEqual(out['m']['mean'], 1.5)
    self.assertNotIn('p50', out['m'])
    self.assertNotIn('p99', out['m'])

  def test_aggregates_empty_matrix_returns_empty(self):
    self.assertEqual(
        metrics_manager._summary_aggregates(np.zeros((0, 0)), []), {}
    )


class MetricsManagerPerHostWriterTest(parameterized.TestCase):
  """Per-host writer plumbing: each process writes to its own subdir."""

  @mock.patch.object(
      metrics_manager.multihost, 'get_process_index', return_value=3
  )
  @mock.patch(
      'orbax.checkpoint._src.testing.benchmarks.core.metrics_manager.metric_writers.create_default_writer'
  )
  def test_per_host_writer_path_includes_host_suffix(
      self, mock_create_writer, unused_mock_idx
  ):
    mock_writer = mock.Mock()
    mock_create_writer.return_value = mock_writer
    temp_dir = epath.Path(self.create_tempdir().full_path)
    manager = metrics_manager.MetricsManager(
        name='Suite',
        num_repeats=1,
        tensorboard_dir=temp_dir,
        enable_per_host_metrics=True,
    )
    m = metric_lib.Metrics()
    m.results['op_0_basics/time_s'] = (1.0, 's')
    manager.add_result('bench1', m)
    args, kwargs = mock_create_writer.call_args
    # Per-host path is encoded in collection (clu plants events under
    # <logdir>/<collection>/), not duplicated in logdir.
    self.assertEqual(args[0], temp_dir)
    self.assertEqual(kwargs.get('collection'), 'bench1/host_3')

  @mock.patch.object(
      metrics_manager.multihost, 'get_process_index', return_value=2
  )
  @mock.patch(
      'orbax.checkpoint._src.testing.benchmarks.core.metrics_manager.metric_writers.create_default_writer'
  )
  def test_per_host_disabled_keeps_legacy_behavior(
      self, mock_create_writer, unused_mock_idx
  ):
    mock_writer = mock.Mock()
    mock_create_writer.return_value = mock_writer
    temp_dir = epath.Path(self.create_tempdir().full_path)
    manager = metrics_manager.MetricsManager(
        name='Suite',
        num_repeats=1,
        tensorboard_dir=temp_dir,
        enable_per_host_metrics=False,
    )
    m = metric_lib.Metrics()
    m.results['op_0_basics/time_s'] = (1.0, 's')
    manager.add_result('bench1', m)
    args, kwargs = mock_create_writer.call_args
    # No host_<idx> in the leaf segment; non-primary hosts use just_logging.
    self.assertEqual(args[0], temp_dir)
    self.assertEqual(kwargs.get('collection'), 'bench1')
    self.assertTrue(kwargs.get('just_logging'))

  @mock.patch.object(
      metrics_manager.multihost, 'get_process_index', return_value=0
  )
  @mock.patch(
      'orbax.checkpoint._src.testing.benchmarks.core.metrics_manager.metric_writers.create_default_writer'
  )
  def test_per_host_primary_still_writes(
      self, mock_create_writer, unused_mock_idx
  ):
    mock_writer = mock.Mock()
    mock_create_writer.return_value = mock_writer
    temp_dir = epath.Path(self.create_tempdir().full_path)
    manager = metrics_manager.MetricsManager(
        name='Suite',
        num_repeats=1,
        tensorboard_dir=temp_dir,
        enable_per_host_metrics=True,
    )
    m = metric_lib.Metrics()
    m.results['op_0_basics/time_s'] = (1.0, 's')
    manager.add_result('bench1', m)
    args, kwargs = mock_create_writer.call_args
    self.assertEqual(args[0], temp_dir)
    self.assertEqual(kwargs.get('collection'), 'bench1/host_0')
    self.assertFalse(kwargs.get('just_logging', False))


@dataclasses.dataclass(frozen=True)
class _HpOpts:
  async_enabled: bool = True
  chunk_byte_size: int | None = None
  notes: str = 'baseline'


class MetricsManagerHparamsTest(parameterized.TestCase):
  """The HParams summary lands on the summary writer at generate_report time."""

  @mock.patch(
      'orbax.checkpoint._src.testing.benchmarks.core.metrics_manager.metric_writers.create_default_writer'
  )
  def test_generate_report_writes_hparams_from_options(
      self, mock_create_writer
  ):
    mock_writer = mock.Mock()
    mock_create_writer.return_value = mock_writer
    temp_dir = epath.Path(self.create_tempdir().full_path)
    manager = metrics_manager.MetricsManager(
        name='HpSuite',
        num_repeats=1,
        tensorboard_dir=temp_dir,
        enable_per_host_metrics=False,
    )
    m = metric_lib.Metrics()
    m.results['op_0_basics/time_s'] = (1.0, 's')
    manager.add_result(
        'bench1',
        m,
        benchmark_options=_HpOpts(
            async_enabled=False, chunk_byte_size=256, notes='x'
        ),
    )
    manager.generate_report()

    mock_writer.write_hparams.assert_called_once_with(
        {
            'async_enabled': False,
            'chunk_byte_size': 256,
            'notes': 'x',
        }
    )

  @mock.patch(
      'orbax.checkpoint._src.testing.benchmarks.core.metrics_manager.metric_writers.create_default_writer'
  )
  def test_multiple_repeats_write_hparams_once(self, mock_create_writer):
    mock_writer = mock.Mock()
    mock_create_writer.return_value = mock_writer
    temp_dir = epath.Path(self.create_tempdir().full_path)
    manager = metrics_manager.MetricsManager(
        name='HpSuite',
        num_repeats=2,
        tensorboard_dir=temp_dir,
        enable_per_host_metrics=False,
    )
    opts = _HpOpts()
    for _ in range(2):
      m = metric_lib.Metrics()
      m.results['op_0_basics/time_s'] = (1.0, 's')
      manager.add_result('bench1', m, benchmark_options=opts)
    manager.generate_report()
    mock_writer.write_hparams.assert_called_once()

  @mock.patch(
      'orbax.checkpoint._src.testing.benchmarks.core.metrics_manager.metric_writers.create_default_writer'
  )
  def test_no_options_does_not_call_write_hparams(self, mock_create_writer):
    mock_writer = mock.Mock()
    mock_create_writer.return_value = mock_writer
    temp_dir = epath.Path(self.create_tempdir().full_path)
    manager = metrics_manager.MetricsManager(
        name='HpSuite',
        num_repeats=1,
        tensorboard_dir=temp_dir,
        enable_per_host_metrics=False,
    )
    m = metric_lib.Metrics()
    m.results['op_0_basics/time_s'] = (1.0, 's')
    manager.add_result('bench1', m, benchmark_options=None)
    manager.generate_report()
    mock_writer.write_hparams.assert_not_called()

  @mock.patch(
      'orbax.checkpoint._src.testing.benchmarks.core.metrics_manager.metric_writers.create_default_writer'
  )
  def test_none_field_value_serialized_as_string(self, mock_create_writer):
    mock_writer = mock.Mock()
    mock_create_writer.return_value = mock_writer
    temp_dir = epath.Path(self.create_tempdir().full_path)
    manager = metrics_manager.MetricsManager(
        name='HpSuite',
        num_repeats=1,
        tensorboard_dir=temp_dir,
        enable_per_host_metrics=False,
    )
    m = metric_lib.Metrics()
    m.results['op_0_basics/time_s'] = (1.0, 's')
    manager.add_result(
        'bench1', m, benchmark_options=_HpOpts(chunk_byte_size=None)
    )
    manager.generate_report()
    args, _ = mock_writer.write_hparams.call_args
    self.assertEqual(args[0]['chunk_byte_size'], 'None')


if __name__ == '__main__':
  absltest.main()
