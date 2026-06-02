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

"""Tests for JaxMonitoringMetric — subscribing to jax.monitoring during a measure() block."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from orbax.checkpoint._src.testing.benchmarks.core import metric as metric_lib


class JaxMonitoringMetricTest(parameterized.TestCase):

  def tearDown(self):
    # Defensive: ensure no listener survives a test that fails mid-flight.
    jax.monitoring.clear_event_listeners()
    super().tearDown()

  def test_captures_duration_with_known_event_name(self):
    metrics = metric_lib.Metrics()
    with metrics.measure('op', ['jax_monitoring']):
      jax.monitoring.record_event_duration_secs(
          '/jax/orbax/write/blocking_duration_secs', 1.5
      )
    self.assertIn('op_2_save_breakdown/blocking_s', metrics.results)
    value, unit = metrics.results['op_2_save_breakdown/blocking_s']
    self.assertEqual(value, 1.5)
    self.assertEqual(unit, 's')

  def test_captures_scalar_with_known_event_name(self):
    metrics = metric_lib.Metrics()
    with metrics.measure('op', ['jax_monitoring']):
      jax.monitoring.record_scalar(
          '/jax/orbax/write/blocking_gbytes_per_sec', 12.5
      )
    self.assertIn('op_4_throughput/save_blocking_gbps', metrics.results)
    value, unit = metrics.results['op_4_throughput/save_blocking_gbps']
    self.assertEqual(value, 12.5)
    self.assertEqual(unit, 'GiB/s')

  def test_filters_out_events_with_non_orbax_prefix(self):
    metrics = metric_lib.Metrics()
    with metrics.measure('op', ['jax_monitoring']):
      jax.monitoring.record_event_duration_secs('/jax/some/other/event', 7.7)
    capture_keys = [k for k in metrics.results if 'other/event' in k]
    self.assertEqual(capture_keys, [])

  def test_unknown_orbax_event_falls_through_to_default_tag(self):
    metrics = metric_lib.Metrics()
    with metrics.measure('op', ['jax_monitoring']):
      jax.monitoring.record_event_duration_secs(
          '/jax/orbax/write/some_unknown_event_secs', 0.3
      )
    self.assertIn(
        'op_9_other/orbax_write_some_unknown_event_secs',
        metrics.results,
    )

  def test_storage_type_label_ignored_for_tag(self):
    metrics = metric_lib.Metrics()
    with metrics.measure('op', ['jax_monitoring']):
      jax.monitoring.record_event_duration_secs(
          '/jax/orbax/write/blocking_duration_secs', 2.0, storage_type='gs'
      )
    self.assertIn('op_2_save_breakdown/blocking_s', metrics.results)
    value, _ = metrics.results['op_2_save_breakdown/blocking_s']
    self.assertEqual(value, 2.0)

  def test_does_not_capture_after_block_exits(self):
    metrics = metric_lib.Metrics()
    with metrics.measure('op', ['jax_monitoring']):
      pass
    jax.monitoring.record_event_duration_secs(
        '/jax/orbax/write/blocking_duration_secs', 99.0
    )
    leaked = [
        k
        for k in metrics.results
        if 'blocking_s' in k and metrics.results[k][0] == 99.0
    ]
    self.assertEqual(leaked, [])

  def test_two_blocks_do_not_cross_contaminate(self):
    m_a = metric_lib.Metrics()
    m_b = metric_lib.Metrics()
    with m_a.measure('op_a', ['jax_monitoring']):
      jax.monitoring.record_event_duration_secs(
          '/jax/orbax/write/blocking_duration_secs', 1.0
      )
    with m_b.measure('op_b', ['jax_monitoring']):
      jax.monitoring.record_event_duration_secs(
          '/jax/orbax/read/blocking_duration_secs', 2.0
      )
    self.assertEqual(m_a.results['op_a_2_save_breakdown/blocking_s'][0], 1.0)
    self.assertEqual(m_b.results['op_b_3_load_breakdown/blocking_s'][0], 2.0)
    self.assertNotIn('op_b_2_save_breakdown/blocking_s', m_b.results)
    self.assertNotIn('op_a_3_load_breakdown/blocking_s', m_a.results)

  def test_ocdbt_merge_tagged_under_save_breakdown(self):
    metrics = metric_lib.Metrics()
    with metrics.measure('op', ['jax_monitoring']):
      jax.monitoring.record_event_duration_secs(
          '/jax/checkpoint/write/async/ocdbt_merge_duration_secs', 0.7
      )
    self.assertIn('op_2_save_breakdown/ocdbt_merge_s', metrics.results)

  def test_sync_global_devices_tagged_under_overhead(self):
    metrics = metric_lib.Metrics()
    with metrics.measure('op', ['jax_monitoring']):
      jax.monitoring.record_event_duration_secs(
          '/jax/checkpoint/sync_global_devices_duration_sec', 0.05
      )
    self.assertIn(
        'op_7_overhead/sync_global_devices_s',
        metrics.results,
    )

  def test_threaded_delete_tagged_under_overhead(self):
    metrics = metric_lib.Metrics()
    with metrics.measure('op', ['jax_monitoring']):
      jax.monitoring.record_event_duration_secs(
          '/jax/orbax/checkpoint_manager/threaded_checkpoint_deleter/duration',
          0.12,
      )
    self.assertIn('op_7_overhead/delete_threaded_s', metrics.results)

  def test_checkpoint_read_gbytes_per_sec_tagged_under_throughput(self):
    metrics = metric_lib.Metrics()
    with metrics.measure('op', ['jax_monitoring']):
      jax.monitoring.record_scalar('/jax/checkpoint/read/gbytes_per_sec', 18.9)
    self.assertIn('op_4_throughput/load_total_gbps', metrics.results)

  def test_checkpoint_read_gbytes_tagged_under_inventory(self):
    metrics = metric_lib.Metrics()
    with metrics.measure('op', ['jax_monitoring']):
      jax.monitoring.record_scalar('/jax/checkpoint/read/gbytes', 140.2)
    self.assertIn('op_5_inventory/load_total_gb', metrics.results)

  def test_standard_delete_tagged_under_overhead(self):
    metrics = metric_lib.Metrics()
    with metrics.measure('op', ['jax_monitoring']):
      jax.monitoring.record_event_duration_secs(
          '/jax/orbax/checkpoint_manager/standard_checkpoint_deleter/duration',
          0.08,
      )
    self.assertIn('op_7_overhead/delete_standard_s', metrics.results)

  def test_multiple_events_in_one_block_all_captured(self):
    metrics = metric_lib.Metrics()
    with metrics.measure('op', ['jax_monitoring']):
      jax.monitoring.record_event_duration_secs(
          '/jax/orbax/write/blocking_duration_secs', 1.0
      )
      jax.monitoring.record_event_duration_secs(
          '/jax/checkpoint/read/broadcast_duration_secs', 0.5
      )
      jax.monitoring.record_scalar('/jax/orbax/write/gbytes', 100.0)
    self.assertIn('op_2_save_breakdown/blocking_s', metrics.results)
    self.assertIn('op_3_load_breakdown/broadcast_s', metrics.results)
    self.assertIn('op_5_inventory/save_total_gb', metrics.results)


if __name__ == '__main__':
  absltest.main()
