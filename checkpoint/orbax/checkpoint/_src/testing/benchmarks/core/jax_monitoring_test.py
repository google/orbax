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
    self.assertIn('op::2_save_breakdown/blocking_s', metrics.results)
    value, unit = metrics.results['op::2_save_breakdown/blocking_s']
    self.assertEqual(value, 1.5)
    self.assertEqual(unit, 's')

  def test_captures_scalar_with_known_event_name(self):
    metrics = metric_lib.Metrics()
    with metrics.measure('op', ['jax_monitoring']):
      jax.monitoring.record_scalar(
          '/jax/orbax/write/blocking_gbytes_per_sec', 12.5
      )
    self.assertIn('op::4_throughput/save_blocking_gbps', metrics.results)
    value, unit = metrics.results['op::4_throughput/save_blocking_gbps']
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
        'op::9_other/orbax_write_some_unknown_event_secs',
        metrics.results,
    )

  def test_storage_type_label_ignored_for_tag(self):
    metrics = metric_lib.Metrics()
    with metrics.measure('op', ['jax_monitoring']):
      jax.monitoring.record_event_duration_secs(
          '/jax/orbax/write/blocking_duration_secs', 2.0, storage_type='gs'
      )
    self.assertIn('op::2_save_breakdown/blocking_s', metrics.results)
    value, _ = metrics.results['op::2_save_breakdown/blocking_s']
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
    self.assertEqual(m_a.results['op_a::2_save_breakdown/blocking_s'][0], 1.0)
    self.assertEqual(m_b.results['op_b::3_load_breakdown/blocking_s'][0], 2.0)
    self.assertNotIn('op_b::2_save_breakdown/blocking_s', m_b.results)
    self.assertNotIn('op_a::3_load_breakdown/blocking_s', m_a.results)

  def test_opt_out_when_not_in_metric_keys(self):
    metrics = metric_lib.Metrics()
    with metrics.measure('op', ['time']):
      jax.monitoring.record_event_duration_secs(
          '/jax/orbax/write/blocking_duration_secs', 1.5
      )
    self.assertFalse(any('save_breakdown' in k for k in metrics.results))

  def test_nested_blocks_each_capture(self):
    # jax.monitoring fans out to every registered listener, so an event inside
    # the inner block is recorded by the inner AND the enclosing outer block.
    metrics = metric_lib.Metrics()
    with metrics.measure('outer', ['jax_monitoring']):
      with metrics.measure('inner', ['jax_monitoring']):
        jax.monitoring.record_event_duration_secs(
            '/jax/orbax/read/blocking_duration_secs', 1.5
        )
    self.assertEqual(
        metrics.results['inner::3_load_breakdown/blocking_s'][0], 1.5
    )
    self.assertEqual(
        metrics.results['outer::3_load_breakdown/blocking_s'][0], 1.5
    )

  def test_async_completion_pools_into_same_name(self):
    # Background save: bracket the deferred completion in a second same-named
    # block; the distinct breakdown tags pool into the one "save" card.
    metrics = metric_lib.Metrics()
    with metrics.measure('save', ['jax_monitoring']):
      jax.monitoring.record_event_duration_secs(
          '/jax/orbax/write/blocking_duration_secs', 2.0
      )
    with metrics.measure('save', ['jax_monitoring']):
      jax.monitoring.record_event_duration_secs(
          '/jax/checkpoint/write/async/total_duration_secs', 9.0
      )
    self.assertEqual(
        metrics.results['save::2_save_breakdown/blocking_s'][0], 2.0
    )
    self.assertEqual(
        metrics.results['save::2_save_breakdown/total_async_s'][0], 9.0
    )

  def test_ocdbt_merge_tagged_under_save_breakdown(self):
    metrics = metric_lib.Metrics()
    with metrics.measure('op', ['jax_monitoring']):
      jax.monitoring.record_event_duration_secs(
          '/jax/checkpoint/write/async/ocdbt_merge_duration_secs', 0.7
      )
    self.assertIn('op::2_save_breakdown/ocdbt_merge_s', metrics.results)

  def test_sync_global_devices_tagged_under_overhead(self):
    metrics = metric_lib.Metrics()
    with metrics.measure('op', ['jax_monitoring']):
      jax.monitoring.record_event_duration_secs(
          '/jax/checkpoint/sync_global_devices_duration_sec', 0.05
      )
    self.assertIn(
        'op::7_overhead/sync_global_devices_s',
        metrics.results,
    )

  def test_threaded_delete_tagged_under_overhead(self):
    metrics = metric_lib.Metrics()
    with metrics.measure('op', ['jax_monitoring']):
      jax.monitoring.record_event_duration_secs(
          '/jax/orbax/checkpoint_manager/threaded_checkpoint_deleter/duration',
          0.12,
      )
    self.assertIn('op::7_overhead/delete_threaded_s', metrics.results)

  def test_checkpoint_read_gbytes_per_sec_tagged_under_throughput(self):
    metrics = metric_lib.Metrics()
    with metrics.measure('op', ['jax_monitoring']):
      jax.monitoring.record_scalar('/jax/checkpoint/read/gbytes_per_sec', 18.9)
    self.assertIn('op::4_throughput/load_total_gbps', metrics.results)

  def test_checkpoint_read_gbytes_tagged_under_inventory(self):
    metrics = metric_lib.Metrics()
    with metrics.measure('op', ['jax_monitoring']):
      jax.monitoring.record_scalar('/jax/checkpoint/read/gbytes', 140.2)
    self.assertIn('op::5_inventory/load_total_gb', metrics.results)

  def test_standard_delete_tagged_under_overhead(self):
    metrics = metric_lib.Metrics()
    with metrics.measure('op', ['jax_monitoring']):
      jax.monitoring.record_event_duration_secs(
          '/jax/orbax/checkpoint_manager/standard_checkpoint_deleter/duration',
          0.08,
      )
    self.assertIn('op::7_overhead/delete_standard_s', metrics.results)

  def test_compile_cache_hits_counted(self):
    metrics = metric_lib.Metrics()
    with metrics.measure('op', ['jax_monitoring']):
      for _ in range(3):
        jax.monitoring.record_event('/jax/compilation_cache/cache_hits')
    self.assertIn('op::8_jax/cache_hits_count', metrics.results)
    self.assertEqual(metrics.results['op::8_jax/cache_hits_count'][0], 3)

  def test_compile_cache_misses_counted(self):
    metrics = metric_lib.Metrics()
    with metrics.measure('op', ['jax_monitoring']):
      jax.monitoring.record_event('/jax/compilation_cache/cache_misses')
      jax.monitoring.record_event('/jax/compilation_cache/cache_misses')
    self.assertEqual(metrics.results['op::8_jax/cache_misses_count'][0], 2)

  def test_compile_cache_hit_rate_derived(self):
    metrics = metric_lib.Metrics()
    with metrics.measure('op', ['jax_monitoring']):
      for _ in range(3):
        jax.monitoring.record_event('/jax/compilation_cache/cache_hits')
      jax.monitoring.record_event('/jax/compilation_cache/cache_misses')
    self.assertIn('op::8_jax/cache_hit_rate', metrics.results)
    rate, _ = metrics.results['op::8_jax/cache_hit_rate']
    self.assertAlmostEqual(rate, 0.75)

  def test_compile_cache_durations_mapped(self):
    metrics = metric_lib.Metrics()
    with metrics.measure('op', ['jax_monitoring']):
      jax.monitoring.record_event_duration_secs(
          '/jax/compilation_cache/cache_retrieval_time_sec', 0.42
      )
      jax.monitoring.record_event_duration_secs(
          '/jax/compilation_cache/compile_time_saved_sec', 1.7
      )
    self.assertEqual(
        metrics.results['op::8_jax/cache_retrieval_s'], (0.42, 's')
    )
    self.assertEqual(
        metrics.results['op::8_jax/compile_time_saved_s'], (1.7, 's')
    )

  def test_unmapped_discrete_events_are_dropped(self):
    metrics = metric_lib.Metrics()
    with metrics.measure('op', ['jax_monitoring']):
      jax.monitoring.record_event('/jax/orbax/write/start')
      jax.monitoring.record_event('/jax/orbax/write/success')
      jax.monitoring.record_event('/jax/orbax/checkpointer/init')
    bloat = [k for k in metrics.results if '9_other' in k]
    self.assertEqual(bloat, [], msg=f'unmapped events leaked: {bloat}')

  def test_compile_cache_events_outside_prefix_ignored(self):
    metrics = metric_lib.Metrics()
    with metrics.measure('op', ['jax_monitoring']):
      jax.monitoring.record_event('/jax/some/other/event')
    leaked = [k for k in metrics.results if 'other_event' in k]
    self.assertEqual(leaked, [])

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
    self.assertIn('op::2_save_breakdown/blocking_s', metrics.results)
    self.assertIn('op::3_load_breakdown/broadcast_s', metrics.results)
    self.assertIn('op::5_inventory/save_total_gb', metrics.results)


if __name__ == '__main__':
  absltest.main()
