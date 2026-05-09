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

"""Tests for Orbax Prometheus metrics telemetry."""

from unittest import mock

from absl.testing import absltest
from jax import monitoring
from orbax.checkpoint import monitoring as orbax_monitoring
import prometheus_client


class PrometheusMetricsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # Use mock.patch to reset internal state without production hooks.
    # This is standard Google practice for keeping production code clean.
    self.enter_context(
        mock.patch.object(orbax_monitoring, '_initialized', False)
    )
    self.enter_context(mock.patch.object(orbax_monitoring, '_metrics', {}))
    monitoring.clear_event_listeners()

    # Clear registry for hermetic tests.
    registry = prometheus_client.REGISTRY
    # pylint: disable=protected-access
    if hasattr(registry, '_collector_to_names'):
      for collector in list(registry._collector_to_names):
        registry.unregister(collector)
    # pylint: enable=protected-access

  def test_initialize(self):
    with mock.patch.object(orbax_monitoring, '_USE_PROMETHEUS', True):
      orbax_monitoring.initialize(port=0)
      # Verify behavior (e.g. listeners registered)
      monitoring.record_scalar('/jax/orbax/test_init', 1)

  def test_multiple_initializations(self):
    with mock.patch.object(orbax_monitoring, '_USE_PROMETHEUS', True):
      orbax_monitoring.initialize(port=0)
      # The second initialization shouldn't throw any errors or wipe states.
      orbax_monitoring.initialize(port=0)

  def test_record_before_initialize(self):
    with mock.patch.object(orbax_monitoring, '_USE_PROMETHEUS', True):
      monitoring.record_scalar('/jax/orbax/test_scalar_early', 123)
      monitoring.record_event_duration_secs(
          '/jax/orbax/test_duration_early', 0.5
      )
      monitoring.record_event('/jax/orbax/test_event_early')
      # No metrics should be registered in prometheus if not initialized.
      self.assertIsNone(
          prometheus_client.REGISTRY.get_sample_value(
              'jax_orbax_test_scalar_early'
          )
      )

  def test_handler_scalar_metric(self):
    with mock.patch.object(
        orbax_monitoring, '_USE_PROMETHEUS', True
    ), mock.patch.object(
        orbax_monitoring,
        '_PROMETHEUS_ALLOWED_METRICS',
        {'/jax/orbax/test_scalar'},
    ):
      orbax_monitoring.initialize(port=0)
      monitoring.record_scalar('/jax/orbax/test_scalar', 123)
      metric_name = 'jax_orbax_test_scalar'
      self.assertEqual(
          prometheus_client.REGISTRY.get_sample_value(metric_name), 123.0
      )

  def test_scalar_metric_updates(self):
    with mock.patch.object(
        orbax_monitoring, '_USE_PROMETHEUS', True
    ), mock.patch.object(
        orbax_monitoring,
        '_PROMETHEUS_ALLOWED_METRICS',
        {'/jax/orbax/test_scalar'},
    ):
      orbax_monitoring.initialize(port=0)
      monitoring.record_scalar('/jax/orbax/test_scalar', 123)
      monitoring.record_scalar('/jax/orbax/test_scalar', 456)
      metric_name = 'jax_orbax_test_scalar'
      self.assertEqual(
          prometheus_client.REGISTRY.get_sample_value(metric_name), 456.0
      )

  def test_handler_duration_metric(self):
    with mock.patch.object(
        orbax_monitoring, '_USE_PROMETHEUS', True
    ), mock.patch.object(
        orbax_monitoring,
        '_PROMETHEUS_ALLOWED_METRICS',
        {'/jax/orbax/test_duration_secs'},
    ):
      orbax_monitoring.initialize(port=0)
      monitoring.record_event_duration_secs(
          '/jax/orbax/test_duration_secs', 0.5
      )
      metric_name = 'jax_orbax_test_duration_secs'
      self.assertEqual(
          prometheus_client.REGISTRY.get_sample_value(metric_name + '_count'),
          1.0,
      )
      self.assertEqual(
          prometheus_client.REGISTRY.get_sample_value(metric_name + '_sum'), 0.5
      )

  def test_duration_metric_updates(self):
    with mock.patch.object(
        orbax_monitoring, '_USE_PROMETHEUS', True
    ), mock.patch.object(
        orbax_monitoring,
        '_PROMETHEUS_ALLOWED_METRICS',
        {'/jax/orbax/test_duration_secs'},
    ):
      orbax_monitoring.initialize(port=0)
      monitoring.record_event_duration_secs(
          '/jax/orbax/test_duration_secs', 0.5
      )
      monitoring.record_event_duration_secs(
          '/jax/orbax/test_duration_secs', 1.5
      )
      metric_name = 'jax_orbax_test_duration_secs'
      self.assertEqual(
          prometheus_client.REGISTRY.get_sample_value(metric_name + '_count'),
          2.0,
      )
      self.assertEqual(
          prometheus_client.REGISTRY.get_sample_value(metric_name + '_sum'), 2.0
      )

  def test_handler_event_metric(self):
    with mock.patch.object(
        orbax_monitoring, '_USE_PROMETHEUS', True
    ), mock.patch.object(
        orbax_monitoring,
        '_PROMETHEUS_ALLOWED_METRICS',
        {'/jax/orbax/test_event'},
    ):
      orbax_monitoring.initialize(port=0)
      monitoring.record_event('/jax/orbax/test_event')
      metric_name = 'jax_orbax_test_event_total'
      self.assertEqual(
          prometheus_client.REGISTRY.get_sample_value(metric_name), 1.0
      )

  def test_event_metric_increments(self):
    with mock.patch.object(
        orbax_monitoring, '_USE_PROMETHEUS', True
    ), mock.patch.object(
        orbax_monitoring,
        '_PROMETHEUS_ALLOWED_METRICS',
        {'/jax/orbax/test_event'},
    ):
      orbax_monitoring.initialize(port=0)
      monitoring.record_event('/jax/orbax/test_event')
      monitoring.record_event('/jax/orbax/test_event')
      monitoring.record_event('/jax/orbax/test_event')
      metric_name = 'jax_orbax_test_event_total'
      self.assertEqual(
          prometheus_client.REGISTRY.get_sample_value(metric_name), 3.0
      )

  def test_ignore_unrelated_metrics(self):
    with mock.patch.object(orbax_monitoring, '_USE_PROMETHEUS', True):
      orbax_monitoring.initialize(port=0)
      monitoring.record_scalar('/jax/compilation/time', 123)
      metric_name = 'jax_compilation_time'
      self.assertIsNone(
          prometheus_client.REGISTRY.get_sample_value(metric_name)
      )

  def test_handler_second_prefix(self):
    with mock.patch.object(
        orbax_monitoring, '_USE_PROMETHEUS', True
    ), mock.patch.object(
        orbax_monitoring,
        '_PROMETHEUS_ALLOWED_METRICS',
        {'/jax/checkpoint/test_scalar'},
    ):
      orbax_monitoring.initialize(port=0)
      monitoring.record_scalar('/jax/checkpoint/test_scalar', 123)
      metric_name = 'jax_checkpoint_test_scalar'
      self.assertEqual(
          prometheus_client.REGISTRY.get_sample_value(metric_name), 123.0
      )


if __name__ == '__main__':
  absltest.main()
