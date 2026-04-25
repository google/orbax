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

from __future__ import annotations

from unittest import mock

from absl.testing import absltest
from jax import monitoring
from orbax.checkpoint._src.monitoring import monitoring as orbax_monitoring
import prometheus_client


class PrometheusMetricsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.enter_context(
        mock.patch.object(orbax_monitoring, '_initialized', False)
    )
    self.enter_context(mock.patch.object(orbax_monitoring, '_metrics', {}))

    # Mock JAX monitoring registration to prevent accumulating listeners across
    # tests.
    self.enter_context(mock.patch.object(monitoring, 'register_event_listener'))
    self.enter_context(
        mock.patch.object(monitoring, 'register_scalar_listener')
    )
    self.enter_context(
        mock.patch.object(monitoring, 'register_event_duration_secs_listener')
    )

    # Clear registry for hermetic tests.
    registry = prometheus_client.REGISTRY
    # pylint: disable=protected-access
    if hasattr(registry, '_collector_to_names'):
      for collector in list(registry._collector_to_names):
        registry.unregister(collector)
    # pylint: enable=protected-access

  def test_initialize_prometheus_server_called_once(self):
    if prometheus_client is None:
      self.skipTest('prometheus-client not installed')
    with mock.patch.object(
        prometheus_client, 'start_http_server', autospec=True
    ) as mock_start_http_server:
      with mock.patch.object(orbax_monitoring, '_USE_PROMETHEUS', True):
        orbax_monitoring.initialize(port=9431)
        mock_start_http_server.assert_called_once_with(9431)
        orbax_monitoring.initialize(port=9431)
        mock_start_http_server.assert_called_once_with(9431)

  def test_initialize(self):
    with mock.patch.object(orbax_monitoring, '_USE_PROMETHEUS', True):
      orbax_monitoring.initialize(port=0)
      monitoring.register_scalar_listener.assert_called_once()  # pytype: disable=attribute-error

  def test_multiple_initializations(self):
    with mock.patch.object(orbax_monitoring, '_USE_PROMETHEUS', True):
      orbax_monitoring.initialize(port=0)
      orbax_monitoring.initialize(port=0)

  def test_record_before_initialize(self):
    with mock.patch.object(orbax_monitoring, '_USE_PROMETHEUS', True):
      orbax_monitoring._record_scalar('/jax/orbax/write/test_scalar_early', 123)
      orbax_monitoring._record_duration(
          '/jax/orbax/write/test_duration_early', 0.5
      )
      orbax_monitoring._record_event('/jax/orbax/write/test_event_early')
      self.assertIsNone(
          prometheus_client.REGISTRY.get_sample_value(
              'jax_orbax_write_test_scalar_early_sum'
          )
      )

  def test_handler_scalar_metric(self):
    with mock.patch.object(orbax_monitoring, '_USE_PROMETHEUS', True):
      orbax_monitoring.initialize(port=0)
      orbax_monitoring._record_scalar('/jax/orbax/write/test_scalar', 123)
      metric_name = 'jax_orbax_write_test_scalar'
      self.assertEqual(
          prometheus_client.REGISTRY.get_sample_value(metric_name + '_sum'),
          123.0,
      )
      self.assertEqual(
          prometheus_client.REGISTRY.get_sample_value(metric_name + '_count'),
          1.0,
      )

  def test_scalar_metric_updates(self):
    with mock.patch.object(orbax_monitoring, '_USE_PROMETHEUS', True):
      orbax_monitoring.initialize(port=0)
      orbax_monitoring._record_scalar('/jax/orbax/write/test_scalar', 123)
      orbax_monitoring._record_scalar('/jax/orbax/write/test_scalar', 456)
      metric_name = 'jax_orbax_write_test_scalar'
      self.assertEqual(
          prometheus_client.REGISTRY.get_sample_value(metric_name + '_sum'),
          579.0,
      )

  def test_handler_duration_metric(self):
    with mock.patch.object(orbax_monitoring, '_USE_PROMETHEUS', True):
      orbax_monitoring.initialize(port=0)
      orbax_monitoring._record_duration(
          '/jax/orbax/write/test_duration_secs', 0.5
      )
      metric_name = 'jax_orbax_write_test_duration_secs'
      self.assertEqual(
          prometheus_client.REGISTRY.get_sample_value(metric_name + '_count'),
          1.0,
      )
      self.assertEqual(
          prometheus_client.REGISTRY.get_sample_value(metric_name + '_sum'), 0.5
      )

  def test_duration_metric_updates(self):
    with mock.patch.object(orbax_monitoring, '_USE_PROMETHEUS', True):
      orbax_monitoring.initialize(port=0)
      orbax_monitoring._record_duration(
          '/jax/orbax/write/test_duration_secs', 0.5
      )
      orbax_monitoring._record_duration(
          '/jax/orbax/write/test_duration_secs', 1.5
      )
      metric_name = 'jax_orbax_write_test_duration_secs'
      self.assertEqual(
          prometheus_client.REGISTRY.get_sample_value(metric_name + '_count'),
          2.0,
      )

  def test_handler_event_metric(self):
    with mock.patch.object(orbax_monitoring, '_USE_PROMETHEUS', True):
      orbax_monitoring.initialize(port=0)
      orbax_monitoring._record_event('/jax/orbax/write/test_event')
      metric_name = 'jax_orbax_write_test_event_total'
      self.assertEqual(
          prometheus_client.REGISTRY.get_sample_value(metric_name), 1.0
      )

  def test_event_metric_increments(self):
    with mock.patch.object(orbax_monitoring, '_USE_PROMETHEUS', True):
      orbax_monitoring.initialize(port=0)
      orbax_monitoring._record_event('/jax/orbax/write/test_event')
      orbax_monitoring._record_event('/jax/orbax/write/test_event')
      orbax_monitoring._record_event('/jax/orbax/write/test_event')
      metric_name = 'jax_orbax_write_test_event_total'
      self.assertEqual(
          prometheus_client.REGISTRY.get_sample_value(metric_name), 3.0
      )

  def test_ignore_unrelated_metrics(self):
    with mock.patch.object(orbax_monitoring, '_USE_PROMETHEUS', True):
      orbax_monitoring.initialize(port=0)
      orbax_monitoring._record_scalar('/jax/compilation/time', 123)
      metric_name = 'jax_compilation_time'
      self.assertIsNone(
          prometheus_client.REGISTRY.get_sample_value(metric_name)
      )

  def test_handler_second_prefix(self):
    with mock.patch.object(orbax_monitoring, '_USE_PROMETHEUS', True):
      orbax_monitoring.initialize(port=0)
      orbax_monitoring._record_scalar('/jax/checkpoint/write/test_scalar', 123)
      metric_name = 'jax_checkpoint_write_test_scalar'
      self.assertEqual(
          prometheus_client.REGISTRY.get_sample_value(metric_name + '_sum'),
          123.0,
      )

  def test_labels(self):
    with mock.patch.object(orbax_monitoring, '_USE_PROMETHEUS', True):
      orbax_monitoring.initialize(port=0)
      orbax_monitoring._record_event(
          '/jax/orbax/write/test_event_label', key1='val1'
      )
      metric_name = 'jax_orbax_write_test_event_label_total'
      self.assertEqual(
          prometheus_client.REGISTRY.get_sample_value(
              metric_name, {'key1': 'val1'}
          ),
          1.0,
      )


if __name__ == '__main__':
  absltest.main()
