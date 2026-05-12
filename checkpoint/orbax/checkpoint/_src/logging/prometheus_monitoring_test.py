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

import os
import shutil
import tempfile
from unittest import mock

from absl.testing import absltest
from jax import monitoring as jax_monitoring
from orbax.checkpoint._src.logging import monitoring


class MonitoringTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    monitoring._recorders = []
    monitoring._initialized = False

    self.mock_register_event = self.enter_context(
        mock.patch.object(jax_monitoring, 'register_event_listener')
    )
    self.mock_register_scalar = self.enter_context(
        mock.patch.object(jax_monitoring, 'register_scalar_listener')
    )
    self.mock_register_duration = self.enter_context(
        mock.patch.object(
            jax_monitoring, 'register_event_duration_secs_listener'
        )
    )

  def test_proxy_initialization(self):
    fake_recorder = mock.create_autospec(monitoring.MetricRecorder)
    monitoring.initialize(fake_recorder)

    self.assertTrue(monitoring._initialized)
    self.assertIn(fake_recorder, monitoring._recorders)

    self.mock_register_event.assert_called_once()
    self.mock_register_scalar.assert_called_once()
    self.mock_register_duration.assert_called_once()

  def test_proxy_forwarding(self):
    fake_recorder = mock.create_autospec(monitoring.MetricRecorder)
    monitoring.initialize(fake_recorder)

    proxy_event_fn = self.mock_register_event.call_args[0][0]
    proxy_scalar_fn = self.mock_register_scalar.call_args[0][0]
    proxy_duration_fn = self.mock_register_duration.call_args[0][0]

    proxy_event_fn('test_event', foo='bar')
    fake_recorder.record_event.assert_called_once_with('test_event', foo='bar')

    proxy_scalar_fn('test_scalar', 123, bar='baz')
    fake_recorder.record_scalar.assert_called_once_with(
        'test_scalar', 123, bar='baz'
    )

    proxy_duration_fn('test_duration', 0.5, baz='qux')
    fake_recorder.record_duration.assert_called_once_with(
        'test_duration', 0.5, baz='qux'
    )

  def test_multiple_recorders(self):
    r1 = mock.create_autospec(monitoring.MetricRecorder)
    r2 = mock.create_autospec(monitoring.MetricRecorder)

    monitoring.initialize(r1)
    monitoring.initialize(r2)

    self.assertLen(monitoring._recorders, 2)

    # Proxy should only be registered once
    self.mock_register_event.assert_called_once()

    proxy_event_fn = self.mock_register_event.call_args[0][0]
    proxy_event_fn('test_event')

    r1.record_event.assert_called_once_with('test_event')
    r2.record_event.assert_called_once_with('test_event')


class PrometheusMonitoringTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    # Set environment variable before importing prometheus_monitoring
    self.enter_context(
        mock.patch.dict(
            os.environ, {'ENABLE_ORBAX_PROMETHEUS_TELEMETRY': 'true'}
        )
    )

    # Ensure PROMETHEUS_MULTIPROC_DIR is set for all tests, as module-level
    # initialization might only run once due to module caching.
    if 'PROMETHEUS_MULTIPROC_DIR' not in os.environ:
      temp_dir = tempfile.mkdtemp(prefix='prometheus_multiproc_test_')
      os.environ['PROMETHEUS_MULTIPROC_DIR'] = temp_dir
      self.addCleanup(shutil.rmtree, temp_dir, ignore_errors=True)

    # pylint: disable=g-import-not-at-top
    from orbax.checkpoint._src.logging import prometheus_monitoring
    import prometheus_client
    # pylint: enable=g-import-not-at-top

    self.prometheus_monitoring = prometheus_monitoring
    self.prometheus_client = prometheus_client

    registry = self.prometheus_client.REGISTRY
    if hasattr(registry, '_collector_to_names'):
      # pylint: disable=protected-access
      for collector in list(registry._collector_to_names):
        registry.unregister(collector)
      # pylint: enable=protected-access

  def test_initialize_server_called(self):
    if self.prometheus_client is None:
      self.skipTest('prometheus-client not installed')

    # Remove PROMETHEUS_MULTIPROC_DIR to test standard single-process server.
    self.enter_context(mock.patch.dict(os.environ))
    if 'PROMETHEUS_MULTIPROC_DIR' in os.environ:
      del os.environ['PROMETHEUS_MULTIPROC_DIR']

    with mock.patch.object(
        self.prometheus_client, 'start_http_server', autospec=True
    ) as mock_start:
      _ = self.prometheus_monitoring.PrometheusMonitoring(port=9431)
      mock_start.assert_called_once_with(9431)

  def test_handler_scalar_metric(self):
    pm = self.prometheus_monitoring.PrometheusMonitoring(port=0)
    pm.record_scalar('/jax/orbax/write/test_scalar', 123)
    metric_name = 'jax_orbax_write_test_scalar'
    self.assertEqual(
        self.prometheus_client.REGISTRY.get_sample_value(metric_name + '_sum'),
        123.0,
    )

  def test_ignore_unrelated_metrics(self):
    pm = self.prometheus_monitoring.PrometheusMonitoring(port=0)
    pm.record_scalar('/jax/compilation/time', 123)
    metric_name = 'jax_compilation_time'
    self.assertIsNone(
        self.prometheus_client.REGISTRY.get_sample_value(metric_name)
    )

  def test_labels(self):
    pm = self.prometheus_monitoring.PrometheusMonitoring(port=0)
    pm.record_event('/jax/orbax/write/test_event_label', key1='val1')
    metric_name = 'jax_orbax_write_test_event_label_total'
    self.assertEqual(
        self.prometheus_client.REGISTRY.get_sample_value(
            metric_name, {'key1': 'val1'}
        ),
        1.0,
    )


if __name__ == '__main__':
  absltest.main()
