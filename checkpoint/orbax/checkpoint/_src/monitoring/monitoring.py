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

"""Orbax metrics telemetry."""

from __future__ import annotations

# pylint: disable=g-import-not-at-top
# pylint: disable=invalid-name

import logging
import threading

from jax import monitoring

try:
  import prometheus_client  # pytype: disable=import-error

  _prom_counter = prometheus_client.Counter
  _prom_histogram = prometheus_client.Histogram
except (ImportError, AttributeError):
  prometheus_client = None
  _prom_counter = None
  _prom_histogram = None

_USE_PROMETHEUS = True

_initialized = False
_metrics = {}  # name -> metric object
_lock = threading.Lock()

_PROMETHEUS_ALLOWED_METRIC_PREFIXES = (
    '/jax/orbax/write/',
    '/jax/checkpoint/write/',
    '/jax/orbax/read/',
)


def _is_allowed(metric_name: str) -> bool:
  """Returns True if the metric is allowed for Prometheus export."""
  for prefix in _PROMETHEUS_ALLOWED_METRIC_PREFIXES:
    if metric_name.startswith(prefix):
      return True
  return False


def _record_event(metric_name: str, **kwargs):
  """JAX monitoring handler for events to route to prometheus-client."""
  if not _initialized or not _is_allowed(metric_name) or not _prom_counter:
    return
  metric_name_safe = metric_name.strip('/').replace('/', '_')
  sorted_keys = sorted(kwargs.keys())
  labelnames = tuple(sorted_keys)
  labelvalues = tuple(str(kwargs[k]) for k in sorted_keys)

  if metric_name_safe not in _metrics:
    with _lock:
      if metric_name_safe not in _metrics:
        _metrics[metric_name_safe] = _prom_counter(
            metric_name_safe, metric_name, labelnames=labelnames
        )

  metric = _metrics[metric_name_safe]
  if _prom_counter and isinstance(metric, _prom_counter):
    if labelnames:
      metric.labels(*labelvalues).inc()
    else:
      metric.inc()


def _record_scalar(metric_name: str, value: float | int, **kwargs):
  """JAX monitoring handler for scalars to route to prometheus-client."""
  if not _initialized or not _is_allowed(metric_name) or not _prom_histogram:
    return
  metric_name_safe = metric_name.strip('/').replace('/', '_')
  sorted_keys = sorted(kwargs.keys())
  labelnames = tuple(sorted_keys)
  labelvalues = tuple(str(kwargs[k]) for k in sorted_keys)

  if metric_name_safe not in _metrics:
    with _lock:
      if metric_name_safe not in _metrics:
        _metrics[metric_name_safe] = _prom_histogram(
            metric_name_safe, metric_name, labelnames=labelnames
        )

  metric = _metrics[metric_name_safe]
  if _prom_histogram and isinstance(metric, _prom_histogram):
    if labelnames:
      metric.labels(*labelvalues).observe(value)
    else:
      metric.observe(value)


def _record_duration(metric_name: str, duration: float | int, **kwargs):
  """JAX monitoring handler for duration to route to prometheus-client."""
  if not _initialized or not _is_allowed(metric_name) or not _prom_histogram:
    return
  metric_name_safe = metric_name.strip('/').replace('/', '_')
  sorted_keys = sorted(kwargs.keys())
  labelnames = tuple(sorted_keys)
  labelvalues = tuple(str(kwargs[k]) for k in sorted_keys)

  if metric_name_safe not in _metrics:
    with _lock:
      if metric_name_safe not in _metrics:
        _metrics[metric_name_safe] = _prom_histogram(
            metric_name_safe, metric_name, labelnames=labelnames
        )

  metric = _metrics[metric_name_safe]
  if _prom_histogram and isinstance(metric, _prom_histogram):
    if labelnames:
      metric.labels(*labelvalues).observe(duration)
    else:
      metric.observe(duration)


def initialize(port=9431):
  """Initializes Orbax metric reporting."""
  global _initialized
  if _initialized:
    return
  if not _USE_PROMETHEUS:
    return

  if not prometheus_client:
    logging.warning(
        'prometheus-client not found. Orbax metrics will not be reported.'
    )
    return

  with _lock:
    if _initialized:
      return
    try:
      if port > 0:
        prometheus_client.start_http_server(port)
        logging.info('Prometheus metrics server started on port %s.', port)
      _initialized = True
      monitoring.register_event_listener(_record_event)
      monitoring.register_scalar_listener(_record_scalar)
      monitoring.register_event_duration_secs_listener(_record_duration)
      logging.info('Installed JAX monitoring listeners for Prometheus.')
    except (OSError, ValueError) as e:
      # Handle 'already in use' for Linux/macOS and Windows (10048).
      if 'already in use' in str(e) or '10048' in str(e):
        # If the server is already running (e.g. started by Grain), just
        # register listeners.
        _initialized = True
        monitoring.register_event_listener(_record_event)
        monitoring.register_scalar_listener(_record_scalar)
        monitoring.register_event_duration_secs_listener(_record_duration)
        logging.info('Prometheus server already active. Listeners installed.')
      else:
        logging.warning('Failed to initialize Prometheus metrics: %s', e)
