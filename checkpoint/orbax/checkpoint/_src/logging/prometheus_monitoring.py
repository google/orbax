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

"""Orbax metrics telemetry Prometheus implementation."""

from __future__ import annotations

import importlib
import logging
import os
import tempfile
import threading
from typing import Any

from orbax.checkpoint._src.logging import monitoring

# Keep a global reference so the directory is not deleted until the program
# exits.
_prometheus_multiproc_dir = None

if os.environ.get('ENABLE_ORBAX_TELEMETRY', 'false').lower() == 'true':
  if 'PROMETHEUS_MULTIPROC_DIR' not in os.environ:
    import atexit  # pylint: disable=g-import-not-at-top
    import shutil  # pylint: disable=g-import-not-at-top
    # Create a directory for prometheus multiprocessing.
    _prometheus_multiproc_dir = tempfile.mkdtemp(prefix='prometheus_multiproc_')
    os.environ['PROMETHEUS_MULTIPROC_DIR'] = _prometheus_multiproc_dir
    _creator_pid = os.getpid()

    def _Cleanup():
      if os.getpid() == _creator_pid:
        shutil.rmtree(_prometheus_multiproc_dir, ignore_errors=True)

    atexit.register(_Cleanup)

try:
  prometheus_client = importlib.import_module('prometheus_client')

  _prom_counter = prometheus_client.Counter  # pytype: disable=attribute-error
  _prom_histogram = prometheus_client.Histogram  # pytype: disable=attribute-error
except (ImportError, AttributeError):
  prometheus_client = None
  _prom_counter = None
  _prom_histogram = None


class PrometheusMonitoring(monitoring.MetricRecorder):
  """Prometheus implementation of Orbax metric recorder."""

  def __init__(self, port: int = 9431):
    self._initialized = False
    self._metrics = {}
    self._lock = threading.Lock()
    self._port = port
    self._allowed_prefixes = (
        '/jax/orbax/write/',
        '/jax/checkpoint/write/',
        '/jax/orbax/read/',
    )

    if not prometheus_client:
      logging.warning(
          'prometheus-client not found. Orbax metrics will not be reported to'
          ' Prometheus.'
      )
      return

    if port > 0:
      self._start_server(port)
    else:
      # If port is 0, we assume it's a worker process in multiprocess mode,
      # or server is started externally. We mark it initialized so it records.
      self._initialized = True

  def _start_server(self, port: int):
    try:
      multiprocess_started = False
      if 'PROMETHEUS_MULTIPROC_DIR' in os.environ:
        try:
          multiprocess = importlib.import_module(
              'prometheus_client.multiprocess'
          )
          registry = prometheus_client.CollectorRegistry()  # pytype: disable=attribute-error
          multiprocess.MultiProcessCollector(registry)  # pytype: disable=attribute-error
          prometheus_client.start_http_server(port, registry=registry)  # pytype: disable=attribute-error
          logging.info(
              'Prometheus multiprocess metrics server started on port %s.',
              port,
          )
          multiprocess_started = True
        except (ImportError, AttributeError):
          pass

      if not multiprocess_started:
        # Standard single-process server
        prometheus_client.start_http_server(port)  # pytype: disable=attribute-error
        logging.info('Prometheus metrics server started on port %s.', port)
      self._initialized = True
    except (OSError, ValueError) as e:
      # Handle 'already in use' for Linux/macOS and Windows (10048).
      if 'already in use' not in str(e) and '10048' not in str(e):
        logging.warning('Failed to start Prometheus server: %s', e)
        return
      # If the server is already running (e.g. started by Grain), just
      # register listeners.
      logging.info('Prometheus server already active.')
      self._initialized = True

  def _is_allowed(self, metric_name: str) -> bool:
    """Returns True if the metric is allowed for Prometheus export."""
    for prefix in self._allowed_prefixes:
      if metric_name.startswith(prefix):
        return True
    return False

  def record_event(self, metric_name: str, **kwargs: Any) -> None:
    """JAX monitoring handler for events to route to prometheus-client."""
    if (
        not self._initialized
        or not self._is_allowed(metric_name)
        or not _prom_counter
    ):
      return
    metric_name_safe = metric_name.strip('/').replace('/', '_')
    sorted_keys = sorted(kwargs.keys())
    labelnames = tuple(sorted_keys)
    labelvalues = tuple(str(kwargs[k]) for k in sorted_keys)

    if metric_name_safe not in self._metrics:
      with self._lock:
        if metric_name_safe not in self._metrics:
          try:
            self._metrics[metric_name_safe] = _prom_counter(
                metric_name_safe, metric_name, labelnames=labelnames
            )
          except ValueError:
            # pylint: disable=protected-access
            self._metrics[metric_name_safe] = (
                prometheus_client.REGISTRY._names_to_collectors.get(  # pytype: disable=attribute-error
                    metric_name_safe
                )
            )
            # pylint: enable=protected-access

    metric = self._metrics[metric_name_safe]
    if _prom_counter and isinstance(metric, _prom_counter):
      if labelnames:
        metric.labels(*labelvalues).inc()
      else:
        metric.inc()

  def record_scalar(
      self, metric_name: str, value: float | int, **kwargs: Any
  ) -> None:
    """JAX monitoring handler for scalars to route to prometheus-client."""
    if (
        not self._initialized
        or not self._is_allowed(metric_name)
        or not _prom_histogram
    ):
      return
    metric_name_safe = metric_name.strip('/').replace('/', '_')
    sorted_keys = sorted(kwargs.keys())
    labelnames = tuple(sorted_keys)
    labelvalues = tuple(str(kwargs[k]) for k in sorted_keys)

    if metric_name_safe not in self._metrics:
      with self._lock:
        if metric_name_safe not in self._metrics:
          try:
            self._metrics[metric_name_safe] = _prom_histogram(
                metric_name_safe, metric_name, labelnames=labelnames
            )
          except ValueError:
            # pylint: disable=protected-access
            self._metrics[metric_name_safe] = (
                prometheus_client.REGISTRY._names_to_collectors.get(  # pytype: disable=attribute-error
                    metric_name_safe
                )
            )
            # pylint: enable=protected-access

    metric = self._metrics[metric_name_safe]
    if _prom_histogram and isinstance(metric, _prom_histogram):
      if labelnames:
        metric.labels(*labelvalues).observe(value)
      else:
        metric.observe(value)

  def record_duration(
      self, metric_name: str, duration: float | int, **kwargs: Any
  ) -> None:
    """JAX monitoring handler for duration to route to prometheus-client."""
    if (
        not self._initialized
        or not self._is_allowed(metric_name)
        or not _prom_histogram
    ):
      return
    metric_name_safe = metric_name.strip('/').replace('/', '_')
    sorted_keys = sorted(kwargs.keys())
    labelnames = tuple(sorted_keys)
    labelvalues = tuple(str(kwargs[k]) for k in sorted_keys)

    if metric_name_safe not in self._metrics:
      with self._lock:
        if metric_name_safe not in self._metrics:
          try:
            self._metrics[metric_name_safe] = _prom_histogram(
                metric_name_safe, metric_name, labelnames=labelnames
            )
          except ValueError:
            # pylint: disable=protected-access
            self._metrics[metric_name_safe] = (
                prometheus_client.REGISTRY._names_to_collectors.get(  # pytype: disable=attribute-error
                    metric_name_safe
                )
            )
            # pylint: enable=protected-access

    metric = self._metrics[metric_name_safe]
    if _prom_histogram and isinstance(metric, _prom_histogram):
      if labelnames:
        metric.labels(*labelvalues).observe(duration)
      else:
        metric.observe(duration)
