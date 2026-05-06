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

"""Orbax metrics telemetry base."""

from __future__ import annotations

import abc
import logging
import os
import threading
from typing import Any


class MetricRecorder(abc.ABC):
  """Abstract base class for Orbax metric recorders."""

  @abc.abstractmethod
  def record_event(self, metric_name: str, **kwargs: Any) -> None:
    """Records a named event with optional metadata."""
    pass

  @abc.abstractmethod
  def record_scalar(
      self, metric_name: str, value: float | int, **kwargs: Any
  ) -> None:
    """Records a scalar summary value with optional metadata."""
    pass

  @abc.abstractmethod
  def record_duration(
      self, metric_name: str, duration: float | int, **kwargs: Any
  ) -> None:
    """Records an event duration in seconds with optional metadata."""
    pass


_recorders: list[MetricRecorder] = []  # Protected by _init_lock
_initialized = False  # Protected by _init_lock
_init_lock = threading.Lock()


def initialize(recorder: MetricRecorder) -> None:
  """Registers a recorder and binds its methods to JAX monitoring listeners."""
  global _initialized

  with _init_lock:
    _recorders.append(recorder)

    if _initialized:
      return

    from jax import monitoring as jax_monitoring  # pylint: disable=g-import-not-at-top

    def _proxy_record_event(metric_name: str, **kwargs: Any) -> None:
      with _init_lock:
        recorders_snapshot = _recorders[:]
      for r in recorders_snapshot:
        r.record_event(metric_name, **kwargs)

    def _proxy_record_scalar(
        metric_name: str, value: float | int, **kwargs: Any
    ) -> None:
      with _init_lock:
        recorders_snapshot = _recorders[:]
      for r in recorders_snapshot:
        r.record_scalar(metric_name, value, **kwargs)

    def _proxy_record_duration(
        metric_name: str, duration: float | int, **kwargs: Any
    ) -> None:
      with _init_lock:
        recorders_snapshot = _recorders[:]
      for r in recorders_snapshot:
        r.record_duration(metric_name, duration, **kwargs)

    jax_monitoring.register_event_listener(_proxy_record_event)
    jax_monitoring.register_scalar_listener(_proxy_record_scalar)
    jax_monitoring.register_event_duration_secs_listener(_proxy_record_duration)

    _initialized = True
    logging.info('Installed JAX monitoring proxy listeners for Orbax.')


def initialize_from_env() -> None:
  """Initializes monitoring based on environment and build type."""

  if os.environ.get('ENABLE_ORBAX_TELEMETRY', 'false').lower() == 'true':
    try:
      from orbax.checkpoint._src.logging import prometheus_monitoring  # pylint: disable=g-import-not-at-top
      import multiprocessing  # pylint: disable=g-import-not-at-top

      port = (
          9431 if multiprocessing.current_process().name == 'MainProcess' else 0
      )
      initialize(prometheus_monitoring.PrometheusMonitoring(port=port))
    except ImportError as e:
      logging.warning('Failed to import PrometheusMonitoring: %s', e)
