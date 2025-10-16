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

"""Metric classes for benchmarking."""

from collections.abc import MutableMapping
import contextlib
import dataclasses
import linecache  # To show the source code line
import os
import threading
import time
import tracemalloc
from typing import Any

from absl import logging
from orbax.checkpoint._src.multihost import multihost
import psutil


class Metric:
  """Base class for metric context managers."""

  def __init__(self, name: str):
    self.name = name
    self.result = None
    self.unit = None

  def _start_log(self):
    logging.info(
        "[process_id=%s] Starting metric: '%s'...",
        multihost.process_index(),
        self.name,
    )

  def _record(self, value: Any, unit: str):
    """Records metric value and logs completion."""
    self.result = value
    self.unit = unit
    logging.info(
        "[process_id=%s] Finished metric: '%s': %.4f %s",
        multihost.process_index(),
        self.name,
        value,
        unit,
    )

  def __enter__(self):
    pass

  def __exit__(self, *exc):
    pass


class TimeMetric(Metric):
  """Measures execution time."""

  def __init__(self, name: str):
    super().__init__(name + "_time")

  def __enter__(self):
    self._start_log()
    self._start_time = time.perf_counter()
    return self

  def __exit__(self, *exc):
    duration = time.perf_counter() - self._start_time
    self._record(duration, "s")


class RssMetric(Metric):
  """Measures RSS memory difference."""

  def __init__(self, name: str):
    super().__init__(name + "_rss")

  def __enter__(self):
    self._start_log()
    self._start_rss = self._get_process_memory()
    return self

  def __exit__(self, *exc):
    rss_diff = self._get_process_memory() - self._start_rss
    self._record(rss_diff, "MB")

  def _get_process_memory(self):
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # RSS


class TracemallocMetric(Metric):
  """Measures memory allocation differences using tracemalloc."""

  _lock = threading.Lock()
  _active_count = 0

  def __init__(self, name: str):
    super().__init__(name + "_tracemalloc")

  def __enter__(self):
    self._start_log()
    with TracemallocMetric._lock:
      if TracemallocMetric._active_count == 0:
        tracemalloc.start()
      TracemallocMetric._active_count += 1
    self._start_snapshot = tracemalloc.take_snapshot()
    _, self._start_peak = tracemalloc.get_traced_memory()
    return self

  def __exit__(self, *exc):
    end_snapshot = tracemalloc.take_snapshot()
    _, end_peak = tracemalloc.get_traced_memory()
    with TracemallocMetric._lock:
      TracemallocMetric._active_count -= 1
      if TracemallocMetric._active_count == 0:
        tracemalloc.stop()
    self._log_tracemalloc_snapshot_diff(
        self.name,
        multihost.process_index(),
        self._start_snapshot,
        end_snapshot,
        top_n=15,
        peak=end_peak - self._start_peak,
    )
    logging.info(
        "[process_id=%s] Finished metric: '%s'",
        multihost.process_index(),
        self.name,
    )

    self._record((end_peak - self._start_peak) / (1024**2), "MB")

  def _log_tracemalloc_snapshot_diff(
      self,
      name: str,
      process_index: int,
      snapshot1: tracemalloc.Snapshot,
      snapshot2: tracemalloc.Snapshot,
      top_n: int,
      peak: float,
  ):
    """Compares two tracemalloc snapshots and logs the differences using logging.info.

    Args:
        name: The name of the metric.
        process_index: The process index of the metric.
        snapshot1: The earlier tracemalloc Snapshot.
        snapshot2: The later tracemalloc Snapshot.
        top_n: Number of top differences to log, sorted by memory difference.
        peak: The peak memory usage of the process.
    """
    if not isinstance(snapshot1, tracemalloc.Snapshot) or not isinstance(
        snapshot2, tracemalloc.Snapshot
    ):
      logging.error(
          "Invalid input: Both inputs must be tracemalloc.Snapshot objects."
      )
      return

    logging.info("--- Comparing tracemalloc snapshots ---")

    # Compare snapshots, grouping by file and line number
    stats = snapshot2.compare_to(snapshot1, "lineno")

    if not stats:
      logging.info(
          "[process_id=%s][name=%s] No memory differences found between"
          " snapshots.",
          process_index,
          name,
      )
      return

    total_diff_bytes = sum(stat.size_diff for stat in stats)
    total_new_allocs = sum(stat.count_diff for stat in stats)

    logging.info(
        "[process_id=%s][name=%s] Total memory difference: %.2f KiB, peak:"
        " %.4f GiB",
        process_index,
        name,
        total_diff_bytes / 1024,
        peak / (1024 * 1024 * 1024),
    )
    logging.info(
        "[process_id=%s][name=%s] Total new allocations: %s",
        process_index,
        name,
        total_new_allocs,
    )

    logging.info(
        "[process_id=%s][name=%s] Top %d line-item memory differences:",
        process_index,
        name,
        top_n,
    )
    for index, stat in enumerate(stats[:top_n]):
      size_diff_kb = stat.size_diff / 1024
      if size_diff_kb == 0 and stat.count_diff == 0:
        continue

      frame = stat.traceback[0]
      filename = os.path.basename(frame.filename)
      logging.info(
          "[process_id=%s][name=%s]   #%d: %+.2f KiB, %+d new allocs | at"
          " %s:%s",
          process_index,
          name,
          index + 1,
          size_diff_kb,
          stat.count_diff,
          filename,
          frame.lineno,
      )

      # Get the line from the source file
      line = linecache.getline(frame.filename, frame.lineno).strip()
      if line:
        logging.info("      >> %s", line)

    logging.info(
        "[process_id=%s][name=%s] --- End of snapshot comparison ---",
        process_index,
        name,
    )


@dataclasses.dataclass
class Metrics:
  """A simple dataclass to store and report test metrics."""

  results: MutableMapping[str, tuple[Any, str]] = dataclasses.field(
      default_factory=dict
  )
  name: str = ""

  @contextlib.contextmanager
  def _measure(self, metric_cls: type[Metric], name: str):
    """Helper context manager to run a metric and record results."""
    metric = metric_cls(name)
    with metric:
      yield
    if metric.result is not None:
      self.results[metric.name] = (metric.result, metric.unit)

  def time(self, name: str):
    """A context manager to time a block of code and record it."""
    return self._measure(TimeMetric, name)

  def process_rss(self, name: str):
    """A context manager to calculate the RSS difference of a block of code and record it."""
    return self._measure(RssMetric, name)

  def tracemalloc(self, name: str):
    """A context manager to calculate the difference of tracemalloc snapshots."""
    return self._measure(TracemallocMetric, name)

  def report(self):
    """Logs a formatted report of all collected metrics."""
    report_lines = []
    report_lines.append(
        f"---[process_id={multihost.process_index()}] {self.name} Metrics"
        " Report ---"
    )
    if not self.results:
      report_lines.append(
          f"[process_id={multihost.process_index()}] No metrics recorded."
      )
    else:
      for name, (value, unit) in self.results.items():
        report_lines.append(f"{name}: {value:.4f} {unit}")
    report_lines.append("----------------------")
    logging.info("\n".join(report_lines))
