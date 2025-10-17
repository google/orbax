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
import tensorstore as ts


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


class TensorstoreMetric(Metric):
  """Measures tensorstore metrics."""

  def __init__(self, name: str):
    super().__init__(name + "_ts")

  def _collect_metrics(self) -> dict[str, dict[str, Any]]:
    """Collects tensorstore metrics for interested metrics."""

    interested_metric_paths = ["/tensorstore", "/mallocz", "/tcmalloc/"]
    metrics_list = []
    for path in interested_metric_paths:
      try:
        metrics_list += ts.experimental_collect_matching_metrics(path)
      except Exception as e:  # pylint: disable=broad-except
        logging.warning(
            "Failed to collect tensorstore metrics for path %s: %s", path, e
        )
    metrics_dict = {}
    for m in metrics_list:
      if m and "name" in m and m.get("values"):
        # For now, only consider metrics with a single value entry that
        # contains 'value' or 'count'.
        if len(m["values"]) == 1:
          metrics_dict[m["name"]] = m["values"][0]
    return metrics_dict

  def __enter__(self):
    self._start_log()
    self._start_metrics = self._collect_metrics()
    return self

  def __exit__(self, *exc):
    end_metrics = self._collect_metrics()
    diff = self._diff_metrics(self._start_metrics, end_metrics)
    logging.info(
        "[process_id=%s] Finished metric: %s, num_diffs=%d",
        multihost.process_index(),
        self.name,
        len(diff),
    )
    # log all start metrics
    for key, values in self._start_metrics.items():
      logging.info(
          "TensorstoreMetric[%s] start for %s: %s", self.name, key, values
      )
    logging.info("----------------------------------------------------------")

    # log all end metrics
    for key, values in end_metrics.items():
      logging.info(
          "TensorstoreMetric[%s] end for %s: %s", self.name, key, values
      )
    logging.info("----------------------------------------------------------")

    for key, values in diff.items():
      logging.info(
          "TensorstoreMetric[%s] diff for %s: %s", self.name, key, values
      )

    # Log the number of metrics that have a non-zero diff.
    self._record(len(diff), f"{self.name}_diff_cnt")

  def _diff_metrics(
      self,
      start_metrics: dict[str, dict[str, Any]],
      end_metrics: dict[str, dict[str, Any]],
  ) -> dict[str, dict[str, Any]]:
    """Diffs two dictionaries of metrics."""
    diff_metrics = {}
    all_keys = set(start_metrics.keys()) | set(end_metrics.keys())
    for key in all_keys:
      start_vals = start_metrics.get(key, {})
      end_vals = end_metrics.get(key, {})

      diff = {}
      if "value" in end_vals or "value" in start_vals:
        start_v = start_vals.get("value", 0)
        end_v = end_vals.get("value", 0)
        if isinstance(start_v, (int, float)) and isinstance(
            end_v, (int, float)
        ):
          val_diff = end_v - start_v
          if val_diff != 0:
            diff["value"] = val_diff

      if "count" in end_vals or "count" in start_vals:
        start_c = start_vals.get("count", 0)
        end_c = end_vals.get("count", 0)
        if isinstance(start_c, (int, float)) and isinstance(
            end_c, (int, float)
        ):
          count_diff = end_c - start_c
          if count_diff != 0:
            diff["count"] = count_diff

      if diff:
        diff_metrics[key] = diff

    return diff_metrics


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

  def tensorstore(self, name: str):
    """A context manager to collect tensorstore metrics."""
    return self._measure(TensorstoreMetric, name)

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
