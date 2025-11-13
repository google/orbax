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


class BaseMetric:
  """Base class for a metric type."""

  def __init__(self, name: str):
    self.name = name
    self._start_time = 0

  def start(self):
    """Start the metric collection."""
    self._start_time = time.perf_counter()
    logging.info(
        "[process_id=%s] Starting metric: '%s'...",
        multihost.process_index(),
        self.name,
    )

  def stop(self) -> dict[str, tuple[Any, str]]:
    """Stop the metric collection and return results."""
    duration = time.perf_counter() - self._start_time
    logging.info(
        "[process_id=%s] Finished metric: '%s' (took %.4fs)",
        multihost.process_index(),
        self.name,
        duration,
    )
    return {}


class TimeMetric(BaseMetric):
  """Measures execution time."""

  def stop(self) -> dict[str, tuple[Any, str]]:
    duration = time.perf_counter() - self._start_time
    results = super().stop()
    results["duration"] = (duration, "s")
    return results


class RssMetric(BaseMetric):
  """Measures RSS memory difference."""

  _start_rss: float = 0

  def start(self):
    super().start()
    self._start_rss = self._get_process_memory()

  def stop(self) -> dict[str, tuple[Any, str]]:
    rss_diff = self._get_process_memory() - self._start_rss
    results = super().stop()
    results["diff"] = (rss_diff, "MB")
    return results

  def _get_process_memory(self):
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


class IOBytesMetric(BaseMetric):
  """Measures process I/O read/write bytes and throughput."""

  _process: psutil.Process
  _start_io: Any = None

  def start(self):
    super().start()
    self._process = psutil.Process(os.getpid())
    try:
      self._start_io = self._process.io_counters()  # pytype: disable=attribute-error
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.warning("Failed to get initial IO counters: %s", e)
      self._start_io = None

  def stop(self) -> dict[str, tuple[Any, str]]:
    results = super().stop()
    if not self._start_io:
      return results

    try:
      end_io = self._process.io_counters()  # pytype: disable=attribute-error
      duration = time.perf_counter() - self._start_time

      read_bytes = end_io.read_bytes - self._start_io.read_bytes
      write_bytes = end_io.write_bytes - self._start_io.write_bytes

      results["read_bytes"] = (read_bytes, "bytes")
      results["write_bytes"] = (write_bytes, "bytes")

      if duration > 0:
        read_mb_s = (read_bytes / (1024 * 1024)) / duration
        write_mb_s = (write_bytes / (1024 * 1024)) / duration
        results["read_throughput"] = (read_mb_s, "MB/s")
        results["write_throughput"] = (write_mb_s, "MB/s")
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.warning("Failed to get final IO counters: %s", e)
    return results


class TracemallocMetric(BaseMetric):
  """Measures memory allocation differences using tracemalloc."""

  _lock = threading.Lock()
  _active_count = 0
  _start_snapshot: Any = None
  _start_peak: int = 0

  def start(self):
    super().start()
    with TracemallocMetric._lock:
      if TracemallocMetric._active_count == 0:
        tracemalloc.start()
      TracemallocMetric._active_count += 1
    self._start_snapshot = tracemalloc.take_snapshot()
    _, self._start_peak = tracemalloc.get_traced_memory()

  def stop(self) -> dict[str, tuple[Any, str]]:
    results = super().stop()
    if self._start_snapshot is None:
      return results

    _, end_peak = tracemalloc.get_traced_memory()
    end_snapshot = tracemalloc.take_snapshot()

    with TracemallocMetric._lock:
      TracemallocMetric._active_count -= 1
      if TracemallocMetric._active_count == 0:
        tracemalloc.stop()

    peak_diff = end_peak - self._start_peak
    results["peak_diff"] = (peak_diff / (1024**2), "MB")

    self._log_tracemalloc_snapshot_diff(
        self.name,
        multihost.process_index(),
        self._start_snapshot,
        end_snapshot,
        top_n=15,
        peak=peak_diff,
    )
    return results

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

    logging.info("--- Comparing tracemalloc snapshots for %s ---", name)
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


class TensorstoreMetric(BaseMetric):
  """Measures tensorstore metrics."""

  _start_metrics: dict[str, dict[str, Any]]

  def start(self):
    super().start()
    self._start_metrics = self._collect_metrics()

  def stop(self) -> dict[str, tuple[Any, str]]:
    results = super().stop()
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
    results["diff_count"] = (len(diff), f"{self.name}_diff_cnt")
    return results

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


# Registry of available metric types
METRIC_REGISTRY: dict[str, type[BaseMetric]] = {
    "time": TimeMetric,
    "rss": RssMetric,
    "io": IOBytesMetric,
    "tracemalloc": TracemallocMetric,
    "tensorstore": TensorstoreMetric,
}

DEFAULT_METRICS = ["time"]


@dataclasses.dataclass
class Metrics:
  """Container and manager for all metric results from a profiling block."""

  results: MutableMapping[str, tuple[Any, str]] = dataclasses.field(
      default_factory=dict
  )
  name: str = ""

  def _add_results(
      self,
      metric_name: str,
      metric_key: str,
      metric_results: dict[str, tuple[Any, str]],
  ):
    for key, (value, unit) in metric_results.items():
      full_key = f"{metric_name}_{metric_key}_{key}"
      self.results[full_key] = (value, unit)

  @contextlib.contextmanager
  def measure(self, operation_name: str, metric_keys: list[str] | None = None):
    """Context manager to measure a block of code with the specified metrics.

    Args:
      operation_name: The name of the operation to measure.
      metric_keys: The keys of the metrics to measure. If None, the default
        metrics (time) will be measured.
    """
    if metric_keys is None:
      metric_keys = DEFAULT_METRICS

    collector = _MetricsCollector(self, operation_name, metric_keys)
    with collector:
      yield

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
      for name, (value, unit) in sorted(self.results.items()):
        if isinstance(value, float):
          report_lines.append(f"{name}: {value:.4f} {unit}")
        else:
          report_lines.append(f"{name}: {value} {unit}")
    report_lines.append("----------------------")
    logging.info("\n".join(report_lines))


class _MetricsCollector:
  """Internal context manager to collect specified metrics."""

  def __init__(
      self, metrics_obj: Metrics, operation_name: str, metric_keys: list[str]
  ):
    self.metrics_obj = metrics_obj
    self.operation_name = operation_name
    self._metrics: dict[str, BaseMetric] = {}

    for key in metric_keys:
      if key in METRIC_REGISTRY:
        metric_class = METRIC_REGISTRY[key]
        self._metrics[key] = metric_class(operation_name)
      else:
        logging.warning("Unknown metric key: %s", key)

  def __enter__(self):
    for metric in self._metrics.values():
      metric.start()
    return self

  def __exit__(self, *exc):
    for key, metric in self._metrics.items():
      try:
        metric_results = metric.stop()
        self.metrics_obj._add_results(metric.name, key, metric_results)
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.exception("Error stopping metric %s: %s", metric.name, e)
