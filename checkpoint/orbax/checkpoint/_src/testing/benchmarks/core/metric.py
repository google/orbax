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

"""Metric classes for benchmarking."""

import collections
from collections.abc import MutableMapping
import contextlib
import dataclasses
import json
import linecache  # To show the source code line
import os
import threading
import time
import tracemalloc
from typing import Any

from absl import logging
from clu import metric_writers
from etils import epath
import numpy as np
from orbax.checkpoint._src.testing.benchmarks.core import multihost
import psutil
import tensorstore as ts


class BaseMetric:
  """Base class for a metric type.

  Subclass override knobs:
    OMIT_REGISTRY_KEY_PREFIX: when True, the metric's result keys are taken
      as final and the METRIC_REGISTRY key (e.g. "jax_monitoring") is NOT
      spliced into the TB tag. Use when the metric already namespaces its
      own keys (e.g. JaxMonitoringMetric returns "2_save_breakdown/...").
  """

  OMIT_REGISTRY_KEY_PREFIX: bool = False

  def __init__(self, name: str):
    self.name = name
    self._start_time = 0

  def start(self):
    """Start the metric collection."""
    self._start_time = time.perf_counter()
    logging.info(
        "[process_id=%s] Starting metric: '%s'...",
        multihost.get_process_index(),
        self.name,
    )

  def stop(self) -> dict[str, tuple[Any, str]]:
    """Stop the metric collection and return results."""
    duration = time.perf_counter() - self._start_time
    logging.info(
        "[process_id=%s] Finished metric: '%s' (took %.4fs)",
        multihost.get_process_index(),
        self.name,
        duration,
    )
    return {}


class TimeMetric(BaseMetric):
  """Measures execution time."""

  OMIT_REGISTRY_KEY_PREFIX = True

  def stop(self) -> dict[str, tuple[Any, str]]:
    duration = time.perf_counter() - self._start_time
    results = super().stop()
    results["0_basics/time_s"] = (duration, "s")
    return results


class RssMetric(BaseMetric):
  """Measures RSS memory difference."""

  OMIT_REGISTRY_KEY_PREFIX = True
  _start_rss: float = 0

  def start(self):
    super().start()
    self._start_rss = self._get_process_memory()

  def stop(self) -> dict[str, tuple[Any, str]]:
    rss_diff = self._get_process_memory() - self._start_rss
    results = super().stop()
    results["0_basics/host_rss_diff_mb"] = (rss_diff, "MB")
    return results

  def _get_process_memory(self):
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


class TracemallocMetric(BaseMetric):
  """Measures memory allocation differences using tracemalloc."""

  OMIT_REGISTRY_KEY_PREFIX = True
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
    results["7_memory/tracemalloc_peak_diff_mb"] = (peak_diff / (1024**2), "MB")

    self._log_tracemalloc_snapshot_diff(
        self.name,
        multihost.get_process_index(),
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

  OMIT_REGISTRY_KEY_PREFIX = True
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
        multihost.get_process_index(),
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
    results["6_tensorstore/diff_count"] = (len(diff), "count")
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
      if metric_key:
        full_key = f"{metric_name}_{metric_key}_{key}"
      else:
        full_key = f"{metric_name}_{key}"
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
        f"---[process_id={multihost.get_process_index()}] {self.name} Metrics"
        " Report ---"
    )
    if not self.results:
      report_lines.append(
          f"[process_id={multihost.get_process_index()}] No metrics recorded."
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
        tag_key = "" if metric.OMIT_REGISTRY_KEY_PREFIX else key
        self.metrics_obj._add_results(metric.name, tag_key, metric_results)
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.exception("Error stopping metric %s: %s", metric.name, e)


################################################################################
# Aggregation and Reporting
################################################################################


def _options_to_hparams(options: Any) -> dict[str, bool | int | float | str]:
  """Flattens a benchmark options object into a TB HParams-acceptable dict.

  HParams values must be primitives (bool / int / float / str). Anything else
  (None, list, tuple, nested) is rendered via str() so the run still appears
  in the Parallel Coordinates view rather than getting dropped.

  Args:
    options: A dataclass instance or dict of benchmark options to flatten;
      anything else yields an empty dict.

  Returns:
    A dict of primitive HParam values keyed by option name.
  """
  if dataclasses.is_dataclass(options):
    raw = dataclasses.asdict(options)
  elif isinstance(options, dict):
    raw = dict(options)
  else:
    return {}
  out: dict[str, bool | int | float | str] = {}
  for k, v in raw.items():
    if isinstance(v, (bool, int, float, str)):
      out[k] = v
    else:
      out[k] = str(v)
  return out


def _render_configuration_markdown(
    benchmark_name: str,
    benchmark_options: dict[str, Any] | None,
    checkpoint_config: dict[str, Any] | None,
) -> str:
  """Renders the run configuration as readable markdown.

  Options + checkpoint_config become two field/value tables; any nested
  dict in checkpoint_config (typically `spec`) is split out into its own
  fenced-JSON block. Replaces the single-line `json.dumps` blob the
  Text-tab card used to show.
  """
  lines = [f"## {benchmark_name}", ""]

  def _table(title: str, items: list[tuple[str, Any]]) -> None:
    lines.append(f"### {title}")
    lines.append("")
    lines.append("| field | value |")
    lines.append("|---|---|")
    for k, v in items:
      lines.append(f"| `{k}` | `{v}` |")
    lines.append("")

  if benchmark_options:
    _table(
        "Benchmark options",
        [(k, v) for k, v in sorted(benchmark_options.items())],
    )

  if checkpoint_config:
    scalar_items = []
    nested_items = []
    for k, v in sorted(checkpoint_config.items()):
      if isinstance(v, dict):
        nested_items.append((k, v))
      else:
        scalar_items.append((k, v))
    if scalar_items:
      _table("Checkpoint config", scalar_items)
    for k, v in nested_items:
      lines.append(f"### Checkpoint config — `{k}`")
      lines.append("")
      lines.append("```json")
      lines.append(json.dumps(v, indent=2, sort_keys=True))
      lines.append("```")
      lines.append("")
  return "\n".join(lines)


def _render_aggregated_metrics_markdown(
    benchmark_name: str,
    aggregated_stats_dict: dict[str, "AggregatedStats"],
    metric_units: dict[str, str],
) -> str:
  """Renders the aggregated metrics as a markdown table grouped by `/` prefix.

  TB's Text dashboard renders markdown — a proper table is dramatically
  more readable than the previous `<pre>` raw dump, and grouping by
  numbered prefix (`1_overview/`, `2_save_breakdown/`, …) mirrors the
  Scalars-view navigation so a reader can locate a metric the same way
  in both surfaces.
  """
  if not aggregated_stats_dict:
    return "_No successful runs to aggregate._"

  groups: dict[str, list[str]] = collections.defaultdict(list)
  for key in sorted(aggregated_stats_dict):
    head, _, _ = key.partition("/")
    section = head if "/" in key else "_other_"
    groups[section].append(key)

  lines = [f"## {benchmark_name} — aggregated metrics", ""]
  for section in sorted(groups):
    lines.append(f"### {section}")
    lines.append("")
    lines.append("| metric | mean | ± std | min | max | n | unit |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for key in groups[section]:
      stats = aggregated_stats_dict[key]
      unit = metric_units.get(key, "")
      leaf = key.split("/", 1)[1] if "/" in key else key
      lines.append(
          f"| `{leaf}` | {stats.mean:.4f} | {stats.std:.4f} |"
          f" {stats.min:.4f} | {stats.max:.4f} | {stats.count} | {unit} |"
      )
    lines.append("")
  return "\n".join(lines)


@dataclasses.dataclass
class AggregatedStats:
  """Statistics aggregated over multiple benchmark repetitions.

  Attributes:
    mean: Mean value.
    std: Standard deviation.
    min: Minimum value.
    max: Maximum value.
    count: Number of values aggregated.
  """

  mean: float
  std: float
  min: float
  max: float
  count: int


class MetricsManager:
  """Manages metrics aggregation and reporting for a test suite.

  This class collects metrics from multiple benchmark runs and repetitions,
  computes aggregate statistics (mean, std, min, max), generates a
  human-readable report for logging, and exports metrics to TensorBoard
  if configured.
  """

  def __init__(
      self,
      name: str,
      num_repeats: int,
      tensorboard_dir: epath.Path | None = None,
  ):
    """Initializes the MetricsManager.

    Args:
      name: The name of the test suite.
      num_repeats: The number of repetitions for each benchmark configuration.
      tensorboard_dir: The directory to write TensorBoard events to. If None,
        metrics will not be written to TensorBoard during the run.
    """
    self._name = name
    self._num_repeats = num_repeats
    self._runs: dict[str, list[tuple[Metrics, Exception | None]]] = (
        collections.defaultdict(list)
    )
    self._benchmark_options: dict[str, Any] = {}
    self._checkpoint_configs: dict[str, Any] = {}
    self._tensorboard_dir = tensorboard_dir
    self._writers: dict[str, Any] = {}

  def add_result(
      self,
      benchmark_name: str,
      metrics: Metrics,
      *,
      benchmark_options: Any | None = None,
      checkpoint_config: Any | None = None,
      error: Exception | None = None,
  ):
    """Adds metrics from a single benchmark run/repetition.

    Args:
      benchmark_name: The name of the benchmark configuration.
      metrics: The Metrics object containing results for this run.
      benchmark_options: The BenchmarkOptions used for this run.
      checkpoint_config: The CheckpointConfig used for this run.
      error: An exception if the run failed, otherwise None.
    """
    self._runs[benchmark_name].append((metrics, error))
    if benchmark_name not in self._benchmark_options:
      self._benchmark_options[benchmark_name] = benchmark_options
    if benchmark_name not in self._checkpoint_configs:
      self._checkpoint_configs[benchmark_name] = checkpoint_config

    if self._tensorboard_dir:
      self._write_result_to_tensorboard(
          benchmark_name,
          metrics,
          error,
          len(self._runs[benchmark_name]) - 1,
          benchmark_options,
          checkpoint_config,
      )

  def _get_writer(self, benchmark_name: str) -> Any:
    """Gets or creates a TensorBoard writer for the given benchmark."""
    if benchmark_name not in self._writers:
      is_primary_host = multihost.get_process_index() == 0
      self._writers[benchmark_name] = metric_writers.create_default_writer(
          self._tensorboard_dir,
          just_logging=not is_primary_host,
          collection=benchmark_name,
      )
    return self._writers[benchmark_name]

  def _write_result_to_tensorboard(
      self,
      benchmark_name: str,
      metrics: Metrics,
      error: Exception | None,
      step: int,
      benchmark_options: Any | None = None,
      checkpoint_config: Any | None = None,
  ):
    """Writes a single result to TensorBoard."""
    writer = self._get_writer(benchmark_name)
    if error is None:
      for key, (value, unit) in metrics.results.items():
        # Hierarchical keys (e.g. "2_save_breakdown/blocking_s") already
        # encode the unit in their suffix; appending "_s" again gives the
        # ugly "..._blocking_s_s". Skip the suffix for those; flat legacy
        # keys keep the existing "{key}_{unit}" shape.
        if "/" in key:
          tag = key
        else:
          tag = f'{key}_{unit.replace("/", "_")}'
        if isinstance(value, (int, float)):
          writer.write_scalars(step=step, scalars={tag: value})
        else:
          writer.write_texts(step=step, texts={tag: str(value)})
    else:
      tag = "error"
      writer.write_texts(step=step, texts={tag: f"<pre>{repr(error)}</pre>"})

    # Write configuration if it's the first step
    if step == 0 and benchmark_options:
      if dataclasses.is_dataclass(benchmark_options):
        opt_dict = dataclasses.asdict(benchmark_options)
      else:
        opt_dict = benchmark_options

      if dataclasses.is_dataclass(checkpoint_config):
        config_dict = dataclasses.asdict(checkpoint_config)
      elif isinstance(checkpoint_config, dict):
        config_dict = checkpoint_config
      else:
        config_dict = None

      writer.write_texts(
          step=0,
          texts={
              "configuration": _render_configuration_markdown(
                  benchmark_name, opt_dict, config_dict
              ),
          },
      )
      if hparams_dict := _options_to_hparams(benchmark_options):
        writer.write_hparams(hparams_dict)
    writer.flush()

  def _aggregate_metrics(
      self, results: list[tuple[Metrics, Exception | None]]
  ) -> tuple[dict[str, AggregatedStats], dict[str, str]]:
    """Computes aggregate stats (mean, std, etc.) for successful runs.

    Args:
      results: A list of (Metrics, error) tuples for a benchmark configuration.

    Returns:
      A tuple containing:
        - A dict mapping metric keys to AggregatedStats.
        - A dict mapping metric keys to their units.
    """
    metrics_collector = collections.defaultdict(list)
    metric_units = {}
    for metrics, error in results:
      if error is None:
        for key, (value, unit) in metrics.results.items():
          if isinstance(value, (int, float)):
            metrics_collector[key].append(value)
            metric_units[key] = unit

    aggregated_stats_dict = {}
    for key, values in metrics_collector.items():
      aggregated_stats_dict[key] = AggregatedStats(
          mean=np.mean(values),
          std=np.std(values),
          min=np.min(values),
          max=np.max(values),
          count=len(values),
      )
    return aggregated_stats_dict, metric_units

  def _count_runs(self) -> tuple[int, int, int]:
    total = passed = failed = 0
    for _, results in self._runs.items():
      total += len(results)
      for _, error in results:
        if error is None:
          passed += 1
        else:
          failed += 1
    return total, passed, failed

  def _format_stats_lines(
      self,
      aggregated_stats_dict: dict[str, AggregatedStats],
      metric_units: dict[str, str],
      indent: str,
  ) -> list[str]:
    """Formats aggregated stats into human-readable report lines."""
    lines = []
    for key, stats in aggregated_stats_dict.items():
      unit = metric_units[key]
      lines.append(
          f"{indent}{key}: {stats.mean:.4f} +/- {stats.std:.4f} {unit} (min:"
          f" {stats.min:.4f}, max: {stats.max:.4f}, n={stats.count})"
      )
    return lines

  def _format_aggregated_report_section(self) -> list[str]:
    """Builds the per-benchmark aggregated-metrics section of the report."""
    lines = ["\n" + "-" * 80, "--- Aggregated Metrics per Benchmark ---"]
    for benchmark_name, results in self._runs.items():
      if not results:
        continue
      lines.append(f"\nBenchmark: {benchmark_name}")
      aggregated_stats_dict, metric_units = self._aggregate_metrics(results)
      if not aggregated_stats_dict:
        lines.append("  No successful runs to aggregate.")
        continue
      lines.extend(
          self._format_stats_lines(aggregated_stats_dict, metric_units, "  ")
      )
    return lines

  def _format_failed_runs_section(self) -> list[str]:
    lines = ["\n" + "-" * 80, "--- Failed Runs ---"]
    for _, results in self._runs.items():
      for metrics, error in results:
        if error is None:
          continue
        error_repr = repr(error)
        # Limit error length to avoid flooding logs.
        if len(error_repr) > 1000:
          error_repr = error_repr[:1000] + "..."
        lines.append(f"Test: {metrics.name}, Error: {error_repr}")
    return lines

  def _write_aggregated_to_tensorboard(self) -> None:
    """Writes each benchmark's aggregated metrics to TensorBoard as text."""
    logging.info("Writing aggregated metrics to TensorBoard...")
    for benchmark_name, results in self._runs.items():
      writer = self._get_writer(benchmark_name)
      aggregated_stats_dict, metric_units = self._aggregate_metrics(results)
      aggregated_metrics_str = _render_aggregated_metrics_markdown(
          benchmark_name, aggregated_stats_dict, metric_units
      )
      writer.write_texts(
          step=0,
          texts={"aggregated_metrics": aggregated_metrics_str},
      )
      writer.flush()
      writer.close()
    # Clear writers after closing to prevent reuse of closed writers if called
    # again.
    self._writers.clear()
    logging.info("Finished writing metrics to TensorBoard.")

  def generate_report(self) -> None:
    """Generates a final string report containing aggregated metrics.

    And exports aggregated metrics to TensorBoard if configured.
    """
    title = f" Test Suite Report: {self._name} "
    report_lines = [f"\n{title:=^80}"]
    total_runs, passed_runs, failed_runs = self._count_runs()
    report_lines.append(f"Total benchmark configurations: {len(self._runs)}")
    report_lines.append(
        f"Total runs ({self._num_repeats} repeats): {total_runs}, Passed:"
        f" {passed_runs}, Failed: {failed_runs}"
    )
    if self._num_repeats > 1:
      report_lines.extend(self._format_aggregated_report_section())
    if failed_runs > 0:
      report_lines.extend(self._format_failed_runs_section())
    report_lines.append("\n" + "=" * 80)
    logging.info("\n".join(report_lines))
    if self._tensorboard_dir:
      self._write_aggregated_to_tensorboard()
