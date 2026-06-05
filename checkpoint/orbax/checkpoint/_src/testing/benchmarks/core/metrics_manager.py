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

"""MetricsManager — aggregates benchmark metrics and writes the TB summary."""

import collections
import dataclasses
import json
from typing import Any

from absl import logging
from clu import metric_writers
from etils import epath
import numpy as np
from orbax.checkpoint._src.testing.benchmarks.core import markdown_cards
from orbax.checkpoint._src.testing.benchmarks.core import metric as metric_lib
from orbax.checkpoint._src.testing.benchmarks.core import multihost
from orbax.checkpoint._src.testing.benchmarks.core import run_manifest as run_manifest_lib

# Type aliases for cross-host metric aggregation
MetricStats = dict[
    str, float
]  # stat_name -> value (e.g. {"mean": 1.5, "max": 2.0})
BenchmarkSummary = dict[str, MetricStats]  # metric_key -> MetricStats


def _summary_aggregates(
    per_host_matrix: np.ndarray, keys: list[str]
) -> BenchmarkSummary:
  """Computes per-metric max/min/mean (p50/p99 with >1 host) across hosts.

  NaN entries (a host that didn't report a key) are dropped per column, so
  primary-only events still aggregate from the hosts that did report. "max" is
  the MLPerf-honest headline (slowest rank for time, smallest for throughput —
  callers translate per metric).

  Args:
    per_host_matrix: (host_count, metric_count) values; NaN where unreported.
    keys: Metric keys, one per matrix column.

  Returns:
    Metric key -> {stat: value}; empty if the matrix or keys are empty.
  """
  if per_host_matrix.size == 0 or len(keys) == 0:
    return {}
  out: dict[str, dict[str, float]] = {}
  for j, key in enumerate(keys):
    column = per_host_matrix[:, j]
    column = column[~np.isnan(column)]
    if column.size == 0:
      continue
    entry = {
        "max": float(np.max(column)),
        "min": float(np.min(column)),
        "mean": float(np.mean(column)),
    }
    if column.size > 1:
      entry["p50"] = float(np.percentile(column, 50))
      entry["p99"] = float(np.percentile(column, 99))
    out[key] = entry
  return out


def _as_plain_dict(obj: Any) -> dict[str, Any] | None:
  """Returns a dataclass or dict as a plain dict, else None."""
  if dataclasses.is_dataclass(obj):
    return dataclasses.asdict(obj)
  return obj if isinstance(obj, dict) else None


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
      enable_per_host_metrics: bool = True,
  ):
    """Initializes the MetricsManager.

    Args:
      name: The name of the test suite.
      num_repeats: The number of repetitions for each benchmark configuration.
      tensorboard_dir: The directory to write TensorBoard events to. If None,
        metrics will not be written to TensorBoard during the run.
      enable_per_host_metrics: When True, every process opens its own writer at
        <tensorboard_dir>/<benchmark>/host_<idx>/ so per-host scalars are
        visible in the TB Scalars view as sibling runs. When False, only the
        primary host writes (legacy behavior).
    """
    self._name = name
    self._num_repeats = num_repeats
    self._runs: dict[str, list[tuple[metric_lib.Metrics, Exception | None]]] = (
        collections.defaultdict(list)
    )
    self._benchmark_options: dict[str, Any] = {}
    self._checkpoint_configs: dict[str, Any] = {}
    self._tensorboard_dir = tensorboard_dir
    self._enable_per_host_metrics = enable_per_host_metrics
    self._writers: dict[str, Any] = {}
    # Suite-level: first non-None inventory per benchmark wins.
    self._inventories: dict[str, Any] = {}
    # Cross-host metric aggregate per benchmark; filled by the primary during
    # generate_report's gather, read by baseline capture/compare.
    self._cross_host_aggregates: dict[str, BenchmarkSummary] = {}
    # Captured once so the manifest reflects the reported run, not the machine
    # state at generate_report time.
    self._suite_run_manifest = run_manifest_lib.capture_run_manifest()

  @property
  def run_manifest(self) -> run_manifest_lib.RunManifest:
    """The environment snapshot captured once when this manager was created."""
    return self._suite_run_manifest

  def add_result(
      self,
      benchmark_name: str,
      metrics: metric_lib.Metrics,
      *,
      benchmark_options: Any | None = None,
      checkpoint_config: Any | None = None,
      error: Exception | None = None,
      inventory: Any | None = None,
  ):
    """Adds metrics from a single benchmark run/repetition.

    Args:
      benchmark_name: The name of the benchmark configuration.
      metrics: The Metrics object containing results for this run.
      benchmark_options: The BenchmarkOptions used for this run.
      checkpoint_config: The CheckpointConfig used for this run.
      error: An exception if the run failed, otherwise None.
      inventory: Optional post-save CheckpointInventory; the first non-None
        provided per benchmark wins (subsequent repeats overwrite the same
        target dir, so the inventory is invariant across repeats).
    """
    self._runs[benchmark_name].append((metrics, error))
    if benchmark_name not in self._benchmark_options:
      self._benchmark_options[benchmark_name] = benchmark_options
    if benchmark_name not in self._checkpoint_configs:
      self._checkpoint_configs[benchmark_name] = checkpoint_config
    if inventory is not None and benchmark_name not in self._inventories:
      self._inventories[benchmark_name] = inventory

    if self._tensorboard_dir:
      self._write_result_to_tensorboard(
          benchmark_name,
          metrics,
          error,
          len(self._runs[benchmark_name]) - 1,
      )

  def _get_writer(self, benchmark_name: str) -> Any:
    """Gets or creates the cached per-benchmark TensorBoard writer.

    Per-host mode opens a writer per process at <benchmark>/host_<idx>/;
    legacy mode writes only from the primary host.

    Args:
      benchmark_name: The benchmark whose writer is fetched.

    Returns:
      The cached or newly created writer.
    """
    if benchmark_name in self._writers:
      return self._writers[benchmark_name]

    host_idx = multihost.get_process_index()
    if self._enable_per_host_metrics:
      # clu plants events at <logdir>/<collection>/, so the host suffix goes in
      # the collection to keep the layout flat at <benchmark>/host_<idx>/.
      writer = metric_writers.create_default_writer(
          self._tensorboard_dir,
          collection=f"{benchmark_name}/host_{host_idx}",
      )
    else:
      writer = metric_writers.create_default_writer(
          self._tensorboard_dir,
          just_logging=host_idx != 0,
          collection=benchmark_name,
      )
    self._writers[benchmark_name] = writer
    return writer

  def _write_result_to_tensorboard(
      self,
      benchmark_name: str,
      metrics: metric_lib.Metrics,
      error: Exception | None,
      step: int,
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

    # Config/HParams/aggregated_metrics are suite-level cards written once to
    # __summary__, so per-host writers carry scalars only.
    writer.flush()

  def _aggregate_metrics(
      self, results: list[tuple[metric_lib.Metrics, Exception | None]]
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

  def mean_metrics(self, benchmark_name: str) -> dict[str, float]:
    """Returns each metric's mean across a benchmark's successful repeats.

    Args:
      benchmark_name: The benchmark configuration to aggregate.

    Returns:
      Mapping of metric key to its mean value over the successful repeats.
    """
    stats, _ = self._aggregate_metrics(self._runs[benchmark_name])
    return {key: float(value.mean) for key, value in stats.items()}

  def cross_host_aggregates(
      self, benchmark_name: str
  ) -> BenchmarkSummary | None:
    """Returns the cross-host metric aggregate for a benchmark, if gathered.

    Populated on the primary host during generate_report when a TensorBoard
    dir is configured (the sidecar gather writes there); None otherwise.

    Args:
      benchmark_name: The benchmark to look up.

    Returns:
      Metric key -> {stat: value} across hosts, or None if not gathered.
    """
    return self._cross_host_aggregates.get(benchmark_name)

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
      self._aggregate_and_write_summary(benchmark_name, results)
    for w in self._writers.values():
      w.flush()
      w.close()
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

  def _aggregate_and_write_summary(
      self,
      benchmark_name: str,
      results: list[tuple[metric_lib.Metrics, Exception | None]],
  ) -> None:
    """Gathers per-host means and writes the cross-host __summary__ run.

    Each host writes a means sidecar; after a barrier the primary host unions
    the keys, computes max/min/mean (p50/p99 with >1 host), and writes the
    summary scalars and cards. "max" is the MLPerf-shape headline (slowest rank
    for time, smallest for throughput — callers pick the relevant column).

    Args:
      benchmark_name: The benchmark being summarized.
      results: The (Metrics, error) tuples for that benchmark.
    """
    if self._tensorboard_dir is None:
      return
    self._write_per_host_means_sidecar(benchmark_name, results)
    multihost.sync_global_processes(f"metrics:summary:{benchmark_name}")
    if multihost.get_process_index() == 0:
      aggregates = self._gather_cross_host_aggregates(benchmark_name)
      if aggregates:
        self._cross_host_aggregates[benchmark_name] = aggregates
        self._write_summary_cards(benchmark_name, results, aggregates)
    # Final barrier so non-primaries don't exit while the primary writes —
    # otherwise the coordinator marks them gone and the shutdown barrier fails.
    multihost.sync_global_processes(f"metrics:summary-done:{benchmark_name}")

  def _sidecar_gather_root(self, benchmark_name: str) -> epath.Path:
    """Returns the directory holding per-host means sidecars for a benchmark.

    Args:
      benchmark_name: The benchmark whose sidecar root is resolved.

    Returns:
      The gather root; legacy mode nests it under _per_host_means/ so it does
      not collide with the single benchmark writer's events at <benchmark>/.
    """
    assert self._tensorboard_dir is not None
    root = self._tensorboard_dir / benchmark_name
    return root if self._enable_per_host_metrics else root / "_per_host_means"

  def _write_per_host_means_sidecar(
      self,
      benchmark_name: str,
      results: list[tuple[metric_lib.Metrics, Exception | None]],
  ) -> None:
    """Writes this host's per-metric means as a JSON sidecar.

    Filesystem-based gather: sidecars avoid jax.distributed allgather and work
    on any shared filesystem. Every host writes even with no successful metrics
    so all hosts reach the caller's barrier; an empty sidecar is harmless.

    Args:
      benchmark_name: The benchmark whose means are written.
      results: This host's (Metrics, error) tuples for that benchmark.
    """
    collector: dict[str, list[float]] = collections.defaultdict(list)
    for m, error in results:
      if error is None:
        for k, (v, _) in m.results.items():
          if isinstance(v, (int, float)):
            collector[k].append(float(v))
    keys = sorted(collector)
    means = [float(np.mean(collector[k])) for k in keys]
    host_dir = (
        self._sidecar_gather_root(benchmark_name)
        / f"host_{multihost.get_process_index()}"
    )
    try:
      host_dir.mkdir(parents=True, exist_ok=True)
      (host_dir / "_per_host_means.json").write_text(
          json.dumps({"keys": keys, "means": means})
      )
    except OSError as e:
      logging.warning("Failed to write per-host means sidecar: %s", e)

  def _gather_cross_host_aggregates(
      self, benchmark_name: str
  ) -> BenchmarkSummary:
    """Reads every host's sidecar and aggregates the values across hosts.

    Keys absent on some hosts become NaN, so primary-only events (e.g.
    metadata_write_s) still aggregate from the hosts that reported them.

    Args:
      benchmark_name: The benchmark whose sidecars are gathered.

    Returns:
      Metric key -> aggregate stats, or empty if no sidecars were found.
    """
    per_host: list[dict[str, float]] = []
    for host_dir in sorted(self._sidecar_gather_root(benchmark_name).iterdir()):
      sidecar = host_dir / "_per_host_means.json"
      if host_dir.name.startswith("host_") and sidecar.exists():
        data = json.loads(sidecar.read_text())
        per_host.append(dict(zip(data["keys"], data["means"])))
    if not per_host:
      return {}
    union_keys = sorted(set().union(*(d.keys() for d in per_host)))
    matrix = np.full((len(per_host), len(union_keys)), np.nan, dtype=np.float64)
    for i, d in enumerate(per_host):
      for j, key in enumerate(union_keys):
        if key in d:
          matrix[i, j] = d[key]
    return _summary_aggregates(matrix, union_keys)

  def _write_config_and_hparams(self, writer: Any, benchmark_name: str) -> None:
    """Writes the configuration text card and HParams for a benchmark.

    Args:
      writer: The TensorBoard writer to emit to.
      benchmark_name: The benchmark whose options/config are rendered.
    """
    benchmark_options = self._benchmark_options.get(benchmark_name)
    if benchmark_options is None:
      return
    opt_dict = _as_plain_dict(benchmark_options)
    if opt_dict is not None:
      writer.write_texts(
          step=0,
          texts={
              "configuration": markdown_cards.render_configuration(
                  benchmark_name,
                  opt_dict,
                  _as_plain_dict(self._checkpoint_configs.get(benchmark_name)),
              )
          },
      )
    hparams_dict = markdown_cards.options_to_hparams(benchmark_options)
    if hparams_dict:
      writer.write_hparams(hparams_dict)

  def _write_summary_cards(
      self,
      benchmark_name: str,
      results: list[tuple[metric_lib.Metrics, Exception | None]],
      aggregates: BenchmarkSummary,
  ) -> None:
    """Writes the __summary__ run: aggregate scalars, text cards, and HParams.

    Per-host mode opens a sibling __summary__ writer; legacy mode reuses the
    benchmark's single writer.

    Args:
      benchmark_name: The benchmark being summarized.
      results: The (Metrics, error) tuples for that benchmark.
      aggregates: Cross-host aggregate stats per metric key.
    """
    assert self._tensorboard_dir is not None
    if self._enable_per_host_metrics:
      writer = metric_writers.create_default_writer(
          self._tensorboard_dir,
          collection=f"{benchmark_name}/__summary__",
      )
      owned = True
    else:
      writer = self._get_writer(benchmark_name)
      owned = False
    try:
      # Aggregate scalars only — TB's Distributions/Histograms views plot over
      # a step axis and collapse to empty at step 0, so histograms are skipped.
      for key, stats in aggregates.items():
        writer.write_scalars(
            step=0, scalars={f"{key}_{stat}": v for stat, v in stats.items()}
        )
      writer.write_texts(
          step=0,
          texts={
              "scorecard": markdown_cards.render_scorecard(
                  benchmark_name,
                  aggregates,
                  self._inventories.get(benchmark_name),
                  self._suite_run_manifest,
              )
          },
      )
      self._write_config_and_hparams(writer, benchmark_name)
      stats_dict, units_dict = self._aggregate_metrics(results)
      writer.write_texts(
          step=0,
          texts={
              "aggregated_metrics": markdown_cards.render_aggregated_metrics(
                  benchmark_name, stats_dict, units_dict
              )
          },
      )
      writer.flush()
    finally:
      if owned:
        writer.close()
