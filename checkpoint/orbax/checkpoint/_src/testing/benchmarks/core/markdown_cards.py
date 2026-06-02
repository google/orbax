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

"""Pure markdown rendering for the per-benchmark text cards.

Three cards, all rendered as markdown that TB's Text dashboard displays:

  - scorecard: headline numbers + inventory + run manifest. The
    screenshot-able card.
  - configuration: options + checkpoint config tables, plus pretty-printed
    JSON for nested fields like `spec`.
  - aggregated_metrics: every metric's mean / std / min / max / n / unit,
    grouped by the numeric section prefix (`2_save_breakdown/` etc.).

Everything in this module is a pure function — no I/O, no MetricsManager
state. Tests are correspondingly cheap.
"""

from __future__ import annotations

import collections
import dataclasses
import json
from typing import Any

_SCORECARD_HEADLINE_KEYS: tuple[tuple[str, str, str], ...] = (
    # (aggregate key, label, headline stat). Keys carry the measure() operation
    # prefix (save_blocking_/save_background_/load_) spliced by _add_results.
    (
        "save_blocking_4_throughput/save_blocking_gbps",
        "Save blocking throughput (max GiB/s)",
        "max",
    ),
    (
        "save_background_4_throughput/save_total_gbps",
        "Save total throughput (max GiB/s)",
        "max",
    ),
    ("load_4_throughput/load_total_gbps", "Load throughput (max GiB/s)", "max"),
    (
        "load_4_throughput/load_per_host_gbps",
        "Load per-host throughput (max GiB/s)",
        "max",
    ),
    (
        "save_background_5_inventory/save_total_gb",
        "Save total per host (GiB)",
        "max",
    ),
    ("load_5_inventory/load_total_gb", "Load total per host (GiB)", "max"),
    (
        "save_blocking_2_save_breakdown/blocking_async_s",
        "Save blocking (slowest host, s)",
        "max",
    ),
    (
        "load_3_load_breakdown/blocking_s",
        "Load blocking (slowest host, s)",
        "max",
    ),
    (
        "save_blocking_7_overhead/sync_global_devices_s",
        "Sync-barrier overhead (slowest host, s)",
        "max",
    ),
)


def render_scorecard(
    benchmark_name: str,
    aggregates: dict[str, dict[str, float]],
    inventory: Any | None,
    manifest: Any | None,
) -> str:
  """Headline numbers + inventory + run manifest as one markdown blob."""
  lines = [f"## {benchmark_name} — scorecard", ""]

  headline_rows = []
  for agg_key, label, stat in _SCORECARD_HEADLINE_KEYS:
    if agg_key in aggregates and stat in aggregates[agg_key]:
      headline_rows.append((label, aggregates[agg_key][stat]))

  if headline_rows:
    lines.extend(["### Headline numbers", ""])
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    for label, value in headline_rows:
      lines.append(f"| {label} | {value:.4f} |")
    lines.append("")

  if inventory is not None:
    lines.extend(["### Inventory", ""])
    lines.append("| field | value |")
    lines.append("|---|---:|")
    total_gb = inventory.total_bytes / (1024**3)
    lines.append(f"| total bytes | {total_gb:.2f} GiB |")
    lines.append(f"| file count | {inventory.file_count:,} |")
    small_pct = inventory.small_file_pct * 100
    canary = "✓" if small_pct < 10 else "⚠ chunk_byte_size too small?"
    lines.append(f"| small files <1 MiB | {small_pct:.1f}% {canary} |")
    if inventory.largest_file_bytes > 0:
      lines.append(
          f"| largest file | {inventory.largest_file_bytes / (1024**2):.2f}"
          " MiB |"
      )
    if inventory.format:
      fmt_str = ", ".join(
          f"{k}={v}" for k, v in sorted(inventory.format.items())
      )
      lines.append(f"| format breakdown | {fmt_str} |")
    lines.append("")

  if manifest is not None:
    lines.append(manifest.as_markdown())

  return "\n".join(lines)


def render_configuration(
    benchmark_name: str,
    benchmark_options: dict[str, Any] | None,
    checkpoint_config: dict[str, Any] | None,
) -> str:
  """Renders the run configuration as readable markdown.

  Options and checkpoint_config become field/value tables; any nested dict in
  checkpoint_config (typically `spec`) becomes its own fenced-JSON block.

  Args:
    benchmark_name: Title rendered as the top-level heading.
    benchmark_options: Flat option name/value pairs, or None to omit.
    checkpoint_config: Checkpoint config; scalars form a table and nested dicts
      each become a fenced-JSON block.

  Returns:
    The configuration rendered as a markdown string.
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


def render_aggregated_metrics(
    benchmark_name: str,
    aggregated_stats_dict: dict[str, Any],
    metric_units: dict[str, str],
    host_label: str | None = None,
) -> str:
  """Renders mean/std/min/max/n/unit per metric, grouped by `/` prefix.

  The numbered section prefix (`2_save_breakdown/`, …) mirrors the
  Scalars-view navigation so a reader finds a metric the same way in both
  surfaces.

  Args:
    benchmark_name: Title rendered as the top-level heading.
    aggregated_stats_dict: Metric key -> AggregatedStats to tabulate.
    metric_units: Metric key -> unit string.
    host_label: When set, appended to the header to identify the host.

  Returns:
    The aggregated metrics rendered as a markdown string.
  """
  if not aggregated_stats_dict:
    return "_No successful runs to aggregate._"

  groups: dict[str, list[str]] = collections.defaultdict(list)
  for key in sorted(aggregated_stats_dict):
    head, _, _ = key.partition("/")
    section = head if "/" in key else "_other_"
    groups[section].append(key)

  suffix = f" — {host_label}" if host_label else ""
  lines = [f"## {benchmark_name} — aggregated metrics{suffix}", ""]
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


def options_to_hparams(options: Any) -> dict[str, bool | int | float | str]:
  """Flattens an options dataclass/dict into a TB HParams-acceptable form.

  HParams values must be primitives; anything else (None, list, tuple, nested)
  is `str()`-ified so the run still appears in Parallel Coordinates instead of
  being dropped.

  Args:
    options: An options dataclass instance or dict.

  Returns:
    A flat dict of primitive HParams values; empty if options is neither a
    dataclass nor a dict.
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
