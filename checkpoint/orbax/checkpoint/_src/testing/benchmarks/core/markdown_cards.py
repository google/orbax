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

  - at_a_glance: the screenshot-able headline — total time, throughput,
    per-host bytes (with each host's share of the on-disk size), inventory,
    and the run manifest. Headline rows are generic candidate-key lookups, so
    the same renderer serves a load card and a save card: whichever of the
    candidate keys the card's metrics populate is the one shown.
  - configuration: options + checkpoint config tables, plus pretty-printed
    JSON for nested fields like `spec`.
  - aggregated_metrics: every metric's mean / std / min / max / n / unit,
    grouped by the numeric section prefix (`2_save_breakdown/` etc.). This is
    the per-stage breakdown — it is built straight from the jax-exported tags.

Everything in this module is a pure function — no I/O, no MetricsManager
state. Tests are correspondingly cheap.
"""

from __future__ import annotations

import collections
import dataclasses
import json
import math
from typing import Any

import humanize


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


def _glance_num(value: float | None, fmt: str = "{:.2f}") -> str:
  """Formats a value for a glance card, or an em-dash when missing."""
  return fmt.format(value) if value is not None else "—"


def _humanize_bytes(num_bytes: float | None) -> str:
  """Formats a byte count using the humanize library, or em-dash if missing."""
  if num_bytes is None:
    return "—"
  if abs(num_bytes) < 1024:
    return f"{num_bytes:.0f} B"
  return humanize.naturalsize(num_bytes, binary=True, format="%.2f")


def _humanize_gib(value_gib: float | None) -> str:
  """Humanizes a value expressed in GiB into the most readable binary unit."""
  return _humanize_bytes(None if value_gib is None else value_gib * 1024**3)


def _glance_agg(
    aggregates: dict[str, dict[str, float]],
    keys: list[str],
    stat: str = "mean",
) -> float | None:
  """Returns the first non-NaN `stat` across candidate keys, or None."""
  for key in keys:
    stats = aggregates.get(key)
    if stats is not None:
      value = stats.get(stat)
      if value is not None and not math.isnan(value):
        if "bytes" in key and "gb" not in key:
          return value / (1024**3)
        return value
  return None


def _glance_host(
    host_values: dict[str, float], keys: list[str] | None
) -> float | None:
  """Returns the first non-NaN value across candidate keys for one host."""
  for key in keys or []:
    if key in host_values and not math.isnan(host_values[key]):
      value = host_values[key]
      if "bytes" in key and "gb" not in key:
        return value / (1024**3)
      return value
  return None


def _per_host_table(
    per_host_values: list[tuple[int, dict[str, float]]],
    columns: list[tuple[str, list[str], str]],
    pct_keys: list[str] | None = None,
    pct_of: float | None = None,
) -> list[str]:
  """Renders one row per host for the columns at least one host reported.

  Args:
    per_host_values: (host_index, metric->mean) per host, sorted by index.
    columns: (column label, candidate metric keys, kind) tuples; kind is "bytes"
      (GiB value, humanized), "count" (integer), or "num" (decimal).
    pct_keys: Candidate keys whose value is shown as a percentage of `pct_of`.
    pct_of: Denominator for the percentage column; omitted when falsy.

  Returns:
    Markdown lines for the table, or empty if there is nothing to show.
  """
  present = [
      (label, keys, kind)
      for label, keys, kind in columns
      if any(_glance_host(v, keys) is not None for _, v in per_host_values)
  ]
  if not per_host_values or not present:
    return []
  show_pct = False
  if pct_keys is not None and pct_of:
    show_pct = any(
        _glance_host(v, pct_keys) is not None for _, v in per_host_values
    )
  header = "| host |" + "".join(f" {label} |" for label, _, _ in present)
  separator = "|---|" + "---:|" * len(present)
  if show_pct:
    header += " % of total |"
    separator += "---:|"
  lines = ["#### Per-host", "", header, separator]
  for idx, values in per_host_values:
    row = f"| {idx} |"
    for _, keys, kind in present:
      value = _glance_host(values, keys)
      if kind == "bytes":
        cell = _humanize_gib(value)
      elif kind == "count":
        cell = _glance_num(value, "{:.0f}")
      else:
        cell = _glance_num(value)
      row += f" {cell} |"
    if show_pct and pct_keys is not None and pct_of:
      part = _glance_host(values, pct_keys)
      row += f" {part / pct_of * 100:.1f}% |" if part is not None else " — |"
    lines.append(row)
  lines.append("")
  return lines


def render_glance_card(
    benchmark_name: str,
    aggregates: dict[str, dict[str, float]],
    per_host_values: list[tuple[int, dict[str, float]]],
    inventory: Any | None,
    manifest: Any | None,
) -> str:
  """Quick-glance headline card: a few generic rows + per-host + inventory.

  The headline rows are candidate-key lookups whose lists cover both load and
  save tags, so the renderer needs no per-kind template: a load card's metrics
  populate the load candidates, a save card's the save candidates, and only the
  resolved ones render. The per-host byte rows (and each host's share of the
  on-disk size) are the sharding check: each host should move only its slice.
  The full per-stage breakdown lives in the `aggregated_metrics` card; the
  inventory and run manifest are appended here for provenance.

  Args:
    benchmark_name: Title rendered as the top-level heading.
    aggregates: Cross-host aggregate stats per metric key (already sliced to one
      operation name, so keys are bare tags).
    per_host_values: (host_index, metric->mean) per host, sorted by index.
    inventory: Optional directory inventory; supplies on-disk total bytes, file
      count, small-file canary, and format breakdown.
    manifest: Optional suite run manifest; appended via `as_markdown()`.

  Returns:
    The card rendered as a markdown string.
  """
  on_disk_gb = None
  if inventory is not None and getattr(inventory, "total_bytes", 0):
    on_disk_gb = inventory.total_bytes / (1024**3)
  lines = [
      f"## {benchmark_name} — at a glance",
      "",
      "| metric | value |",
      "|---|---:|",
  ]
  total = _glance_agg(aggregates, _TOTAL_TIME_KEYS, "max")
  lines.append(f"| Total time | {_glance_num(total)} s |")
  throughput = _glance_agg(aggregates, _THROUGHPUT_KEYS, "max")
  if throughput is not None:
    lines.append(f"| Throughput | {_glance_num(throughput)} GiB/s |")
  bytes_per_host = _glance_agg(aggregates, _BYTES_KEYS, "mean")
  if bytes_per_host is not None:
    lines.append(f"| Bytes / host (mean) | {_humanize_gib(bytes_per_host)} |")
  if on_disk_gb is not None:
    lines.append(f"| Checkpoint size (on-disk) | {_humanize_gib(on_disk_gb)} |")
  hbm = _glance_agg(aggregates, ["7_memory/device_hbm_peak_diff_gb"], "max")
  if hbm is not None:
    lines.append(f"| HBM peak / host (max) | {_humanize_gib(hbm)} |")
  lines.append("")
  lines += _per_host_table(
      per_host_values, _PER_HOST_COLS, pct_keys=_BYTES_KEYS, pct_of=on_disk_gb
  )
  lines += _glance_inventory(inventory)
  if manifest is not None:
    lines.append(manifest.as_markdown())
  return "\n".join(lines).rstrip() + "\n"


def _glance_inventory(inventory: Any | None) -> list[str]:
  """Renders the checkpoint inventory block (folded in from the scorecard)."""
  if inventory is None:
    return []
  out = ["### Inventory", "", "| field | value |", "|---|---:|"]
  out.append(f"| total bytes | {_humanize_bytes(inventory.total_bytes)} |")
  out.append(f"| file count | {inventory.file_count:,} |")
  small_pct = inventory.small_file_pct * 100
  canary = "✓" if small_pct < 10 else "⚠ chunk_byte_size too small?"
  out.append(f"| small files <1 MiB | {small_pct:.1f}% {canary} |")
  if inventory.largest_file_bytes > 0:
    out.append(
        f"| largest file | {_humanize_bytes(inventory.largest_file_bytes)} |"
    )
  if inventory.format:
    fmt_str = ", ".join(f"{k}={v}" for k, v in sorted(inventory.format.items()))
    out.append(f"| format breakdown | {fmt_str} |")
  out.append("")
  return out


_TOTAL_TIME_KEYS = [
    "3_load_breakdown/total_s",
    "2_save_breakdown/total_async_s",
    "2_save_breakdown/total_s",
    "0_basics/time_s",
]
_THROUGHPUT_KEYS = [
    "4_throughput/load_total_gbps",
    "4_throughput/save_total_gbps",
    "4_throughput/load_per_host_gbps",
    "4_throughput/save_blocking_gbps",
]
_BYTES_KEYS = [
    "6_io/file_bytes_read_per_host",
    "6_tensorstore/tensorstore_kvstore_file_bytes_read_value_diff",
    "6_tensorstore/tensorstore_kvstore_gcs_bytes_read_value_diff",
    "6_tensorstore/tensorstore_kvstore_file_bytes_written_value_diff",
    "6_tensorstore/tensorstore_kvstore_gcs_bytes_written_value_diff",
]
_READ_WRITE_COUNT_KEYS = [
    "6_io/file_reads_per_host_count",
    "6_tensorstore/tensorstore_kvstore_file_read_value_diff",
    "6_tensorstore/tensorstore_kvstore_gcs_read_value_diff",
    "6_tensorstore/tensorstore_kvstore_file_write_value_diff",
    "6_tensorstore/tensorstore_kvstore_gcs_write_value_diff",
]
_BLOCKING_TIME_KEYS = [
    "3_load_breakdown/blocking_s",
    "2_save_breakdown/blocking_s",
]
# Per-host columns for the at-a-glance card. Candidate-key lists cover load and
# save flavours; `_BYTES_KEYS` also feeds the "% of on-disk" sharding column.
_PER_HOST_COLS = [
    ("bytes", _BYTES_KEYS, "bytes"),
    ("reads/writes", _READ_WRITE_COUNT_KEYS, "count"),
    ("blocking s", _BLOCKING_TIME_KEYS, "num"),
    ("total s", _TOTAL_TIME_KEYS, "num"),
]


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
