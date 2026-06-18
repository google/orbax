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

  - at_a_glance: the screenshot-able card — load/save total time + per-stage
    breakdown, per-host byte counts (the sharding check), inventory detail,
    and the run manifest.
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
  """Quick-glance load/save card: headline + per-stage + per-host breakdown.

  Surfaces the jax.monitoring load/save breakdown, throughput, and per-host
  byte counts already captured by the default metrics in one screenshot-able
  card. The per-host byte rows (and their share of the total) are the sharding
  check: each host should read/write only its slice. The checkpoint inventory
  and run manifest are appended for provenance.

  Args:
    benchmark_name: Title rendered as the top-level heading.
    aggregates: Cross-host aggregate stats per metric key.
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
  lines = [f"## {benchmark_name} — at a glance", ""]
  lines += _glance_section(_LOAD_SPEC, aggregates, per_host_values, on_disk_gb)
  lines += _glance_section(_SAVE_SPEC, aggregates, per_host_values, on_disk_gb)
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


@dataclasses.dataclass(frozen=True)
class _OpSpec:
  """Declarative spec for one operation (load/save) section of the card."""

  title: str
  bytes_verb: str  # "read" / "written"
  blocking_keys: list[str]
  total_time_keys: list[str]
  bytes_keys: list[str]  # per-host byte counter(s)
  op_count_keys: list[str]  # per-host physical read/write op count
  tput_label: str
  tput_keys: list[list[str]]  # [left candidates, right candidates]
  hbm_keys: list[str]
  breakdown: list[tuple[str, list[str]]]
  per_host_cols: list[tuple[str, list[str], str]]  # (label, keys, kind)


_LOAD_SPEC = _OpSpec(
    title="Load",
    bytes_verb="read",
    blocking_keys=["load_3_load_breakdown/blocking_s"],
    total_time_keys=["load_3_load_breakdown/total_s"],
    bytes_keys=[
        "load_6_tensorstore/tensorstore_kvstore_file_bytes_read_value_diff",
        "load_6_tensorstore/tensorstore_kvstore_gcs_bytes_read_value_diff",
    ],
    op_count_keys=[
        "load_6_tensorstore/tensorstore_kvstore_file_read_value_diff",
        "load_6_tensorstore/tensorstore_kvstore_gcs_read_value_diff",
    ],
    tput_label="Throughput (host / total)",
    tput_keys=[
        ["load_4_throughput/load_per_host_gbps"],
        ["load_4_throughput/load_total_gbps"],
    ],
    hbm_keys=["load_7_memory/device_hbm_peak_diff_gb"],
    breakdown=[
        ("worker I/O (storage read)", ["load_3_load_breakdown/worker_io_s"]),
        (
            "primary deserialize",
            ["load_3_load_breakdown/primary_deserialization_s"],
        ),
        ("broadcast", ["load_3_load_breakdown/broadcast_s"]),
        ("blocking", ["load_3_load_breakdown/blocking_s"]),
        ("total", ["load_3_load_breakdown/total_s"]),
    ],
    per_host_cols=[
        (
            "bytes read",
            [
                "load_6_tensorstore/tensorstore_kvstore_file_bytes_read_value_diff",
                "load_6_tensorstore/tensorstore_kvstore_gcs_bytes_read_value_diff",
            ],
            "bytes",
        ),
        (
            "reads",
            [
                "load_6_tensorstore/tensorstore_kvstore_file_read_value_diff",
                "load_6_tensorstore/tensorstore_kvstore_gcs_read_value_diff",
            ],
            "count",
        ),
        ("blocking s", ["load_3_load_breakdown/blocking_s"], "num"),
        ("total s", ["load_3_load_breakdown/total_s"], "num"),
        ("worker I/O s", ["load_3_load_breakdown/worker_io_s"], "num"),
    ],
)


_SAVE_SPEC = _OpSpec(
    title="Save",
    bytes_verb="written",
    blocking_keys=["save_blocking_2_save_breakdown/blocking_s"],
    total_time_keys=[
        "save_background_2_save_breakdown/total_async_s",
        "save_background_2_save_breakdown/total_s",
        "save_blocking_2_save_breakdown/total_s",
    ],
    bytes_keys=[
        "save_background_6_tensorstore/tensorstore_kvstore_file_bytes_written_value_diff",
        "save_blocking_6_tensorstore/tensorstore_kvstore_file_bytes_written_value_diff",
        "save_background_6_tensorstore/tensorstore_kvstore_gcs_bytes_written_value_diff",
        "save_blocking_6_tensorstore/tensorstore_kvstore_gcs_bytes_written_value_diff",
    ],
    op_count_keys=[
        "save_background_6_tensorstore/tensorstore_kvstore_file_write_value_diff",
        "save_blocking_6_tensorstore/tensorstore_kvstore_file_write_value_diff",
        "save_background_6_tensorstore/tensorstore_kvstore_gcs_write_value_diff",
        "save_blocking_6_tensorstore/tensorstore_kvstore_gcs_write_value_diff",
    ],
    tput_label="Throughput (blocking / total)",
    tput_keys=[
        ["save_blocking_4_throughput/save_blocking_gbps"],
        ["save_background_4_throughput/save_total_gbps"],
    ],
    hbm_keys=[
        "save_background_7_memory/device_hbm_peak_diff_gb",
        "save_blocking_7_memory/device_hbm_peak_diff_gb",
    ],
    breakdown=[
        (
            "directory creation",
            [
                "save_blocking_2_save_breakdown/directory_creation_s",
                "save_background_2_save_breakdown/async_directory_creation_s",
            ],
        ),
        (
            "metadata write",
            [
                "save_background_2_save_breakdown/metadata_write_s",
                "save_blocking_2_save_breakdown/metadata_write_s",
            ],
        ),
        (
            "commit",
            [
                "save_background_2_save_breakdown/commit_s",
                "save_blocking_2_save_breakdown/commit_s",
            ],
        ),
        (
            "ocdbt merge",
            [
                "save_background_2_save_breakdown/ocdbt_merge_s",
                "save_blocking_2_save_breakdown/ocdbt_merge_s",
            ],
        ),
        ("blocking", ["save_blocking_2_save_breakdown/blocking_s"]),
        (
            "total",
            [
                "save_background_2_save_breakdown/total_async_s",
                "save_background_2_save_breakdown/total_s",
            ],
        ),
    ],
    per_host_cols=[
        (
            "bytes written",
            [
                "save_background_6_tensorstore/tensorstore_kvstore_file_bytes_written_value_diff",
                "save_blocking_6_tensorstore/tensorstore_kvstore_file_bytes_written_value_diff",
                "save_background_6_tensorstore/tensorstore_kvstore_gcs_bytes_written_value_diff",
                "save_blocking_6_tensorstore/tensorstore_kvstore_gcs_bytes_written_value_diff",
            ],
            "bytes",
        ),
        (
            "writes",
            [
                "save_background_6_tensorstore/tensorstore_kvstore_file_write_value_diff",
                "save_blocking_6_tensorstore/tensorstore_kvstore_file_write_value_diff",
                "save_background_6_tensorstore/tensorstore_kvstore_gcs_write_value_diff",
                "save_blocking_6_tensorstore/tensorstore_kvstore_gcs_write_value_diff",
            ],
            "count",
        ),
        ("blocking s", ["save_blocking_2_save_breakdown/blocking_s"], "num"),
    ],
)


def _glance_section(
    spec: _OpSpec,
    aggregates: dict[str, dict[str, float]],
    per_host_values: list[tuple[int, dict[str, float]]],
    on_disk_gb: float | None,
) -> list[str]:
  """Renders one operation (load/save) section from its `_OpSpec`."""
  blocking = _glance_agg(aggregates, spec.blocking_keys, "max")
  total_time = _glance_agg(aggregates, spec.total_time_keys, "max")
  per_host_mean = _glance_agg(aggregates, spec.bytes_keys, "mean")
  per_host_min = _glance_agg(aggregates, spec.bytes_keys, "min")
  per_host_max = _glance_agg(aggregates, spec.bytes_keys, "max")
  if not any(v is not None for v in (blocking, total_time, per_host_mean)):
    return [
        f"### {spec.title}",
        "",
        f"_(no {spec.title.lower()} in this run)_",
        "",
    ]

  reads_mean = _glance_agg(aggregates, spec.op_count_keys, "mean")
  reads_min = _glance_agg(aggregates, spec.op_count_keys, "min")
  reads_max = _glance_agg(aggregates, spec.op_count_keys, "max")
  tput_left = _glance_agg(aggregates, spec.tput_keys[0], "max")
  tput_right = _glance_agg(aggregates, spec.tput_keys[1], "max")
  hbm = _glance_agg(aggregates, spec.hbm_keys, "max")

  out = [f"### {spec.title}", "", "| metric | value |", "|---|---:|"]
  out.append(
      f"| Total time (blocking / total) | {_glance_num(blocking)} /"
      f" {_glance_num(total_time)} s |"
  )
  if per_host_mean is not None:
    out.append(
        f"| Bytes {spec.bytes_verb} / host (mean · min · max) |"
        f" {_humanize_gib(per_host_mean)} · {_humanize_gib(per_host_min)} ·"
        f" {_humanize_gib(per_host_max)} |"
    )
  if reads_mean is not None:
    verb = "Reads" if spec.bytes_verb == "read" else "Writes"
    rmean = _glance_num(reads_mean, "{:.0f}")
    rmin = _glance_num(reads_min, "{:.0f}")
    rmax = _glance_num(reads_max, "{:.0f}")
    out.append(
        f"| {verb} / host (mean · min · max) | {rmean} · {rmin} · {rmax} |"
    )
  if on_disk_gb is not None:
    out.append(f"| Checkpoint size (on-disk) | {_humanize_gib(on_disk_gb)} |")
  # Sharding check: per-host bytes vs the on-disk checkpoint size. ~1 means
  # every host moved ~the whole checkpoint (replicated); ~1/N means each host
  # handled only its shard. (Dividing by the per-device-inflated logical-read
  # counter would be wrong — it tracks local-device copies, not sharding.)
  if per_host_mean is not None and on_disk_gb:
    ratio = per_host_mean / on_disk_gb
    flag = (
        "✓ sharded"
        if ratio <= 0.9
        else "⚠ each host handles ~the whole checkpoint"
    )
    out.append(f"| Per-host ÷ checkpoint | {ratio:.3f}  {flag} |")
  if tput_left is not None or tput_right is not None:
    out.append(
        f"| {spec.tput_label} | {_glance_num(tput_left)} /"
        f" {_glance_num(tput_right)} GiB/s |"
    )
  if hbm is not None:
    out.append(f"| HBM peak / host (max) | {_humanize_gib(hbm)} |")
  out.append("")

  rows = [
      (label, _glance_agg(aggregates, keys, "mean"))
      for label, keys in spec.breakdown
  ]
  rows = [(label, v) for label, v in rows if v is not None]
  if rows:
    out += [
        f"#### {spec.title} time breakdown (mean s)",
        "",
        "| stage | s |",
        "|---|---:|",
    ]
    out += [f"| {label} | {_glance_num(v)} |" for label, v in rows]
    out.append("")

  out += _per_host_table(
      per_host_values,
      spec.per_host_cols,
      pct_keys=spec.bytes_keys,
      pct_of=on_disk_gb,
  )
  return out


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
