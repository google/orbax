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

"""Benchmark baselines — schema, recorder, and comparer.

A `Baseline` is the captured state of one benchmark run that future runs
diff against. The schema versioning is intentional — a captured baseline
may outlive several benchmark-framework changes, and `BaselineComparer`
refuses silently-incompatible payloads rather than mis-aligning.

`BaselineRecorder` writes a payload to a sink; `BaselineComparer` reads one
back and diffs metrics against a current run. Paths are resolved with
`etils.epath`, so local and `gs://` sinks work transparently.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Any

from etils import epath

_SCHEMA_VERSION = 1


@dataclasses.dataclass
class Baseline:
  """Snapshot of a benchmark run that future runs compare against.

  Field semantics:
    benchmark_name: Stable identifier matching the generator's class name.
    captured_at:    ISO 8601 UTC timestamp.
    captured_at_sha: Code SHA at capture time (from the run manifest).
    fixture:        Path / repo id of the input the benchmark was driven by.
    config:         Full options + ckpt_config + mesh_config dict.
    metrics:        Per-metric-key → {stat: value} cross-host aggregate
                    (max/min/mean across hosts; p50/p99 with >1 host).
    digests:        Per-leaf SHA-256 hex (for L2 digest-mode comparison).
                    Empty dict if the benchmark didn't capture digests.
    inventory:      CheckpointInventory snapshot dict (post-save).
    manifest:       RunManifest snapshot dict (code SHA, library versions,
                    env, topology).
  """

  benchmark_name: str
  captured_at: str
  captured_at_sha: str
  fixture: str
  config: dict[str, Any]
  metrics: dict[str, dict[str, float]]
  digests: dict[str, str] = dataclasses.field(default_factory=dict)
  inventory: dict[str, Any] = dataclasses.field(default_factory=dict)
  manifest: dict[str, Any] = dataclasses.field(default_factory=dict)
  schema_version: int = _SCHEMA_VERSION

  def to_json(self, *, indent: int | None = 2) -> str:
    return json.dumps(dataclasses.asdict(self), indent=indent, sort_keys=True)

  @classmethod
  def from_json(cls, payload: str) -> "Baseline":
    """Parses a Baseline from JSON, rejecting incompatible schema versions."""
    raw = json.loads(payload)
    version = raw.pop("schema_version", None)
    if version != _SCHEMA_VERSION:
      raise ValueError(
          f"Baseline schema_version {version} != supported {_SCHEMA_VERSION}; "
          "regenerate the baseline against the current framework."
      )
    # Coerce metric values to float — JSON parses ints as ints which would
    # otherwise propagate and confuse the speedup ratio calculations.
    raw["metrics"] = {
        key: {stat: float(v) for stat, v in stats.items()}
        for key, stats in raw.get("metrics", {}).items()
    }
    return cls(**raw)


class BaselineRecorder:
  """Writes Baseline payloads to a chosen sink (local path or gs://-mirror)."""

  def __init__(self, sink_path: str):
    """`sink_path` is a directory; baselines land at `<dir>/<sha>.json`."""
    self._sink = epath.Path(sink_path)

  def write(self, baseline: Baseline) -> pathlib.Path:
    self._sink.mkdir(parents=True, exist_ok=True)
    sha = baseline.captured_at_sha or "unknown"
    out = self._sink / f"{sha}.json"
    out.write_text(baseline.to_json())
    return pathlib.Path(str(out))


@dataclasses.dataclass(frozen=True)
class MetricDelta:
  """Per-metric diff between a current value and a baseline value.

  `ratio` is "speedup" for time-shaped metrics — baseline_value / current_value
  when smaller is better (the usual case). The caller renders it: 2.0 means
  the current run took half as long, displayed as "2.0× faster".
  We don't know per-metric which direction is "better" — that judgement is
  the caller's. The Comparer just reports the raw ratio + diff.
  """

  key: str
  baseline: float
  current: float
  delta_abs: float
  ratio: float | None  # None when baseline is 0 (avoids divide-by-zero)


@dataclasses.dataclass
class ComparisonReport:
  baseline: Baseline
  deltas: list[MetricDelta]
  missing_in_current: list[str]
  missing_in_baseline: list[str]


class BaselineComparer:
  """Loads a stored baseline and produces deltas against a current run."""

  def __init__(self, source_path: str):
    """`source_path` is the full path to a baseline JSON file."""
    self._source = epath.Path(source_path)

  def load(self) -> Baseline:
    return Baseline.from_json(self._source.read_text())

  def compare(
      self,
      current_metrics: dict[str, dict[str, float]],
      baseline: Baseline | None = None,
      *,
      stat: str = "mean",
  ) -> ComparisonReport:
    """Diffs `current_metrics` against the baseline on one stat per key.

    Args:
      current_metrics: Per-metric-key → {stat: value} for the current run.
      baseline: Baseline to compare against; loaded from disk if None.
      stat: Which cross-host stat to diff (e.g. "mean", "max").

    Returns:
      A ComparisonReport with one MetricDelta per shared key that carries the
      requested stat on both sides.
    """
    if baseline is None:
      baseline = self.load()
    deltas: list[MetricDelta] = []
    common = sorted(set(baseline.metrics) & set(current_metrics))
    for key in common:
      if stat not in baseline.metrics[key] or stat not in current_metrics[key]:
        continue
      b = float(baseline.metrics[key][stat])
      c = float(current_metrics[key][stat])
      ratio = (b / c) if c != 0 else None
      deltas.append(
          MetricDelta(
              key=key,
              baseline=b,
              current=c,
              delta_abs=c - b,
              ratio=ratio,
          )
      )
    return ComparisonReport(
        baseline=baseline,
        deltas=deltas,
        missing_in_current=sorted(set(baseline.metrics) - set(current_metrics)),
        missing_in_baseline=sorted(
            set(current_metrics) - set(baseline.metrics)
        ),
    )
