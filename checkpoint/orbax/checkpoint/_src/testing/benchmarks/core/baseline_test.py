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

"""Tests for the Baseline schema, recorder, and comparer."""

import json
import tempfile

from absl.testing import absltest
from orbax.checkpoint._src.testing.benchmarks.core import baseline as baseline_lib


def _example() -> baseline_lib.Baseline:
  return baseline_lib.Baseline(
      benchmark_name="MyBench",
      captured_at="2026-05-26T10:00:00Z",
      captured_at_sha="abc1234",
      fixture="/tmp/fixture",
      config={"async_enabled": True, "use_ocdbt": True},
      metrics={
          "save_0_basics/time_s": {"max": 1.6, "mean": 1.5},
          "load_0_basics/time_s": {"max": 2.1, "mean": 2.0},
      },
      digests={"layer0/W": "deadbeef" * 8},
      inventory={"total_bytes": 1234},
      manifest={"jax_version": "0.4.99"},
  )


def _baseline(sha: str = "abc1234", **overrides) -> baseline_lib.Baseline:
  defaults = dict(
      benchmark_name="MyBench",
      captured_at="2026-05-26T10:00:00Z",
      captured_at_sha=sha,
      fixture="/tmp/fixture",
      config={"async_enabled": True},
      metrics={
          "save_0_basics/time_s": {"mean": 2.0},
          "load_0_basics/time_s": {"mean": 4.0},
      },
  )
  defaults.update(overrides)
  return baseline_lib.Baseline(**defaults)


class BaselineRoundTripTest(absltest.TestCase):

  def test_to_from_json_round_trip(self):
    original = _example()
    restored = baseline_lib.Baseline.from_json(original.to_json())
    self.assertEqual(restored, original)

  def test_json_is_sorted_for_diffable_writes(self):
    payload = _example().to_json()
    parsed = json.loads(payload)
    # sort_keys=True must place 'benchmark_name' alphabetically.
    self.assertEqual(list(parsed.keys())[0], "benchmark_name")

  def test_metrics_coerced_to_float_on_load(self):
    raw = json.loads(_example().to_json())
    raw["metrics"]["save_0_basics/time_s"]["mean"] = 3  # int, not float
    restored = baseline_lib.Baseline.from_json(json.dumps(raw))
    self.assertIsInstance(
        restored.metrics["save_0_basics/time_s"]["mean"], float
    )

  def test_unsupported_schema_version_raises(self):
    raw = json.loads(_example().to_json())
    raw["schema_version"] = 99
    with self.assertRaises(ValueError):
      baseline_lib.Baseline.from_json(json.dumps(raw))

  def test_missing_optional_fields_defaults(self):
    minimal = baseline_lib.Baseline(
        benchmark_name="X",
        captured_at="2026-05-26T10:00:00Z",
        captured_at_sha="x",
        fixture="x",
        config={},
        metrics={"a": {"mean": 1.0}},
    )
    self.assertEqual(minimal.digests, {})
    self.assertEqual(minimal.inventory, {})
    self.assertEqual(minimal.manifest, {})

  def test_schema_version_persisted_in_payload(self):
    payload = json.loads(_example().to_json())
    self.assertEqual(payload["schema_version"], 1)


class BaselineRecorderTest(absltest.TestCase):

  def test_writes_named_by_sha(self):
    with tempfile.TemporaryDirectory() as d:
      r = baseline_lib.BaselineRecorder(d)
      written = r.write(_baseline(sha="abc1234"))
      self.assertTrue(written.exists())
      self.assertEqual(written.name, "abc1234.json")

  def test_round_trip_via_recorder_and_comparer(self):
    with tempfile.TemporaryDirectory() as d:
      r = baseline_lib.BaselineRecorder(d)
      original = _baseline(sha="xyz9999")
      written = r.write(original)
      restored = baseline_lib.BaselineComparer(str(written)).load()
      self.assertEqual(restored, original)

  def test_unknown_sha_persists_under_unknown(self):
    with tempfile.TemporaryDirectory() as d:
      r = baseline_lib.BaselineRecorder(d)
      written = r.write(_baseline(sha=""))
      self.assertEqual(written.name, "unknown.json")


class BaselineComparerTest(absltest.TestCase):

  def test_speedup_ratio_for_faster_run(self):
    with tempfile.TemporaryDirectory() as d:
      r = baseline_lib.BaselineRecorder(d)
      written = r.write(_baseline(sha="abc"))
      c = baseline_lib.BaselineComparer(str(written))
      report = c.compare({
          "save_0_basics/time_s": {"mean": 1.0},
          "load_0_basics/time_s": {"mean": 2.0},
      })
      ratios = {d.key: d.ratio for d in report.deltas}
      # Baseline 2.0 / current 1.0 → 2.0× speedup
      self.assertAlmostEqual(ratios["save_0_basics/time_s"], 2.0)  # pyrefly: ignore[no-matching-overload]
      self.assertAlmostEqual(ratios["load_0_basics/time_s"], 2.0)  # pyrefly: ignore[no-matching-overload]

  def test_slowdown_ratio_for_slower_run(self):
    with tempfile.TemporaryDirectory() as d:
      r = baseline_lib.BaselineRecorder(d)
      written = r.write(_baseline(sha="abc"))
      c = baseline_lib.BaselineComparer(str(written))
      report = c.compare({
          "save_0_basics/time_s": {"mean": 4.0},
          "load_0_basics/time_s": {"mean": 8.0},
      })
      ratios = {d.key: d.ratio for d in report.deltas}
      # Baseline 2.0 / current 4.0 → 0.5 (we got slower)
      self.assertAlmostEqual(ratios["save_0_basics/time_s"], 0.5)  # pyrefly: ignore[no-matching-overload]

  def test_missing_metrics_reported(self):
    with tempfile.TemporaryDirectory() as d:
      r = baseline_lib.BaselineRecorder(d)
      written = r.write(_baseline(sha="abc"))
      c = baseline_lib.BaselineComparer(str(written))
      report = c.compare({
          "save_0_basics/time_s": {"mean": 1.0},
          "new_only_in_current": {"mean": 5.0},
      })
      self.assertIn("load_0_basics/time_s", report.missing_in_current)
      self.assertIn("new_only_in_current", report.missing_in_baseline)

  def test_zero_current_returns_none_ratio(self):
    with tempfile.TemporaryDirectory() as d:
      r = baseline_lib.BaselineRecorder(d)
      written = r.write(_baseline(sha="abc"))
      c = baseline_lib.BaselineComparer(str(written))
      report = c.compare({
          "save_0_basics/time_s": {"mean": 0.0},
          "load_0_basics/time_s": {"mean": 1.0},
      })
      ratios = {d.key: d.ratio for d in report.deltas}
      self.assertIsNone(ratios["save_0_basics/time_s"])
      self.assertAlmostEqual(ratios["load_0_basics/time_s"], 4.0)  # pyrefly: ignore[no-matching-overload]

  def test_delta_abs_signed_against_baseline(self):
    with tempfile.TemporaryDirectory() as d:
      r = baseline_lib.BaselineRecorder(d)
      written = r.write(_baseline(sha="abc"))
      c = baseline_lib.BaselineComparer(str(written))
      report = c.compare({
          "save_0_basics/time_s": {"mean": 2.5},
          "load_0_basics/time_s": {"mean": 3.0},
      })
      deltas = {d.key: d.delta_abs for d in report.deltas}
      self.assertAlmostEqual(deltas["save_0_basics/time_s"], 0.5)
      self.assertAlmostEqual(deltas["load_0_basics/time_s"], -1.0)

  def test_stat_selects_aggregate_column(self):
    with tempfile.TemporaryDirectory() as d:
      r = baseline_lib.BaselineRecorder(d)
      written = r.write(
          _baseline(
              sha="abc",
              metrics={"save_0_basics/time_s": {"mean": 2.0, "max": 3.0}},
          )
      )
      c = baseline_lib.BaselineComparer(str(written))
      report = c.compare(
          {"save_0_basics/time_s": {"mean": 1.0, "max": 1.0}}, stat="max"
      )
      ratios = {d.key: d.ratio for d in report.deltas}
      # max column: baseline 3.0 / current 1.0 → 3.0
      self.assertAlmostEqual(ratios["save_0_basics/time_s"], 3.0)  # pyrefly: ignore[no-matching-overload]


if __name__ == "__main__":
  absltest.main()
