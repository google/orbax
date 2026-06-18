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

"""Tests for the OrbaxLoadBenchmark generator class."""

import os
import tempfile

from absl.testing import absltest
from orbax.checkpoint import v1 as ocp
from orbax.checkpoint._src.testing.benchmarks.core import checkpoint_generation
from orbax.checkpoint._src.testing.benchmarks.core import configs as benchmark_configs
from orbax.checkpoint._src.testing.benchmarks.orbax_load import load_benchmark as olb


def _small_fixture(d: str) -> str:
  """Writes a tiny synthetic Orbax checkpoint and returns its path."""
  ckpt = os.path.join(d, "ckpt")
  config = benchmark_configs.CheckpointConfig(
      random_seed=0,
      spec={
          "w": {"dtype": "float32", "shape": [32, 32]},
          "b": {"dtype": "float32", "shape": [32]},
      },
  )
  checkpoint_generation.generate_and_save_checkpoint(config, ckpt, mesh=None)
  return ckpt


class OptionsTest(absltest.TestCase):

  def test_requires_checkpoint_path(self):
    self.assertFalse(olb.OrbaxLoadBenchmarkOptions().is_valid())

  def test_with_path_is_valid(self):
    opts = olb.OrbaxLoadBenchmarkOptions(checkpoint_path="/tmp/x")
    self.assertTrue(opts.is_valid())

  def test_context_inherits_orbax_default(self):
    opts = olb.OrbaxLoadBenchmarkOptions(checkpoint_path="/tmp/x")
    self.assertIsInstance(opts.context, ocp.Context)


class GenerationTest(absltest.TestCase):

  def test_generator_emits_one_benchmark_per_option_combo(self):
    opts = olb.OrbaxLoadBenchmarkOptions(checkpoint_path="/tmp/x")
    gen = olb.OrbaxLoadBenchmark(
        checkpoint_configs=[benchmark_configs.CheckpointConfig()],
        options=opts,
    )
    benchmarks = gen.generate(skip_incompatible_mesh_configs=False)
    self.assertGreaterEqual(len(benchmarks), 1)


class FixtureTest(absltest.TestCase):
  """The synthetic fixture (generated via checkpoint_generation) round-trips."""

  def test_generate_and_save_round_trips(self):
    with tempfile.TemporaryDirectory() as d:
      ckpt = _small_fixture(d)
      restored = ocp.load(ckpt)
      self.assertEqual(set(restored), {"w", "b"})
      self.assertEqual(restored["w"].shape, (32, 32))


class LoadFlowTest(absltest.TestCase):
  """Smoke test that the test_fn actually loads an Orbax checkpoint.

  Runs against a small synthetic fixture on a single process. Multi-host
  behaviour is covered by the docker smoke harness, not by a unit test.
  """

  def test_loads_synthetic_fixture(self):
    with tempfile.TemporaryDirectory() as d:
      ckpt = _small_fixture(d)
      opts = olb.OrbaxLoadBenchmarkOptions(checkpoint_path=ckpt)
      gen = olb.OrbaxLoadBenchmark(
          checkpoint_configs=[benchmark_configs.CheckpointConfig()],
          options=opts,
      )
      bench = gen.generate(skip_incompatible_mesh_configs=False)[0]
      result = bench.run(repeat_index=0)
      self.assertTrue(result.is_successful(), msg=str(result.error))
      # Time scalar surfaces under the standard 0_basics/ namespace.
      time_keys = [
          k
          for k in result.metrics.results
          if k.startswith("load_0_basics/time_s")
      ]
      self.assertNotEmpty(time_keys)

  def test_digests_captured_to_path(self):
    with tempfile.TemporaryDirectory() as d:
      ckpt = _small_fixture(d)
      digests_dir = os.path.join(d, "digests")
      opts = olb.OrbaxLoadBenchmarkOptions(
          checkpoint_path=ckpt,
          capture_digests_path=digests_dir,
      )
      gen = olb.OrbaxLoadBenchmark(
          checkpoint_configs=[benchmark_configs.CheckpointConfig()],
          options=opts,
      )
      bench = gen.generate(skip_incompatible_mesh_configs=False)[0]
      result = bench.run(repeat_index=0)
      self.assertTrue(result.is_successful(), msg=str(result.error))
      # This host's per-leaf digests were written to host_<idx>.json.
      host_file = os.path.join(digests_dir, "host_00000.json")
      self.assertTrue(os.path.exists(host_file))


if __name__ == "__main__":
  absltest.main()
