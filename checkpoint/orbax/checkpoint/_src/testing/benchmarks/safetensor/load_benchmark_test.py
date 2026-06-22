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

"""Tests for the SafetensorLoadBenchmark generator class."""

import os
import tempfile

from absl.testing import absltest
import numpy as np
from orbax.checkpoint._src.testing.benchmarks.core import configs as benchmark_configs
from orbax.checkpoint._src.testing.benchmarks.safetensor import load_benchmark as slb
import safetensors.numpy as st_np


def _small_fixture(d: str) -> str:
  """Writes a tiny 2-tensor `model.safetensors` for the load smoke test."""
  st_np.save_file(
      {
          "w": np.zeros((32, 32), dtype=np.float32),
          "b": np.zeros((32,), dtype=np.float32),
      },
      os.path.join(d, "model.safetensors"),
  )
  return d


class OptionsTest(absltest.TestCase):

  def test_requires_checkpoint_path(self):
    opts = slb.SafetensorLoadBenchmarkOptions()
    self.assertFalse(opts.is_valid())

  def test_with_path_is_valid(self):
    opts = slb.SafetensorLoadBenchmarkOptions(checkpoint_path="/tmp/x")
    self.assertTrue(opts.is_valid())

  def test_context_pins_safetensors_layout(self):
    from orbax.checkpoint import v1 as ocp

    opts = slb.SafetensorLoadBenchmarkOptions(checkpoint_path="/tmp/x")
    ctx = opts.context
    self.assertEqual(
        ctx.checkpoint_layout,
        ocp.options.CheckpointLayout.SAFETENSORS,
    )


class GenerationTest(absltest.TestCase):

  def test_generator_emits_one_benchmark_per_option_combo(self):
    opts = slb.SafetensorLoadBenchmarkOptions(checkpoint_path="/tmp/x")
    gen = slb.SafetensorLoadBenchmark(
        checkpoint_configs=[benchmark_configs.CheckpointConfig()],
        options=opts,
    )
    benchmarks = gen.generate(skip_incompatible_mesh_configs=False)
    self.assertGreaterEqual(len(benchmarks), 1)


class LoadFlowTest(absltest.TestCase):
  """Smoke test that the test_fn actually loads through SafetensorsLayout.

  Runs against a tiny local safetensors file on a single process.
  Multi-host behaviour is covered by the docker smoke harness, not here.
  """

  def test_loads_local_safetensors(self):
    with tempfile.TemporaryDirectory() as d:
      _small_fixture(d)
      opts = slb.SafetensorLoadBenchmarkOptions(checkpoint_path=d)
      gen = slb.SafetensorLoadBenchmark(
          checkpoint_configs=[benchmark_configs.CheckpointConfig()],
          options=opts,
      )
      generated = gen.generate(skip_incompatible_mesh_configs=False)
      bench = generated[0]
      result = bench.run(repeat_index=0)
      self.assertTrue(result.is_successful(), msg=str(result.error))
      # Time scalar surfaces under the standard 0_basics/ namespace.
      time_keys = [
          k
          for k in result.metrics.results
          if k.startswith("load::0_basics/time_s")
      ]
      self.assertNotEmpty(time_keys)
      # The loader self-reports per-host reads via jax.monitoring, so the load
      # card populates even though there are no TensorStore counters.
      self.assertIn(
          "load::6_io/file_bytes_read_per_host", result.metrics.results
      )
      self.assertIn(
          "load::6_io/file_reads_per_host_count", result.metrics.results
      )

  def test_digests_captured_to_path(self):
    with tempfile.TemporaryDirectory() as d:
      _small_fixture(d)
      digests_dir = os.path.join(d, "digests")
      opts = slb.SafetensorLoadBenchmarkOptions(
          checkpoint_path=d,
          capture_digests_path=digests_dir,
      )
      gen = slb.SafetensorLoadBenchmark(
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
