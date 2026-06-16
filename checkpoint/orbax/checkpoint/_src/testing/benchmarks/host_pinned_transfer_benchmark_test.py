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

# Copyright 2024 The Orbax Authors.
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

"""Unit tests verifying host-pinned D2H/H2D memory transfer microbenchmarks."""

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
from orbax.checkpoint._src.testing.benchmarks import host_pinned_transfer_benchmark
from orbax.checkpoint._src.testing.benchmarks.core import configs as benchmarks_configs
from orbax.checkpoint._src.testing.benchmarks.core import core as benchmarks_core

HostPinnedTransferOptions = (
    host_pinned_transfer_benchmark.HostPinnedTransferOptions
)
HostPinnedTransferBenchmark = (
    host_pinned_transfer_benchmark.HostPinnedTransferBenchmark
)


class HostPinnedTransferBenchmarkTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(
          options=HostPinnedTransferOptions(mode='eager'),
          expected_len=1,
      ),
      dict(
          options=HostPinnedTransferOptions(mode=['eager', 'jit']),
          expected_len=2,
      ),
  )
  def test_generate_benchmarks(self, options, expected_len):
    generator = HostPinnedTransferBenchmark(
        checkpoint_configs=[benchmarks_configs.CheckpointConfig(spec={})],
        options=options,
    )
    benchmarks = generator.generate()
    self.assertLen(benchmarks, expected_len)
    for benchmark in benchmarks:
      self.assertIsInstance(benchmark.options, HostPinnedTransferOptions)

  @parameterized.parameters('eager', 'jit')
  def test_benchmark_test_fn(self, mode):
    generator = HostPinnedTransferBenchmark(
        checkpoint_configs=[benchmarks_configs.CheckpointConfig(spec={})],
        options=HostPinnedTransferOptions(),
    )
    test_path = epath.Path(self.create_tempdir().full_path)
    test_options = HostPinnedTransferOptions(
        shape=(128, 128),
        iterations=2,
        warmup=1,
        mode=mode,
    )
    context = benchmarks_core.TestContext(
        pytree=None, path=test_path, options=test_options
    )

    result = generator.test_fn(context)
    self.assertIsInstance(result, benchmarks_core.TestResult)
    if mode == 'eager':
      self.assertContainsSubset(
          {
              'h2d_pinned_latency_s',
              'h2d_pinned_throughput_gbps',
              'h2d_unpinned_latency_s',
              'h2d_unpinned_throughput_gbps',
              'd2h_pinned_latency_s',
              'd2h_pinned_throughput_gbps',
              'd2h_unpinned_latency_s',
              'd2h_unpinned_throughput_gbps',
          },
          result.metrics.results.keys(),
      )
    elif mode == 'jit':
      self.assertContainsSubset(
          {
              'd2h_pinned_jit_latency_s',
              'd2h_pinned_jit_throughput_gbps',
              'd2h_unpinned_jit_latency_s',
              'd2h_unpinned_jit_throughput_gbps',
          },
          result.metrics.results.keys(),
      )


if __name__ == '__main__':
  absltest.main()
