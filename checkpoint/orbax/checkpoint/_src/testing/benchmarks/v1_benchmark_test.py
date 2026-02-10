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

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import jax.numpy as jnp
from orbax.checkpoint._src.testing.benchmarks import v1_benchmark
from orbax.checkpoint._src.testing.benchmarks.core import configs as benchmarks_configs
from orbax.checkpoint._src.testing.benchmarks.core import core as benchmarks_core


V1BenchmarkOptions = v1_benchmark.V1BenchmarkOptions
V1Benchmark = v1_benchmark.V1Benchmark


class V1BenchmarkTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.directory = epath.Path(self.create_tempdir().full_path)

  @parameterized.parameters(
      dict(
          options=V1BenchmarkOptions(use_ocdbt=False, use_zarr3=True),
          expected_len=1,
      ),
      dict(
          options=V1BenchmarkOptions(use_ocdbt=[False, True], use_zarr3=True),
          expected_len=2,
      ),
      dict(
          options=V1BenchmarkOptions(
              use_ocdbt=[False, True], use_zarr3=[False, True]
          ),
          expected_len=4,
      ),
  )
  def test_generate_benchmarks(self, options, expected_len):
    generator = V1Benchmark(
        checkpoint_configs=[benchmarks_configs.CheckpointConfig(spec={})],
        options=options,
    )
    benchmarks = generator.generate()
    self.assertLen(benchmarks, expected_len)
    for benchmark in benchmarks:
      self.assertIsInstance(benchmark.options, V1BenchmarkOptions)

  @parameterized.product(
      use_ocdbt=(False, True),
      use_zarr3=(False, True),
      use_compression=(False, True),
      save_concurrent_gb=(None, 1),
      restore_concurrent_gb=(None, 2),
      use_replica_parallel=(False,),  # Simplified for single process test
      enable_replica_parallel_separate_folder=(False,),
  )
  def test_benchmark_test_fn(
      self,
      use_ocdbt,
      use_zarr3,
      use_compression,
      save_concurrent_gb,
      restore_concurrent_gb,
      use_replica_parallel,
      enable_replica_parallel_separate_folder,
  ):
    # Skip invalid combinations
    if enable_replica_parallel_separate_folder and (
        not use_replica_parallel or not use_ocdbt
    ):
      return

    generator = V1Benchmark(
        checkpoint_configs=[benchmarks_configs.CheckpointConfig(spec={})],
        options=V1BenchmarkOptions(),
    )

    pytree = {
        'a': jnp.arange(10),
        'b': {'c': jnp.ones((5, 5))},
    }

    test_options = V1BenchmarkOptions(
        use_ocdbt=use_ocdbt,
        use_zarr3=use_zarr3,
        use_compression=use_compression,
        save_concurrent_gb=save_concurrent_gb,
        restore_concurrent_gb=restore_concurrent_gb,
        use_replica_parallel=use_replica_parallel,
        enable_replica_parallel_separate_folder=enable_replica_parallel_separate_folder,
    )

    # Create unique path for each parameter set
    test_subdir = (
        self.directory / f'test_{use_ocdbt}_{use_zarr3}_{use_compression}'
    )
    test_subdir.mkdir(exist_ok=True, parents=True)

    context = benchmarks_core.TestContext(
        pytree=pytree, path=test_subdir, options=test_options
    )

    result = generator.test_fn(context)

    self.assertIsInstance(result, benchmarks_core.TestResult)
    # Check for expected metrics keys based on _metrics_to_measure
    # in v1_benchmark.py and the metric.measure calls.
    # The benchmark records "save_blocking", "save_background", "load".
    # Metric "time" is always added.

    # We expect roughly:
    # save_blocking_time_duration
    # save_background_time_duration
    # load_time_duration

    metrics = result.metrics.results
    self.assertIn('save_blocking_time_duration', metrics)
    self.assertIn('save_background_time_duration', metrics)
    self.assertIn('load_time_duration', metrics)


if __name__ == '__main__':
  absltest.main()
