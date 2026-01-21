# Copyright 2025 The Orbax Authors.
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

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import jax.numpy as jnp
from orbax.checkpoint._src.testing.benchmarks import safetensors_benchmark
from orbax.checkpoint._src.testing.benchmarks.core import configs as benchmarks_configs
from orbax.checkpoint._src.testing.benchmarks.core import core as benchmarks_core
from orbax.checkpoint.experimental.v1._src.layout import safetensors_layout

SafetensorsBenchmarkOptions = safetensors_benchmark.SafetensorsBenchmarkOptions
SafetensorsBenchmark = safetensors_benchmark.SafetensorsBenchmark


class SafetensorsBenchmarkTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_layout_cls = self.enter_context(
        mock.patch.object(
            safetensors_layout, 'SafetensorsLayout', autospec=True
        )
    )
    self.mock_safetensors_numpy = self.enter_context(
        mock.patch.object(safetensors_benchmark, 'safetensors', autospec=True)
    )

  def test_generate_benchmarks(self):
    generator = SafetensorsBenchmark(
        checkpoint_configs=[benchmarks_configs.CheckpointConfig(spec={})],
        options=SafetensorsBenchmarkOptions(),
    )
    benchmarks = generator.generate()
    self.assertLen(benchmarks, 1)
    self.assertIsInstance(benchmarks[0].options, SafetensorsBenchmarkOptions)

  def test_benchmark_test_fn(self):
    generator = SafetensorsBenchmark(
        checkpoint_configs=[benchmarks_configs.CheckpointConfig(spec={})],
        options=SafetensorsBenchmarkOptions(),
    )

    # Mock Layout instance and its load method
    mock_layout_instance = self.mock_layout_cls.return_value

    # Mock load to return an awaitable that returns the restore_fn
    # restore_fn itself is an awaitable that returns the pytree
    async def mock_restore_fn():
      return {'a': jnp.arange(10)}

    async def mock_load(path):
      del path
      return mock_restore_fn()

    mock_layout_instance.load.side_effect = mock_load

    pytree = {
        'a': jnp.arange(10),
    }
    test_path = epath.Path(self.create_tempdir().full_path)
    test_options = SafetensorsBenchmarkOptions()
    context = benchmarks_core.TestContext(
        pytree=pytree, path=test_path, options=test_options
    )

    result = generator.test_fn(context)

    # Verify Save (Setup)
    self.mock_safetensors_numpy.numpy.save_file.assert_called_once()
    # Check that we converted jax arrays to numpy
    save_args = self.mock_safetensors_numpy.numpy.save_file.call_args
    # First arg is dictionary, we can check keys
    self.assertIn('a', save_args[0][0])

    # Verify Restore (Benchmark Target)
    mock_layout_instance.load.assert_called_once_with(
        test_path / 'safetensors_ckpt'
    )

    # Verify Results
    self.assertIsInstance(result, benchmarks_core.TestResult)
    self.assertContainsSubset(
        {
            'save_time_duration',
            'load_safetensor_time_duration',
        },
        result.metrics.results.keys(),
    )

  def test_options_class(self):
    self.assertEqual(
        SafetensorsBenchmarkOptions,
        SafetensorsBenchmark.options_class,
    )


if __name__ == '__main__':
  absltest.main()
