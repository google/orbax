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
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from orbax.checkpoint._src.testing.benchmarks import checkpoint_manager_perf_benchmark
from orbax.checkpoint._src.testing.benchmarks.core import configs as benchmarks_configs
from orbax.checkpoint._src.testing.benchmarks.core import core as benchmarks_core


CheckpointManagerPerfBenchmarkOptions = (
    checkpoint_manager_perf_benchmark.CheckpointManagerPerfBenchmarkOptions
)
CheckpointManagerPerfBenchmark = (
    checkpoint_manager_perf_benchmark.CheckpointManagerPerfBenchmark
)


class CheckpointManagerPerfBenchmarkTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_checkpoint_manager = self.enter_context(
        mock.patch.object(ocp, 'CheckpointManager', autospec=True)
    )

  @parameterized.parameters(
      dict(
          options=CheckpointManagerPerfBenchmarkOptions(
              use_ocdbt=False, use_zarr3=True
          ),
          expected_len=1,
      ),
      dict(
          options=CheckpointManagerPerfBenchmarkOptions(
              use_ocdbt=[False, True], use_zarr3=True
          ),
          expected_len=2,
      ),
      dict(
          options=CheckpointManagerPerfBenchmarkOptions(
              use_ocdbt=[False, True], use_zarr3=[False, True]
          ),
          expected_len=4,
      ),
  )
  def test_generate_benchmarks(self, options, expected_len):
    generator = CheckpointManagerPerfBenchmark(
        checkpoint_configs=[benchmarks_configs.CheckpointConfig(spec={})],
        options=options,
    )
    benchmarks = generator.generate()
    self.assertLen(benchmarks, expected_len)
    for benchmark in benchmarks:
      self.assertIsInstance(
          benchmark.options, CheckpointManagerPerfBenchmarkOptions
      )

  @parameterized.parameters(
      dict(use_ocdbt=False, use_zarr3=True),
      dict(use_ocdbt=True, use_zarr3=False),
      dict(use_ocdbt=True, use_zarr3=True),
  )
  def test_benchmark_test_fn(self, use_ocdbt, use_zarr3):
    self.enter_context(mock.patch('time.time', return_value=0.0))
    generator = CheckpointManagerPerfBenchmark(
        checkpoint_configs=[benchmarks_configs.CheckpointConfig(spec={})],
        options=CheckpointManagerPerfBenchmarkOptions(),
    )
    mock_save = self.mock_checkpoint_manager.return_value.save
    mock_restore = self.mock_checkpoint_manager.return_value.restore
    mock_close = self.mock_checkpoint_manager.return_value.close
    mock_all_steps = self.mock_checkpoint_manager.return_value.all_steps
    mock_all_steps.return_value = list(range(200))
    pytree = {
        'a': jnp.arange(10),
    }
    test_path = epath.Path(self.create_tempdir().full_path)
    test_options = CheckpointManagerPerfBenchmarkOptions(
        use_ocdbt=use_ocdbt,
        use_zarr3=use_zarr3,
    )
    context = benchmarks_core.TestContext(
        pytree=pytree, path=test_path, options=test_options
    )

    result = generator.test_fn(context)

    self.mock_checkpoint_manager.assert_called_once()
    self.assertIsInstance(result, benchmarks_core.TestResult)
    self.assertEqual(mock_save.call_count, 20)
    save_args = mock_save.call_args[1]['args']
    self.assertIsInstance(save_args, ocp.args.Composite)
    self.assertIn('pytree_item', save_args)
    self.assertIsInstance(save_args['pytree_item'], ocp.args.StandardSave)
    mock_restore.assert_called_once()
    restore_args = mock_restore.call_args[1]['args']
    self.assertIsInstance(restore_args, ocp.args.Composite)
    self.assertIn('pytree_item', restore_args)
    self.assertIsInstance(restore_args['pytree_item'], ocp.args.StandardRestore)
    mock_close.assert_called_once()

  def test_options_class(self):
    self.assertEqual(
        CheckpointManagerPerfBenchmarkOptions,
        CheckpointManagerPerfBenchmark.options_class,
    )

  def test_train_step(self):
    pytree = {'a': jnp.array([1.0, 2.0])}
    generator = CheckpointManagerPerfBenchmark(
        checkpoint_configs=[benchmarks_configs.CheckpointConfig(spec={})],
        options=CheckpointManagerPerfBenchmarkOptions(),
    )
    result = generator._train_step(pytree)
    expected = {'a': jnp.array([1.5, 2.5])}
    np.testing.assert_array_equal(result['a'], expected['a'])

  def test_clear_pytree(self):
    pytree = {'a': jnp.array([1.0, 2.0]), 'b': 1}
    generator = CheckpointManagerPerfBenchmark(
        checkpoint_configs=[benchmarks_configs.CheckpointConfig(spec={})],
        options=CheckpointManagerPerfBenchmarkOptions(),
    )
    generator._clear_pytree(pytree)
    self.assertTrue(pytree['a'].is_deleted())


if __name__ == '__main__':
  jax.config.update('jax_platform_name', 'cpu')
  absltest.main()
