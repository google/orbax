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
import orbax.checkpoint as ocp
from orbax.checkpoint._src.testing.benchmarks import pytree_checkpoint_benchmark
from orbax.checkpoint._src.testing.benchmarks.core import configs as benchmarks_configs
from orbax.checkpoint._src.testing.benchmarks.core import core as benchmarks_core


PyTreeCheckpointOptions = pytree_checkpoint_benchmark.PyTreeCheckpointOptions

PyTreeCheckpointBenchmark = (
    pytree_checkpoint_benchmark.PyTreeCheckpointBenchmark
)


class PyTreeCheckpointBenchmarkTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_checkpointer = self.enter_context(
        mock.patch.object(ocp, 'AsyncCheckpointer', autospec=True)
    )
    self.mock_handler = self.enter_context(
        mock.patch.object(ocp, 'PyTreeCheckpointHandler', autospec=True)
    )

  @parameterized.parameters(
      dict(
          options=PyTreeCheckpointOptions(use_ocdbt=False, use_zarr3=True),
          expected_len=1,
      ),
      dict(
          options=PyTreeCheckpointOptions(
              use_ocdbt=[False, True], use_zarr3=True
          ),
          expected_len=2,
      ),
      dict(
          options=PyTreeCheckpointOptions(
              use_ocdbt=[False, True], use_zarr3=[False, True]
          ),
          expected_len=4,
      ),
  )
  def test_generate_benchmarks(self, options, expected_len):
    generator = PyTreeCheckpointBenchmark(
        checkpoint_configs=[benchmarks_configs.CheckpointConfig(spec={})],
        options=options,
    )
    benchmarks = generator.generate()
    self.assertLen(benchmarks, expected_len)
    for benchmark in benchmarks:
      self.assertIsInstance(benchmark.options, PyTreeCheckpointOptions)

  @parameterized.product(
      use_ocdbt=(False, True),
      use_zarr3=(False, True),
      use_compression=(False, True),
      save_concurrent_gb=(None, 1),
      restore_concurrent_gb=(None, 2),
      save_device_host_concurrent_gb=(None, 1),
  )
  def test_benchmark_test_fn(
      self,
      use_ocdbt,
      use_zarr3,
      use_compression,
      save_concurrent_gb,
      restore_concurrent_gb,
      save_device_host_concurrent_gb,
  ):
    generator = PyTreeCheckpointBenchmark(
        checkpoint_configs=[benchmarks_configs.CheckpointConfig(spec={})],
        options=PyTreeCheckpointOptions(),
    )
    mock_save = self.mock_checkpointer.return_value.save
    mock_wait = self.mock_checkpointer.return_value.wait_until_finished
    mock_restore = self.mock_checkpointer.return_value.restore
    mock_close = self.mock_checkpointer.return_value.close
    pytree = {
        'a': jnp.arange(10),
    }
    test_path = epath.Path(self.create_tempdir().full_path)
    test_options = PyTreeCheckpointOptions(
        use_ocdbt=use_ocdbt,
        use_zarr3=use_zarr3,
        use_compression=use_compression,
        save_concurrent_gb=save_concurrent_gb,
        restore_concurrent_gb=restore_concurrent_gb,
        save_device_host_concurrent_gb=save_device_host_concurrent_gb,
    )
    context = benchmarks_core.TestContext(
        pytree=pytree, path=test_path, options=test_options
    )

    result = generator.test_fn(context)

    self.mock_handler.assert_called_once_with(
        use_zarr3=use_zarr3,
        use_ocdbt=use_ocdbt,
        use_compression=use_compression,
        save_concurrent_gb=save_concurrent_gb,
        restore_concurrent_gb=restore_concurrent_gb,
        save_device_host_concurrent_gb=save_device_host_concurrent_gb,
        is_prioritized_key_fn=mock.ANY,
    )
    self.mock_checkpointer.assert_called_once_with(
        self.mock_handler.return_value
    )
    self.assertIsInstance(result, benchmarks_core.TestResult)
    self.assertContainsSubset(
        {'save_time', 'wait_until_finished_time', 'restore_time'},
        result.metrics.results.keys(),
    )
    mock_save.assert_called_once()
    save_args = mock_save.call_args[1]['args']
    self.assertIsInstance(save_args, ocp.args.PyTreeSave)
    self.assertEqual(save_args.item, pytree)
    mock_wait.assert_called_once()
    mock_restore.assert_called_once()
    restore_args = mock_restore.call_args[1]['args']
    self.assertIsInstance(restore_args, ocp.args.PyTreeRestore)
    self.assertIsNotNone(restore_args.restore_args)
    mock_close.assert_called_once()

  def test_options_class(self):
    self.assertEqual(
        PyTreeCheckpointOptions,
        PyTreeCheckpointBenchmark.options_class,
    )


if __name__ == '__main__':
  absltest.main()
