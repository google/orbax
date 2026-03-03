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

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
from orbax.checkpoint._src.testing.benchmarks import pytorch_checkpoint_benchmark
from orbax.checkpoint._src.testing.benchmarks.core import configs as benchmarks_configs
from orbax.checkpoint._src.testing.benchmarks.core import core as benchmarks_core
from safetensors.torch import save_file
import torch
import torch.distributed.checkpoint as dcp


PyTorchCheckpointOptions = pytorch_checkpoint_benchmark.PyTorchCheckpointOptions

PyTorchCheckpointBenchmark = (
    pytorch_checkpoint_benchmark.PyTorchCheckpointBenchmark
)


class PyTorchCheckpointBenchmarkTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_save = self.enter_context(
        mock.patch.object(dcp, 'save', autospec=True)
    )
    self.mock_load = self.enter_context(
        mock.patch.object(dcp, 'load', autospec=True)
    )
    self.mock_dist_initialized = self.enter_context(
        mock.patch('torch.distributed.is_initialized', return_value=True)
    )
    self.mock_dist_get_rank = self.enter_context(
        mock.patch.object(torch.distributed, 'get_rank', return_value=0)
    )
    self.mock_dist_get_world_size = self.enter_context(
        mock.patch.object(torch.distributed, 'get_world_size', return_value=1)
    )
    self.mock_set_device = self.enter_context(
        mock.patch.object(torch.cuda, 'set_device', autospec=True)
    )
    self.mock_cuda_is_available = self.enter_context(
        mock.patch.object(torch.cuda, 'is_available', return_value=True)
    )
    self.mock_cuda_memory_allocated = self.enter_context(
        mock.patch.object(torch.cuda, 'memory_allocated', return_value=0)
    )
    self.mock_cuda_memory_reserved = self.enter_context(
        mock.patch.object(torch.cuda, 'memory_reserved', return_value=0)
    )
    self.mock_cuda_memory_summary = self.enter_context(
        mock.patch.object(torch.cuda, 'memory_summary', return_value='')
    )
    self.mock_cuda_empty_cache = self.enter_context(
        mock.patch.object(torch.cuda, 'empty_cache')
    )
    self.mock_dist_barrier = self.enter_context(
        mock.patch.object(torch.distributed, 'barrier')
    )
    mock_mesh = mock.MagicMock()
    mock_mesh.size.return_value = 1
    self.mock_init_device_mesh = self.enter_context(
        mock.patch(
            'torch.distributed.device_mesh.init_device_mesh',
            return_value=mock_mesh,
        )
    )
    self.mock_empty = self.enter_context(
        mock.patch.object(torch, 'empty', return_value=mock.MagicMock())
    )
    self.mock_dtensor = self.enter_context(
        mock.patch('torch.distributed.tensor.DTensor.from_local')
    )

  @parameterized.parameters(
      dict(
          options=PyTorchCheckpointOptions(
              reference_checkpoint_path='/tmp/path',
              metric_tracemalloc_enabled=False,
              enable_async_save=False,
              save_thread_count=1,
              save_per_thread_copy_ahead_mb=10,
          ),
          expected_len=1,
      ),
  )
  def test_generate_benchmarks(self, options, expected_len):
    generator = PyTorchCheckpointBenchmark(
        checkpoint_configs=[benchmarks_configs.CheckpointConfig()],
        options=options,
    )
    benchmarks = generator.generate()
    self.assertLen(benchmarks, expected_len)
    for benchmark in benchmarks:
      self.assertIsInstance(benchmark.options, PyTorchCheckpointOptions)

  def test_benchmark_test_fn(
      self,
  ):
    safetensor_dir = epath.Path(self.create_tempdir('safetensors').full_path)
    save_file(
        {'a': torch.arange(10)}, safetensor_dir / 'a.safetensors'
    )
    test_path = epath.Path(self.create_tempdir().full_path)
    generator = PyTorchCheckpointBenchmark(
        checkpoint_configs=[benchmarks_configs.CheckpointConfig()],
        options=PyTorchCheckpointOptions(
            reference_checkpoint_path=str(safetensor_dir),
            metric_tracemalloc_enabled=False,
            enable_async_save=False,
            save_thread_count=1,
            save_per_thread_copy_ahead_mb=10,
        ),
    )
    test_options = PyTorchCheckpointOptions(
        reference_checkpoint_path=str(safetensor_dir),
        metric_tracemalloc_enabled=False,
        enable_async_save=False,
        save_thread_count=1,
        save_per_thread_copy_ahead_mb=10,
    )
    context = benchmarks_core.TestContext(
        pytree=None,
        path=test_path,
        options=test_options,
    )

    result = generator.test_fn(context)

    self.assertIsInstance(result, benchmarks_core.TestResult)
    self.assertContainsSubset(
        {
            'save_time_duration',
            'restore_time_duration',
        },
        result.metrics.results.keys(),
    )
    self.mock_save.assert_called_once()
    save_args = self.mock_save.call_args
    # Verify state_dict structure
    state_dict = save_args[0][0]
    self.assertIn('a', state_dict)

    self.mock_load.assert_called_once()

  def test_options_class(self):
    self.assertEqual(
        PyTorchCheckpointOptions,
        PyTorchCheckpointBenchmark.options_class,
    )


if __name__ == '__main__':
  absltest.main()
