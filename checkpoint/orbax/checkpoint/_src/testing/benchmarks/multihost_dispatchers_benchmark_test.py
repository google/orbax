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
from orbax.checkpoint._src.multihost import dispatchers
from orbax.checkpoint._src.testing.benchmarks import multihost_dispatchers_benchmark
from orbax.checkpoint._src.testing.benchmarks.core import configs
from orbax.checkpoint._src.testing.benchmarks.core import core


MultihostDispatchersBenchmarkOptions = (
    multihost_dispatchers_benchmark.MultihostDispatchersBenchmarkOptions
)
MultihostDispatchersBenchmark = (
    multihost_dispatchers_benchmark.MultihostDispatchersBenchmark
)


def _dispatch_side_effect(
    func, *, input_arrays, result_specs=None, func_args=(), func_kwargs=None
):
  del func, func_kwargs
  if result_specs is None:
    return dispatchers._make_dummy_result_array(input_arrays)
  else:
    return multihost_dispatchers_benchmark.build_jax_array(
        input_arrays, *func_args
    )


class MultihostDispatchersBenchmarkTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.directory = epath.Path(self.create_tempdir('ckpt').full_path)
    self.mesh = jax.sharding.Mesh(np.array(jax.devices()), ('x',))
    self.mock_log_pytree = self.enter_context(
        mock.patch.object(
            multihost_dispatchers_benchmark.pytree_utils,
            'log_pytree',
            autospec=True,
        )
    )
    self.mock_pretty_log_mesh = self.enter_context(
        mock.patch.object(
            multihost_dispatchers_benchmark.mesh_utils,
            'pretty_log_mesh',
            autospec=True,
        )
    )
    self.mock_assert_pytree_equal = self.enter_context(
        mock.patch.object(
            multihost_dispatchers_benchmark.pytree_utils,
            'assert_pytree_equal',
            autospec=True,
        )
    )

    self.mock_colocated_dispatcher = mock.create_autospec(
        dispatchers.ColocatedPythonDispatcher,
        instance=True,
        spec_set=True,
    )
    self.mock_colocated_dispatcher.dispatch.side_effect = _dispatch_side_effect
    self.enter_context(
        mock.patch.object(
            dispatchers,
            'ColocatedPythonDispatcher',
            return_value=self.mock_colocated_dispatcher,
        )
    )

    self.checkpoint_config = configs.CheckpointConfig(
        spec={
            'array': {
                'shape': (16,),
                'dtype': jnp.float32,
                'sharding': ('x',),
            }
        }
    )
    sharding = jax.sharding.NamedSharding(
        self.mesh, jax.sharding.PartitionSpec('x')
    )
    self.arr = jax.device_put(np.arange(16, dtype=jnp.float32), sharding)


  def test_build_jax_array_returns_correct_array(self):
    shape = (16,)
    sharding = self.arr.sharding
    specs = jax.ShapeDtypeStruct(shape, self.arr.dtype, sharding=sharding)

    arr = multihost_dispatchers_benchmark.build_jax_array(
        self.arr, shape, specs
    )

    self.assertEqual(arr.shape, shape)
    self.assertEqual(arr.dtype, jnp.float32)
    self.assertEqual(arr.sharding, sharding)
    np.testing.assert_array_equal(arr, np.zeros(shape, dtype=jnp.float32))

  def test_log_pytree_fn_calls_logging(self):
    metadata = {'array_sharding': self.arr.sharding}
    multihost_dispatchers_benchmark.log_pytree_fn(self.arr, metadata)
    self.mock_log_pytree.assert_called_once_with('array_in_worker', self.arr)
    self.mock_pretty_log_mesh.assert_called_once_with(
        'array mesh in worker: ', self.arr.sharding.mesh
    )

  def test_colocated_benchmark_returns_test_result_with_basic_metrics(self):
    options = MultihostDispatchersBenchmarkOptions(
        use_colocated=True,
    )
    benchmark_generator = MultihostDispatchersBenchmark(
        checkpoint_configs=[self.checkpoint_config],
        mesh_configs=None,
        options=options,
        output_dir=str(self.directory),
    )
    context = core.TestContext(
        pytree={'array': self.arr},
        path=self.directory,
        options=options,
        mesh=self.mesh,
    )

    benchmarks = benchmark_generator.generate()
    self.assertLen(benchmarks, 1)
    result = benchmarks[0].test_fn(context)

    self.assertIsInstance(result, core.TestResult)
    self.assertContainsSubset(
        {
            'dispatch_without_result_specs_time',
            'dispatch_without_result_specs_block_until_ready_time',
            'dispatch_with_dummy_result_array_time',
            'dispatch_with_dummy_result_array_block_until_ready_time',
            'dispatch_with_result_specs_time',
        },
        result.metrics.results.keys(),
    )

  @parameterized.named_parameters(
      dict(testcase_name='no_device_count', device_count=None),
      dict(testcase_name='with_device_count', device_count=1),
  )
  def test_benchmark_colocated_dispatcher_called(self, device_count):
    options = MultihostDispatchersBenchmarkOptions(
        use_colocated=True,
        device_count=device_count,
    )
    benchmark_generator = MultihostDispatchersBenchmark(
        checkpoint_configs=[self.checkpoint_config],
        mesh_configs=None,
        options=options,
        output_dir=str(self.directory),
    )
    context = core.TestContext(
        pytree={'array': self.arr},
        path=self.directory,
        options=options,
        mesh=self.mesh,
    )
    benchmarks = benchmark_generator.generate()
    self.assertLen(benchmarks, 1)
    benchmark = benchmarks[0]

    benchmark.test_fn(context)

    self.assertEqual(self.mock_colocated_dispatcher.dispatch.call_count, 3)


  def test_generate_benchmarks_creates_multiple_benchmark_configs(self):
    num_benchmarks = 2
    options = MultihostDispatchersBenchmarkOptions(
        use_colocated=[True], device_count=[1, 2]
    )
    b = MultihostDispatchersBenchmark(
        checkpoint_configs=[configs.CheckpointConfig()],
        options=options,
    )

    benchmarks = b.generate()

    self.assertLen(benchmarks, num_benchmarks)
    for bench in benchmarks:
      self.assertIsInstance(bench.options, MultihostDispatchersBenchmarkOptions)


if __name__ == '__main__':
  absltest.main()
