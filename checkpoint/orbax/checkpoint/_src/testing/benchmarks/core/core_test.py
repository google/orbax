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

import dataclasses
import time
from typing import List
from unittest import mock

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import jax
import numpy as np
from orbax.checkpoint import test_utils
from orbax.checkpoint._src.testing.benchmarks.core import checkpoint_generation
from orbax.checkpoint._src.testing.benchmarks.core import configs
from orbax.checkpoint._src.testing.benchmarks.core import core
from orbax.checkpoint._src.testing.benchmarks.core import device_mesh
from orbax.checkpoint._src.testing.benchmarks.core import directory_setup


@dataclasses.dataclass
class MyBenchmarkOptions(core.BenchmarkOptions):
  opt1: int | List[int] = 1
  opt2: str | List[str] = 'a'


@core.benchmark_options(MyBenchmarkOptions)
class MyGenerator(core.BenchmarksGenerator):

  def test_fn(self, test_context: core.TestContext) -> core.TestResult:
    return core.TestResult(metrics=core.Metrics())


class MetricsTest(parameterized.TestCase):

  def test_time(self):
    metrics = core.Metrics()

    with mock.patch.object(time, 'perf_counter', side_effect=[1.0, 3.0]):
      with metrics.time('test_metric'):
        pass

    self.assertEqual(metrics.timings, {'test_metric': 2.0})

  def test_report(self):
    metrics = core.Metrics(name='TestMetrics')
    metrics.timings = {'metric1': 1.23, 'metric2': 4.56}
    expected_report = """---[process_id=0] TestMetrics Metrics Report ---
metric1: 1.2300 seconds
metric2: 4.5600 seconds
----------------------"""

    with mock.patch.object(logging, 'info') as mock_log:
      metrics.report()

      mock_log.assert_any_call(expected_report)


class BenchmarkTest(parameterized.TestCase):

  def _get_test_tree(self):
    return {'a': 1, 'b': np.arange(10)}

  @mock.patch.object(directory_setup, 'setup_test_directory')
  @mock.patch.object(checkpoint_generation, 'generate_checkpoint')
  @mock.patch.object(device_mesh, 'create_mesh')
  @mock.patch.object(core.Metrics, 'report')
  def test_run_with_generate_checkpoint(
      self,
      mock_metrics_report,
      mock_create_mesh,
      mock_generate_checkpoint,
      mock_setup_test_directory,
  ):
    path = epath.Path(self.create_tempdir().full_path)
    mock_setup_test_directory.return_value = path
    mock_generate_checkpoint.return_value = self._get_test_tree()
    mock_mesh = mock.create_autospec(jax.sharding.Mesh, instance=True)
    mock_create_mesh.return_value = mock_mesh
    options = MyBenchmarkOptions()

    def test_fn(context):
      test_utils.assert_tree_equal(self, context.pytree, self._get_test_tree())
      self.assertEqual(context.path, path)
      self.assertEqual(context.options, options)
      self.assertEqual(context.mesh, mock_mesh)
      metrics = core.Metrics()
      metrics.timings['save'] = 1.0
      return core.TestResult(metrics=metrics)

    ckpt_config = configs.CheckpointConfig(path=None, spec={})
    mesh_config = configs.MeshConfig(
        mesh_axes=['x'],
        ici_parallelism={'x': 1},
        dcn_parallelism={},
    )
    benchmark = core.Benchmark(
        test_fn=test_fn,
        checkpoint_config=ckpt_config,
        options=options,
        name='test_benchmark',
        mesh_config=mesh_config,
    )

    result = benchmark.run()

    self.assertEqual(result.metrics.timings, {'save': 1.0})
    mock_setup_test_directory.assert_called_once_with('test_benchmark', None)
    mock_create_mesh.assert_called_once_with(mesh_config)
    mock_generate_checkpoint.assert_called_once_with(
        ckpt_config, mesh=mock_mesh
    )
    self.assertEqual(mock_metrics_report.call_count, 2)

  @mock.patch.object(directory_setup, 'setup_test_directory')
  @mock.patch.object(checkpoint_generation, 'load_checkpoint')
  @mock.patch.object(device_mesh, 'create_mesh')
  @mock.patch.object(core.Metrics, 'report')
  def test_run_with_load_checkpoint(
      self,
      mock_metrics_report,
      mock_create_mesh,
      mock_load_checkpoint,
      mock_setup_test_directory,
  ):
    path = epath.Path(self.create_tempdir().full_path)
    mock_setup_test_directory.return_value = path
    mock_load_checkpoint.return_value = self._get_test_tree()
    mock_mesh = mock.create_autospec(jax.sharding.Mesh, instance=True)
    mock_create_mesh.return_value = mock_mesh
    options = MyBenchmarkOptions()

    def test_fn(context):
      test_utils.assert_tree_equal(self, context.pytree, self._get_test_tree())
      self.assertEqual(context.path, path)
      self.assertEqual(context.options, options)
      self.assertEqual(context.mesh, mock_mesh)
      metrics = core.Metrics()
      metrics.timings['restore'] = 2.0
      return core.TestResult(metrics=metrics)

    ckpt_config = configs.CheckpointConfig(path='/tmp/path', spec={})
    benchmark = core.Benchmark(
        test_fn=test_fn,
        checkpoint_config=ckpt_config,
        options=options,
        name='test_benchmark',
        mesh_config=configs.MeshConfig(
            mesh_axes=['x'],
            ici_parallelism={'x': 1},
            dcn_parallelism={},
        ),
    )

    result = benchmark.run()

    self.assertEqual(result.metrics.timings, {'restore': 2.0})
    mock_setup_test_directory.assert_called_once_with('test_benchmark', None)
    mock_load_checkpoint.assert_called_once_with('/tmp/path')
    mock_create_mesh.assert_called_once_with(benchmark.mesh_config)
    self.assertEqual(mock_metrics_report.call_count, 2)


class BenchmarksGeneratorTest(parameterized.TestCase):

  def test_get_options_product(self):
    gen = MyGenerator(
        checkpoint_config=configs.CheckpointConfig(),
        options=MyBenchmarkOptions(opt1=[1, 2], opt2='a'),
    )

    options_product = gen._get_options_product()

    self.assertCountEqual(
        options_product,
        [
            MyBenchmarkOptions(opt1=1, opt2='a'),
            MyBenchmarkOptions(opt1=2, opt2='a'),
        ],
    )

  def test_generate(self):
    gen = MyGenerator(
        checkpoint_config=configs.CheckpointConfig(),
        options=MyBenchmarkOptions(opt1=[1, 2], opt2=['a', 'b']),
    )

    benchmarks = gen.generate()

    self.assertLen(benchmarks, 4)
    names = [b.name for b in benchmarks]
    self.assertIn('MyGenerator_opt1_1_opt2_a', names)
    self.assertIn('MyGenerator_opt1_1_opt2_b', names)
    self.assertIn('MyGenerator_opt1_2_opt2_a', names)
    self.assertIn('MyGenerator_opt1_2_opt2_b', names)

  def test_options_class(self):
    self.assertEqual(MyGenerator.options_class, MyBenchmarkOptions)

  def test_generator_missing_decorator(self):
    class UndecoratedGenerator(core.BenchmarksGenerator):

      def test_fn(self, test_context: core.TestContext) -> core.TestResult:
        return core.TestResult(metrics=core.Metrics())

    with self.assertRaisesRegex(
        TypeError, 'must be decorated with @benchmark_options'
    ):
      UndecoratedGenerator(
          checkpoint_config=configs.CheckpointConfig(),
          options=MyBenchmarkOptions(),
      )

  def test_generator_mismatched_options(self):
    with self.assertRaisesRegex(
        TypeError, 'Expected options of type MyBenchmarkOptions'
    ):
      MyGenerator(
          checkpoint_config=configs.CheckpointConfig(),
          options=core.BenchmarkOptions(),
      )


class TestSuiteTest(parameterized.TestCase):

  @mock.patch.object(core.Benchmark, 'run')
  def test_run(self, mock_benchmark_run):
    gen = MyGenerator(
        checkpoint_config=configs.CheckpointConfig(),
        options=MyBenchmarkOptions(opt1=[1, 2]),
    )
    suite = core.TestSuite(name='my_suite', benchmarks_generators=[gen])

    suite.run()

    self.assertEqual(mock_benchmark_run.call_count, 2)

  @mock.patch.object(core.Benchmark, 'run')
  def test_run_multiple_generators(self, mock_benchmark_run):
    gen1 = MyGenerator(
        checkpoint_config=configs.CheckpointConfig(),
        options=MyBenchmarkOptions(opt1=[1, 2]),
    )
    gen2 = MyGenerator(
        checkpoint_config=configs.CheckpointConfig(),
        options=MyBenchmarkOptions(opt2=['c', 'd']),
    )
    suite = core.TestSuite(name='my_suite', benchmarks_generators=[gen1, gen2])

    suite.run()

    # gen1 produces 2 benchmarks, gen2 produces 2 benchmarks
    self.assertEqual(mock_benchmark_run.call_count, 4)

  @mock.patch.object(core.Benchmark, 'run')
  @mock.patch.object(logging, 'warning')
  def test_run_no_benchmarks_generated(
      self, mock_logging_warning, mock_benchmark_run
  ):
    # This generator will produce no benchmarks because opt1 is an empty list
    gen = MyGenerator(
        checkpoint_config=configs.CheckpointConfig(),
        options=MyBenchmarkOptions(opt1=[], opt2='a'),
    )
    suite = core.TestSuite(name='empty_suite', benchmarks_generators=[gen])

    suite.run()

    mock_logging_warning.assert_any_call(
        'Generator %s produced no benchmarks.', 'MyGenerator'
    )
    mock_logging_warning.assert_any_call(
        'No benchmarks were run for this suite.'
    )
    mock_benchmark_run.assert_not_called()


if __name__ == '__main__':
  absltest.main()
