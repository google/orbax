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
from orbax.checkpoint._src.testing.benchmarks.core import metric as metric_lib


@dataclasses.dataclass(frozen=True)
class MyBenchmarkOptions(core.BenchmarkOptions):
  opt1: int | List[int] = 1
  opt2: str | List[str] = 'a'

  def is_valid(self) -> bool:
    return not (self.opt1 == 2 and self.opt2 == 'a')


@core.benchmark_options(MyBenchmarkOptions)
class MyGenerator(core.BenchmarksGenerator):

  def test_fn(self, test_context: core.TestContext) -> core.TestResult:
    return core.TestResult(metrics=metric_lib.Metrics())


class TestResultTest(parameterized.TestCase):

  def test_is_successful_with_error(self):
    result = core.TestResult(metrics=metric_lib.Metrics(), error=ValueError())
    self.assertFalse(result.is_successful())

  def test_is_successful_without_error(self):
    result = core.TestResult(metrics=metric_lib.Metrics(), error=None)
    self.assertTrue(result.is_successful())


class BenchmarkTest(parameterized.TestCase):

  def _get_test_tree(self):
    return {'a': 1, 'b': np.arange(10)}

  @mock.patch.object(directory_setup, 'setup_test_directory')
  @mock.patch.object(checkpoint_generation, 'generate_checkpoint')
  @mock.patch.object(device_mesh, 'create_mesh')
  @mock.patch.object(metric_lib.Metrics, 'report')
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
      metrics = metric_lib.Metrics()
      metrics.results['save'] = (1.0, 's')
      return core.TestResult(metrics=metrics)

    ckpt_config = configs.CheckpointConfig(path=None, spec={})
    mesh_config = configs.MeshConfig(
        mesh_axes=['x'],
        ici_parallelism={'x': 1},
        dcn_parallelism={},
    )
    mesh = device_mesh.create_mesh(mesh_config)
    benchmark = core.Benchmark(
        test_fn=test_fn,
        checkpoint_config=ckpt_config,
        options=options,
        name='test_benchmark',
        mesh=mesh,
    )

    result = benchmark.run()

    self.assertEqual(result.metrics.results, {'save': (1.0, 's')})
    mock_setup_test_directory.assert_called_once_with(
        'test_benchmark', None, None
    )
    mock_create_mesh.assert_called_once_with(mesh_config)
    mock_generate_checkpoint.assert_called_once_with(
        ckpt_config, mesh=mock_mesh
    )
    self.assertEqual(mock_metrics_report.call_count, 2)

  @mock.patch.object(directory_setup, 'setup_test_directory')
  @mock.patch.object(checkpoint_generation, 'load_checkpoint')
  @mock.patch.object(device_mesh, 'create_mesh')
  @mock.patch.object(metric_lib.Metrics, 'report')
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
      metrics = metric_lib.Metrics()
      metrics.results['restore'] = (2.0, 's')
      return core.TestResult(metrics=metrics)

    ckpt_config = configs.CheckpointConfig(path='/tmp/path', spec={})
    mesh_config = configs.MeshConfig(
        mesh_axes=['x'],
        ici_parallelism={'x': 1},
        dcn_parallelism={},
    )
    mesh = device_mesh.create_mesh(mesh_config)
    benchmark = core.Benchmark(
        test_fn=test_fn,
        checkpoint_config=ckpt_config,
        options=options,
        name='test_benchmark',
        mesh=mesh,
    )

    result = benchmark.run()

    self.assertEqual(result.metrics.results, {'restore': (2.0, 's')})
    mock_setup_test_directory.assert_called_once_with(
        'test_benchmark', None, None
    )
    mock_load_checkpoint.assert_called_once_with('/tmp/path')
    mock_create_mesh.assert_called_once_with(mesh_config)
    self.assertEqual(mock_metrics_report.call_count, 2)

  @mock.patch.object(directory_setup, 'setup_test_directory')
  @mock.patch.object(checkpoint_generation, 'generate_checkpoint')
  @mock.patch.object(device_mesh, 'create_mesh')
  @mock.patch.object(metric_lib.Metrics, 'report')
  def test_run_with_repeat_index(
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

    def test_fn(_):
      return core.TestResult(metrics=metric_lib.Metrics())

    ckpt_config = configs.CheckpointConfig(path=None, spec={})
    mesh_config = configs.MeshConfig(
        mesh_axes=['x'],
        ici_parallelism={'x': 1},
        dcn_parallelism={},
    )
    mesh = device_mesh.create_mesh(mesh_config)
    benchmark = core.Benchmark(
        test_fn=test_fn,
        checkpoint_config=ckpt_config,
        options=options,
        name='test_benchmark',
        mesh=mesh,
    )

    result = benchmark.run(repeat_index=0)
    mock_setup_test_directory.assert_called_once_with(
        'test_benchmark', None, 0
    )
    self.assertEqual(mock_metrics_report.call_count, 2)
    self.assertEqual(result.metrics.name, 'test_benchmark_repeat_0')


class BenchmarksGeneratorTest(parameterized.TestCase):

  def test_get_options_product(self):
    gen = MyGenerator(
        checkpoint_configs=[configs.CheckpointConfig()],
        options=MyBenchmarkOptions(opt1=[1, 2], opt2='a'),
    )

    options_product = gen._get_options_product()

    self.assertCountEqual(
        options_product,
        [
            MyBenchmarkOptions(opt1=1, opt2='a'),
        ],
    )

  def test_generate_benchmark_name(self):
    ckpt_config = configs.CheckpointConfig()
    gen = MyGenerator(
        checkpoint_configs=[ckpt_config],
        options=MyBenchmarkOptions(),
    )
    options1 = MyBenchmarkOptions(opt1=1, opt2='a')
    options2 = MyBenchmarkOptions(opt1=1, opt2='b')
    mesh = mock.create_autospec(jax.sharding.Mesh, instance=True)

    name1 = gen.generate_benchmark_name(options1, mesh, ckpt_config)
    name2 = gen.generate_benchmark_name(options2, mesh, ckpt_config)

    self.assertRegex(name1, r'^MyGenerator_[a-f0-9]{12}$')
    self.assertRegex(name2, r'^MyGenerator_[a-f0-9]{12}$')
    self.assertNotEqual(name1, name2)

    options3 = MyBenchmarkOptions(opt1=1, opt2='a')
    mesh2 = mock.create_autospec(jax.sharding.Mesh, instance=True)
    name3 = gen.generate_benchmark_name(options3, mesh2, ckpt_config)
    self.assertRegex(name3, r'^MyGenerator_[a-f0-9]{12}$')
    self.assertNotEqual(name1, name3)

    ckpt_config2 = configs.CheckpointConfig(path='/some/path')
    name4 = gen.generate_benchmark_name(options1, mesh, ckpt_config2)
    self.assertRegex(name4, r'^MyGenerator_[a-f0-9]{12}$')
    self.assertNotEqual(name1, name4)

  @mock.patch.object(device_mesh, 'create_mesh')
  @mock.patch.object(logging, 'warning')
  def test_get_meshes_skip_incompatible(
      self, mock_logging_warning, mock_create_mesh
  ):
    mesh_config1 = configs.MeshConfig(mesh_axes=['x'], ici_parallelism={'x': 1})
    mesh_config2 = configs.MeshConfig(mesh_axes=['y'], ici_parallelism={'y': 1})
    mock_mesh = mock.create_autospec(jax.sharding.Mesh, instance=True)
    mock_create_mesh.return_value = mock_mesh
    exception = ValueError('Incompatible')
    mock_create_mesh.side_effect = [exception, mock_mesh]

    gen = MyGenerator(
        checkpoint_configs=[configs.CheckpointConfig()],
        options=MyBenchmarkOptions(),
        mesh_configs=[mesh_config1, mesh_config2],
    )

    meshes = gen._get_meshes(skip_incompatible_mesh_configs=True)

    self.assertLen(meshes, 1)
    self.assertEqual(meshes[0], mock_mesh)
    mock_create_mesh.assert_has_calls(
        [mock.call(mesh_config1), mock.call(mesh_config2)]
    )
    mock_logging_warning.assert_called_once_with(
        'Failed to create mesh with config %s: %s', mesh_config1, exception
    )

  @mock.patch.object(device_mesh, 'create_mesh')
  def test_get_meshes_no_skip_incompatible(self, mock_create_mesh):
    mesh_config1 = configs.MeshConfig(mesh_axes=['x'], ici_parallelism={'x': 1})
    mesh_config2 = configs.MeshConfig(mesh_axes=['y'], ici_parallelism={'y': 1})
    mock_mesh = mock.create_autospec(jax.sharding.Mesh, instance=True)
    mock_create_mesh.return_value = mock_mesh
    mock_create_mesh.side_effect = [ValueError('Incompatible'), mock_mesh]

    gen = MyGenerator(
        checkpoint_configs=[configs.CheckpointConfig()],
        options=MyBenchmarkOptions(),
        mesh_configs=[mesh_config1, mesh_config2],
    )

    with self.assertRaisesRegex(ValueError, 'Failed to create mesh'):
      gen._get_meshes(skip_incompatible_mesh_configs=False)

  def test_generate(self):
    gen = MyGenerator(
        checkpoint_configs=[configs.CheckpointConfig()],
        options=MyBenchmarkOptions(opt1=[1, 2], opt2=['a', 'b']),
    )

    benchmarks = gen.generate()

    # Combinations: (1,a), (1,b), (2,b). (2,a) is invalid.
    self.assertLen(benchmarks, 3)

  def test_options_class(self):
    self.assertEqual(MyGenerator.options_class, MyBenchmarkOptions)

  def test_generator_missing_decorator(self):
    class UndecoratedGenerator(core.BenchmarksGenerator):

      def test_fn(self, test_context: core.TestContext) -> core.TestResult:
        return core.TestResult(metrics=metric_lib.Metrics())

    with self.assertRaisesRegex(
        TypeError, 'must be decorated with @benchmark_options'
    ):
      UndecoratedGenerator(
          checkpoint_configs=[configs.CheckpointConfig()],
          options=MyBenchmarkOptions(),
      )

  def test_generator_mismatched_options(self):
    with self.assertRaisesRegex(
        TypeError, 'Expected options of type MyBenchmarkOptions'
    ):
      MyGenerator(
          checkpoint_configs=[configs.CheckpointConfig()],
          options=core.BenchmarkOptions(),
      )


class TestSuiteTest(parameterized.TestCase):

  @mock.patch.object(core.Benchmark, 'run')
  def test_run(self, mock_benchmark_run):
    gen = MyGenerator(
        checkpoint_configs=[configs.CheckpointConfig()],
        options=MyBenchmarkOptions(opt1=[1, 2]),
    )
    suite = core.TestSuite(name='my_suite', benchmarks_generators=[gen])

    suite.run()

    self.assertEqual(mock_benchmark_run.call_count, 1)

  @mock.patch.object(core.Benchmark, 'run')
  def test_run_multiple_generators(self, mock_benchmark_run):
    gen1 = MyGenerator(
        checkpoint_configs=[configs.CheckpointConfig()],
        options=MyBenchmarkOptions(opt1=[1, 2]),
    )
    gen2 = MyGenerator(
        checkpoint_configs=[configs.CheckpointConfig()],
        options=MyBenchmarkOptions(opt2=['c', 'd']),
    )
    suite = core.TestSuite(name='my_suite', benchmarks_generators=[gen1, gen2])

    suite.run()

    # gen1 produces 1 benchmark (2,a is invalid), gen2 produces 2 benchmarks
    self.assertEqual(mock_benchmark_run.call_count, 3)

  @mock.patch.object(core.Benchmark, 'run')
  def test_run_with_num_repeats(self, mock_benchmark_run):
    gen = MyGenerator(
        checkpoint_configs=[configs.CheckpointConfig()],
        options=MyBenchmarkOptions(opt1=1),
    )
    suite = core.TestSuite(
        name='my_suite', benchmarks_generators=[gen], num_repeats=3
    )

    suite.run()

    self.assertEqual(mock_benchmark_run.call_count, 3)
    mock_benchmark_run.assert_has_calls([
        mock.call(repeat_index=0),
        mock.call(repeat_index=1),
        mock.call(repeat_index=2),
    ])

  @mock.patch.object(core.Benchmark, 'run')
  @mock.patch.object(logging, 'warning')
  def test_run_no_benchmarks_generated(
      self, mock_logging_warning, mock_benchmark_run
  ):
    # This generator will produce no benchmarks because opt1 is an empty list
    gen = MyGenerator(
        checkpoint_configs=[configs.CheckpointConfig()],
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

  @mock.patch.object(directory_setup, 'setup_test_directory')
  @mock.patch.object(checkpoint_generation, 'generate_checkpoint')
  @mock.patch.object(logging, 'info')
  def test_run_generates_report_with_failures(
      self,
      mock_logging_info,
      mock_generate_checkpoint,
      mock_setup_test_directory,
  ):
    path = epath.Path(self.create_tempdir().full_path)
    mock_setup_test_directory.return_value = path
    mock_generate_checkpoint.return_value = {}

    @core.benchmark_options(MyBenchmarkOptions)
    class MyGeneratorWithFailure(core.BenchmarksGenerator):

      def test_fn(self, test_context: core.TestContext) -> core.TestResult:
        if test_context.options.opt1 == 2 and test_context.options.opt2 == 'b':  # pytype: disable=attribute-error
          raise ValueError('opt1=2, opt2=b failed')
        metrics = metric_lib.Metrics()
        metrics.results['fake_metric'] = (0.1, 's')
        return core.TestResult(metrics=metrics)

    gen = MyGeneratorWithFailure(
        checkpoint_configs=[configs.CheckpointConfig()],
        options=MyBenchmarkOptions(opt1=[1, 2], opt2=['a', 'b']),
    )
    suite = core.TestSuite(name='report_suite', benchmarks_generators=[gen])
    suite.run()

    # 3 benchmarks generated: (1,a), (1,b), (2,b).
    # (1,a), (1,b) pass. (2,b) fails because (2,a) is invalid.

    # The last call to logging.info should be the report
    report_log_call = mock_logging_info.call_args_list[-1]
    report_log = report_log_call[0][0]

    self.assertIn(' Test Suite Report: report_suite ', report_log)
    self.assertIn('Total benchmark configurations: 3', report_log)
    self.assertIn('Total runs (1 repeats): 3, Passed: 2, Failed: 1', report_log)
    self.assertIn('--- Failed Runs ---', report_log)
    self.assertIn("Error: ValueError('opt1=2, opt2=b failed')", report_log)


if __name__ == '__main__':
  absltest.main()
