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

import dataclasses
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
from orbax.checkpoint._src.testing.benchmarks.core import config_parsing
from orbax.checkpoint._src.testing.benchmarks.core import configs as config_lib
from orbax.checkpoint._src.testing.benchmarks.core import core
from orbax.checkpoint._src.testing.benchmarks.core import metric as metric_lib
import yaml



@dataclasses.dataclass(frozen=True)
class MockOptions(core.BenchmarkOptions):
  param1: int = 1
  param2: str = 'a'


@core.benchmark_options(MockOptions)
class MockGenerator(core.BenchmarksGenerator):

  def test_fn(self, test_context: core.TestContext) -> core.TestResult:
    return core.TestResult(metrics=metric_lib.Metrics())


class UndecoratedMockGenerator(core.BenchmarksGenerator):

  def test_fn(self, test_context: core.TestContext) -> core.TestResult:
    return core.TestResult(metrics=metric_lib.Metrics())


class TestLoadYamlConfig(parameterized.TestCase):

  def test_load_valid_yaml(self):
    content = 'suite_name: test'
    mocker = mock.mock_open(read_data=content)
    with mock.patch.object(epath, 'Path', autospec=True) as mock_path_cls:
      mock_path_cls.return_value.open = mocker
      config = config_parsing._load_yaml_config('fake/path.yaml')
      mock_path_cls.assert_called_once_with('fake/path.yaml')
      mocker.assert_called_once_with('r')
      self.assertEqual(config, {'suite_name': 'test'})


  def test_file_not_found(self):
    with mock.patch.object(epath, 'Path', autospec=True) as mock_path_cls:
      mock_path_cls.return_value.open.side_effect = FileNotFoundError
      with self.assertRaises(FileNotFoundError):
        config_parsing._load_yaml_config('bad/path.yaml')

  def test_yaml_error(self):
    with mock.patch.object(epath, 'Path', autospec=True) as mock_path_cls:
      mock_path_cls.return_value.open = mock.mock_open(
          read_data='key: value: error'
      )
      with self.assertRaises(yaml.YAMLError):
        config_parsing._load_yaml_config('fake/path.yaml')


class TestValidateConfig(parameterized.TestCase):

  def _get_valid_config(self):
    return {
        'suite_name': 'valid',
        'checkpoint_config': {'spec': {}},
        'benchmarks': [{
            'generator': 'my.module.MyGenerator',
            'options': {'param1': 1},
        }],
    }

  def test_valid(self):
    try:
      config_parsing._validate_config(self._get_valid_config())
    except ValueError:
      self.fail('Validation failed on a valid config.')

  def test_valid_with_checkpoint_configs(self):
    config = self._get_valid_config()
    del config['checkpoint_config']
    config['checkpoint_configs'] = [{'spec': {}}]
    try:
      config_parsing._validate_config(config)
    except ValueError:
      self.fail('Validation failed on config with checkpoint_configs.')

  @parameterized.parameters('suite_name', 'benchmarks')
  def test_missing_required_keys(self, key_to_remove):
    config = self._get_valid_config()
    del config[key_to_remove]
    with self.assertRaisesRegex(
        ValueError, f'Missing required key.*{key_to_remove}'
    ):
      config_parsing._validate_config(config)

  def test_missing_checkpoint_config_and_configs(self):
    config = self._get_valid_config()
    del config['checkpoint_config']
    with self.assertRaisesRegex(
        ValueError,
        'Missing required key in YAML config: checkpoint_config or'
        ' checkpoint_configs',
    ):
      config_parsing._validate_config(config)

  def test_benchmarks_not_list(self):
    config = self._get_valid_config()
    config['benchmarks'] = {}

    with self.assertRaisesRegex(ValueError, "'benchmarks' must be a list"):
      config_parsing._validate_config(config)

  def test_benchmark_item_not_dict(self):
    config = self._get_valid_config()
    config['benchmarks'].append('not_a_dict')

    with self.assertRaisesRegex(
        ValueError, "Each item in 'benchmarks' must be a dict"
    ):
      config_parsing._validate_config(config)

  @parameterized.parameters('generator', 'options')
  def test_benchmark_item_missing_keys(self, key_to_remove):
    config = self._get_valid_config()
    del config['benchmarks'][0][key_to_remove]

    with self.assertRaisesRegex(ValueError, f"Missing '{key_to_remove}'"):
      config_parsing._validate_config(config)

  def test_options_not_dict(self):
    config = self._get_valid_config()
    config['benchmarks'][0]['options'] = 'not_a_dict'  # type: ignore

    with self.assertRaisesRegex(ValueError, "'options' must be a dict"):
      config_parsing._validate_config(config)


class TestCreateTestSuiteFromConfig(parameterized.TestCase):

  def _get_valid_yaml_content(self):
    return """
suite_name: Full Test Suite
checkpoint_config:
  spec: { 'a': 'numpy.ndarray:float32:10' }
benchmarks:
  -
    generator: MockGenerator
    options:
      param1: 10
      param2: 'test'
  -
    generator: MockGenerator
    options:
      param1: [20, 30]
"""

  @mock.patch.object(config_parsing, '_load_yaml_config', autospec=True)
  @mock.patch.object(config_parsing, '_import_class', autospec=True)
  def test_valid_creation(self, mock_import, mock_load):
    mock_load.return_value = yaml.safe_load(self._get_valid_yaml_content())
    mock_import.return_value = MockGenerator

    test_suite = config_parsing.create_test_suite_from_config('fake.yaml')

    self.assertEqual(mock_import.call_count, 2)
    mock_import.assert_called_with('MockGenerator')
    self.assertIsInstance(test_suite, core.TestSuite)
    self.assertEqual(test_suite._name, 'Full Test Suite')
    self.assertEqual(test_suite._num_repeats, 1)
    self.assertLen(test_suite._benchmarks_generators, 2)
    self.assertIsInstance(test_suite._benchmarks_generators[0], MockGenerator)
    self.assertEqual(
        test_suite._benchmarks_generators[0]._options,
        MockOptions(param1=10, param2='test'),
    )
    self.assertIsInstance(
        test_suite._benchmarks_generators[1]._options, MockOptions
    )
    opts = test_suite._benchmarks_generators[1]._options
    assert isinstance(opts, MockOptions)
    self.assertEqual(opts.param1, [20, 30])
    self.assertIsNone(test_suite._benchmarks_generators[0]._mesh_configs)
    self.assertIsNone(test_suite._benchmarks_generators[1]._mesh_configs)

  @mock.patch.object(config_parsing, '_load_yaml_config', autospec=True)
  @mock.patch.object(config_parsing, '_import_class', autospec=True)
  def test_valid_creation_with_num_repeats(self, mock_import, mock_load):
    yaml_content = """
suite_name: Repeated Test Suite
num_repeats: 5
checkpoint_config:
  spec: { 'a': 'numpy.ndarray:float32:10' }
benchmarks:
  -
    generator: MockGenerator
    options:
      param1: 10
"""
    mock_load.return_value = yaml.safe_load(yaml_content)
    mock_import.return_value = MockGenerator

    test_suite = config_parsing.create_test_suite_from_config('fake.yaml')
    self.assertEqual(test_suite._num_repeats, 5)

  def _get_yaml_with_mesh_config(self):
    return """
suite_name: Full Test Suite
checkpoint_config:
  spec: { 'a': 'numpy.ndarray:float32:10' }
mesh_config:
  mesh_axes: ['data', 'model']
  ici_parallelism: {'data': 2, 'model': 2}
benchmarks:
  -
    generator: MockGenerator
    options:
      param1: 10
      param2: 'test'
"""

  def _get_yaml_with_mesh_configs(self):
    return """
suite_name: Full Test Suite
checkpoint_config:
  spec: { 'a': 'numpy.ndarray:float32:10' }
mesh_configs:
  -
    mesh_axes: ['data', 'model']
    ici_parallelism: {'data': 2, 'model': 2}
  -
    mesh_axes: ['data', 'model']
    ici_parallelism: {'data': 4, 'model': 1}
benchmarks:
  -
    generator: MockGenerator
    options:
      param1: 10
      param2: 'test'
"""

  def _get_yaml_with_checkpoint_configs(self):
    return """
suite_name: Full Test Suite
checkpoint_configs:
  -
    spec: { 'a': 'numpy.ndarray:float32:10' }
  -
    spec: { 'b': 'numpy.ndarray:float32:20' }
benchmarks:
  -
    generator: MockGenerator
    options:
      param1: 10
      param2: 'test'
"""

  @mock.patch.object(config_parsing, '_load_yaml_config', autospec=True)
  @mock.patch.object(config_parsing, '_import_class', autospec=True)
  def test_valid_creation_with_mesh_config(self, mock_import, mock_load):
    mock_load.return_value = yaml.safe_load(self._get_yaml_with_mesh_config())
    mock_import.return_value = MockGenerator

    test_suite = config_parsing.create_test_suite_from_config('fake.yaml')

    self.assertEqual(mock_import.call_count, 1)
    mock_import.assert_called_with('MockGenerator')
    self.assertIsInstance(test_suite, core.TestSuite)
    self.assertEqual(test_suite._name, 'Full Test Suite')
    self.assertLen(test_suite._benchmarks_generators, 1)
    self.assertIsInstance(test_suite._benchmarks_generators[0], MockGenerator)
    self.assertEqual(
        test_suite._benchmarks_generators[0]._options,
        MockOptions(param1=10, param2='test'),
    )
    self.assertEqual(
        test_suite._benchmarks_generators[0]._mesh_configs,
        [
            config_lib.MeshConfig(
                mesh_axes=['data', 'model'],
                ici_parallelism={'data': 2, 'model': 2},
            )
        ],
    )

  @mock.patch.object(config_parsing, '_load_yaml_config', autospec=True)
  @mock.patch.object(config_parsing, '_import_class', autospec=True)
  def test_valid_creation_with_mesh_configs(self, mock_import, mock_load):
    mock_load.return_value = yaml.safe_load(self._get_yaml_with_mesh_configs())
    mock_import.return_value = MockGenerator

    test_suite = config_parsing.create_test_suite_from_config('fake.yaml')

    self.assertEqual(mock_import.call_count, 1)
    mock_import.assert_called_with('MockGenerator')
    self.assertIsInstance(test_suite, core.TestSuite)
    self.assertEqual(test_suite._name, 'Full Test Suite')
    self.assertLen(test_suite._benchmarks_generators, 1)
    self.assertIsInstance(test_suite._benchmarks_generators[0], MockGenerator)
    self.assertEqual(
        test_suite._benchmarks_generators[0]._options,
        MockOptions(param1=10, param2='test'),
    )
    self.assertEqual(
        test_suite._benchmarks_generators[0]._mesh_configs,
        [
            config_lib.MeshConfig(
                mesh_axes=['data', 'model'],
                ici_parallelism={'data': 2, 'model': 2},
            ),
            config_lib.MeshConfig(
                mesh_axes=['data', 'model'],
                ici_parallelism={'data': 4, 'model': 1},
            ),
        ],
    )

  @mock.patch.object(config_parsing, '_load_yaml_config', autospec=True)
  @mock.patch.object(config_parsing, '_import_class', autospec=True)
  def test_valid_creation_with_checkpoint_configs(self, mock_import, mock_load):
    mock_load.return_value = yaml.safe_load(
        self._get_yaml_with_checkpoint_configs()
    )
    mock_import.return_value = MockGenerator

    test_suite = config_parsing.create_test_suite_from_config('fake.yaml')

    self.assertEqual(mock_import.call_count, 1)
    mock_import.assert_called_with('MockGenerator')
    self.assertIsInstance(test_suite, core.TestSuite)
    self.assertEqual(test_suite._name, 'Full Test Suite')
    self.assertLen(test_suite._benchmarks_generators, 1)
    self.assertIsInstance(test_suite._benchmarks_generators[0], MockGenerator)
    self.assertEqual(
        test_suite._benchmarks_generators[0]._options,
        MockOptions(param1=10, param2='test'),
    )
    self.assertLen(
        test_suite._benchmarks_generators[0]._checkpoint_configs,
        2,
    )
    self.assertEqual(
        test_suite._benchmarks_generators[0]._checkpoint_configs,
        [
            config_lib.CheckpointConfig(
                spec={'a': 'numpy.ndarray:float32:10'}
            ),
            config_lib.CheckpointConfig(
                spec={'b': 'numpy.ndarray:float32:20'}
            ),
        ],
    )

  @mock.patch.object(config_parsing, '_load_yaml_config', autospec=True)
  @mock.patch.object(config_parsing, '_import_class', autospec=True)
  def test_generator_import_fail(self, mock_import, mock_load):
    config = yaml.safe_load(self._get_valid_yaml_content())
    mock_load.return_value = config
    mock_import.side_effect = ImportError('Test Import Error')

    with self.assertRaisesRegex(ImportError, 'Test Import Error'):
      config_parsing.create_test_suite_from_config('fake.yaml')

  @mock.patch.object(config_parsing, '_load_yaml_config', autospec=True)
  def test_generator_not_subclass(self, mock_load):
    yaml_content = """
suite_name: Bad Generator
checkpoint_config: {spec: {}}
benchmarks:
  - generator: yaml.YAMLObject # Not a BenchmarksGenerator
    options: {}
"""

    mock_load.return_value = yaml.safe_load(yaml_content)

    with self.assertRaisesRegex(
        TypeError, 'is not a subclass of BenchmarksGenerator'
    ):
      config_parsing.create_test_suite_from_config('fake.yaml')

  @mock.patch.object(config_parsing, '_load_yaml_config', autospec=True)
  def test_generator_not_decorated(self, mock_load):
    yaml_content = """
suite_name: Not Decorated
checkpoint_config: {spec: {}}
benchmarks:
  - generator: orbax.checkpoint._src.testing.benchmarks.core.config_parsing_test.UndecoratedMockGenerator
    options: {}
"""

    mock_load.return_value = yaml.safe_load(yaml_content)

    with self.assertRaisesRegex(
        TypeError, 'is not decorated with @benchmark_options'
    ):
      config_parsing.create_test_suite_from_config('fake.yaml')

  @mock.patch.object(config_parsing, '_load_yaml_config', autospec=True)
  def test_invalid_options_for_dataclass(self, mock_load):
    yaml_content = """
suite_name: Bad Options
checkpoint_config: {spec: {}}
benchmarks:
  -
    generator: orbax.checkpoint._src.testing.benchmarks.core.config_parsing_test.MockGenerator
    options: { 'bad_param': 1 }
"""

    mock_load.return_value = yaml.safe_load(yaml_content)

    with self.assertRaises(TypeError):
      config_parsing.create_test_suite_from_config('fake.yaml')

  @parameterized.named_parameters(
      (
          'missing_suite_name',
          {'checkpoint_config': {}, 'benchmarks': []},
          'Missing required key.*suite_name',
      ),
      (
          'benchmark_options_not_dict',
          {
              'suite_name': 'test',
              'checkpoint_config': {},
              'benchmarks': [{'generator': 'gen', 'options': 'not_a_dict'}],
          },
          "'options' must be a dict",
      ),
  )
  @mock.patch.object(config_parsing, '_load_yaml_config', autospec=True)
  def test_validation_error_in_create(
      self, config_dict, expected_regex, mock_load
  ):
    mock_load.return_value = config_dict
    with self.assertRaisesRegex(ValueError, expected_regex):
      config_parsing.create_test_suite_from_config('fake.yaml')


if __name__ == '__main__':
  absltest.main()
