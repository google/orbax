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

"""Parses benchmark YAML configurations to create a core.TestSuite.

YAML Structure:

suite_name: str
checkpoint_config:  # See configs.CheckpointConfig
  path: Optional[str]  # Path to load, null to generate.
  spec: Dict[str, Any]  # Spec to generate. {"dtype": ..., "shape": ...,
  "sharding": ...}
mesh_config: Optional[Dict[str, Any]]  # See configs.MeshConfig
benchmarks: List[Dict[str, Any]]
  - generator: str  # Import path to core.BenchmarksGenerator subclass.
    options: Dict[str, Any]  # Kwargs for generator's options dataclass.

Example:

suite_name: My Awesome Benchmark Suite
checkpoint_config:
  spec:
    params: {dtype: float32, shape: [2048, 2048], sharding: [data, model]}
    step: int
mesh_config:
  mesh_axes: [data, model]
  ici_parallelism: {data: 2, model: 2}
  dcn_parallelism: {}
benchmarks:
  - generator: my.module.MyBenchmarkGenerator
    options: {batch_sizes: [64, 128], use_ssd: false}
"""

import importlib
from typing import Any, Dict, List, Type

from absl import logging
from orbax.checkpoint._src.testing.benchmarks.core import configs as config_lib
from orbax.checkpoint._src.testing.benchmarks.core import core
import yaml



def _import_class(class_path: str) -> Type[Any]:
  """Dynamically imports a class from a string path."""
  try:
    module_path, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
  except (ImportError, ValueError, AttributeError) as e:
    raise ImportError(f'Failed to import class {class_path}: {e}') from e


def _load_yaml_config(config_path: str) -> Dict[str, Any]:
  """Loads a YAML configuration file."""
  logging.info('Loading configuration from: %s', config_path)
  try:
    with open(config_path, 'r') as f:
      return yaml.safe_load(f)
  except yaml.YAMLError as e:
    logging.error('Error parsing YAML file: %s', e)
    raise
  except FileNotFoundError:
    logging.error('Configuration file not found: %s', config_path)
    raise


def _validate_config(config: Dict[str, Any]) -> None:
  """Performs basic validation on the loaded YAML config."""
  required_keys = ['suite_name', 'checkpoint_config', 'benchmarks']
  for key in required_keys:
    if key not in config:
      raise ValueError(f'Missing required key in YAML config: {key}')

  if not isinstance(config['benchmarks'], list):
    raise ValueError("'benchmarks' must be a list.")

  for i, benchmark_group in enumerate(config['benchmarks']):
    if not isinstance(benchmark_group, dict):
      raise ValueError(
          "Each item in 'benchmarks' must be a dict, got"
          f' {type(benchmark_group)} at index {i}'
      )
    if 'generator' not in benchmark_group:
      raise ValueError(f"Missing 'generator' in benchmarks entry at index {i}")
    if 'options' not in benchmark_group:
      raise ValueError(f"Missing 'options' in benchmarks entry at index {i}")
    if not isinstance(benchmark_group['options'], dict):
      raise ValueError(
          f"'options' must be a dict in benchmarks entry at index {i}"
      )


def create_test_suite_from_config(
    config_path: str, output_dir: str | None = None
) -> core.TestSuite:
  """Creates a single TestSuite object from the benchmark configuration.

  Args:
    config_path: Path to the YAML configuration file.
    output_dir: Optional directory to store benchmark results in. If None,
      results will be stored in a temporary directory.

  Returns:
    A TestSuite object containing all benchmarks generated from the config.
  """

  config = _load_yaml_config(config_path)

  _validate_config(config)

  suite_name = config['suite_name']
  checkpoint_config = config_lib.CheckpointConfig(**config['checkpoint_config'])
  mesh_config = None
  if 'mesh_config' in config:
    mesh_config = config_lib.MeshConfig(**config['mesh_config'])
  generators: List[core.BenchmarksGenerator] = []

  for i, benchmark_group in enumerate(config['benchmarks']):
    generator_class_path = benchmark_group['generator']
    logging.info(
        'Processing benchmark group %d: %s', i + 1, generator_class_path
    )
    try:
      generator_class = _import_class(generator_class_path)
    except ImportError as e:
      logging.error('Failed to import generator class: %s', e)
      raise

    if not issubclass(generator_class, core.BenchmarksGenerator):
      raise TypeError(
          f'Class {generator_class_path} is not a subclass of'
          ' BenchmarksGenerator.'
      )

    options_class = generator_class.options_class
    if options_class is None:
      raise TypeError(
          f'Generator class {generator_class_path} is not decorated with'
          ' @benchmark_options.'
      )

    generator_options_dict = benchmark_group['options']
    try:
      generator_options = options_class(**generator_options_dict)
    except TypeError as e:
      logging.error(
          'Failed to instantiate options class %s with provided options %s: %s',
          options_class.__name__,
          generator_options_dict,
          e,
      )
      raise

    generator = generator_class(
        checkpoint_config=checkpoint_config,
        options=generator_options,
        output_dir=output_dir,
        mesh_config=mesh_config,
    )
    generators.append(generator)

  return core.TestSuite(name=suite_name, benchmarks_generators=generators)
