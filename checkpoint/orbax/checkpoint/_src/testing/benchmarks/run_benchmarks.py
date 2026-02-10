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

r"""Main script to run Orbax checkpoint benchmarks based on YAML configurations.


When running in an open-source environment (e.g., on a cluster with mpirun),
jax.distributed.initialize() will be called to set up the distributed system
using standard environment variables like JAX_COORDINATOR_ADDRESS, JAX_PROCESS_ID,
and JAX_NUM_PROCESSES.
"""

import os
from typing import List

from absl import app
from absl import flags
from absl import logging
from etils import epath
import jax
from orbax.checkpoint._src.testing.benchmarks.core import config_parsing


# Core Flags
_CONFIG_FILE = flags.DEFINE_string(
    'config_file',
    None,
    'Path to the YAML configuration file for the benchmark.',
    required=True,
)
_OUTPUT_DIRECTORY = flags.DEFINE_string(
    'output_directory',
    None,
    'Output directory for benchmark results.',
    required=True,
)
_LOCAL_DIRECTORY = flags.DEFINE_string(
    'local_directory',
    None,
    'Local directory for benchmark results. This is used for ECM benchmarks.',
)
_ENABLE_HLO_DUMP = flags.DEFINE_bool(
    'enable_hlo_dump',
    False,
    'Enables HLO dumping to a subdirectory within --output_directory.',
)




def _init_jax_distributed():
  """Initializes JAX distributed system if not managed by XManager."""

  try:
    jax.distributed.initialize()
    logging.info('JAX distributed system initialized.')
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.warning(
        'Failed to initialize JAX distributed system: %s. '
        'This is expected if running in a single-process environment. '
        'Continuing as single-process.',
        e,
        exc_info=False,
    )

  logging.info('JAX process index: %d', jax.process_index())
  logging.info('JAX process count: %d', jax.process_count())
  logging.info('JAX device count: %d', jax.device_count())
  logging.info('JAX local device count: %d', jax.local_device_count())


def _configure_hlo_dump(output_directory: str):
  """Sets the XLA_FLAGS environment variable to enable HLO dumping."""
  hlo_dump_path = epath.Path(output_directory) / 'hlo_dump'
  try:
    hlo_dump_path.mkdir(parents=True, exist_ok=True)
    logging.info('Created HLO dump directory: %s', hlo_dump_path)
  except OSError as e:
    logging.exception(
        'Failed to create HLO dump directory %s: %s', hlo_dump_path, e
    )
    raise

  xla_flags = os.environ.get('XLA_FLAGS', '')
  # Options: as_proto, as_text, as_url
  dump_flags = f'--xla_dump_to={hlo_dump_path} --xla_dump_hlo_as_proto'

  new_xla_flags = f'{xla_flags} {dump_flags}'.strip()
  os.environ['XLA_FLAGS'] = new_xla_flags
  logging.info('Set XLA_FLAGS for HLO dump: %s', new_xla_flags)


def _run_benchmarks(
    config_file: str, output_directory: str, local_directory: str | None = None
) -> None:
  """Runs Orbax checkpoint benchmarks based on a generator class and a config file.

  Args:
    config_file: Path to the YAML configuration file.
    output_directory: Directory to store benchmark results in.
    local_directory: Local directory for benchmark results. This is used for ECM
      benchmarks.

  Raises:
    RuntimeError: If any benchmark test fails.
  """
  logging.info('Running benchmarks from config: %s', config_file)
  logging.info('Output directory: %s', output_directory)
  try:
    epath.Path(output_directory).mkdir(parents=True, exist_ok=True)
    logging.info('Ensured output directory exists: %s', output_directory)
  except OSError as e:
    logging.exception(
        'Failed to create output directory %s: %s', output_directory, e
    )
    raise

  try:
    test_suite = config_parsing.create_test_suite_from_config(
        config_file,
        output_dir=output_directory,
        local_directory=local_directory,
    )
  except Exception as e:
    logging.error('Failed to create test suite from config: %s', e)
    raise

  logging.info('Benchmark test suite created successfully.')
  results = test_suite.run()
  failed_results = [result for result in results if not result.is_successful()]
  if not failed_results:
    logging.info('Benchmark test suite run completed successfully.')
  else:
    error_messages = []
    for result in failed_results:
      error_messages.append(
          f'Test: {result.metrics.name}, Error: {repr(result.error)}'
      )
    exception_message = (
        'Benchmark test suite run failed with following errors:\n'
        + '\n'.join(error_messages)
    )
    raise RuntimeError(exception_message)


def main(argv: List[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError(f'Too many command-line arguments: {argv[1:]}')

  logging.info('run_benchmarks.py started.')

  _init_jax_distributed()

  if _ENABLE_HLO_DUMP.value:
    _configure_hlo_dump(_OUTPUT_DIRECTORY.value)
  else:
    logging.info('HLO dump is disabled.')

  xla_flags = os.environ.get('XLA_FLAGS')
  if xla_flags:
    logging.info('XLA_FLAGS is set to: %s', xla_flags)
  else:
    logging.info('XLA_FLAGS is not set in environment.')

  jax.config.update('jax_enable_x64', True)
  logging.info('Set jax_enable_x64=True')

  _run_benchmarks(
      _CONFIG_FILE.value, _OUTPUT_DIRECTORY.value, _LOCAL_DIRECTORY.value
  )

  logging.info('run_benchmarks.py finished.')


if __name__ == '__main__':
  jax.config.config_with_absl()
  app.run(main)
