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

r"""Main script to run Orbax checkpoint benchmarks based on YAML configurations.

"""

import os
import re
from typing import List

from absl import app
from absl import flags
from absl import logging
import jax
from orbax.checkpoint._src.testing.benchmarks.core import config_parsing


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



def run_benchmarks(config_file: str, output_directory: str | None) -> None:
  """Runs Orbax checkpoint benchmarks based on a generator class and a config file.

  Args:
    config_file: Path to the YAML configuration file.
    output_directory: Directory to store benchmark results in.
  """
  try:
    test_suite = config_parsing.create_test_suite_from_config(
        config_file, output_dir=output_directory
    )
  except Exception as e:
    logging.error('Failed to create test suite from config: %s', e)
    raise

  test_suite.run()


def main(argv: List[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  jax.config.update('jax_enable_x64', True)
  run_benchmarks(_CONFIG_FILE.value, _OUTPUT_DIRECTORY.value)


if __name__ == '__main__':
  jax.config.config_with_absl()
  app.run(main)
