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

"""Script to run pytest on test files from a YAML config based on num_processes."""

from collections.abc import Sequence
import os
import subprocess
import sys

from absl import app
from absl import flags
from absl import logging
import pytest
import yaml


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'filename', None, 'Path to yaml file containing test files.'
)
flags.DEFINE_integer(
    'processes',
    None,
    'Number of processes to select test list from yaml file.',
)


def install_deps():
  """Installs required dependencies if not present."""
  deps_map = {'pytest': 'pytest', 'chex': 'chex', 'pyyaml': 'yaml'}
  to_install = []
  for pip_name, import_name in deps_map.items():
    try:
      __import__(import_name)
    except ImportError:
      to_install.append(pip_name)

  if to_install:
    logging.info('Installing dependencies: %s', ', '.join(to_install))
    try:
      subprocess.check_call(
          [sys.executable, '-m', 'pip', 'install'] + to_install
      )
    except subprocess.CalledProcessError as e:
      logging.error('Failed to install dependencies: %s', e)
      sys.exit(1)


def _find_test_path(test_file_yaml):
  """Returns existing path for test file, or None."""
  if os.path.exists(test_file_yaml):
    return test_file_yaml

  path_suffix = test_file_yaml
  if path_suffix.startswith('//third_party/py/'):
    path_suffix = path_suffix.removeprefix('//third_party/py/')

  if ':' in path_suffix:
    path_suffix = path_suffix.replace(':', '/') + '.py'

  candidate1 = path_suffix
  candidate2 = os.path.join('/app/orbax_repo/checkpoint', path_suffix)
  if os.path.exists(candidate1):
    return candidate1
  elif os.path.exists(candidate2):
    return candidate2
  return None


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  install_deps()

  try:
    with open(FLAGS.filename, 'r') as f:
      try:
        tests_by_process_count = yaml.safe_load(f)
      except yaml.YAMLError as e:
        logging.error('Failed to parse yaml file %s: %s', FLAGS.filename, e)
        sys.exit(1)
  except FileNotFoundError:
    logging.error('YAML file not found: %s', FLAGS.filename)
    sys.exit(1)

  key = f'processes:{FLAGS.processes}'
  if key not in tests_by_process_count:
    logging.error(
        'key=%s (from processes=%d) not found as a key in %s. Available'
        ' keys: %s',
        key,
        FLAGS.processes,
        FLAGS.filename,
        list(tests_by_process_count.keys()),
    )
    sys.exit(1)

  test_files = tests_by_process_count[key]
  if not test_files:
    logging.warning(
        'No test files found for processes=%d in %s.',
        FLAGS.processes,
        FLAGS.filename,
    )
    return

  results = {}
  failed_tests = []

  for test_file_yaml in test_files:
    test_path = _find_test_path(test_file_yaml)
    if test_path is None:
      results[test_file_yaml] = 'SKIPPED'
      logging.warning('Skipping %s: file not found.', test_file_yaml)
      continue

    logging.info('Running test: %s (found from %s)', test_path, test_file_yaml)
    try:
      exit_code = pytest.main([test_path])
      if exit_code == 0:
        results[test_file_yaml] = 'PASSED'
        logging.info('%s: PASSED', test_path)
      else:
        results[test_file_yaml] = 'FAILED'
        failed_tests.append(test_file_yaml)
        logging.error('%s: FAILED with exit code %s', test_path, exit_code)
    except Exception as e:  # Catching broad Exception to log any unexpected failure during pytest execution. # pylint: disable=broad-exception-caught
      results[test_file_yaml] = 'FAILED'
      failed_tests.append(test_file_yaml)
      logging.error('%s: FAILED with exception: %s', test_path, e)

  print('--- Test Summary ---')
  for test_file, status in results.items():
    print(f'{test_file}: {status}')
  print('--------------------')

  if failed_tests:
    logging.error('%d test(s) failed: %s', len(failed_tests), failed_tests)
    sys.exit(1)
  else:
    logging.info('All tests passed or skipped.')


if __name__ == '__main__':
  flags.mark_flag_as_required('filename')
  flags.mark_flag_as_required('processes')
  app.run(main)
