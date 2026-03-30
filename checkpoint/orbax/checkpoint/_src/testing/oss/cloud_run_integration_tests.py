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

"""Runs Orbax benchmarks on GCP."""

import datetime
import os
import subprocess
import sys
import yaml


def run_benchmark(test_config):
  """Runs a single benchmark test based on the given config.

  Args:
    test_config: A dictionary containing the test configuration.

  Returns:
    True if benchmark ran successfully, False otherwise.
  """
  print(f"Running benchmark: {test_config['name']}")

  # Build command
  output_dir = os.path.join(
      test_config['output_directory'],
      datetime.datetime.now().strftime('%Y%m%d'),
  )

  cmd = [
      'python3',
      'orbax/checkpoint/_src/testing/benchmarks/xpk/launch_xpk.py',
      '--cluster_name',
      test_config['cluster_name'],
      '--tpu_type',
      test_config['tpu_type'],
      '--zone',
      test_config['zone'],
      '--config_file',
      test_config['config_file'],
      '--docker_image',
      test_config['docker_image'],
      '--output_directory',
      output_dir,
      '--num_slices',
      str(test_config['num_slices']),
  ]
  if test_config.get('nodelete_cluster_on_completion'):
    cmd.append('--nodelete_cluster_on_completion')
  if test_config.get('ramdisk_directory'):
    cmd.extend(['--ramdisk_directory', test_config['ramdisk_directory']])
  if test_config.get('test_restart_workflow'):
    cmd.append('--test_restart_workflow')
  if test_config.get('verbose'):
    cmd.append('--verbose')
  if test_config.get('skip_validation'):
    cmd.append('--skip_validation')
  if test_config.get('enable_pathways'):
    cmd.append('--enable_pathways')
  if test_config.get('gcp_region'):
    cmd.extend(['--region', test_config['gcp_region']])

  print(f"Executing command: {' '.join(cmd)}")
  try:
    subprocess.run(cmd, check=True)
  except subprocess.CalledProcessError as e:
    print(f'Benchmark script failed: {e}')
    return False

  return True


def main():
  """Loads test configurations and runs benchmarks."""
  config_path = 'orbax/checkpoint/_src/testing/oss/cloud_run_integration_tests.yaml'
  with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

  failures = 0
  for test in config.get('tests', []):
    if not run_benchmark(test):
      failures += 1

  if failures:
    print(f'{failures} benchmarks failed.')
    sys.exit(1)


if __name__ == '__main__':
  main()
