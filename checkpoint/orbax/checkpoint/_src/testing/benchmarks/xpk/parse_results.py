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

# Copyright 2024 The Orbax Authors.
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

"""Parses Orbax benchmark run results from GCS."""

import argparse
import json
from etils import epath
import numpy as np


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--output_dir',
      type=str,
      required=True,
      help=(
          'GCS root directory of the runs (e.g.'
          ' gs://bucket/runs/20260616-1703UTC)'
      ),
  )
  return parser.parse_args()


def parse_run_results(run_dir_path: epath.Path):
  """Parses and aggregates means from host sidecar JSON files in a run."""
  print('\n==================================================')
  print(f'Parsing results in: {run_dir_path}')
  print('==================================================')

  tb_dir = run_dir_path / 'tensorboard'
  if not tb_dir.exists():
    print('No tensorboard directory found!')
    return

  for benchmark_dir in tb_dir.iterdir():
    if not benchmark_dir.is_dir():
      continue

    benchmark_name = benchmark_dir.name
    print(f'\nBenchmark: {benchmark_name}')

    # Locate per-host means JSON files
    means_dir = benchmark_dir / '_per_host_means'
    if not means_dir.exists():
      # Check if per-host metrics were enabled directly
      means_dir = benchmark_dir

    # Gather all hosts
    hosts_data = []
    for host_dir in means_dir.iterdir():
      if host_dir.name.startswith('host_'):
        sidecar = host_dir / '_per_host_means.json'
        if sidecar.exists():
          try:
            hosts_data.append(json.loads(sidecar.read_text()))
          except Exception as e:  # pylint: disable=broad-exception-caught
            print(f'Failed to read {sidecar}: {e}')

    if not hosts_data:
      print('No host metrics found.')
      continue

    # Aggregate across hosts
    all_keys = sorted(list(set().union(*(d['keys'] for d in hosts_data))))

    # Print metrics
    print(f'Aggregated Metrics (across {len(hosts_data)} hosts):')
    for key in all_keys:
      values = []
      for d in hosts_data:
        if key in d['keys']:
          idx = d['keys'].index(key)
          values.append(d['means'][idx])

      if values:
        mean_val = np.mean(values)
        max_val = np.max(values)
        min_val = np.min(values)

        # Highlight important bandwidth/throughput metrics
        if 'throughput' in key or 'ratio' in key or 'gbps' in key:
          print(
              f'  \033[1;32m{key:60s}: mean={mean_val:10.4f} |'
              f' max={max_val:10.4f} | min={min_val:10.4f}\033[0m'
          )
        else:
          print(f'  {key:60s}: mean={mean_val:10.4f} s')


def main():
  args = parse_args()
  root_path = epath.Path(args.output_dir)
  if not root_path.exists():
    print(f'Error: {root_path} does not exist!')
    return

  # Walk subdirectories (e.g. v5e/micro, v5e/bandwidth, v5p/micro...)
  for tpu_gen_dir in root_path.iterdir():
    if tpu_gen_dir.is_dir():
      for run_type_dir in tpu_gen_dir.iterdir():
        if run_type_dir.is_dir():
          parse_run_results(run_type_dir)


if __name__ == '__main__':
  main()
