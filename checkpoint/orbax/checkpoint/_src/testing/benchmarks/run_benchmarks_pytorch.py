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

"""Benchmark launcher for torch.distributed.checkpoint (DCP) using XManager."""

from collections.abc import Sequence
import os

from absl import app
from absl import flags
from absl import logging
from etils import epath
from orbax.checkpoint._src.testing.benchmarks.core import config_parsing
import torch.distributed as dist
from torch.google import distributed as gdist

from .learning.deepmind.xmanager2.client import xmanager_api
from .pyglib.contrib.g3_multiprocessing import g3_multiprocessing


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




def _init_torch_distributed() -> None:
  """Initializes Torch distributed system if not managed by XManager."""
  if not dist.is_initialized():
    if 'TEST_TMPDIR' not in os.environ:
      logging.info('torch.distributed not initialized, setting env vars...')
      os.environ.setdefault('MASTER_ADDR', 'localhost')
      os.environ.setdefault('MASTER_PORT', '12355')
      dist.init_process_group(backend='nccl', rank=0, world_size=1)
    else:
      try:
        dist.init_process_group(backend='cpu:gloo,cuda:nccl')
        logging.info(
            "Initialized torch.distributed with backend 'cpu:gloo,cuda:nccl',"
            ' rank %d, world size %d',
            dist.get_rank(),
            dist.get_world_size(),
        )
      except Exception as e:
        logging.exception(
            'Failed to initialize torch.distributed via dist: %s', e
        )
        raise


def _run_benchmarks(config_file: str, output_directory: str) -> None:
  """Runs the benchmarks."""
  logging.info('Running benchmarks from config: %s', config_file)
  logging.info('Output directory: %s', output_directory)
  try:
    epath.Path(output_directory).mkdir(parents=True, exist_ok=True)
    logging.info('Output directory created: %s', output_directory)
  except OSError as e:
    logging.exception('Failed to create output directory: %s', e)
    raise

  try:
    test_suite = config_parsing.create_test_suite_from_config(
        config_file, output_dir=output_directory
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


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  logging.info('run_benchmarks_pytorch.py started.')
  _init_torch_distributed()
  _run_benchmarks(_CONFIG_FILE.value, _OUTPUT_DIRECTORY.value)
  logging.info('run_benchmarks_pytorch.py finished.')


if __name__ == '__main__':
  if 'GPU' not in os.environ:
    app.run(main)
  else:
    g3_multiprocessing.handle_main(gdist.torchrun(main))
