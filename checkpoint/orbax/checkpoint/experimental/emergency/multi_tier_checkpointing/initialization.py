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

"""Initialization for multi-tier checkpointing."""
import os
import time
from typing import List, Optional

from absl import logging
from etils import epath
import jax
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.multihost import multislice

_REPLICATOR_FILE = 'replicator.yaml'
_TEMP_REPLICATOR_FILE_NAME = _REPLICATOR_FILE + '.tmp'
_JAX_INIT_INFO_FILE = 'jax-init-info.txt'


def _wait_for_replicator_file_to_disappear(
    local_checkpoint_directory: epath.Path, *, timeout_seconds: int = 300
):
  """Waits for a file to disappear."""
  replicator_file = epath.Path(local_checkpoint_directory) / _REPLICATOR_FILE
  for _ in range(timeout_seconds):
    if not replicator_file.exists():
      logging.info('replicator.yaml no longer exists.')
      return
    time.sleep(1)
  raise TimeoutError(
      f'Timeout reached ({timeout_seconds} seconds) while waiting for'
      f' {_REPLICATOR_FILE} to disappear.'
  )


def _create_replicator_file(
    file_path: epath.Path,
    *,
    run_name: str,
    num_nodes: int,
    num_slices: int,
    node_rank: int,
    peer_ranks: List[int],
    backup_interval_minutes: int,
):
  """Creates a replicator file."""
  temp_file = epath.Path(file_path) / _TEMP_REPLICATOR_FILE_NAME
  replicator_file = epath.Path(file_path) / _REPLICATOR_FILE
  replicator_yaml = f"""job-name: {run_name}
  framework: orbax
  assume-data-parallelism: {num_slices}
  node-rank: {node_rank}
  nodes: {num_nodes}
  peer-ranks: {peer_ranks}
  backup-interval-minutes: {backup_interval_minutes}"""
  temp_file.write_text(
      '\n'.join([l.strip() for l in replicator_yaml.split('\n')])
  )
  os.rename(temp_file, replicator_file)


def initialize_multi_tier_checkpointing(
    local_checkpoint_directory: epath.Path,
    *,
    backup_interval_minutes: int = 10,
    num_slices: Optional[int] = None,
    run_name: Optional[str] = None,
    jax_initialization_timeout_seconds: int = 900,
):
  """Initializes multi-tier checkpointing.

  Args:
    local_checkpoint_directory: The local checkpoint directory.
    backup_interval_minutes: The backup interval for the replicator service, in
      minutes.
    num_slices: The number of slices.
    run_name: The name of the run.
    jax_initialization_timeout_seconds: The timeout for JAX initialization.
  """
  local_checkpoint_directory = epath.Path(local_checkpoint_directory)
  process_id, coordinator_address = _retrieve_jax_init_info(
      local_checkpoint_directory
  )
  if not process_id or not coordinator_address:
    raise ValueError(
        'Data is missing from the JAX init info file: Current values:'
        f' process_id: {process_id}, coordinator_address: {coordinator_address}'
    )
  logging.info(
      'Using process_id %s and coordinator_address %s to initialize JAX'
      ' distributed runtime...',
      process_id,
      coordinator_address,
  )
  jax.distributed.initialize(
      process_id=int(process_id),
      coordinator_address=coordinator_address,
      initialization_timeout=jax_initialization_timeout_seconds,
  )
  multihost.initialize_runtime_to_distributed_ids()
  multihost.initialize_distributed_to_device_ids()
  _wait_for_replicator_file_to_disappear(local_checkpoint_directory)
  num_slices = (
      num_slices
      or multislice.slice_count()
  )
  num_nodes = jax.process_count()
  nodes_per_slice = num_nodes // num_slices
  node_rank = jax._src.distributed.global_state.process_id  # pylint: disable=protected-access
  my_process_index = jax.process_index()
  process_index_to_node_rank = (
      multihost.runtime_to_distributed_ids()
  )
  logging.info(
      'Mapping of IDs: jax-init-info.txt=%s, NodeRank=%s, ProcessIndex=%s,'
      ' ProcessIndex->NodeRank=%s',
      process_id,
      node_rank,
      my_process_index,
      process_index_to_node_rank,
  )
  my_in_pipeline_index = my_process_index % nodes_per_slice
  peer_ranks = []
  for i in range(num_slices):
    peer_process_index = i * nodes_per_slice + my_in_pipeline_index
    if peer_process_index != my_process_index:
      peer_process_rank = process_index_to_node_rank[peer_process_index]
      peer_ranks.append(peer_process_rank)
  logging.info('Peers for NodeRank %s: %s', node_rank, peer_ranks)
  run_name = run_name if run_name else os.environ.get('JOBSET_NAME')
  if not run_name:
    raise ValueError(
        'Run name is not set and JOBSET_NAME is not set in the environment.'
    )
  _create_replicator_file(
      local_checkpoint_directory,
      run_name=run_name,
      num_nodes=num_nodes,
      num_slices=num_slices,
      node_rank=node_rank,
      peer_ranks=peer_ranks,
      backup_interval_minutes=backup_interval_minutes,
  )
  _wait_for_replicator_file_to_disappear(local_checkpoint_directory)
  _block_and_process_restore_dir(local_checkpoint_directory)


def _retrieve_jax_init_info(
    local_checkpoint_directory: epath.Path, *, timeout_seconds: int = 900
) -> List[str]:
  """Retrieve JAX init info from a local file.

  Args:
    local_checkpoint_directory: The local checkpoint directory.
    timeout_seconds: The timeout in seconds.

  Returns:
    A list of strings containing the JAX init info (process id and coordinator
    address).

  Raises:
    TimeoutError: if the JAX init info file is not found within the timeout.
    ValueError: if the JAX init info file is found but the values are not
    valid.

  Allow time for the JAX init info file to be populated by GKE. This is needed
  because the file is only populated when the worker with process id of 0 is
  determined. After a disruption, although some workers might be up and
  running, the init info file won't be populated until the node with process
  id of 0 is known and this could take time. Using 900 seconds for now and it
  needs to be increased if the "repair" time is longer.
  """
  local_jax_init_info_file = (
      epath.Path(local_checkpoint_directory) / _JAX_INIT_INFO_FILE
  )

  for i in range(timeout_seconds):
    if local_jax_init_info_file.exists():
      values = local_jax_init_info_file.read_text().split('\n')
      if len(values) < 2:
        raise ValueError(
            'JAX init info file doesnt have required process id and'
            f' coordinator address data: Current values: {values}'
        )
      return values[:2]
    logging.info(
        'Unable to locate %s after %d seconds,'
        ' sleeping for 1 second before retrying...',
        _JAX_INIT_INFO_FILE,
        i,
    )
    time.sleep(1)
  raise TimeoutError(
      f'Unable to locate {_JAX_INIT_INFO_FILE} after {timeout_seconds} seconds,'
  )


def _block_and_process_restore_dir(
    local_checkpoint_directory: epath.Path, *, timeout_seconds: int = 300
):
  """Block until a file ending with `.restore` appears, then extract the step number and rename the directory using the step number.

  Args:
    local_checkpoint_directory: The local checkpoint directory.
    timeout_seconds: The timeout in seconds.

  Raises:
    TimeoutError: if no .restore file is found within the timeout.
  """
  for _ in range(timeout_seconds):
    files = [f.name for f in local_checkpoint_directory.glob('*.restore')]
    logging.info('block_and_process_restore_dir: restore files: %s', files)
    for f in files:
      step = _extract_step(f)
      if step != '0':
        os.rename(
            epath.Path(local_checkpoint_directory) / f,
            epath.Path(local_checkpoint_directory) / step,
        )
        logging.info(
            'Found a restore directory at step %s and renamed it to %s.',
            step,
            epath.Path(local_checkpoint_directory) / step,
        )
      else:
        logging.info(
            'Found a restore directory at step 0, skipping renaming.'
        )
      return
    time.sleep(1)
  raise TimeoutError(
      f'{timeout_seconds} seconds have passed but no .restore file was found.'
  )


def _extract_step(f):
  # The base file name is formatted as:
  # {job_name}-s{step}-n{node_rank}-w{worker_rank}
  return f.rsplit('-', 3)[1][1:]
