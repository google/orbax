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

"""Initialization for multi-tier checkpointing."""

import os
import time
from typing import Optional

from absl import logging
from etils import epath
import jax
from jax.experimental import colocated_python
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint._src.multihost import colocated_transport
from orbax.checkpoint._src.multihost import dispatchers
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.multihost import multislice

_REPLICATOR_FILE = 'replicator.yaml'
_TEMP_REPLICATOR_FILE_NAME = _REPLICATOR_FILE + '.tmp'
_JAX_INIT_INFO_FILE = 'jax-init-info.txt'


def _wait_for_replicator_file_to_disappear(
    local_checkpoint_directory: epath.Path, *, timeout_seconds: int = 300
):
  """Waits for the MTC daemonset to consume `replicator.yaml`."""
  replicator_file = epath.Path(local_checkpoint_directory) / _REPLICATOR_FILE
  logging.info(
      'Waiting for %s to disappear (timeout=%ss)...',
      replicator_file,
      timeout_seconds,
  )
  for t in range(timeout_seconds):
    if not replicator_file.exists():
      logging.info('replicator.yaml no longer exists (waited %ss).', t)
      return
    time.sleep(1)
  raise TimeoutError(
      f'Timeout reached ({timeout_seconds} seconds) while waiting for'
      f' {_REPLICATOR_FILE} to disappear. The MTC replicator daemon likely'
      ' failed during coordinator election (get_coordinator). Check MTC'
      ' coordinator and replication-worker logs.'
  )


def _create_replicator_file(
    file_path: epath.Path,
    *,
    run_name: str,
    num_nodes: int,
    data_parallelism: int,
    node_rank: int,
    peer_ranks: list[int],
    backup_interval_minutes: int,
):
  """Creates a replicator file."""
  temp_file = epath.Path(file_path) / _TEMP_REPLICATOR_FILE_NAME
  replicator_file = epath.Path(file_path) / _REPLICATOR_FILE
  replicator_yaml = f"""job-name: {run_name}
  framework: orbax
  assume-data-parallelism: {data_parallelism}
  node-rank: {node_rank}
  nodes: {num_nodes}
  peer-ranks: {peer_ranks}
  backup-interval-minutes: {backup_interval_minutes}"""
  final_yaml = '\n'.join([l.strip() for l in replicator_yaml.split('\n')])
  logging.info(
      'Writing replicator file to %s (via temp %s)',
      replicator_file,
      temp_file,
  )
  temp_file.write_text(final_yaml)
  os.replace(temp_file, replicator_file)
  logging.info('Replicator file written and renamed successfully.')


def _initialize_mtc_colocated(
    local_checkpoint_directory: epath.Path,
    backup_interval_minutes: int,
    num_slices: int,
    run_name: str,
    data_parallelism: int,
    timeout_seconds: int,
):
  """Initializes multi-tier checkpointing with colocated Python sidecars.

  Args:
    local_checkpoint_directory: The local checkpoint directory.
    backup_interval_minutes: The backup interval in minutes.
    num_slices: The number of slices.
    run_name: The run name.
    data_parallelism: The data parallelism.
    timeout_seconds: The timeout in seconds.
  """
  logging.info(
      'Initializing colocated MTC setup: process_count=%s, process_index=%s, '
      'device_count=%s',
      jax.process_count(),
      jax.process_index(),
      jax.device_count(),
  )
  colocated_transport.install_pathways_colocated_serialization_patch()
  all_devices = jax.devices()

  # 2. Get ALL unique CPU devices for dispatch to all sidecars.
  #
  # Dispatch to every unique colocated CPU device directly. Worker grouping
  # based on virtual_task_index can collapse an entire slice into one worker
  # when that metadata is unavailable.
  #
  # Instead, we get all unique CPU devices via colocated_cpu_devices()
  # and create a replicated sharding across all of them. With replicated
  # PartitionSpec(), each sidecar executes the closure exactly once
  # regardless of how many CPU devices it owns.
  all_cpu_devices = colocated_python.colocated_cpu_devices(all_devices)

  seen_ids = set()
  unique_cpu_devices = []
  for d in all_cpu_devices:
    if d.id not in seen_ids:
      seen_ids.add(d.id)
      unique_cpu_devices.append(d)
  logging.info(
      'unique_cpu_devices (len=%s), first 8: %s',
      len(unique_cpu_devices),
      [(d.id, d.process_index) for d in unique_cpu_devices[:8]],
  )

  # 3. Use replicated dispatch.
  dummy_in = dispatchers.get_dummy_input_array(unique_cpu_devices)
  logging.info(
      'Dispatch target devices (%s): %s',
      len(unique_cpu_devices),
      [d.id for d in unique_cpu_devices],
  )

  # Convert epath.Path to str before closure capture to ensure
  # clean serialization via cloudpickle.
  local_dir_str = str(local_checkpoint_directory)

  # 4. Define the SPMD closure that runs on each remote worker.
  def _setup(dummy_arg):
    num_nodes = jax.process_count()
    if num_nodes % num_slices != 0:
      raise ValueError(
          'num_nodes must be divisible by num_slices, got '
          f'num_nodes={num_nodes}, num_slices={num_slices}.'
      )
    nodes_per_slice = num_nodes // num_slices
    node_rank = jax.process_index()
    if not (0 <= node_rank < num_nodes):
      raise ValueError(
          f'Invalid node_rank={node_rank} for num_nodes={num_nodes}.'
      )
    my_in_pipeline_index = node_rank % nodes_per_slice
    peer_ranks = [
        i * nodes_per_slice + my_in_pipeline_index
        for i in range(num_slices)
        if (i * nodes_per_slice + my_in_pipeline_index) != node_rank
    ]
    loc_dir = epath.Path(local_dir_str)

    replicator_file = epath.Path(loc_dir) / _REPLICATOR_FILE
    if replicator_file.exists():
      logging.info('Found stale replicator.yaml from previous run. Removing.')
      try:
        replicator_file.unlink()
      except FileNotFoundError:
        pass

    _create_replicator_file(
        loc_dir,
        run_name=run_name,
        num_nodes=num_nodes,
        data_parallelism=data_parallelism,
        node_rank=node_rank,
        peer_ranks=peer_ranks,
        backup_interval_minutes=backup_interval_minutes,
    )
    _wait_for_replicator_file_to_disappear(
        loc_dir, timeout_seconds=timeout_seconds
    )
    _block_and_process_restore_dir(loc_dir, timeout_seconds=timeout_seconds)

    # Construct a fresh array from local data only.
    return jax.make_array_from_callback(
        dummy_arg.shape,
        dummy_arg.sharding,
        lambda _: np.array(True),
        dtype=jnp.bool_,
    )

  # 5. Wrap and dispatch using native JAX SPMD.
  wrapped_setup_fn = colocated_python.colocated_python(_setup)
  wrapped_setup_fn = wrapped_setup_fn.specialize(out_specs_fn=lambda x: x)

  # Triggers concurrent execution across all sidecars.
  dispatch_start = time.time()
  result = wrapped_setup_fn(dummy_in)
  jax.block_until_ready(result)
  logging.info(
      'All shards ready (%.1fs total). Setup complete on all hosts.',
      time.time() - dispatch_start,
  )


def _initialize_jax_from_mtc(
    local_checkpoint_directory: epath.Path,
    jax_initialization_timeout_seconds: int = 900,
) -> str:
  """Initialize jax with jax_init_info."""
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
  return process_id


def initialize_multi_tier_checkpointing(
    local_checkpoint_directory: epath.Path,
    *,
    backup_interval_minutes: int = 30,
    num_slices: Optional[int] = None,
    run_name: Optional[str] = None,
    data_parallelism: Optional[int] = None,
    jax_initialization_timeout_seconds: int = 900,
    use_mtc_process_ids: bool = True,
    use_colocated_python: bool = False,
):
  """Initializes multi-tier checkpointing.

  Args:
    local_checkpoint_directory: The local checkpoint directory.
    backup_interval_minutes: The backup interval for the replicator service, in
      minutes.
    num_slices: The number of slices.
    run_name: The name of the run.
    data_parallelism: Number of identical pipelines in job, should be equal to
      ICI data parallelism * DCN data parallelism. If not provided, it will be
      inferred from the number of slices.
    jax_initialization_timeout_seconds: The timeout for JAX initialization.
    use_mtc_process_ids: Use the MTC rank server to calculate process ids.
    use_colocated_python: Whether to use Colocated Python for initialization.
  """
  run_name = run_name if run_name else os.environ.get('JOBSET_NAME')
  if not run_name:
    raise ValueError(
        'Run name is not set and JOBSET_NAME is not set in the environment.'
    )

  if use_colocated_python:
    num_slices = num_slices or multislice.slice_count()
    data_parallelism = data_parallelism or num_slices
    logging.info(
        'Initializing multi-tier checkpointing via Colocated Python:'
        ' run_name=%s, num_slices=%s,'
        ' data_parallelism=%s.',
        run_name,
        num_slices,
        data_parallelism,
    )
    _initialize_mtc_colocated(
        local_checkpoint_directory=local_checkpoint_directory,
        backup_interval_minutes=backup_interval_minutes,
        num_slices=num_slices,
        run_name=run_name,
        data_parallelism=data_parallelism,
        timeout_seconds=jax_initialization_timeout_seconds,
    )
    return

  # Standard Multi-Controller Path
  if use_mtc_process_ids:
    process_id = _initialize_jax_from_mtc(
        local_checkpoint_directory, jax_initialization_timeout_seconds
    )
  else:
    process_id = None
    jax.distributed.initialize(
        initialization_timeout=jax_initialization_timeout_seconds,
    )

  num_slices = num_slices or multislice.slice_count()
  data_parallelism = data_parallelism or num_slices
  logging.info(
      'Initializing multi-tier checkpointing:'
      ' run_name=%s, num_slices=%s,'
      ' data_parallelism=%s.',
      run_name,
      num_slices,
      data_parallelism,
  )

  multihost.initialize_runtime_to_distributed_ids()
  multihost.initialize_distributed_to_device_ids()
  _wait_for_replicator_file_to_disappear(
      local_checkpoint_directory,
      timeout_seconds=jax_initialization_timeout_seconds,
  )
  num_nodes = jax.process_count()
  nodes_per_slice = num_nodes // num_slices
  node_rank = jax._src.distributed.global_state.process_id  # pylint: disable=protected-access
  my_process_index = jax.process_index()
  process_index_to_node_rank = (
      multihost.runtime_to_distributed_ids()
  )
  if use_mtc_process_ids:
    logging.info(
        'Mapping of IDs: jax-init-info.txt=%s, NodeRank=%s, ProcessIndex=%s,'
        ' ProcessIndex->NodeRank=%s',
        process_id,
        node_rank,
        my_process_index,
        process_index_to_node_rank,
    )
  else:
    logging.info(
        'Mapping of IDs (jax-init-info not used): NodeRank=%s, ProcessIndex=%s,'
        ' ProcessIndex->NodeRank=%s',
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

  _create_replicator_file(
      local_checkpoint_directory,
      run_name=run_name,
      num_nodes=num_nodes,
      data_parallelism=data_parallelism,
      node_rank=node_rank,
      peer_ranks=peer_ranks,
      backup_interval_minutes=backup_interval_minutes,
  )
  _wait_for_replicator_file_to_disappear(
      local_checkpoint_directory,
      timeout_seconds=jax_initialization_timeout_seconds,
  )
  _block_and_process_restore_dir(local_checkpoint_directory)


def _retrieve_jax_init_info(
    local_checkpoint_directory: epath.Path, *, timeout_seconds: int = 900
) -> list[str]:
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
    if i % 30 == 0:
      logging.info(
          'Waiting for %s... elapsed=%ss',
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
  """Blocks until MTC exposes a `.restore` artifact and installs it locally.

  Args:
    local_checkpoint_directory: The local checkpoint directory.
    timeout_seconds: The timeout in seconds.

  Raises:
    TimeoutError: if no .restore file is found within the timeout.

  MTC creates a `*.restore` artifact and Orbax renames it into the numeric step
  directory the backend already understands.
  """
  for _ in range(timeout_seconds):
    files = [f.name for f in local_checkpoint_directory.glob('*.restore')]
    for f in files:
      step = _extract_step(f)
      if step != '0':
        step_dir = epath.Path(local_checkpoint_directory) / step
        os.replace(
            epath.Path(local_checkpoint_directory) / f,
            step_dir,
        )
        logging.info(
            'Found a restore directory at step %s and renamed it to %s.',
            step,
            step_dir,
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
