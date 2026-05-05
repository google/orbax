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
# pylint: disable=logging-fstring-interpolation
import os
import re
import time
from typing import List, Optional

from absl import logging
from etils import epath
import jax
from jax.experimental import colocated_python
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint._src.futures import signaling_client
from orbax.checkpoint._src.multihost import colocated_transport
from orbax.checkpoint._src.multihost import dispatchers
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.multihost import multislice
from orbax.checkpoint.experimental.emergency.multi_tier_checkpointing import (
    colocated_utils,
)


_REPLICATOR_FILE = 'replicator.yaml'
_TEMP_REPLICATOR_FILE_NAME = _REPLICATOR_FILE + '.tmp'
_JAX_INIT_INFO_FILE = 'jax-init-info.txt'
_RESTORE_DIR_RE = re.compile(r'^.+-s(?P<step>\d+)-n\d+-w\d+\.restore$')


def _wait_for_replicator_file_to_disappear(
    local_checkpoint_directory: epath.Path, *, timeout_seconds: int = 300
):
  """Waits for the MTC daemonset to consume `replicator.yaml`."""
  replicator_file = epath.Path(local_checkpoint_directory) / _REPLICATOR_FILE
  logging.info(
      f'Waiting for {replicator_file} to disappear '
      f'(timeout={timeout_seconds}s)...'
  )
  for t in range(timeout_seconds):
    if not replicator_file.exists():
      logging.info('replicator.yaml no longer exists (waited %ds).', t)
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
    data_parallelism: int,
    node_rank: int,
    peer_ranks: List[int],
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
  final_yaml = '\n'.join(
      line.strip() for line in replicator_yaml.split('\n')
  )
  logging.info(
      f'Writing replicator file to {replicator_file} (via temp {temp_file})'
  )
  temp_file.write_text(final_yaml)
  os.replace(temp_file, replicator_file)
  logging.info('Replicator file written and renamed successfully.')


def _node_rank_input_array(
    colocated_cpu_devices: tuple[jax.Device, ...],
) -> jax.Array:
  """Builds a per-worker rank array over colocated CPU devices."""
  node_ranks = np.arange(len(colocated_cpu_devices), dtype=np.int32)
  mesh = jax.sharding.Mesh(
      np.array(colocated_cpu_devices, dtype=object), ('worker',)
  )
  sharding = jax.sharding.NamedSharding(
      mesh, jax.sharding.PartitionSpec('worker')
  )
  return jax.make_array_from_callback(
      node_ranks.shape,
      sharding,
      lambda idx: node_ranks[idx],
      dtype=jnp.int32,
  )


def _initialize_mtc_colocated(
    local_checkpoint_directory: epath.Path,
    backup_interval_minutes: int,
    num_slices: int,
    run_name: str,
    data_parallelism: int,
    timeout_seconds: int,
) -> None:
  """Initializes multi-tier checkpointing with a colocated Python sidecar on all workers.

  Args:
    local_checkpoint_directory: The local checkpoint directory on the
      worker's filesystem.
    backup_interval_minutes: The backup interval in minutes.
    num_slices: The number of slices.
    run_name: The run name.
    data_parallelism: The data parallelism.
    timeout_seconds: The timeout in seconds.
  """
  logging.info(
      'Initializing colocated MTC setup: '
      f'controller_device_count={jax.device_count()}'
  )
  colocated_transport.install_pathways_colocated_serialization_patch()
  all_devices = jax.devices()

  colocated_cpu_devices = colocated_utils.colocated_cpu_devices_by_worker(
      tuple(all_devices)
  )
  num_nodes = len(colocated_cpu_devices)
  if num_nodes == 0:
    raise ValueError('No colocated CPU devices found for MTC initialization.')
  logging.info(
      f'Dispatching MTC initialization to {num_nodes} colocated CPU devices.'
  )

  dummy_in = dispatchers.get_dummy_input_array(colocated_cpu_devices)
  node_rank_in = _node_rank_input_array(colocated_cpu_devices)

  local_dir_str = str(local_checkpoint_directory)

  def _setup(dummy_arg: jax.Array, node_rank_arg: jax.Array) -> jax.Array:
    signaling_client.mark_pathways_colocated_runtime_active()
    if num_nodes % num_slices != 0:
      raise ValueError(
          'num_nodes must be divisible by num_slices, got '
          f'num_nodes={num_nodes}, num_slices={num_slices}.'
      )
    nodes_per_slice = num_nodes // num_slices
    node_rank = int(
        colocated_utils.require_single_local_scalar_result(
            node_rank_arg, op_name='mtc_node_rank'
        )
    )
    if not 0 <= node_rank < num_nodes:
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
    try:
      replicator_file.unlink()
      logging.info('Removed stale replicator.yaml from previous run.')
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
    _block_and_process_restore_dir(
        loc_dir, timeout_seconds=timeout_seconds
    )

    # Construct a fresh array from local data only.
    return jax.make_array_from_callback(
        dummy_arg.shape,
        dummy_arg.sharding,
        lambda _: np.array(True),
        dtype=jnp.bool_,
    )

  wrapped_setup_fn = colocated_python.colocated_python(_setup)
  wrapped_setup_fn = wrapped_setup_fn.specialize(
      out_specs_fn=lambda dummy_arg, _node_rank_arg: dummy_arg
  )

  dispatch_start = time.time()
  result = wrapped_setup_fn(dummy_in, node_rank_in)
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
    data_parallelism: Number of identical pipelines in job, should be
      equal to ICI data parallelism * DCN data parallelism. If not provided, it
      will be inferred from the number of slices.
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
        'Initializing multi-tier checkpointing via Colocated Python: '
        f'run_name={run_name}, num_slices={num_slices}, '
        f'data_parallelism={data_parallelism}.'
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
      'Initializing multi-tier checkpointing: '
      f'run_name={run_name}, num_slices={num_slices}, '
      f'data_parallelism={data_parallelism}.'
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
  node_rank_by_process_index = (
      multihost.runtime_to_distributed_ids()
  )
  if use_mtc_process_ids:
    logging.info(
        f'Mapping of IDs: jax-init-info.txt={process_id}, '
        f'NodeRank={node_rank}, ProcessIndex={my_process_index}, '
        f'ProcessIndex->NodeRank={node_rank_by_process_index}'
    )
  else:
    logging.info(
        f'Mapping of IDs (jax-init-info not used): NodeRank={node_rank}, '
        f'ProcessIndex={my_process_index}, '
        f'ProcessIndex->NodeRank={node_rank_by_process_index}'
    )

  my_in_pipeline_index = my_process_index % nodes_per_slice
  peer_ranks = []
  for i in range(num_slices):
    peer_process_index = i * nodes_per_slice + my_in_pipeline_index
    if peer_process_index != my_process_index:
      peer_process_rank = node_rank_by_process_index[peer_process_index]
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
            "JAX init info file doesn't have required process id and"
            f' coordinator address data: Current values: {values}'
        )
      logging.info('Found %s after %ds.', _JAX_INIT_INFO_FILE, i)
      return values[:2]
    if i % 30 == 0:
      logging.info(f'Waiting for {_JAX_INIT_INFO_FILE}... elapsed={i}s')
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

  MTC creates a `*.restore` symlink to the directory and Orbax renames it into
  the numeric step directory the backend already understands.
  """
  local_checkpoint_directory = epath.Path(local_checkpoint_directory)
  for _ in range(timeout_seconds):
    files = [f.name for f in local_checkpoint_directory.glob('*.restore')]
    logging.info('block_and_process_restore_dir: restore files: %s', files)
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


def _extract_step(f: str) -> str:
  """Extracts the checkpoint step from an MTC restore file name."""
  match = _RESTORE_DIR_RE.fullmatch(f)
  if match is None:
    raise ValueError(
        'Unexpected restore artifact name. Expected '
        '{job_name}-s{step}-n{node_rank}-w{worker_rank}.restore, got '
        f'{f!r}.'
    )
  return match.group('step')
