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


def _validate_replicator_ranks(
    *, num_nodes: int, node_rank: int, peer_ranks: List[int]
) -> None:
  """Validates the rank fields written to `replicator.yaml`."""
  if num_nodes <= 0:
    raise ValueError(f'num_nodes must be positive, got {num_nodes}.')
  if not 0 <= node_rank < num_nodes:
    raise ValueError(
        f'Invalid node_rank={node_rank} for num_nodes={num_nodes}.'
    )
  invalid_peer_ranks = [
      rank for rank in peer_ranks if not 0 <= rank < num_nodes
  ]
  if invalid_peer_ranks:
    raise ValueError(
        f'Invalid peer_ranks={invalid_peer_ranks} for num_nodes={num_nodes}.'
    )
  if node_rank in peer_ranks:
    raise ValueError(
        f'peer_ranks must not include node_rank={node_rank}: {peer_ranks}.'
    )
  if len(peer_ranks) != len(set(peer_ranks)):
    raise ValueError(f'peer_ranks must be unique, got {peer_ranks}.')


def _validate_node_rank_by_process_index(
    node_rank_by_process_index: List[int], *, num_nodes: int
) -> None:
  """Validates a ProcessIndex -> NodeRank mapping."""
  if len(node_rank_by_process_index) != num_nodes:
    raise ValueError(
        'ProcessIndex->NodeRank mapping must have one entry per node, got '
        f'{node_rank_by_process_index} for num_nodes={num_nodes}.'
    )
  invalid_entries = [
      (process_index, node_rank)
      for process_index, node_rank in enumerate(node_rank_by_process_index)
      if not 0 <= node_rank < num_nodes
  ]
  if invalid_entries:
    raise ValueError(
        'ProcessIndex->NodeRank mapping contains invalid entries for '
        f'num_nodes={num_nodes}: {invalid_entries}.'
    )
  if len(set(node_rank_by_process_index)) != num_nodes:
    raise ValueError(
        'ProcessIndex->NodeRank mapping must be one-to-one, got '
        f'{node_rank_by_process_index}.'
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
  _validate_replicator_ranks(
      num_nodes=num_nodes, node_rank=node_rank, peer_ranks=peer_ranks
  )
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
  logging.info('Replicator YAML contents:\n%s', final_yaml)
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
      f'process_count={jax.process_count()}, device_count={jax.device_count()}'
  )
  colocated_transport.install_pathways_colocated_serialization_patch()
  all_devices = jax.devices()

  worker_cpu_devices = colocated_utils.colocated_cpu_devices_by_worker(
      tuple(all_devices)
  )
  logging.info(
      'Dispatching MTC initialization to %d worker colocated CPU devices '
      'from %d JAX devices.',
      len(worker_cpu_devices),
      len(all_devices),
  )

  dummy_in = dispatchers.get_dummy_input_array(worker_cpu_devices)

  local_dir_str = str(local_checkpoint_directory)

  def _setup(dummy_arg: jax.Array) -> jax.Array:
    signaling_client.mark_pathways_colocated_runtime_active()
    num_nodes = jax.process_count()
    if num_nodes % num_slices != 0:
      raise ValueError(
          'num_nodes must be divisible by num_slices, got '
          f'num_nodes={num_nodes}, num_slices={num_slices}.'
      )
    nodes_per_slice = num_nodes // num_slices
    node_rank = jax.process_index()
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
        loc_dir,
        timeout_seconds=timeout_seconds,
        allow_missing_restore=True,
    )

    # Construct a fresh array from local data only.
    return jax.make_array_from_callback(
        dummy_arg.shape,
        dummy_arg.sharding,
        lambda _: np.array(True),
        dtype=jnp.bool_,
    )

  wrapped_setup_fn = colocated_python.colocated_python(_setup)
  wrapped_setup_fn = wrapped_setup_fn.specialize(out_specs_fn=lambda x: x)

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
  if num_nodes % num_slices != 0:
    raise ValueError(
        'num_nodes must be divisible by num_slices, got '
        f'num_nodes={num_nodes}, num_slices={num_slices}.'
    )
  nodes_per_slice = num_nodes // num_slices
  my_process_index = jax.process_index()
  if not 0 <= my_process_index < num_nodes:
    raise ValueError(
        f'Invalid ProcessIndex={my_process_index} for num_nodes={num_nodes}.'
    )
  node_rank_by_process_index = multihost.runtime_to_distributed_ids()
  _validate_node_rank_by_process_index(
      node_rank_by_process_index, num_nodes=num_nodes
  )
  node_rank = node_rank_by_process_index[my_process_index]
  jax_process_id = (
      jax._src.distributed.global_state.process_id  # pylint: disable=protected-access
  )
  if use_mtc_process_ids:
    logging.info(
        f'Mapping of IDs: jax-init-info.txt={process_id}, '
        f'JaxProcessId={jax_process_id}, NodeRank={node_rank}, '
        f'ProcessIndex={my_process_index}, '
        f'ProcessIndex->NodeRank={node_rank_by_process_index}'
    )
  else:
    logging.info(
        'Mapping of IDs (jax-init-info not used): '
        f'JaxProcessId={jax_process_id}, NodeRank={node_rank}, '
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
    local_checkpoint_directory: epath.Path,
    *,
    timeout_seconds: int = 300,
    allow_missing_restore: bool = False,
) -> bool:
  """Block until a `.restore` marker appears, then normalize it.

  Args:
    local_checkpoint_directory: The local checkpoint directory.
    timeout_seconds: The timeout in seconds.
    allow_missing_restore: If true, return `False` instead of raising when no
      restore marker is found.

  Returns:
    `True` if a restore marker was processed, `False` if no restore marker was
    found and `allow_missing_restore=True`.

  Raises:
    TimeoutError: if no .restore file is found within the timeout and
      `allow_missing_restore=False`.

  MTC creates a `*.restore` symlink to the directory and Orbax renames it into
  the numeric step directory the backend already understands.
  """
  local_checkpoint_directory = epath.Path(local_checkpoint_directory)
  for elapsed_seconds in range(timeout_seconds):
    marker_paths = sorted(
        local_checkpoint_directory.glob('*.restore'), key=lambda p: p.name
    )
    files = [f.name for f in marker_paths]
    if files:
      logging.info('block_and_process_restore_dir: restore files: %s', files)
    elif elapsed_seconds % 60 == 0:
      logging.info(
          'Waiting for MTC restore marker in %s... elapsed=%ds',
          local_checkpoint_directory,
          elapsed_seconds,
      )
    restore_markers = []
    no_checkpoint_markers = []
    for marker_path in marker_paths:
      step = _extract_step(marker_path.name)
      if step == '0' and marker_path.is_file():
        no_checkpoint_markers.append(marker_path)
      else:
        restore_markers.append((int(step), marker_path))

    if restore_markers:
      step, marker_path = max(restore_markers, key=lambda item: item[0])
      step_dir = local_checkpoint_directory / str(step)
      os.replace(marker_path, step_dir)
      logging.info(
          'Found a restore directory at step %s and renamed it to %s.',
          step,
          step_dir,
      )
      for stale_marker_path in [
          p for _, p in restore_markers if p != marker_path
      ] + no_checkpoint_markers:
        try:
          stale_marker_path.unlink()
          logging.info(
              'Removed stale MTC restore marker %s.', stale_marker_path
          )
        except FileNotFoundError:
          pass
      return True

    if no_checkpoint_markers:
      for marker_path in no_checkpoint_markers:
        try:
          marker_path.unlink()
        except FileNotFoundError:
          pass
        logging.info(
            'Found MTC no-checkpoint restore marker %s and removed it.',
            marker_path,
        )
      return True
    time.sleep(1)
  if allow_missing_restore:
    logging.warning(
        'No MTC restore marker appeared in %s after %ds. Continuing without '
        'local restore; this is expected when the replicator has no checkpoint '
        'to restore.',
        local_checkpoint_directory,
        timeout_seconds,
    )
    return False
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
