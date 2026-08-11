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
import logging as python_logging
import os
import re
import time
from typing import List, Optional, Sequence

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
from orbax.checkpoint.experimental.emergency.multi_tier_checkpointing import (
    pathways_topology,
)
from orbax.checkpoint.experimental.emergency.multi_tier_checkpointing.time_block import (
    TimeBlock,
)


_REPLICATOR_FILE = 'replicator.yaml'
_REPLICATOR_ERRORS_FILE = 'replicator.errors'
_REPLICATOR_FAILED_FILE = 'replicator.failed'
_TEMP_REPLICATOR_FILE_NAME = _REPLICATOR_FILE + '.tmp'
_JAX_INIT_INFO_FILE = 'jax-init-info.txt'
_RESTORE_DIR_RE = re.compile(r'^.+-s(?P<step>\d+)-n\d+-w\d+\.restore$')
_PATHWAYS_REPLICATOR_FILE_TIMEOUT_SECONDS = 600
# Abseil maps standard levels below DEBUG to increasing VLOG levels.
_VLOG2_LEVEL = python_logging.DEBUG - 1


def _wait_for_replicator_file_to_disappear(
    local_checkpoint_directory: epath.Path,
    *,
    timeout_seconds: int = 300,
    check_for_errors: bool = True,
):
  """Waits for the MTC daemonset to consume `replicator.yaml`."""
  replicator_file = epath.Path(local_checkpoint_directory) / _REPLICATOR_FILE
  with TimeBlock(
      f'Wait for {replicator_file} to disappear '
      f'(timeout={timeout_seconds}s)',
      level=_VLOG2_LEVEL,
  ):
    for _ in range(timeout_seconds):
      if not replicator_file.exists():
        # only AFTER replicator.yaml disappears we can be sure that
        # errors are coming from current invocation of Replicator
        if check_for_errors:
          _check_for_replicator_errors(local_checkpoint_directory)
        return
      time.sleep(1)
    raise TimeoutError(
        f'Timeout reached ({timeout_seconds} seconds) while waiting for'
        f' {_REPLICATOR_FILE} to disappear.'
    )


def _read_replicator_error_file(error_file: epath.Path) -> Optional[str]:
  """Read replicator errors file."""
  try:
    error_data = epath.Path(error_file).read_text()
    logging.info(f'Contents of replicator error file:\n{error_data}')
    return error_data
  except (OSError, ValueError) as e:
    logging.info(
        'check_for_replicator_errors: Failed to read contents of failed'
        f' file: {e}'
    )
    return None


def _cleanup_replicator_error_file(error_file: epath.Path) -> None:
  """Clean up replicator errors file."""
  try:
    epath.Path(error_file).unlink()
  except (OSError, ValueError) as e:
    logging.info(
        'check_for_replicator_errors: Failed to remove replicator errors'
        f' file: {e}'
    )


def _process_replicator_error_file(error_file: epath.Path) -> Optional[str]:
  """Handles replicator errors by reading, logging, cleaning the error file."""
  error_text = None
  if epath.Path(error_file).exists():
    logging.info(f'check_for_replicator_errors: file found: {error_file}.')
    error_text = _read_replicator_error_file(error_file)
    _cleanup_replicator_error_file(error_file)

  return error_text


def _check_for_replicator_errors(
    local_checkpoint_directory: epath.Path,
) -> None:
  """Check for errors in replicator service."""
  local_dir = epath.Path(local_checkpoint_directory)

  replicator_errors_file = local_dir / _REPLICATOR_ERRORS_FILE
  errors = _process_replicator_error_file(replicator_errors_file)
  if errors:
    logging.error(f'Replicator errors: {errors}')
    # continue, regular errors may be recoverable

  replicator_failed_file = local_dir / _REPLICATOR_FAILED_FILE
  fatal = _process_replicator_error_file(replicator_failed_file)
  if fatal:
    msg = f'Replicator fatal errors: {fatal}'
    logging.log(python_logging.CRITICAL, msg)
    raise RuntimeError(msg)


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
    backup_interval_minutes: Optional[int],
    backup_interval_steps: Optional[int],
):
  """Creates a replicator file."""
  _validate_replicator_ranks(
      num_nodes=num_nodes, node_rank=node_rank, peer_ranks=peer_ranks
  )
  if (backup_interval_minutes is None) == (backup_interval_steps is None):
    raise ValueError(
        'Exactly one of backup_interval_minutes or backup_interval_steps '
        'must be specified.'
    )
  if backup_interval_minutes is not None and backup_interval_minutes <= 0:
    raise ValueError('backup_interval_minutes must be > 0.')
  if backup_interval_steps is not None and backup_interval_steps <= 0:
    raise ValueError('backup_interval_steps must be > 0.')

  temp_file = epath.Path(file_path) / _TEMP_REPLICATOR_FILE_NAME
  replicator_file = epath.Path(file_path) / _REPLICATOR_FILE
  backup_interval_yaml = (
      f'backup-interval-minutes: {backup_interval_minutes}'
      if backup_interval_minutes is not None
      else f'backup-interval-steps: {backup_interval_steps}'
  )
  replicator_yaml = f"""job-name: {run_name}
  framework: orbax
  assume-data-parallelism: {data_parallelism}
  node-rank: {node_rank}
  nodes: {num_nodes}
  peer-ranks: {peer_ranks}
  {backup_interval_yaml}"""
  final_yaml = '\n'.join(
      line.strip() for line in replicator_yaml.split('\n')
  )
  logging.vlog(
      2,
      f'Writing replicator file to {replicator_file} (via temp {temp_file})'
  )
  logging.info('Replicator YAML contents:\n%s', final_yaml)
  temp_file.write_text(final_yaml)
  os.replace(temp_file, replicator_file)
  logging.vlog(2, 'Replicator file written and renamed successfully.')


def _initialize_mtc_colocated(
    local_checkpoint_directory: epath.Path,
    backup_interval_minutes: Optional[int],
    backup_interval_steps: Optional[int],
    num_slices: int,
    run_name: str,
    data_parallelism: int,
    timeout_seconds: int,
    devices: Optional[Sequence[jax.Device]] = None,
) -> None:
  """Initializes multi-tier checkpointing with a colocated Python sidecar on all workers.

  Args:
    local_checkpoint_directory: The local checkpoint directory on the worker's
      filesystem.
    backup_interval_minutes: The backup interval in minutes. Exactly one of
      `backup_interval_minutes` or `backup_interval_steps` must be specified.
    backup_interval_steps: The backup interval in steps. Exactly one of
      `backup_interval_minutes` or `backup_interval_steps` must be specified.
    num_slices: The number of slices.
    run_name: The run name.
    data_parallelism: The data parallelism.
    timeout_seconds: The timeout in seconds.
    devices: Optional JAX devices to initialize on. If unset, all devices
      visible to the controller are used.
  """
  logging.info(
      'Initializing colocated MTC setup: '
      f'process_count={jax.process_count()}, device_count={jax.device_count()}'
  )
  colocated_transport.install_pathways_colocated_serialization_patch()
  all_devices = tuple(devices) if devices is not None else tuple(jax.devices())

  topology = pathways_topology.Topology.from_devices(all_devices)
  worker_cpu_devices = topology.worker_cpu_devices()
  worker_rank_in = topology.worker_rank_array(worker_cpu_devices)
  num_nodes = topology.num_workers
  worker_keys = tuple(tuple(worker.key) for worker in topology.workers)
  worker_tpu_device_ids = tuple(
      tuple(int(device_id) for device_id in worker.device_ids)
      for worker in topology.workers
  )
  worker_cpu_device_ids = tuple(int(device.id) for device in worker_cpu_devices)
  peer_ranks_by_worker_rank = tuple(
      tuple(int(rank) for rank in peer_ranks)
      for peer_ranks in topology.peer_ranks_by_worker_rank(num_slices)
  )
  logging.info(
      'Dispatching MTC initialization to %d worker colocated CPU devices '
      'from %d JAX devices.',
      len(worker_cpu_devices),
      len(all_devices),
  )
  logging.info(
      'Pathways MTC initialization topology: num_nodes=%d, num_slices=%d, '
      'data_parallelism=%d, worker_cpu_ids=%s, worker_tpu_ids=%s, '
      'peer_ranks=%s.',
      num_nodes,
      num_slices,
      data_parallelism,
      colocated_utils.value_sample(worker_cpu_device_ids),
      colocated_utils.nested_id_sample(worker_tpu_device_ids),
      colocated_utils.nested_id_sample(peer_ranks_by_worker_rank),
  )

  dummy_in = dispatchers.get_dummy_input_array(worker_cpu_devices)

  local_dir_str = str(local_checkpoint_directory)

  def _setup(dummy_arg: jax.Array, worker_rank_arg: jax.Array) -> jax.Array:
    """Sets up the initial MTC sidecar and processes restore tasks.

    Args:
      dummy_arg: A dummy JAX array holding dependencies to force order.
      worker_rank_arg: The worker's node rank.

    Returns:
      A JAX array signaling completion, acting as a dependency for further
      setup.
    """
    signaling_client.mark_pathways_colocated_runtime_active()
    deadline = time.time() + timeout_seconds

    def _remaining_timeout_seconds() -> int:
      remaining = int(deadline - time.time())
      if remaining <= 0:
        raise TimeoutError('Timed out while initializing colocated MTC setup.')
      return remaining

    node_rank = pathways_topology.worker_rank_from_array(worker_rank_arg)
    if not 0 <= node_rank < num_nodes:
      raise ValueError(
          f'Invalid node_rank={node_rank} for num_nodes={num_nodes}.'
      )
    worker_key = worker_keys[node_rank]
    tpu_device_ids = worker_tpu_device_ids[node_rank]
    worker_cpu_id = worker_cpu_device_ids[node_rank]
    peer_ranks = list(peer_ranks_by_worker_rank[node_rank])
    loc_dir = epath.Path(local_dir_str)
    logging.vlog(
        2,
        'Pathways MTC worker identity: '
        'logical_worker_rank=%d/%d, worker_key=%s, '
        'tpu_device_ids=%s, worker_cpu_id=%d, peer_ranks=%s, hostname=%s, '
        'kube_node_name=%s, worker_rank_sharding=%s',
        node_rank,
        num_nodes,
        worker_key,
        tpu_device_ids,
        worker_cpu_id,
        peer_ranks,
        os.environ.get('HOSTNAME'),
        os.environ.get('KUBE_NODE_NAME'),
        getattr(worker_rank_arg, 'sharding', None),
    )

    replicator_file = epath.Path(loc_dir) / _REPLICATOR_FILE
    try:
      replicator_file.unlink()
      logging.vlog(2, 'Removed stale replicator.yaml from previous run.')
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
        backup_interval_steps=backup_interval_steps,
    )
    _wait_for_replicator_file_to_disappear(
        loc_dir,
        timeout_seconds=min(
            _remaining_timeout_seconds(),
            _PATHWAYS_REPLICATOR_FILE_TIMEOUT_SECONDS,
        ),
    )
    _block_and_process_restore_dir(
        loc_dir, timeout_seconds=_remaining_timeout_seconds()
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
      out_specs_fn=lambda dummy_arg, _worker_rank_arg: dummy_arg
  )

  with TimeBlock(
      'Initialize colocated MTC on all workers',
      level=python_logging.INFO,
  ):
    result = wrapped_setup_fn(dummy_in, worker_rank_in)
    jax.block_until_ready(result)


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
  logging.vlog(
      2,
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
    backup_interval_minutes: Optional[int] = None,
    backup_interval_steps: Optional[int] = None,
    num_slices: Optional[int] = None,
    run_name: Optional[str] = None,
    data_parallelism: Optional[int] = None,
    jax_initialization_timeout_seconds: int = 900,
    use_mtc_process_ids: bool = True,
    use_colocated_python: bool = False,
    devices: Optional[Sequence[jax.Device]] = None,
):
  """Initializes multi-tier checkpointing.

  Args:
    local_checkpoint_directory: The local checkpoint directory.
    backup_interval_minutes: The backup interval for the replicator service, in
      minutes. Exactly one of `backup_interval_minutes` or
      `backup_interval_steps` must be specified.
    backup_interval_steps: The backup interval for the replicator service, in
      steps. Exactly one of `backup_interval_minutes` or `backup_interval_steps`
      must be specified.
    num_slices: The number of slices.
    run_name: The name of the run.
    data_parallelism: Number of identical pipelines in job, should be equal to
      ICI data parallelism * DCN data parallelism. If not provided, it will be
      inferred from the number of slices.
    jax_initialization_timeout_seconds: The timeout for JAX initialization.
    use_mtc_process_ids: Use the MTC rank server to calculate process ids.
    use_colocated_python: Whether to use Colocated Python for initialization.
    devices: Optional JAX devices for Colocated Python initialization. This is
      useful when the caller has already filtered controller-visible devices,
      such as after an elastic restart.
  """
  # Preserve previous default behavior where backup_interval_minutes defaulted
  # to 30.
  if backup_interval_minutes is None and backup_interval_steps is None:
    backup_interval_minutes = 30

  run_name = run_name if run_name else os.environ.get('JOBSET_NAME')
  if not run_name:
    raise ValueError(
        'Run name is not set and JOBSET_NAME is not set in the environment.'
    )

  def _resolve_parallelism_args():
    nonlocal num_slices, data_parallelism
    num_slices = (
        multislice.slice_count()
        if num_slices is None or num_slices <= 0
        else num_slices
    )
    data_parallelism = (
        num_slices
        if data_parallelism is None or data_parallelism <= 0
        else data_parallelism
    )

    logging.info(
        'Initializing multi-tier checkpointing: '
        f'{run_name=}, {num_slices=}, '
        f'{data_parallelism=}, {use_colocated_python=}.'
    )

  if use_colocated_python:
    _resolve_parallelism_args()
    _initialize_mtc_colocated(
        local_checkpoint_directory=local_checkpoint_directory,
        backup_interval_minutes=backup_interval_minutes,
        backup_interval_steps=backup_interval_steps,
        num_slices=num_slices,  # pyrefly: ignore[bad-argument-type]
        run_name=run_name,
        data_parallelism=data_parallelism,  # pyrefly: ignore[bad-argument-type]
        timeout_seconds=jax_initialization_timeout_seconds,
        devices=devices,
    )
    return

  if devices is not None:
    raise ValueError(
        '`devices` is only supported when use_colocated_python=True.'
    )

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

  # must be called after jax.distributed.initialize
  _resolve_parallelism_args()

  multihost.initialize_runtime_to_distributed_ids()
  multihost.initialize_distributed_to_device_ids()

  # We haven't initialized Replicator yet, but it's possible that
  # previous initialization of Replicator (not by us) is still pending,
  # wait for it to finish and ignore errors
  _wait_for_replicator_file_to_disappear(
      local_checkpoint_directory,
      timeout_seconds=jax_initialization_timeout_seconds,
      check_for_errors=False,
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
    logging.vlog(
        2,
        f'Mapping of IDs: jax-init-info.txt={process_id}, '
        f'JaxProcessId={jax_process_id}, NodeRank={node_rank}, '
        f'ProcessIndex={my_process_index}, '
        f'ProcessIndex->NodeRank={node_rank_by_process_index}',
    )
  else:
    logging.vlog(
        2,
        'Mapping of IDs (jax-init-info not used): '
        f'JaxProcessId={jax_process_id}, NodeRank={node_rank}, '
        f'ProcessIndex={my_process_index}, '
        f'ProcessIndex->NodeRank={node_rank_by_process_index}',
    )

  my_in_pipeline_index = my_process_index % nodes_per_slice
  peer_ranks = []
  for i in range(num_slices):  # pyrefly: ignore[bad-argument-type]
    peer_process_index = i * nodes_per_slice + my_in_pipeline_index
    if peer_process_index != my_process_index:
      peer_process_rank = node_rank_by_process_index[peer_process_index]
      peer_ranks.append(peer_process_rank)
  logging.vlog(2, 'Peers for NodeRank %s: %s', node_rank, peer_ranks)

  _create_replicator_file(
      local_checkpoint_directory,
      run_name=run_name,
      num_nodes=num_nodes,
      data_parallelism=data_parallelism,  # pyrefly: ignore[bad-argument-type]
      node_rank=node_rank,
      peer_ranks=peer_ranks,
      backup_interval_minutes=backup_interval_minutes,
      backup_interval_steps=backup_interval_steps,
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

  with TimeBlock(
      f'Wait for {_JAX_INIT_INFO_FILE}', level=_VLOG2_LEVEL
  ):
    for i in range(timeout_seconds):
      if local_jax_init_info_file.exists():
        values = local_jax_init_info_file.read_text().split('\n')
        if len(values) < 2:
          raise ValueError(
              "JAX init info file doesn't have required process id and"
              f' coordinator address data: Current values: {values}'
          )
        return values[:2]
      if i % 30 == 0:
        logging.vlog(2, 'Waiting for %s.', _JAX_INIT_INFO_FILE)
      time.sleep(1)
    raise TimeoutError(
        f'Unable to locate {_JAX_INIT_INFO_FILE} after {timeout_seconds} '
        'seconds,'
    )


def _block_and_process_restore_dir(
    local_checkpoint_directory: epath.Path,
    *,
    timeout_seconds: int = 300,
) -> bool:
  """Block until a `.restore` marker appears, then normalize it.

  Args:
    local_checkpoint_directory: The local checkpoint directory.
    timeout_seconds: The timeout in seconds.

  Returns:
    `True` if a restore marker or no-checkpoint marker was processed.

  Raises:
    TimeoutError: if no .restore marker is found within the timeout.

  MTC creates a `*.restore` symlink to the directory and Orbax renames it into
  the numeric step directory the backend already understands.
  """
  local_checkpoint_directory = epath.Path(local_checkpoint_directory)

  def _remove_restore_marker(marker_path: epath.Path) -> None:
    try:
      marker_path.unlink()
    except FileNotFoundError:
      pass

  with TimeBlock(
      f'Wait for MTC restore marker in {local_checkpoint_directory}',
      level=_VLOG2_LEVEL,
  ):
    for elapsed_seconds in range(timeout_seconds):
      marker_paths = sorted(
          local_checkpoint_directory.glob('*.restore'), key=lambda p: p.name
      )
      files = [f.name for f in marker_paths]
      if files:
        logging.info(
            'block_and_process_restore_dir: restore files: %s', files
        )
      elif elapsed_seconds % 60 == 0:
        logging.vlog(
            2,
            'Waiting for MTC restore marker in %s.',
            local_checkpoint_directory,
        )

      _check_for_replicator_errors(local_checkpoint_directory)

      restore_markers = []
      no_checkpoint_markers = []
      for marker_path in marker_paths:
        step = _extract_step(marker_path.name)
        # Replicator writes a zero-sized file for "no checkpoint" and a symlink
        # for an actual restore marker.
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
          _remove_restore_marker(stale_marker_path)
          logging.vlog(
              2, 'Removed stale MTC restore marker %s.', stale_marker_path
          )
        return True

      if no_checkpoint_markers:
        for marker_path in no_checkpoint_markers:
          _remove_restore_marker(marker_path)
          logging.info(
              'Found MTC no-checkpoint restore marker %s and removed it.',
              marker_path,
          )
        return True
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
