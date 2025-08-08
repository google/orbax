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

"""Emergency checkpointing utils for multihost / multislice."""

import os
import time
from typing import List, Optional

from absl import flags
from absl import logging
from etils import epath
import jax
import numpy as np
from orbax.checkpoint._src.multihost import multihost


REPLICATOR_FILE = 'replicator.yaml'
TEMP_REPLICATOR_FILE_NAME = REPLICATOR_FILE + '.tmp'
JAX_INIT_INFO_FILE = 'jax-init-info.txt'

EXPERIMENTAL_USE_DISTRIBUTED_ID_FOR_MESH_CONSISTENCY = flags.DEFINE_bool(
    'experimental_use_distributed_id_for_mesh_consistency',
    True,
    'Decides how to Map how device ids changed across restarts. '
    'If True, use remapping distributed id for consistent restore mesh '
    'logic. If False, remaps mesh according to jax.devices() order.',
)


def _int_list_flip_index_and_value(int_list: List[int]):
  """Reverses indices and values of a list of integers.

  Ex: [ 3, 4, 0, 1, 2 ] -> [ 2, 3, 4, 0, 1 ]
  Old index 0, value: 3 -> New index: 3, value: 0

  Args:
    int_list: List of integers.

  Returns:
    List of integers with reversed indices and values.
  """
  result = [None for _ in range(len(int_list))]
  for index, value in enumerate(int_list):
    result[value] = index
  assert None not in result
  return result


def _get_runtime_id_across_restarts(
    previous_runtime_to_dist_id: Optional[List[int]],
) -> List[Optional[int]]:
  """Get runtime id changes across restarts.

  Args:
    previous_runtime_to_dist_id: mapping from runtime process index to
      distributed process index of the previous incarnation.

  Returns:
    Integer list which maps previous runtime id (index) to current runtime id
    (array element).
  Raises:
    ValueError:
  """
  current_dist_to_runtime_id = _int_list_flip_index_and_value(
      runtime_to_distributed_ids()
  )
  previous_dist_to_runtime_id = _int_list_flip_index_and_value(
      previous_runtime_to_dist_id
  )

  result = [None for _ in range(jax.process_count())]
  # Previous runtime id (index) to current runtime id (value).
  for i in range(jax.process_count()):
    result[previous_dist_to_runtime_id[i]] = current_dist_to_runtime_id[i]
  assert None not in result
  return result


def process_index_from_device_id(device_id: int) -> int:
  """Get process index from device id."""
  if jax.devices()[0].platform == 'gpu':
    return device_id // jax.local_device_count()
  elif jax.devices()[0].platform == 'tpu':
    if hasattr(jax.devices()[0], 'slice_index'):
      # Note that it is possible for single slice TPU devices to have  a slice
      # index.
      num_slices = max([d.slice_index for d in jax.devices()]) + 1
      # Multi-slice TPU workload.
      if num_slices > 1:
        num_processes_per_slice = jax.process_count() // num_slices
        # This is based on how Megascale device ids are assigned.
        # See platforms/xla/megascale/runtime/common/multi_slice_topology.h.
        slice_id = device_id // 100000 - 1
        local_process_id = device_id % 100000 // jax.local_device_count()
        return slice_id * num_processes_per_slice + local_process_id
    # Single slice TPU workload.
    return device_id // jax.local_device_count()
  # CPU workload.
  else:
    # This is based on how CPU device ids are assigned.
    # See tensorflow/compiler/xla/pjrt/cpu/cpu_topology.h.
    return device_id // (1 << 17) // jax.local_device_count()


def consistent_restore_mesh(
    devices: List[jax.Device],
    user_mesh: jax.sharding.Mesh,
    previous_flattened_mesh_device_ids: List[int],
    previous_distributed_to_device_ids: List[List[int]],
    current_distributed_to_device_ids: List[List[int]],
):
  """Create a mesh that is consistent with the previous incarnation.

    1. We can think of mesh as being backed by a list of devices.
    2. Default mesh follows the default device id order [0, ..., n-1]. Or
       the user may permute it according to their needs.
    3. After restart, the user will construct the same software mesh as (2).
    4. But a given hardware device may change its id because of scheduler
       or runtime quirks.
    5. Goal: construct the mesh with the same hardware device order as
       before restart, that may not follow the current software ids.
    6. Thus, we shuffle the device order within the mesh by checking how
       each device's software ids changed across restarts.

  This is to restore the same global array values from local hardware devices
  even if software process and device ids have changed across restarts.

  Args:
    devices: List of Jax devices (usually `jax.devices()`).
    user_mesh: The user mesh.
    previous_flattened_mesh_device_ids: The flattened device ids of the mesh.
    previous_distributed_to_device_ids: The distributed id to range of device
      ids mapping of the previous incarnation.
    current_distributed_to_device_ids: The distributed id to range of device ids
      mapping of the current incarnation.

  Returns:
    The new mesh devices that should be used to create the mesh.
  """
  if EXPERIMENTAL_USE_DISTRIBUTED_ID_FOR_MESH_CONSISTENCY.value:
    # Map how device ids changed across restarts.
    device_id_across_restarts = {}
    for i in range(len(previous_distributed_to_device_ids)):
      for j in range(len(previous_distributed_to_device_ids[i])):
        previous_id = previous_distributed_to_device_ids[i][j]
        current_id = current_distributed_to_device_ids[i][j]
        device_id_across_restarts[previous_id] = current_id
    logging.debug(
        'device_id_across_restarts (key: previous_id, value: current_id): %s',
        device_id_across_restarts,
    )
    # Key devices by id.
    jax_devices_by_id = {d.id: d for d in devices}

    new_flattened_mesh_devices = [
        # Convert old ids to current ids that correspond to the same physical
        # hardware.
        jax_devices_by_id[device_id_across_restarts[id]]
        for id in previous_flattened_mesh_device_ids
    ]
    new_mesh_devices = np.array(new_flattened_mesh_devices).reshape(
        user_mesh.devices.shape
    )
  else:
    # This logic assumes jax.devices() returns the same output across restarts.
    # This assumption may break in future Jax releases.
    device_ids = [x.id for x in devices]
    new_flattened_mesh_devices = [
        devices[device_ids.index(i)] for i in previous_flattened_mesh_device_ids
    ]
    new_mesh_devices = np.array(new_flattened_mesh_devices).reshape(
        user_mesh.devices.shape
    )
  return jax.sharding.Mesh(new_mesh_devices, user_mesh.axis_names)


def runtime_to_distributed_ids() -> List[int]:
  """Returns the runtime to distributed process id mapping."""
  # TODO(b/325293150): Deprecate this after jaxlib contains the fix.
  result = multihost.runtime_to_distributed_ids()
  runtime_and_distributed_ids_are_the_same = all([
      result[i] == i for i in range(len(result))
  ])

  # JAX may choose to overwrite the device process index with the distributed
  # process index. In that case, we have to use the device id to infer the real
  # device process index. This is a hack, the intent is to remove it once this
  # workaround is no longer needed.
  if runtime_and_distributed_ids_are_the_same:
    result = [-1 for _ in range(jax.process_count())]
    devices = jax.devices()
    for i in range(0, jax.device_count(), jax.local_device_count()):
      result[process_index_from_device_id(devices[i].id)] = devices[
          i
      ].process_index
    assert -1 not in result
  return result


def wait_for_replicator_file_to_disappear(
    local_checkpoint_directory: epath.Path, timeout_seconds: int = 300
) -> bool:
  """Waits for a file to disappear."""
  file_exists = True
  replicator_file = epath.Path(local_checkpoint_directory / REPLICATOR_FILE)
  for _ in range(timeout_seconds):
    if not replicator_file.exists():
      file_exists = False
      break
    time.sleep(1)
  if file_exists:
    logging.warning(
        'There is existing replicator.yaml which did not disappear in time.'
    )
  else:
    logging.info(
        'replicator.yaml no longer exists, creating new replicator.yaml.'
    )
  return file_exists


def create_replicator_file(
    file_path: epath.Path,
    run_name: str,
    num_nodes: int,
    num_slices: int,
    node_rank: int,
    peer_ranks: List[int],
    backup_interval_minutes: int,
):
  """Creates a replicator file."""
  temp_file = epath.Path(file_path / TEMP_REPLICATOR_FILE_NAME)
  replicator_file = epath.Path(file_path / REPLICATOR_FILE)
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


def get_num_slices(compile_topology_num_slices: int = 0):
  """Get the number of slices."""
  if jax.devices()[0].platform == 'cpu':
    logging.info('Setting num_slices=1 for CPU hardware type')
    return 1
  if compile_topology_num_slices > 0:
    return compile_topology_num_slices
  else:
    devices = jax.devices()
    try:
      return 1 + max(d.slice_index for d in devices)
    except (ValueError, AttributeError):
      return 1


def initialize_multi_tier_checkpointing(
    local_checkpoint_directory: epath.Path,
    use_replicator_service: bool = False,
    backup_interval_minutes: int = 10,
    num_slices: Optional[int] = None,
    num_nodes: Optional[int] = None,
    nodes_per_slice: Optional[int] = None,
    node_rank: Optional[int] = None,
    run_name: Optional[str] = None,
    jax_initialization_timeout_seconds: int = 900,
    compile_topology_num_slices: int = 0,
):
  """Initializes multi-tier checkpointing.

  Args:
    local_checkpoint_directory: The local checkpoint directory.
    use_replicator_service: Whether to use the replicator service.
    backup_interval_minutes: The backup interval for the replicator service, in
      minutes.
    num_slices: The number of slices.
    num_nodes: The number of nodes.
    nodes_per_slice: The number of nodes per slice.
    node_rank: The rank of the node.
    run_name: The name of the run.
    jax_initialization_timeout_seconds: The timeout for JAX initialization.
    compile_topology_num_slices: The number of slices from the compile topology.
  """
  process_id, coordinator_address = _retrieve_jax_init_info(
      local_checkpoint_directory
  )
  if process_id and coordinator_address:
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
    if use_replicator_service:
      wait_for_replicator_file_to_disappear(local_checkpoint_directory)
      num_slices = num_slices or get_num_slices(
          compile_topology_num_slices
      )
      num_nodes = num_nodes or jax.process_count()
      nodes_per_slice = num_nodes // num_slices or nodes_per_slice
      node_rank = node_rank or jax._src.distributed.global_state.process_id  # pylint: disable=protected-access
      my_process_index = jax.process_index()
      process_index_to_node_rank = runtime_to_distributed_ids()
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
      create_replicator_file(
          local_checkpoint_directory,
          run_name,
          num_nodes,
          num_slices,
          node_rank,
          peer_ranks,
          backup_interval_minutes,
      )
      wait_for_replicator_file_to_disappear(local_checkpoint_directory)
      _block_and_proces_restore_dir(local_checkpoint_directory)
  else:
    logging.warning(
        'Initializing JAX distributed runtime without args when emergency'
        ' checkpointing is enabled. This should not happen and your workload'
        ' may have unexpected behavior.'
    )
    jax.distributed.initialize(
        initialization_timeout=jax_initialization_timeout_seconds
    )
    multihost.initialize_runtime_to_distributed_ids()
    multihost.initialize_distributed_to_device_ids()


def _retrieve_jax_init_info(
    local_checkpoint_directory: epath.Path, timeout_seconds: int = 900
) -> List[str]:
  """Retrieve JAX init info from a local file.

  Args:
    local_checkpoint_directory: The local checkpoint directory.
    timeout_seconds: The timeout in seconds.

  Returns:
    A list of strings containing the JAX init info (process id and coordinator
    address).

  Allow time for the JAX init info file to be populated by GKE. This is needed
  because the file is only populated when the worker with process id of 0 is
  determined. After a disruption, although some workers might be up and
  running, the init info file won't be populated until the node with process
  id of 0 is known and this could take time. Using 900 seconds for now and it
  needs to be increased if the "repair" time is longer.
  """
  local_jax_init_info_file = (
      epath.Path(local_checkpoint_directory) / JAX_INIT_INFO_FILE
  )

  for i in range(timeout_seconds):
    if local_jax_init_info_file.exists():
      return local_jax_init_info_file.read_text().split('\n')[:2]
    logging.info(
        'Unable to locate %s after %d seconds,'
        ' sleeping for 1 second before retrying...',
        JAX_INIT_INFO_FILE,
        i,
    )
    time.sleep(1)
  logging.info(
      'Unable to locate %s after 900 seconds,'
      'returning empty process id and coordinator address.',
      JAX_INIT_INFO_FILE,
  )
  return ['', '']


def _block_and_proces_restore_dir(
    local_checkpoint_directory: epath.Path, timeout=300
):
  """Block until a file ending with `.restore` appears, then extract the step number and rename the directory using the step number.

  Args:
    local_checkpoint_directory: The local checkpoint directory.
    timeout: The timeout in seconds.
  """
  restore_word = '.restore'
  for _ in range(timeout):
    files = os.listdir(local_checkpoint_directory)
    for f in files:
      if f.endswith(restore_word):
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
  logging.info(
      '%s seconds have passed but no .restore file was found.', timeout
  )


def _extract_step(f):
  # The base file name is formatted as:
  # {job_name}-s{step}-n{node_rank}-w{gpu_rank}
  return f.rsplit('-', 3)[1][1:]

