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

"""Emergency checkpointing utils for multihost / multislice."""

from typing import List, Optional

import jax
import numpy as np
from orbax.checkpoint.multihost import utils as multihost_utils


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
    # See third_party/tensorflow/compiler/xla/pjrt/cpu/cpu_topology.h.
    return device_id // (1 << 17) // jax.local_device_count()


def consistent_restore_mesh(
    user_mesh: jax.sharding.Mesh,
    previous_flattened_mesh_device_ids: List[int],
    previous_runtime_to_dist_id: List[int],
):
  """Create a mesh that is consistent with the previous incarnation.

  This is to restore the same global array values even if process ids have
  changed across restarts.

  TODO(b/325293150): This logic can be removed once the bug is resolved by JAX.

  Args:
    user_mesh: The user mesh.
    previous_flattened_mesh_device_ids: The flattened device ids of the mesh.
    previous_runtime_to_dist_id: The runtime to distributed process id mapping
      of the previous incarnation.

  Returns:
    The new mesh devices that should be used to create the mesh.
  """
  runtime_id_across_restarts = _get_runtime_id_across_restarts(
      previous_runtime_to_dist_id
  )
  new_flattened_mesh_device_ids = [
      runtime_id_across_restarts[process_index_from_device_id(raw_id)]
      * jax.local_device_count()
      + raw_id % jax.local_device_count()
      for raw_id in previous_flattened_mesh_device_ids
  ]
  new_flattened_mesh_devices = [
      jax.devices()[id] for id in new_flattened_mesh_device_ids
  ]
  new_mesh_devices = np.array(new_flattened_mesh_devices).reshape(
      user_mesh.devices.shape
  )
  return jax.sharding.Mesh(new_mesh_devices, user_mesh.axis_names)


def runtime_to_distributed_ids() -> List[int]:
  """Returns the runtime to distributed process id mapping."""
  # TODO(b/325293150): Deprecate this after jaxlib contains the fix.
  result = multihost_utils.runtime_to_distributed_ids()
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

