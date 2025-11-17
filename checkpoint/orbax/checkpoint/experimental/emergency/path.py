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

"""Path utilities for emergency checkpointing."""

import collections
from absl import logging
from etils import epath
import jax
from jax.experimental import multihost_utils
import numpy as np
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.multihost import multislice
from orbax.checkpoint._src.path import step as step_lib


def _common_values_per_replica(
    per_process_values: dict[int, set[int]],
    *,
    global_mesh: jax.sharding.Mesh,
    replica_axis_index: int,
) -> dict[int, set[int]]:
  """Obtains values shared in common across all processes in each replica.

  Args:
    per_process_values: A mapping of process index to a list of values local to
      that process.
    global_mesh: The global mesh.
    replica_axis_index: The index of the replica axis in the global mesh.

  Returns:
    A mapping of slice index to a set of values shared in common across all
    processes in that slice. A value appearing in one process but not another
    in the same slice will not appear in the output.
  """
  total_num_replicas = multislice.replica_count(
      global_mesh, replica_axis_index=replica_axis_index
  )
  num_processes_per_replica = (
      global_mesh.devices.size // total_num_replicas // jax.local_device_count()
  )
  per_replica_values = collections.defaultdict(list)
  for pid, values in per_process_values.items():
    replica_id = multislice.process_replica_id(
        pid, global_mesh, replica_axis_index=replica_axis_index
    )
    per_replica_values[replica_id].extend(values)

  for replica_id, values in per_replica_values.items():
    counter = collections.Counter(values)
    common_values = [
        k for k in counter if counter[k] == num_processes_per_replica
    ]
    # Here `len(common_values)`` will be less than or equal to `len(values)`
    # because a value can only appear in `common_values` if it occurs
    # `num_processes_per_slice` times in `values`.
    if len(common_values) > len(values):
      raise AssertionError(
          f' len(common_values) ({common_values}) exceeded length of input'
          f' values ({values}).'
      )
    per_replica_values[replica_id] = common_values

  return {k: set(v) for k, v in per_replica_values.items()}


def get_per_replica_local_steps(
    local_directory: epath.Path,
    *,
    step_name_format: step_lib.NameFormat[step_lib.Metadata],
    global_mesh: jax.sharding.Mesh,
    replica_axis_index: int,
) -> dict[int, set[int]]:
  """Gets the set of steps present in each replica from all hosts."""
  local_steps = set(m.step for m in step_name_format.find_all(local_directory))
  logging.info(
      'Found steps: %s in local host storage: %s.',
      local_steps,
      local_directory,
  )

  num_local_steps = len(local_steps)
  max_num_local_steps = multihost.global_max([num_local_steps])[0]
  # Pad the local steps so all hosts have an array of the same length.
  padded_local_steps = list(local_steps) + [-1] * (
      max_num_local_steps - num_local_steps
  )
  local_steps_per_process_array = np.array(
      [multihost.process_index()] + padded_local_steps, dtype=np.int32
  )

  # Use all_gather to collect the arrays from every host.
  global_steps_per_process = multihost_utils.process_allgather(
      local_steps_per_process_array, tiled=False
  )

  # The rest of the logic works on the gathered NumPy array.
  per_process_steps = {}
  for process_and_steps in global_steps_per_process:
    per_process_steps[process_and_steps[0]] = set(
        s for s in process_and_steps[1:] if s != -1
    )
  per_slice_steps = _common_values_per_replica(
      per_process_steps,
      global_mesh=global_mesh,
      replica_axis_index=replica_axis_index,
  )
  logging.vlog(1, 'per_replica_steps=%s', per_slice_steps)
  return per_slice_steps
