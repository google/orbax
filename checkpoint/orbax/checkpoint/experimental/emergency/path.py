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
import json
from typing import Any
from absl import logging
from etils import epath
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.multihost import multislice
from orbax.checkpoint._src.path import step as step_lib


def sync_global_data(
    local_data: dict[str, Any],
) -> list[dict[str, Any]]:
  """Exchanges arbitrary JSON-serializable data with all hosts.

  Args:
    local_data: A dictionary of JSON-serializable data.

  Returns:
    A list of dictionaries containing the data from all hosts.
  """
  # 1. Serialize
  json_str = json.dumps(local_data)
  local_bytes = np.frombuffer(json_str.encode('utf-8'), dtype=np.uint8)
  local_len = jnp.array([len(local_bytes)], dtype=jnp.int32)

  # 2. Exchange Lengths
  all_lens = multihost_utils.process_allgather(local_len, tiled=False)
  max_len = int(jnp.max(all_lens))

  # 3. Pad to Max Length
  padded_bytes = np.zeros(max_len, dtype=np.uint8)
  padded_bytes[: len(local_bytes)] = local_bytes
  padded_bytes_jax = jnp.array(padded_bytes)

  # 4. Exchange Data
  all_padded_data = multihost_utils.process_allgather(
      padded_bytes_jax, tiled=False
  )

  # 5. Decode
  global_data = []
  all_padded_data_np = np.array(all_padded_data)
  # process_allgather with tiled=False concatenates results into a 1D array.
  # Reshape to (num_processes, max_len) for indexing.
  all_padded_data_np = all_padded_data_np.reshape(jax.process_count(), -1)

  for i in range(len(all_lens)):
    length = int(all_lens[i])
    valid_bytes = all_padded_data_np[i, :length]
    data_str = valid_bytes.tobytes().decode('utf-8')
    global_data.append(json.loads(data_str))

  return global_data


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

  all_processes_data = sync_global_data(
      {
          'process_id': multihost.process_index(),
          'steps': list(local_steps),
      },
  )
  per_process_steps = {}
  for data in all_processes_data:
    per_process_steps[data['process_id']] = set(data['steps'])
  per_slice_steps = _common_values_per_replica(
      per_process_steps,
      global_mesh=global_mesh,
      replica_axis_index=replica_axis_index,
  )
  logging.vlog(1, 'per_replica_steps=%s', per_slice_steps)
  return per_slice_steps
