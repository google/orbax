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

"""Utilities for working with local paths in Pathways.

TODO(b/448471028): Rework using Dispatcher.
"""

import jax
from jax import numpy as jnp
import numpy as np
from orbax.checkpoint._src.multihost import multislice
from orbax.checkpoint._src.path import step as step_lib
from orbax.checkpoint.experimental.v1._src.path import types as path_types
from .learning.deepmind.jax.ocean.remote_python import rp


def _get_max_num_steps(
    local_directory: path_types.Path,
    *,
    step_name_format: step_lib.NameFormat[step_lib.Metadata],
    global_mesh: jax.sharding.Mesh,
) -> int:
  """Returns the maximum number of steps present on any worker."""
  devices = global_mesh.devices.flatten()
  device_count = len(devices)
  fully_sharded_sharding = jax.sharding.NamedSharding(
      jax.sharding.Mesh(global_mesh.devices.flatten(), 'x'),
      jax.sharding.PartitionSpec(
          'x',
      ),
  )

  @rp.stateless_fn
  def _max_steps_per_device(_) -> jax.Array:
    local_steps = set(
        m.step for m in step_name_format.find_all(local_directory)
    )
    local_data = [len(local_steps)] * jax.local_device_count()
    local_data = jnp.asarray(local_data, dtype=np.int64)
    assert local_data.shape == (jax.local_device_count(),)
    return jax.make_array_from_process_local_data(
        fully_sharded_sharding, local_data, (device_count,)
    )

  dummy_input = rp.make_dummy_array(global_mesh.devices.flatten().tolist())
  _max_steps_per_device.register_shape_fn(
      lambda _: jax.ShapeDtypeStruct(
          (device_count,),
          dtype=np.int64,
          sharding=fully_sharded_sharding,
      )
  )
  result = _max_steps_per_device(rp.to_remote_python(dummy_input))
  step_count_per_device = rp.from_remote_python(jax.block_until_ready(result))
  return int(max(step_count_per_device))


def _get_steps_per_device(
    local_directory: path_types.Path,
    max_steps_per_process: int,
    *,
    global_mesh: jax.sharding.Mesh,
    step_name_format: step_lib.NameFormat[step_lib.Metadata],
) -> dict[jax.Device, set[int]]:
  """Returns array (device_count, max_num_steps)."""
  devices = global_mesh.devices.flatten()
  device_count = len(devices)
  fully_replicated_sharding = jax.sharding.NamedSharding(
      jax.sharding.Mesh(devices, 'x'),
      jax.sharding.PartitionSpec(),
  )
  fully_sharded_sharding = jax.sharding.NamedSharding(
      jax.sharding.Mesh(devices, 'x'),
      jax.sharding.PartitionSpec('x', None),
  )
  global_shape = (device_count, max_steps_per_process)

  @rp.stateless_fn
  def _padded_steps_per_device(_) -> jax.Array:
    local_steps = list(
        m.step for m in step_name_format.find_all(local_directory)
    )
    num_local_steps = len(local_steps)
    padded_local_steps = list(local_steps) + [-1] * (
        max_steps_per_process - num_local_steps
    )
    local_data = jnp.asarray(padded_local_steps, dtype=np.int64)
    local_data = jnp.tile(local_data, (jax.local_device_count(), 1))
    assert local_data.shape == (jax.local_device_count(), max_steps_per_process)
    return jax.make_array_from_process_local_data(
        fully_sharded_sharding, local_data, global_shape
    )

  dummy_input = jax.device_put(
      jnp.asarray(0, dtype=np.int64), device=fully_replicated_sharding
  )
  _padded_steps_per_device.register_shape_fn(
      lambda _: jax.ShapeDtypeStruct(
          global_shape,
          dtype=np.int64,
          sharding=fully_sharded_sharding,
      )
  )
  result = _padded_steps_per_device(rp.to_remote_python(dummy_input))
  steps_per_device = rp.from_remote_python(jax.block_until_ready(result))
  device_to_steps = {}
  for shard in steps_per_device.addressable_shards:
    data = np.asarray(shard.data)[0].tolist()
    device_to_steps[shard.device] = set(v for v in data if v != -1)
  return device_to_steps


def per_replica_local_steps(
    local_directory: path_types.Path,
    *,
    step_name_format: step_lib.NameFormat[step_lib.Metadata],
    global_mesh: jax.sharding.Mesh,
    replica_axis_index: int,
) -> dict[int, set[int]]:
  """Returns a mapping of replica index to local steps present on that replica."""
  max_steps_per_process = _get_max_num_steps(
      local_directory,
      global_mesh=global_mesh,
      step_name_format=step_name_format,
  )
  # (device_count, max_steps_per_process)
  steps_per_device = _get_steps_per_device(
      local_directory,
      max_steps_per_process,
      global_mesh=global_mesh,
      step_name_format=step_name_format,
  )

  num_replicas = multislice.replica_count(
      global_mesh, replica_axis_index=replica_axis_index
  )
  result: dict[int, set[int]] = {}
  for replica_id in range(num_replicas):
    replica_devices = multislice.replica_devices(
        global_mesh,
        replica_id=replica_id,
        replica_axis_index=replica_axis_index,
    )
    replica_steps = set()
    for i, d in enumerate(replica_devices):
      if i == 0:
        replica_steps = steps_per_device[d]
      else:
        replica_steps &= steps_per_device[d]
    result[replica_id] = replica_steps

  return result
