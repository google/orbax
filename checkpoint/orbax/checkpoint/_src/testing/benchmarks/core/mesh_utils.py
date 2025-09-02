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

"""Utility functions for mesh operations in benchmarks."""

import collections
from absl import logging
import jax
import numpy as np
from orbax.checkpoint._src.multihost import multislice


def pretty_log_mesh(msg: str, mesh: jax.sharding.Mesh):
  """Logs the mesh in a pretty format."""
  # Log devices in a grid format matching the mesh shape
  device_grid = np.vectorize(
      lambda d: (
          f"(id={d.id},"
          f" proc={d.process_index},"
          f" slice_id={d.slice_index if hasattr(d, 'slice_index') else None}"
      )
  )(mesh.devices)

  # Calculate devices per process in the mesh
  unique_mesh_devices = set(mesh.devices.flatten())
  devices_per_process = collections.defaultdict(int)
  mesh_process_ids = set()
  for device in unique_mesh_devices:
    devices_per_process[device.process_index] += 1
    mesh_process_ids.add(device.process_index)

  # Find devices not in the mesh for processes involved in the mesh
  all_devices = jax.devices()
  not_in_mesh_devices = collections.defaultdict(list)
  mesh_device_ids = {d.id for d in unique_mesh_devices}

  for device in all_devices:
    if (
        device.process_index in mesh_process_ids
        and device.id not in mesh_device_ids
    ):
      not_in_mesh_devices[device.process_index].append(device.id)

  logging.info(
      "%r\nprocess_id=%r: Mesh axes: %r, shape: %r, Device grid:\n%r"
      "\nMesh process IDs: %r"
      "\nDevices per process in mesh: %r"
      "\nDevices not in mesh (for mesh processes): %r",
      msg,
      jax.process_index(),
      mesh.axis_names,
      mesh.shape,
      device_grid,
      sorted(list(mesh_process_ids)),
      dict(devices_per_process),
      dict(not_in_mesh_devices),
  )


def get_local_replica_mesh(
    mesh: jax.sharding.Mesh, replica_axis_index: int
) -> jax.sharding.Mesh:
  """Returns the local replica mesh for the given global mesh."""
  return jax.sharding.Mesh(
      np.expand_dims(
          multislice.local_replica_devices(
              mesh, replica_axis_index=replica_axis_index
          ),
          axis=replica_axis_index,
      ),
      mesh.axis_names,
  )
