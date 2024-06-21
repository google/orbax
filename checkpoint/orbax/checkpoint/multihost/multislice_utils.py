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

"""Multislice utils."""

from typing import Optional
import jax
import numpy as np
from orbax.checkpoint.multihost import utils


def process_slice_id(
    process_index: int,
    global_mesh: jax.sharding.Mesh,
    replica_axis_index: Optional[int] = 0,
) -> int:
  """Returns the slice id that the process_index belongs to."""
  num_slices = global_mesh.devices.shape[replica_axis_index]
  for slice_id in range(num_slices):
    device_slice = np.take(
        global_mesh.devices, slice_id, axis=replica_axis_index
    )
    if in_slice(process_index, device_slice):
      return slice_id

  return -1


def _pid_in_slice(device_slice: np.ndarray) -> np.ndarray:
  pid = np.vectorize(
      lambda d: utils.runtime_to_distributed_process_id(d.process_index)
  )
  return pid(device_slice)


def in_slice(process_index: int, device_slice: np.ndarray) -> bool:
  return process_index in _pid_in_slice(device_slice)


def in_primary_slice(
    process_index: int,
    global_mesh: jax.sharding.Mesh,
    replica_axis_index: Optional[int] = 0,
) -> bool:
  """Returns true if host is in primary slice (the first slice)."""
  primary_slice = np.take(global_mesh.devices, 0, axis=replica_axis_index)

  return in_slice(process_index, primary_slice)


def local_slice_devices(devices_array: np.ndarray) -> np.ndarray:
  for device_slice in devices_array:
    if in_slice(utils.process_index(), device_slice):
      return device_slice
  raise ValueError(
      f'process_index {utils.process_index()} does not exist in provided'
      ' `global_mesh`'
  )