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

"""Test utils for mesh configuration."""

import dataclasses
from absl import logging
import jax
import numpy as np


@dataclasses.dataclass
class MeshConfig:
  """Configuration facilitating mesh generation."""

  replica_count: int
  replica_axis_index: int
  use_device_count: int | None = None

  def __post_init__(self):
    if self.replica_axis_index not in [0, 1]:
      raise ValueError(
          f'replica_axis_index must be 0 or 1. Got: {self.replica_axis_index}'
      )
    if (
        self.use_device_count is not None
        and self.use_device_count < self.replica_count
    ):
      raise ValueError(
          'use_device_count must be greater than or equal to replica_count.'
          f' Got: {self.use_device_count} and {self.replica_count}'
      )
    if (
        self.use_device_count is not None
        and self.use_device_count == self.replica_count
    ):
      raise ValueError(
          'use_device_count must be greater than replica_count. Got:'
          f' {self.use_device_count} and {self.replica_count}'
      )

  @property
  def mesh(self) -> jax.sharding.Mesh:
    """Generates a JAX mesh based on the configuration."""
    if jax.device_count() != 8:
      raise ValueError('Device count must be 8. Got: {jax.device_count()}')
    if jax.device_count() % self.replica_count != 0:
      raise ValueError(
          'Device count must be divisible by replica count. Got:'
          f' {jax.device_count()} and {self.replica_count}'
      )
    use_device_count = self.use_device_count or jax.device_count()
    devices_per_replica = use_device_count // self.replica_count
    axes = (self.replica_count, devices_per_replica)
    axis_names = ('replica', 'data')
    device_array = np.asarray(jax.devices()[:use_device_count]).reshape(axes)
    if self.replica_axis_index == 1:
      axes = axes[::-1]
      axis_names = axis_names[::-1]
      device_array = np.swapaxes(device_array, 0, 1)
    assert (
        device_array.shape == axes
    ), f'Devices: {device_array.shape}, axes: {axes}'
    logging.info('Devices: %s', device_array)
    return jax.sharding.Mesh(device_array, axis_names)
