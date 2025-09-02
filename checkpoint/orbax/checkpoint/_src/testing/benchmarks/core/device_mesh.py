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

"""Functions for JAX device mesh handling in Orbax benchmark tests."""

from absl import logging
import jax
from jax.experimental import mesh_utils
import numpy as np
from orbax.checkpoint._src.testing.benchmarks.core import configs


def _num_slices() -> int:
  """Returns number of slices."""
  if hasattr(jax.devices()[0], 'slice_index'):
    return max(d.slice_index for d in jax.devices()) + 1
  return 1


def create_mesh(config: configs.MeshConfig) -> jax.sharding.Mesh:
  """Creates a jax.sharding.Mesh from a MeshConfig object.

  Args:
      config: The MeshConfig object defining the topology.

  Returns:
      A fully configured jax.sharding.Mesh.
  """
  logging.info('Creating mesh with config: %s', config)
  devices = jax.devices()
  num_devices = len(devices)
  # Convert the user-friendly dict maps into ordered lists based on mesh_axes
  ici_shape = [config.ici_parallelism.get(axis, 1) for axis in config.mesh_axes]
  dcn_shape = [config.dcn_parallelism.get(axis, 1) for axis in config.mesh_axes]

  # --- Validation ---
  if config.process_is_granule:
    process_count = jax.process_count()
    num_devices_per_granule = num_devices // process_count
    if num_devices % process_count != 0:
      raise ValueError(
          f'Total devices ({num_devices}) must be divisible by process_count'
          f' ({process_count}).'
      )
    if np.prod(dcn_shape) != jax.process_count():
      raise ValueError(
          f'The product of DCN parallelism values {np.prod(dcn_shape)} must'
          f' equal process_count {process_count}.'
      )
  else:
    num_slices = _num_slices()
    num_devices_per_granule = num_devices // num_slices
    if num_devices % num_slices != 0:
      raise ValueError(
          f'Total devices ({num_devices}) must be divisible by num_slices'
          f' ({num_slices}).'
      )
    if np.prod(dcn_shape) != num_slices:
      raise ValueError(
          f'The product of DCN parallelism values {np.prod(dcn_shape)} must'
          f' equal num_slices {num_slices}.'
      )
  if np.prod(ici_shape) != num_devices_per_granule:
    raise ValueError(
        f'The product of ICI parallelism values {np.prod(ici_shape)} must'
        f' equal num_devices_per_granule {num_devices_per_granule}.'
    )

  # --- Mesh Creation ---
  devices_array = mesh_utils.create_hybrid_device_mesh(
      ici_shape,
      dcn_shape,
      devices,
      process_is_granule=config.process_is_granule,
      allow_split_physical_axes=config.allow_split_physical_axes,
  )
  logging.info(
      'Creating mesh with axes: %s',
      {axis: dim for axis, dim in zip(config.mesh_axes, devices_array.shape)},
  )
  return jax.sharding.Mesh(devices_array, config.mesh_axes)
