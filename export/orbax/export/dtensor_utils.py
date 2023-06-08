# Copyright 2023 The Orbax Authors.
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

"""Utilities for transforming distributed JAX arrays to DTensors."""
import contextlib
import threading
from typing import Optional

import jax
from jax.experimental import pjit
import numpy as np
import tensorflow as tf
from tensorflow.experimental import dtensor


DTensor = tf.Tensor


_DTENSOR_INITIALIZED = False


def initialize_dtensor(reset_context: bool = False):
  """Initialize a DTensor system for Orbax Export.

  Args:
    reset_context: Reset the tensorflow context along DTensor initialization.
      Behaviors of existing TensorFlow objects (e.g. Tensors) are undefined. Set
      this to True as an escape hatch, if there is no clear way to refactor your
      code to call initialize_dtensor() before calling TensorFlow APIs that
      initialize the context. See also `dtensor.initialize_accelerator_system`.

  Raises:
    RuntimeError: if the number of DTensor clients is not the same as that of
    JAX processes.
  """
  n_jax_devices = jax.device_count()
  n_jax_local_devices = jax.local_device_count()
  n_jax_processes = jax.process_count()

  dtensor.initialize_accelerator_system(
      device_type='CPU',
      num_logical_cpu_devices=n_jax_local_devices,
      experimental_reset_context=reset_context,
  )
  if dtensor.num_clients() != n_jax_processes:
    raise RuntimeError(
        f'The number of DTensor clients ({dtensor.num_clients()}) is not equal'
        f' to the number of JAX processes ({n_jax_processes}. Did you forget to'
        ' set ``DTENSOR_JOBS`` or other DTensor env variables for all the JAX'
        ' processes?'
    )
  assert (
      dtensor.num_local_devices('CPU'),
      dtensor.num_global_devices('CPU'),
  ) == (n_jax_local_devices, n_jax_devices), (
      'DTensor virtual CPU count does not match JAX device count, this is'
      ' impossible.'
  )
  global _DTENSOR_INITIALIZED
  _DTENSOR_INITIALIZED = True


def dtensor_initialized() -> bool:
  """Checks whether DTensor is intialized and matches the JAX device set."""
  return _DTENSOR_INITIALIZED


def shutdown_dtensor() -> None:
  if not dtensor_initialized():
    raise RuntimeError('DTensor is not initialized.')
  dtensor.shutdown_accelerator_system()
  global _DTENSOR_INITIALIZED
  _DTENSOR_INITIALIZED = False


def jax_mesh_to_dtensor_mesh(mesh: jax.sharding.Mesh) -> dtensor.Mesh:
  """Creates a DTensor mesh from a JAX mesh.

  Args:
    mesh: a JAX global mesh for pjit.

  Returns:
    A DTensor host mesh of the same shape and axis names as those of the JAX
    mesh.
  """
  mesh_shape = mesh.devices.shape
  global_device_ids = np.arange(0, np.prod(mesh_shape)).reshape(mesh_shape)
  with mesh:
    # Shard the global device ids so that each process gets the local device ids
    # for the correponding DTensor mesh.
    sharded_device_ids = pjit.pjit(
        lambda x: x,
        out_axis_resources=jax.sharding.PartitionSpec(*mesh.axis_names),
    )(global_device_ids)

  local_device_ids = [
      int(s.data) for s in sharded_device_ids.addressable_shards
  ]
  return dtensor.Mesh(
      list(mesh.shape.keys()),
      global_device_ids=global_device_ids,
      local_device_ids=list(local_device_ids),
      local_devices=dtensor.local_devices(device_type='CPU'),
  )


# TODO(b/261191533): jax.Array contains OpSharding info. Maybe we can get rid
# of jax.sharding.PartitionSpec here.
def jax_array_to_dtensor(
    arr: jax.Array, pspec: jax.sharding.PartitionSpec, dmesh: dtensor.Mesh
) -> DTensor:
  """Converts a jax.Array to a dtensor.

  Args:
    arr: a jax.Array.
    pspec: the partition spec of the input ``array``.
    dmesh: the DTensor mesh where the output dtensor is created.

  Returns:
    A DTensor sharded in the same way as the input ``arr``.

  Raises:
    ValueError: if a dimension of ``arr`` is sharded across more than one axes
      of the mesh.
    ValueError: if
  """
  if pspec is None:
    dspec = [dtensor.UNSHARDED] * len(arr.shape)
  else:
    dspec = list()
    for i, mesh_axis_name in enumerate(pspec):
      if mesh_axis_name is not None:
        if not isinstance(mesh_axis_name, str):
          if not isinstance(mesh_axis_name, tuple):
            raise TypeError(
                'An element in a PartitionSpec must be be a ``None``, a mesh'
                ' axis or a tuple of mesh axes. Got {mesh_axis_name}.'
            )
          if len(mesh_axis_name) > 1:
            raise ValueError(
                f'Dimension {i} of the input array (shape={arr.shape}) is'
                f' sharded across more than one axis ({mesh_axis_name}) of the'
                ' mesh, but jax.Array to DTensor tranform does not support'
                ' partitioning of an array dimension across multiple mesh axes.'
            )
          else:
            mesh_axis_name = mesh_axis_name[0]
        mesh_dim_size = dmesh.dim_size(mesh_axis_name)
        if arr.shape[i] % dmesh.dim_size(mesh_axis_name) != 0:
          raise ValueError(
              f'The size of the dim {i} (={arr.shape[i]}) of the input array'
              f' (shape={arr.shape}) must be a multiple of the size of'
              f' mesh axis "{mesh_axis_name}" (={mesh_dim_size}).)'
          )
        dspec.append(mesh_axis_name)
      else:
        dspec.append(dtensor.UNSHARDED)

    dspec.extend([dtensor.UNSHARDED] * (len(arr.shape) - len(pspec)))
  layout = dtensor.Layout(dspec, dmesh)

  local_data = [s.data for s in arr.addressable_shards]
  return dtensor.pack(local_data, layout)


class _ThreadLocalStack(threading.local):

  def __init__(self):
    super().__init__()
    self.stack = list()


_MESH_STACK = _ThreadLocalStack()


@contextlib.contextmanager
def maybe_enable_dtensor_export_on(mesh: Optional[jax.sharding.Mesh]):
  """Creates a DTensor context from a JAX mesh for Orbax Export.

  If DTensor is not initialized or `mesh` is None, this function is a no-op.

  Args:
    mesh: a JAX pjit Mesh.

  Yields:
    None.
  """
  if not dtensor_initialized() or mesh is None:
    yield
  else:
    _MESH_STACK.stack.append(jax_mesh_to_dtensor_mesh(mesh))
    try:
      yield
    finally:
      _MESH_STACK.stack.pop(-1)


def get_current_dtensor_mesh() -> Optional[dtensor.Mesh]:
  """Returns the DTensor mesh in the current context."""
  return _MESH_STACK.stack[-1] if _MESH_STACK.stack else None
