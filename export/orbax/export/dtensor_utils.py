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

"""Utilities for transforming distributed JAX arrays to DTensors."""

import contextlib
import dataclasses
import threading
from typing import Optional

import jax
from jax.experimental import pjit
import jaxtyping
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
  """Checks whether DTensor is initialized and matches the JAX device set."""
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
    # for the corresponding DTensor mesh.
    sharded_device_ids = pjit.pjit(
        lambda x: x,
        out_shardings=jax.sharding.PartitionSpec(*mesh.axis_names),
    )(global_device_ids)

  local_device_ids = [
      int(s.data.item()) for s in sharded_device_ids.addressable_shards
  ]
  return dtensor.Mesh(
      list(mesh.shape.keys()),
      global_device_ids=global_device_ids,
      local_device_ids=list(local_device_ids),
      local_devices=dtensor.local_devices(device_type='CPU'),
  )


def _reshard_jax_array(
    array: jax.Array, mesh: jax.sharding.Mesh, pspec: jax.sharding.PartitionSpec
) -> jax.Array:
  """Reshards a jax.Array."""
  with mesh:
    return pjit.pjit(lambda x: x, out_shardings=pspec)(array)


# TODO(b/261191533): jax.Array contains OpSharding info. Maybe we can get rid
# of jax.sharding.PartitionSpec here.
def jax_array_to_dtensor(
    arr: jax.Array,
    pspec: jax.sharding.PartitionSpec,
    dmesh: dtensor.Mesh,
    jax_mesh: Optional[jax.sharding.Mesh] = None,
    allow_multi_axis_sharding_conslidation: bool = False,
) -> DTensor:
  """Converts a jax.Array to a dtensor.

  Args:
    arr: A jax.Array.
    pspec: The partition spec of the input ``array``.
    dmesh: The DTensor mesh where the output dtensor is created.
    jax_mesh: The jax mesh for the jax array and partition spec.
    allow_multi_axis_sharding_conslidation: Whether reducing sharding a
      dimension across multiple axis names to one is allowed or not.

  Returns:
    A DTensor sharded in the same way as the input ``arr`` when the partition
    spec does not have product-sharding (i.e., sharding a dimension across
    multiple axis names). Or a DTensor with consolidated/reduced sharding
    when the partition spec has product-sharding and
    `allow_multi_axis_sharding_conslidation` is true (e.g. input array's
    sharding is P(None, ('a','b') and output dtensor sharding gets reduced
    to P(None, 'a')).

  Raises:
    ValueError: When `allow_multi_axis_sharding_conslidation` is false and if a
      dimension of ``arr`` is product-sharded, i.e., sharded across more than
      one axes of the mesh. For example, if a mesh has two axes `'x'` and `'y'`,
      `PartitionSpec((x, y))` is considered product-sharded if the mesh size of
      both axes are greater than 1. If the mesh size of the `'x'` or `'y'` is 1,
      the spec is not considered product-sharded because it is efftively
      sharded on one axis only.
  """
  arr_reshard_needed = False
  if pspec is None:
    dspec = [dtensor.UNSHARDED] * len(arr.shape)
  else:
    dspec = list()
    for i, mesh_axis_name in enumerate(pspec):
      if mesh_axis_name:
        if not isinstance(mesh_axis_name, str):
          if not isinstance(mesh_axis_name, tuple):
            raise TypeError(
                'An element in a PartitionSpec must be be a ``None``, a mesh'
                ' axis or a tuple of mesh axes. Got {mesh_axis_name}.'
            )
          if len(mesh_axis_name) > 1:
            dim_sizes = tuple(dmesh.dim_size(name) for name in mesh_axis_name)
            if dim_sizes.count(1) < len(mesh_axis_name) - 1:
              if not allow_multi_axis_sharding_conslidation:
                raise ValueError(
                    f'Dimension {i} of the input array (shape={arr.shape}) is'
                    f' sharded across more than one axis ({mesh_axis_name},'
                    f' sizes = {dim_sizes}) of the mesh, but jax.Array to'
                    ' DTensor transform does not support partitioning of an'
                    ' array dimension across multiple mesh axes, unless there'
                    ' is at most one axis with size >= 1.'
                )
              else:
                # Reduce/consolidate partition across multiple axis names
                # into one of those axis name. The selected axis name will be
                # the one with the highest dim size. E.g. P(None, ('a', 'b'))
                # will be reduced to P(None, ('a')) for {'a':4, 'b':2}
                # or P(None, ('b')) for {'a':2, 'b':4}.
                max_dim_size = max(
                    tuple(dmesh.dim_size(name) for name in mesh_axis_name)
                )
                max_dim_size_idx = dim_sizes.index(max_dim_size)
                mesh_axis_name = tuple([mesh_axis_name[max_dim_size_idx]])
                arr_reshard_needed = True
            else:
              mesh_axis_name = tuple(
                  filter(lambda x: dmesh.dim_size(x) != 1, mesh_axis_name)
              ) or (mesh_axis_name[0],)

          assert len(mesh_axis_name) == 1, mesh_axis_name
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

  if not arr_reshard_needed:
    local_data = [s.data for s in arr.addressable_shards]
  else:
    resharded_arr = _reshard_jax_array(
        arr,
        jax_mesh,
        jax.sharding.PartitionSpec(*[
            axis_name if axis_name != dtensor.UNSHARDED else None
            for axis_name in dspec
        ]),
    )
    local_data = [s.data for s in resharded_arr.addressable_shards]
    del resharded_arr
  return dtensor.pack(local_data, layout)


class _ThreadLocalStack(threading.local):

  def __init__(self):
    super().__init__()
    self.stack = list()


_MESH_STACK = _ThreadLocalStack()


@dataclasses.dataclass(frozen=True)
class Mesh:
  jax_mesh: jax.sharding.Mesh
  dtensor_mesh: dtensor.Mesh


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
    _MESH_STACK.stack.append(
        Mesh(jax_mesh=mesh, dtensor_mesh=jax_mesh_to_dtensor_mesh(mesh))
    )
    try:
      yield
    finally:
      _MESH_STACK.stack.pop(-1)


def get_current_dtensor_mesh() -> Optional[dtensor.Mesh]:
  """Returns the DTensor mesh in the current context."""
  mesh = get_current_mesh()
  return mesh.dtensor_mesh if mesh else None


def get_current_mesh() -> Optional[Mesh]:
  """Returns the Jax and DTensor mesh in the current context."""
  return _MESH_STACK.stack[-1] if _MESH_STACK.stack else None


def get_pspec_from_jax_arrays(
    nested_jax_arrays: jaxtyping.PyTree,
) -> jaxtyping.PyTree[jax.sharding.PartitionSpec]:
  """Get the partition spec of a nested jax.Array or jax.ShapeDtypeStruct.

  Args:
    nested_jax_arrays: a nested structure of jax.Array or jax.ShapeDtypeStruct.

  Returns:
    A nested structure of jax.sharding.PartitionSpec.

  Raises:
    AssertionError: if the input nested structure contains jax.Array with
    different meshes.
  """
  expected_mesh = None

  def _get_partition_spec(jax_arr):
    nonlocal expected_mesh
    if not hasattr(jax_arr, 'sharding'):
      return jax.sharding.PartitionSpec()

    if not isinstance(jax_arr.sharding, jax.sharding.NamedSharding):
      raise AssertionError(
          f'Unsupported sharding type: {type(jax_arr.sharding)}, only support'
          ' NamedSharding'
      )

    expected_mesh = (
        jax_arr.sharding.mesh if not expected_mesh else expected_mesh
    )
    if expected_mesh != jax_arr.sharding.mesh:
      raise AssertionError(
          'All those NamedShardings must have the same mesh.'
          f' {expected_mesh} != {jax_arr.sharding.mesh}'
      )
    else:
      return jax_arr.sharding.spec

  return jax.tree_util.tree_map(_get_partition_spec, nested_jax_arrays)
