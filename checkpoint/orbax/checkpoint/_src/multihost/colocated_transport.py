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

"""Helpers for transporting values through colocated Python."""

from collections.abc import Mapping
import dataclasses
import functools
import re
import types
from typing import Any, Sequence, cast

from absl import logging
import jax
import jax.experimental.colocated_python as cp
from jax.experimental.colocated_python import serialization as cp_serialization
import numpy as np
from orbax.checkpoint._src.arrays import abstract_arrays
from orbax.checkpoint._src.metadata import sharding as sharding_metadata
from orbax.checkpoint._src.serialization import jax_array_restore_args


PyTree = Any
_PATHWAYS_CPU_DEVICE_LOOKUP_PATCH_INSTALLED = False
_PJRT_IFRT_DEVICE_ID_RE = re.compile(r'PjRtIFRTDeviceId=(\d+)')


def _device_platform(device: jax.Device) -> str:
  platform = getattr(device, 'platform', None)
  if platform is not None:
    return str(platform)
  return str(getattr(device, 'device_kind', 'unknown')).lower()


def _device_identity_key(device: jax.Device) -> tuple[int, int, str]:
  """Returns a stable key for deduping devices across backend scans."""
  return (id(device.client), device.id, _device_platform(device))


def _extract_pjrt_ifrt_device_id(device: jax.Device) -> int | None:
  """Returns the backend-global IFRT CPU id when the backend exposes one.

  Pathways sidecars log CPU devices like:

    `CpuDevice(id=0)[PjRtIFRTDeviceId=120]`

  where `device.id` is local to the remote-Python CPU backend, but the IFRT id
  is the backend-global id that matches controller-side specialization.

  Example:
  - controller serializes a CPU mesh using device id `120`
  - worker Python only exposes local ids `0..3`
  - the same worker device prints as `CpuDevice(id=0)[PjRtIFRTDeviceId=120]`

  JAX does not currently expose that backend-global id as a public Python
  attribute, so Orbax has to parse the repr while waiting for an upstream
  serialization fix.

  Args:
    device: The JAX device to extract the backend-global IFRT ID from.

  Returns:
    The integer PjRt IFRT device ID if present in the repr, otherwise None.
  """
  match = _PJRT_IFRT_DEVICE_ID_RE.search(repr(device))
  if not match:
    return None
  return int(match.group(1))


def _all_backend_devices() -> tuple[jax.Device, ...]:
  """Returns all devices visible to colocated-python serialization backends."""
  devices_by_key: dict[tuple[int, int, str], jax.Device] = {}

  local_devices = cp_serialization.xb.local_devices()
  if local_devices:
    # `local_devices()[0]` is only used to reach the default backend client;
    # `_get_all_devices()` then enumerates every device owned by that client.
    default_backend_client = local_devices[0].client
    for device in default_backend_client._get_all_devices():  # pylint: disable=protected-access
      devices_by_key.setdefault(_device_identity_key(device), device)

  for backend in cp_serialization.xb.backends().values():
    for device in backend._get_all_devices():  # pylint: disable=protected-access
      devices_by_key.setdefault(_device_identity_key(device), device)
  return tuple(devices_by_key.values())


@functools.lru_cache(maxsize=1)
def _get_cpu_device_map() -> Mapping[int, jax.Device]:
  """Builds a worker-side CPU lookup for JAX colocated deserialization.

  JAX colocated Python serializes `Mesh`, `DeviceList`, and
  `SingleDeviceSharding` by integer CPU device id. In Pathways, controller-side
  CPU ids can be backend-global IFRT ids while worker-side Python can expose
  local CPU ids. Worker CPU reprs include the backend-global
  `PjRtIFRTDeviceId`, so register both namespaces when they are unambiguous.
  """
  cpu_device_map: dict[int, jax.Device] = {}
  backend_devices = _all_backend_devices()
  for device in backend_devices:
    if _device_platform(device) != 'cpu':
      continue
    ifrt_device_id = _extract_pjrt_ifrt_device_id(device)
    if ifrt_device_id is None:
      continue
    existing = cpu_device_map.get(ifrt_device_id)
    if existing is not None and existing != device:
      raise ValueError(
          'Multiple CPU devices with PjRt-IFRT id '
          f'{ifrt_device_id} found: {existing} and {device}'
      )
    cpu_device_map[ifrt_device_id] = device

  for device in backend_devices:
    if _device_platform(device) != 'cpu':
      continue
    existing = cpu_device_map.get(device.id)
    if existing is not None and existing != device:
      raise ValueError(
          'CPU device id '
          f'{device.id} is ambiguous: it is both a PjRt-IFRT id for '
          f'{existing} and a local device id for {device}'
      )
    if existing is None:
      cpu_device_map[device.id] = device

  return types.MappingProxyType(cpu_device_map)


def install_pathways_colocated_cpu_device_lookup_patch() -> None:
  """Installs a narrow Pathways CPU lookup patch for colocated Python.

  This deliberately leaves JAX's colocated Python reducers unchanged. It only
  swaps the CPU id lookup used while deserializing already-CPU shardings/device
  lists so Pathways workers can resolve controller-global IFRT CPU ids.
  """
  # pylint: disable=global-statement
  global _PATHWAYS_CPU_DEVICE_LOOKUP_PATCH_INSTALLED
  if _PATHWAYS_CPU_DEVICE_LOOKUP_PATCH_INSTALLED:
    return
  cp_serialization._get_cpu_device_map = _get_cpu_device_map  # pylint: disable=protected-access
  _PATHWAYS_CPU_DEVICE_LOOKUP_PATCH_INSTALLED = True


def unique_colocated_cpu_devices(
    devices: Sequence[jax.Device],
) -> tuple[jax.Device, ...]:
  """Returns one colocated CPU device per worker."""
  all_cpu = tuple(cp.colocated_cpu_devices(tuple(devices)))
  unique_cpu = []
  seen_ids = set()
  for device in all_cpu:
    if device.id in seen_ids:
      continue
    seen_ids.add(device.id)
    unique_cpu.append(device)
  return tuple(unique_cpu)


def colocated_cpu_sharding(
    sharding: jax.sharding.Sharding,
) -> jax.sharding.Sharding:
  """Returns a CPU sharding colocated with the given sharding."""
  if isinstance(sharding, jax.sharding.SingleDeviceSharding):
    cpu_devices = cp.colocated_cpu_devices(list(sharding.device_set))
    return jax.sharding.SingleDeviceSharding(
        cpu_devices[0], memory_kind=sharding.memory_kind
    )
  if isinstance(sharding, jax.sharding.NamedSharding):
    cpu_mesh = cp.colocated_cpu_devices(sharding.mesh)
    return jax.sharding.NamedSharding(
        cpu_mesh, sharding.spec, memory_kind=sharding.memory_kind
    )
  raise TypeError(
      f'Sharding type {type(sharding)} not supported in to_colocated_python.'
  )


def to_colocated_python(input_tree: PyTree) -> PyTree:
  """Copies a pytree of arrays to colocated CPU devices."""

  def _get_sharding(x: Any) -> jax.sharding.Sharding | None:
    if isinstance(x, jax.Array):
      cpu_sharding = colocated_cpu_sharding(x.sharding)
      logging.vlog(
          1,
          'Staging array from %s to colocated CPU sharding %s',
          x.sharding,
          cpu_sharding,
      )
      return cpu_sharding
    return None

  cpu_sharding_tree = jax.tree.map(_get_sharding, input_tree)
  return jax.device_put(input_tree, cpu_sharding_tree, may_alias=True)


def make_scalar_array_like(
    value: Any, like: jax.Array, *, dtype: Any
) -> jax.Array:
  """Builds a scalar array on the same global sharding as `like`.

  Using `jax.device_put(..., sharding=...)` on a non-fully-addressable global
  sharding can trigger multihost consistency checks. Constructing the result via
  callback avoids that path while preserving the target sharding.

  Args:
    value: The scalar value to fill the array with.
    like: An array whose sharding and shape will be copied.
    dtype: The desired dtype of the result.

  Returns:
    A new array with the same shape and sharding as `like`, filled with `value`.
  """
  return jax.make_array_from_callback(
      like.shape,
      like.sharding,
      lambda _: np.asarray(value, dtype=dtype),
      dtype=dtype,
  )


def convert_array_restore_args(
    restore_args: jax_array_restore_args.ArrayRestoreArgs,
) -> jax_array_restore_args.ArrayRestoreArgs:
  """Converts ArrayRestoreArgs to use colocated CPU devices."""
  if restore_args.mesh is not None:
    cpu_mesh = cp.colocated_cpu_devices(restore_args.mesh)
    logging.vlog(
        1,
        'Converting restore mesh with axis names %s to colocated CPU mesh.',
        restore_args.mesh.axis_names,
    )
    restore_args = dataclasses.replace(restore_args, mesh=cpu_mesh)
  if restore_args.sharding is None:
    return restore_args
  if isinstance(restore_args.sharding, jax.sharding.Sharding):
    cpu_sharding = colocated_cpu_sharding(restore_args.sharding)
    logging.vlog(
        1,
        'Converting restore sharding from %s to colocated CPU sharding %s',
        restore_args.sharding,
        cpu_sharding,
    )
    return dataclasses.replace(restore_args, sharding=cpu_sharding)
  if isinstance(restore_args.sharding, sharding_metadata.ShardingMetadata):
    sharding = restore_args.sharding.to_jax_sharding()
    cpu_sharding = colocated_cpu_sharding(sharding)
    logging.vlog(
        1,
        'Converting restore sharding metadata %s to colocated CPU sharding %s',
        type(restore_args.sharding).__name__,
        cpu_sharding,
    )
    return dataclasses.replace(
        restore_args,
        sharding=restore_args.sharding.from_jax_sharding(cpu_sharding),
    )
  raise TypeError(
      f'Sharding type {type(restore_args.sharding)} not supported in'
      ' to_colocated_python.'
  )


def convert_single_replica_restore_args(
    restore_args: jax_array_restore_args.SingleReplicaArrayRestoreArgs,
) -> jax_array_restore_args.SingleReplicaArrayRestoreArgs:
  """Converts SingleReplicaArrayRestoreArgs to use colocated CPU devices."""
  if restore_args.single_replica_sharding is not None:
    cpu_single_replica_sharding = colocated_cpu_sharding(
        restore_args.single_replica_sharding
    )
    assert isinstance(cpu_single_replica_sharding, jax.sharding.NamedSharding)
    restore_args = dataclasses.replace(
        restore_args, single_replica_sharding=cpu_single_replica_sharding
    )
  return cast(
      jax_array_restore_args.SingleReplicaArrayRestoreArgs,
      convert_array_restore_args(restore_args),
  )


def transform_tree_shardings(input_tree: PyTree) -> Any:
  """Converts shardings/specs/restore-args/arrays to colocated CPU devices."""

  def _transform_leaf_sharding(leaf: Any) -> Any:
    if isinstance(leaf, jax.sharding.Sharding):
      return colocated_cpu_sharding(leaf)
    if isinstance(leaf, jax.ShapeDtypeStruct) and hasattr(leaf, 'sharding'):
      cpu_sharding = colocated_cpu_sharding(leaf.sharding)
      return jax.ShapeDtypeStruct(
          leaf.shape, leaf.dtype, sharding=cpu_sharding
      )
    if isinstance(leaf, jax_array_restore_args.SingleReplicaArrayRestoreArgs):
      return convert_single_replica_restore_args(leaf)
    if isinstance(leaf, jax_array_restore_args.ArrayRestoreArgs):
      return convert_array_restore_args(leaf)
    if isinstance(leaf, jax.Array):
      return to_colocated_python(leaf)
    return leaf

  return jax.tree.map(_transform_leaf_sharding, input_tree)


def to_final_specs(
    input_tree: PyTree,
    tpu_or_cpu_specs: PyTree,
) -> PyTree:
  """Transfers jax.Arrays to the final sharding specs."""

  def _to_final_spec(leaf: Any, tpu_or_cpu_spec: Any) -> Any:
    if isinstance(leaf, jax.Array) and hasattr(tpu_or_cpu_spec, 'sharding'):
      logging.vlog(
          1,
          'Transferring array from %s to final sharding %s',
          leaf.sharding,
          tpu_or_cpu_spec.sharding,
      )
    return jax.device_put(leaf, tpu_or_cpu_spec.sharding, may_alias=True)

  return jax.tree.map(_to_final_spec, input_tree, tpu_or_cpu_specs)


def shape_dtype_struct_for_array(array: jax.Array) -> jax.ShapeDtypeStruct:
  """Builds a ShapeDtypeStruct from a jax.Array."""
  return cast(
      jax.ShapeDtypeStruct, abstract_arrays.to_shape_dtype_struct(array)
  )


def zeros_like_spec(spec: jax.ShapeDtypeStruct) -> jax.Array:
  """Builds a zero-valued array matching the given spec without global allocs."""

  def _zeros(index: tuple[slice | int, ...] | None) -> np.ndarray:
    assert index is not None
    local_shape = []
    for dim, size in zip(index, spec.shape):
      if isinstance(dim, slice):
        start = 0 if dim.start is None else dim.start
        stop = size if dim.stop is None else dim.stop
        local_shape.append(stop - start)
      else:
        local_shape.append(1)
    return np.zeros(tuple(local_shape), dtype=spec.dtype)

  return jax.make_array_from_callback(
      spec.shape, spec.sharding, _zeros, dtype=spec.dtype
  )
