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
from typing import Any, cast, Sequence

from absl import logging
import jax
import jax.experimental.colocated_python as cp
from jax.experimental.colocated_python import serialization as cp_serialization
import numpy as np
from orbax.checkpoint._src.arrays import abstract_arrays
from orbax.checkpoint._src.metadata import sharding as sharding_metadata
from orbax.checkpoint._src.serialization import jax_array_restore_args


PyTree = Any
_PATHWAYS_SERIALIZATION_PATCH_INSTALLED = False
_PJRT_IFRT_DEVICE_ID_RE = re.compile(r'PjRtIFRTDeviceId=(\d+)')


def _to_serializable_cpu_device(device: jax.Device) -> jax.Device:
  """Normalizes a device to the CPU device used by colocated Python."""
  if device.platform == 'cpu':
    return device
  return cp.colocated_cpu_devices((device,))[0]


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

  JAX calls this while unreducing every colocated `Mesh`, `DeviceList`, or
  `SingleDeviceSharding`. Backend device topology is stable for this process,
  so the map is cached to avoid repeated backend scans.

  The controller serializes colocated CPU objects by controller-visible
  `device.id`, which is a backend-global IFRT id in the Pathways runtime. The
  worker-side remote-Python CPU backend can expose different local `device.id`
  values, so we first register the backend-global IFRT id parsed from repr and
  then fall back to the local `device.id` without overwriting the global entry.

  When the IFRT id and local `device.id` namespaces do not collide, this keeps
  both lookups working. If the namespaces collide for different devices, the
  map is ambiguous and this function fails instead of returning the wrong CPU.
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


def _normalize_mesh_to_colocated_cpu(
    mesh: jax.sharding.Mesh,
) -> jax.sharding.Mesh:
  devices = tuple(mesh.devices.flat)
  if all(_device_platform(device) == 'cpu' for device in devices):
    return mesh
  cpu_devices = np.vectorize(
      _to_serializable_cpu_device, otypes=[object]
  )(mesh.devices)
  return jax.sharding.Mesh(
      cpu_devices, mesh.axis_names, axis_types=mesh.axis_types
  )


def _normalize_device_list_to_colocated_cpu(
    device_list: cp_serialization.DeviceList,
) -> cp_serialization.DeviceList:
  if all(_device_platform(device) == 'cpu' for device in device_list):
    return device_list
  return cp_serialization.DeviceList(
      tuple(_to_serializable_cpu_device(device) for device in device_list)
  )


def _normalize_single_device_sharding_to_colocated_cpu(
    sharding: jax.sharding.SingleDeviceSharding,
) -> jax.sharding.SingleDeviceSharding:
  device = next(iter(sharding.device_set))
  if _device_platform(device) == 'cpu':
    return sharding
  return jax.sharding.SingleDeviceSharding(
      _to_serializable_cpu_device(device), memory_kind=sharding.memory_kind
  )


def install_pathways_colocated_serialization_patch() -> None:
  """Installs a Pathways-aware colocated-python serialization patch.

  The live Pathways failures are below Orbax checkpoint semantics. They happen
  while JAX is pickling and unpickling callable specializations that contain
  mesh-backed shardings.

  The patch is intentionally narrow:

  1. Keep JAX's existing serialized representation based on integer CPU ids
  2. Normalize any non-CPU mesh/device-list/sharding to colocated CPU devices
     before it reaches JAX's reducers
  3. Teach worker-side CPU lookup to recognize backend-global PjRt-IFRT ids,
     which are what controller-side CPU `device.id` values correspond to in the
     Pathways remote-Python runtime

  This keeps Orbax close to upstream JAX semantics while fixing the exact
  controller/proxy/worker identity mismatch seen in Pathways logs.

  The important constraint is that we are not changing the checkpoint contract
  or inventing a second serialized format. We are only making JAX's existing
  colocated serialization contract portable across the controller/worker CPU-id
  namespace split used by Pathways single-controller.

  Tracked at b/503051746 to make proper changes to JAX.
  """
  # pylint: disable=global-statement
  global _PATHWAYS_SERIALIZATION_PATCH_INSTALLED
  if _PATHWAYS_SERIALIZATION_PATCH_INSTALLED:
    return

  original_reduce_mesh = cp_serialization._reduce_mesh  # pylint: disable=protected-access
  original_reduce_device_list = cp_serialization._reduce_device_list  # pylint: disable=protected-access
  original_reduce_single_device_sharding = cp_serialization._reduce_single_device_sharding  # pylint: disable=protected-access

  def _orbax_reduce_mesh(mesh: jax.sharding.Mesh) -> Any:
    return original_reduce_mesh(_normalize_mesh_to_colocated_cpu(mesh))

  def _orbax_reduce_device_list(
      device_list: cp_serialization.DeviceList,
  ) -> Any:
    return original_reduce_device_list(
        _normalize_device_list_to_colocated_cpu(device_list)
    )

  def _orbax_reduce_single_device_sharding(
      sharding: jax.sharding.SingleDeviceSharding,
  ) -> Any:
    return original_reduce_single_device_sharding(
        _normalize_single_device_sharding_to_colocated_cpu(sharding)
    )

  cp_serialization._reduce_mesh = _orbax_reduce_mesh  # pylint: disable=protected-access
  cp_serialization._reduce_device_list = _orbax_reduce_device_list  # pylint: disable=protected-access
  cp_serialization._reduce_single_device_sharding = _orbax_reduce_single_device_sharding  # pylint: disable=protected-access
  cp_serialization._get_cpu_device_map = _get_cpu_device_map  # pylint: disable=protected-access
  _PATHWAYS_SERIALIZATION_PATCH_INSTALLED = True


def unique_colocated_cpu_devices(
    devices: Sequence[jax.Device],
) -> tuple[jax.Device, ...]:
  """Returns one colocated CPU device per worker."""
  logging.info('unique_colocated_cpu_devices: input devices=%s', devices)
  all_cpu = tuple(cp.colocated_cpu_devices(tuple(devices)))
  logging.info(
      'unique_colocated_cpu_devices: colocated_cpu_devices returned=%s', all_cpu
  )
  unique_cpu = []
  seen_ids = set()
  for device in all_cpu:
    if device.id in seen_ids:
      continue
    seen_ids.add(device.id)
    unique_cpu.append(device)
  logging.info('unique_colocated_cpu_devices: unique_cpu=%s', unique_cpu)
  return tuple(unique_cpu)


def colocated_cpu_sharding(
    sharding: jax.sharding.Sharding,
) -> jax.sharding.Sharding:
  """Returns a CPU sharding colocated with the given sharding."""
  logging.info('colocated_cpu_sharding: input sharding=%s', sharding)
  if isinstance(sharding, jax.sharding.SingleDeviceSharding):
    cpu_devices = cp.colocated_cpu_devices(list(sharding.device_set))
    result = jax.sharding.SingleDeviceSharding(
        cpu_devices[0], memory_kind=sharding.memory_kind
    )
    logging.info(
        'colocated_cpu_sharding: returning SingleDeviceSharding=%s', result
    )
    return result
  if isinstance(sharding, jax.sharding.NamedSharding):
    cpu_mesh = cp.colocated_cpu_devices(sharding.mesh)
    result = jax.sharding.NamedSharding(
        cpu_mesh, sharding.spec, memory_kind=sharding.memory_kind
    )
    logging.info('colocated_cpu_sharding: returning NamedSharding=%s', result)
    return result
  logging.error(
      'colocated_cpu_sharding: unsupported sharding type=%s', type(sharding)
  )
  raise TypeError(
      f'Sharding type {type(sharding)} not supported in to_colocated_python.'
  )


def to_colocated_python(input_tree: PyTree) -> PyTree:
  """Copies a pytree of arrays to colocated CPU devices."""
  logging.info(
      'to_colocated_python: starting with tree structure=%s',
      jax.tree.structure(input_tree),
  )

  def _get_sharding(x: Any) -> jax.sharding.Sharding | None:
    if isinstance(x, jax.Array):
      cpu_sharding = colocated_cpu_sharding(x.sharding)
      logging.info(
          'Staging array from %s to colocated CPU sharding %s',
          x.sharding,
          cpu_sharding,
      )
      return cpu_sharding
    return None

  cpu_sharding_tree = jax.tree.map(_get_sharding, input_tree)
  result = jax.device_put(input_tree, cpu_sharding_tree, may_alias=True)
  logging.info('to_colocated_python: finished device_put')
  return result


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
    logging.info(
        'Converting restore mesh with axis names %s to colocated CPU mesh.',
        restore_args.mesh.axis_names,
    )
    restore_args = dataclasses.replace(restore_args, mesh=cpu_mesh)
  if restore_args.sharding is None:
    return restore_args
  if isinstance(restore_args.sharding, jax.sharding.Sharding):
    cpu_sharding = colocated_cpu_sharding(restore_args.sharding)
    logging.info(
        'Converting restore sharding from %s to colocated CPU sharding %s',
        restore_args.sharding,
        cpu_sharding,
    )
    return dataclasses.replace(restore_args, sharding=cpu_sharding)
  if isinstance(restore_args.sharding, sharding_metadata.ShardingMetadata):
    sharding = restore_args.sharding.to_jax_sharding()
    cpu_sharding = colocated_cpu_sharding(sharding)
    logging.info(
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
  logging.info('transform_tree_shardings: starting mapping')

  def _transform_leaf_sharding(leaf: Any) -> Any:
    if isinstance(leaf, jax.sharding.Sharding):
      logging.info('transform_tree_shardings: converting Sharding=%s', leaf)
      return colocated_cpu_sharding(leaf)
    if isinstance(leaf, jax.ShapeDtypeStruct) and hasattr(leaf, 'sharding'):
      logging.info(
          'transform_tree_shardings: ShapeDtypeStruct sharding=%s',
          leaf.sharding,
      )
      cpu_sharding = colocated_cpu_sharding(leaf.sharding)
      return jax.ShapeDtypeStruct(
          leaf.shape, leaf.dtype, sharding=cpu_sharding
      )
    if isinstance(leaf, jax_array_restore_args.SingleReplicaArrayRestoreArgs):
      logging.info(
          'transform_tree_shardings: SingleReplicaArrayRestoreArgs=%s', leaf
      )
      return convert_single_replica_restore_args(leaf)
    if isinstance(leaf, jax_array_restore_args.ArrayRestoreArgs):
      logging.info('transform_tree_shardings: ArrayRestoreArgs=%s', leaf)
      return convert_array_restore_args(leaf)
    if isinstance(leaf, jax.Array):
      logging.info('transform_tree_shardings: Array of shape %s', leaf.shape)
      return to_colocated_python(leaf)
    return leaf

  result = jax.tree.map(_transform_leaf_sharding, input_tree)
  logging.info('transform_tree_shardings: finished mapping')
  return result


def to_final_specs(
    input_tree: PyTree,
    tpu_or_cpu_specs: PyTree,
) -> PyTree:
  """Transfers jax.Arrays to the final sharding specs."""

  def _to_final_spec(leaf: Any, tpu_or_cpu_spec: Any) -> Any:
    if isinstance(leaf, jax.Array) and hasattr(tpu_or_cpu_spec, 'sharding'):
      logging.info(
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
