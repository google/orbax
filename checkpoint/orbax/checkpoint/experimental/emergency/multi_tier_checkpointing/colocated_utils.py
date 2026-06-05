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

"""Helper utilities for colocated emergency checkpointing."""

from collections.abc import Sequence
from typing import Any

from absl import logging
import jax
import numpy as np
from orbax.checkpoint._src.multihost import colocated_transport
from orbax.checkpoint.experimental.emergency.multi_tier_checkpointing import (
    pathways_topology,
)


PyTree = Any
# The MTC coordinator protocol reserves step 0 as "no checkpoint".
# Valid colocated MTC checkpoints therefore start at step 1.
NO_STEP_SENTINEL = 0
MAX_TRACKED_STEPS = 128


def make_scalar_on_like(
    value: Any, like: jax.Array, *, dtype: Any
) -> jax.Array:
  """Builds a scalar array on the same global sharding as `like`."""
  return colocated_transport.make_scalar_array_like(value, like, dtype=dtype)


def compute_distributed_to_device_ids(
    devices: Sequence[jax.Device],
) -> list[list[int]]:
  """Returns per-worker device ids in slice-major order."""
  topology = pathways_topology.Topology.from_devices(devices)
  distributed_to_device_ids = [
      list(ids) for ids in topology.distributed_to_device_ids
  ]
  logging.vlog(
      1,
      'Computed Pathways distributed_to_device_ids for %d workers: %s',
      len(distributed_to_device_ids),
      distributed_to_device_ids,
  )
  return distributed_to_device_ids


def device_list_signature(
    sharding: jax.sharding.Sharding,
) -> tuple[tuple[str, int], ...]:
  """Builds a stable device-list signature from public sharding APIs."""
  if isinstance(sharding, jax.sharding.NamedSharding):
    return tuple((d.platform, d.id) for d in sharding.mesh.devices.flat)
  if isinstance(sharding, jax.sharding.SingleDeviceSharding):
    d = next(iter(sharding.device_set))
    return ((d.platform, d.id),)
  return tuple(sorted((d.platform, d.id) for d in sharding.device_set))


def assert_arrays_on_platform(
    tree: PyTree,
    *,
    expected_platform: str,
    tree_name: str,
) -> None:
  """Validates that all jax.Array leaves are on the expected platform."""
  leaves = jax.tree.leaves(tree)
  array_leaves = [x for x in leaves if isinstance(x, jax.Array)]
  if not array_leaves:
    return

  for i, leaf in enumerate(array_leaves):
    platforms = sorted({d.platform for d in leaf.sharding.device_set})
    if platforms != [expected_platform]:
      sample_devices = sorted(leaf.sharding.device_set, key=lambda d: d.id)[:4]
      raise ValueError(
          f'{tree_name} contains arrays not confined to '
          f'"{expected_platform}" devices. First mismatch: '
          f'(leaf={i}, platforms={platforms}, '
          f'sample_devices={[(d.id, d.platform) for d in sample_devices]}).'
      )


def assert_arrays_on_allowed_cpu_ids(
    tree: PyTree,
    *,
    allowed_ids: frozenset[int],
    tree_name: str,
) -> None:
  """Validates that all array shardings use only expected CPU device ids."""
  leaves = jax.tree.leaves(tree)
  array_leaves = [x for x in leaves if isinstance(x, jax.Array)]
  if not array_leaves:
    return

  for i, leaf in enumerate(array_leaves):
    ids = {d.id for d in leaf.sharding.device_set}
    invalid = sorted(ids - allowed_ids)
    if invalid:
      sample_devices = sorted(leaf.sharding.device_set, key=lambda d: d.id)[:4]
      raise ValueError(
          f'{tree_name} contains unexpected CPU device ids. First mismatch: '
          f'(leaf={i}, invalid_ids={invalid[:8]}, '
          f'sample_devices={[(d.id, d.platform) for d in sample_devices]}).'
      )


def assert_specs_on_allowed_cpu_ids(
    tree: PyTree,
    *,
    allowed_ids: frozenset[int],
    tree_name: str,
) -> None:
  """Validates ShapeDtypeStruct leaves reference only expected CPU ids."""
  leaves = jax.tree.leaves(tree)
  spec_leaves = [x for x in leaves if isinstance(x, jax.ShapeDtypeStruct)]
  if not spec_leaves:
    return

  for i, spec in enumerate(spec_leaves):
    sharding = spec.sharding
    if sharding is None:
      raise ValueError(
          f'{tree_name} contains non-CPU or unexpected device ids. '
          f'First mismatch: (leaf={i}, sharding=None).'
      )
    platforms = sorted({d.platform for d in sharding.device_set})
    ids = {d.id for d in sharding.device_set}
    invalid = sorted(ids - allowed_ids)
    if platforms != ['cpu'] or invalid:
      sample_devices = sorted(sharding.device_set, key=lambda d: d.id)[:4]
      raise ValueError(
          f'{tree_name} contains non-CPU or unexpected device ids. '
          f'First mismatch: (leaf={i}, platforms={platforms}, '
          f'invalid_ids={invalid[:8]}, '
          f'sample_devices={[(d.id, d.platform) for d in sample_devices]}).'
      )


def require_unanimous_scalar_result(
    result: jax.Array, *, op_name: str
) -> Any:
  """Returns a unanimous scalar value from workers or raises."""
  values = scalar_result_values(result, op_name=op_name)
  unique_values = set(values)
  if len(unique_values) != 1:
    raise RuntimeError(
        f'{op_name}: workers disagreed on scalar result: '
        f'{sorted(unique_values)} (sample={values[:8]})'
    )
  return values[0]


def scalar_result_values(result: jax.Array, *, op_name: str) -> list[Any]:
  """Returns scalar values from workers."""
  values = []
  for shard in result.addressable_shards:
    value = np.asarray(shard.data)
    if value.size != 1:
      raise ValueError(
          f'{op_name}: expected scalar shard value, got shape={value.shape}.'
      )
    values.append(value.item())
  if not values:
    value = np.asarray(result)
    if value.size != 1:
      raise ValueError(
          f'{op_name}: unexpected non-scalar result shape={value.shape}.'
      )
    values.append(value.item())
  return values


def array_result_values(result: jax.Array, *, op_name: str) -> list[np.ndarray]:
  """Returns array values from workers."""
  values = []
  for shard in result.addressable_shards:
    values.append(np.asarray(shard.data))
  if not values:
    values.append(np.asarray(result))
  for value in values:
    if value.ndim == 0:
      raise ValueError(f'{op_name}: expected array shard value, got scalar.')
  return values
