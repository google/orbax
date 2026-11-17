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

"""Utilities for managing worker memory usage."""

import collections
from collections.abc import Iterable, Iterator, Sequence
import functools

from absl import logging
import humanize
import jax
import numpy as np
from orbax.checkpoint._src.multihost import dispatchers
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.serialization import types


@functools.lru_cache(maxsize=1)
def _device_to_worker_ids(
    dispatcher: dispatchers.Dispatcher,
    devices: tuple[jax.Device, ...] | None = None,
) -> dict[int, int]:
  """Returns a mapping from device ID to worker ID.

  Works by using a remote Python function to obtain `multihost.process_index()`
  on
  each worker. This is then returned as a sharded array, where shard `i`,
  located on device `i`, contains the worker ID for which device `i` is local
  (`jax.local_devices()`).

  These contortions are necessary because there is not a straightforward API to
  obtain the mapping on Pathways.

  Args:
    dispatcher: The dispatcher instance to use.
    devices: Optional devices to query. Defaults to `jax.devices()`.

  Returns:
    A mapping from device ID to worker ID.
  """
  if devices is None:
    devices = tuple(jax.devices())
  fully_sharded_sharding = jax.sharding.NamedSharding(
      jax.sharding.Mesh(devices, 'x'),
      jax.sharding.PartitionSpec(
          'x',
      ),
  )

  def _get_worker_ids_impl(device_ids: jax.Array) -> jax.Array:
    return jax.make_array_from_callback(
        device_ids.shape,
        device_ids.sharding,
        lambda _: np.array(multihost.process_index(), dtype=np.int32).reshape(
            1,
        ),
        dtype=np.int32,
    )

  device_ids = jax.device_put(
      np.asarray([d.id for d in devices]),
      device=fully_sharded_sharding,
  )
  result_specs = jax.ShapeDtypeStruct(
      device_ids.shape, dtype=np.int32, sharding=fully_sharded_sharding
  )
  worker_ids = dispatcher.dispatch(
      _get_worker_ids_impl, input_arrays=device_ids, result_specs=result_specs
  )
  jax.block_until_ready(worker_ids)
  return {
      int(device_id): int(worker_id)
      for device_id, worker_id in zip(
          np.asarray(device_ids), np.asarray(worker_ids)
      )
  }


def _estimate_worker_memory_usage(
    arr: jax.Array | jax.ShapeDtypeStruct,
    *,
    replica_id: int | None,
    device_to_worker_ids_map: dict[int, int],
) -> dict[int, int]:
  """Estimates memory used by the array on each worker after transfer.

  Args:
    arr: The array or array spec to estimate memory usage for.
    replica_id: The replica id to use for estimation. If None, all replicas are
      used. Must be None when `arr` is a `jax.ShapeDtypeStruct`.
    device_to_worker_ids_map: A mapping from device ID to worker ID.

  Returns:
    A mapping from worker ID to estimated memory usage.
  """
  worker_memory_usage = dict.fromkeys(device_to_worker_ids_map.values(), 0)
  shard_memory_size = (
      int(np.prod(arr.sharding.shard_shape(arr.shape))) * arr.dtype.itemsize
  )
  if replica_id is None:
    devices = arr.sharding.device_set
  else:
    if not isinstance(arr, jax.Array):
      raise ValueError(
          'Filtering by `replica_id` requires a materialized `jax.Array`.'
      )
    devices = [
        shard.device
        for shard in arr.global_shards
        if shard.replica_id == replica_id
    ]
  for device in devices:
    worker_memory_usage[device_to_worker_ids_map[device.id]] += (
        shard_memory_size
    )
  return worker_memory_usage


def _is_array_under_memory_budget(
    worker_memory_budget: int,
    incremented_worker_memory_usage: dict[int, int],
) -> bool:
  for _, worker_memory_usage in incremented_worker_memory_usage.items():
    if worker_memory_usage >= worker_memory_budget:
      return False
  return True


def _humanize_worker_memory_usage(
    worker_memory_usage: dict[int, int],
) -> dict[str, str]:
  return {
      f'worker_{worker_id}': humanize.naturalsize(memory_usage, binary=True)
      for worker_id, memory_usage in worker_memory_usage.items()
  }


def _batch_indices_with_memory_usage(
    item_usages: Iterable[dict[int, int]],
    worker_memory_budget: int,
) -> Iterator[tuple[list[int], dict[int, int]]]:
  """Groups items into consecutive batches that fit `worker_memory_budget`.

  An item that alone reaches or exceeds `worker_memory_budget` gets its own
  batch.

  Args:
    item_usages: Per-worker byte estimates for each item, keyed by worker ID.
      Workers missing from an estimate use no memory for that item.
    worker_memory_budget: Maximum bytes allowed on any worker in a batch.

  Yields:
    `(indices, worker_memory_usage)` for each batch.
  """
  current_batch: list[int] = []
  current_worker_memory_usage: dict[int, int] = {}
  for i, item_usage in enumerate(item_usages):
    incremented_worker_memory_usage = {
        worker_id: (
            current_worker_memory_usage.get(worker_id, 0)
            + item_usage.get(worker_id, 0)
        )
        for worker_id in current_worker_memory_usage.keys() | item_usage.keys()
    }
    if not current_batch or _is_array_under_memory_budget(
        worker_memory_budget, incremented_worker_memory_usage
    ):
      current_batch.append(i)
      current_worker_memory_usage = incremented_worker_memory_usage
    else:
      # Assumes the previous batch completes and frees worker memory before the
      # next batch starts.
      yield current_batch, current_worker_memory_usage
      current_batch = [i]
      current_worker_memory_usage = dict(item_usage)
  if current_batch:
    yield current_batch, current_worker_memory_usage


def _active_devices(
    arrays: Iterable[jax.Array | jax.ShapeDtypeStruct],
) -> tuple[jax.Device, ...]:
  active_devices = set()
  for arr in arrays:
    active_devices.update(arr.sharding.device_set)
  return tuple(sorted(active_devices, key=lambda d: d.id))


def next_memory_budgeted_batch(
    params: Sequence[tuple[jax.Array, types.ParamInfo, types.SaveArgs]],
    worker_memory_budget: int,
    *,
    replica_ids: Sequence[int | None],
    dispatcher: dispatchers.Dispatcher | None,
):
  """Yields batches of info, args, and arrays that fit within the memory budget.

  Args:
    params: A sequence of (array, info, args) tuples.
    worker_memory_budget: The maximum amount of memory that can be used on any
      worker.
    replica_ids: The replica id to use for estimation for each array. If an
      element is None, all replicas are used for that array.
    dispatcher: The dispatcher instance to use. If None, the default dispatcher
      is used.
  """
  if len(replica_ids) != len(params):
    raise ValueError(
        f'Expected {len(params)} replica_ids, got {len(replica_ids)}'
    )
  if not params:
    return
  if dispatcher is None:
    device_to_worker_ids_map = {
        d.id: multihost.process_index_from_device(d) for d in jax.devices()
    }
  else:
    devices = _active_devices(arr for arr, _, _ in params)
    device_to_worker_ids_map = _device_to_worker_ids(dispatcher, devices)
    # NOTE: We only transfer save shards with replica_id == replica_id, but we
    # are actually redundantly transferring all shards, thanks to remote python
    # / colocated python. So we set replica_ids to None, to estimate memory
    # usage for all replicas.
    replica_ids = [None] * len(params)

  item_usages = (
      _estimate_worker_memory_usage(
          arr,
          replica_id=replica_id,
          device_to_worker_ids_map=device_to_worker_ids_map,
      )
      for (arr, _, _), replica_id in zip(params, replica_ids, strict=True)
  )
  for indices, batch_worker_memory_usage in _batch_indices_with_memory_usage(
      item_usages, worker_memory_budget
  ):
    logging.info(
        '[process=%d] Obtained a memory-limited batch (replica_ids=%s) with %d'
        ' arrays. Per-worker, projected to cost: %s',
        multihost.process_index(),
        collections.Counter(replica_ids[i] for i in indices),
        len(indices),
        _humanize_worker_memory_usage(batch_worker_memory_usage),
    )
    yield [params[i] for i in indices]


def next_memory_budgeted_batch_indices(
    specs: Sequence[jax.ShapeDtypeStruct],
    worker_memory_budget: int,
    *,
    dispatcher: dispatchers.Dispatcher,
) -> Iterator[list[int]]:
  """Yields consecutive batches of `specs` that fit the worker memory budget.

  Like `next_memory_budgeted_batch`, but for arrays that are yet to be
  materialized by `dispatcher`, e.g. when restoring. A spec that alone exceeds
  the budget gets its own batch.

  Args:
    specs: The arrays to materialize, with shardings.
    worker_memory_budget: The maximum amount of memory to use on any worker.
    dispatcher: The dispatcher that materializes the arrays.

  Yields:
    Batches of indices into `specs`.
  """
  if not specs:
    return
  device_to_worker_ids_map = _device_to_worker_ids(
      dispatcher, _active_devices(specs)
  )
  item_usages = (
      _estimate_worker_memory_usage(
          spec,
          replica_id=None,
          device_to_worker_ids_map=device_to_worker_ids_map,
      )
      for spec in specs
  )
  for indices, _ in _batch_indices_with_memory_usage(
      item_usages, worker_memory_budget
  ):
    yield indices
