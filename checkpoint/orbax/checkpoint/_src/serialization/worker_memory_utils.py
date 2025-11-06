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

"""Utilities for managing worker memory usage."""

import collections
from collections.abc import Sequence
import functools

from absl import logging
import humanize
import jax
import numpy as np
from orbax.checkpoint._src.multihost import dispatchers
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.serialization import types


@functools.lru_cache(maxsize=1)
def _device_to_worker_ids(dispatcher: dispatchers.Dispatcher) -> dict[int, int]:
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

  Returns:
    A mapping from device ID to worker ID.
  """
  fully_sharded_sharding = jax.sharding.NamedSharding(
      jax.sharding.Mesh(jax.devices(), 'x'),
      jax.sharding.PartitionSpec(
          'x',
      ),
  )

  def _get_worker_ids_impl(device_ids: jax.Array) -> jax.Array:
    return jax.make_array_from_callback(
        device_ids.shape,
        device_ids.sharding,
        lambda _: np.array(multihost.process_index()).reshape(
            1,
        ),
        dtype=np.int32,
    )

  device_ids = jax.device_put(
      np.asarray([d.id for d in jax.devices()]),
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
    arr: jax.Array,
    *,
    replica_id: int | None,
    device_to_worker_ids_map: dict[int, int],
) -> dict[int, int]:
  """Estimates memory used by the array on each worker after transfer.

  Args:
    arr: The array to estimate memory usage for.
    replica_id: The replica id to use for estimation. If None, all replicas are
      used.
    device_to_worker_ids_map: A mapping from device ID to worker ID.

  Returns:
    A mapping from worker ID to estimated memory usage.
  """
  worker_memory_usage = collections.defaultdict(int)
  shard_memory_size = (
      np.prod(arr.sharding.shard_shape(arr.shape)) * arr.dtype.itemsize
  )
  for shard in arr.global_shards:
    if replica_id is not None and shard.replica_id != replica_id:
      continue
    worker_id = device_to_worker_ids_map[shard.device.id]
    worker_memory_usage[worker_id] += shard_memory_size
  return worker_memory_usage


def _increment_worker_memory_usage(
    arr: jax.Array,
    current_worker_memory_usage: dict[int, int],
    *,
    replica_id: int | None,
    device_to_worker_ids_map: dict[int, int],
) -> dict[int, int]:
  """Increments memory used by the array on each worker after transfer."""
  estimate = _estimate_worker_memory_usage(
      arr,
      replica_id=replica_id,
      device_to_worker_ids_map=device_to_worker_ids_map,
  )
  return {
      worker_id: current_worker_memory_usage[worker_id] + estimate[worker_id]
      for worker_id in current_worker_memory_usage
  }


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


def next_memory_budgeted_batch(
    params: Sequence[tuple[jax.Array, types.ParamInfo, types.SaveArgs]],
    worker_memory_budget: int,
    *,
    replica_id: int | None,
    dispatcher: dispatchers.Dispatcher | None,
):
  """Yields batches of info, args, and arrays that fit within the memory budget.

  Args:
    params: A sequence of (array, info, args) tuples.
    worker_memory_budget: The maximum amount of memory that can be used on any
      worker.
    replica_id: The replica id to use for estimation. If None, all replicas are
      used.
    dispatcher: The dispatcher instance to use. If None, the default dispatcher
      is used.
  """
  arrays, infos, args = zip(*params, strict=True)
  if dispatcher is None:
    device_to_worker_ids_map = {
        d.id: multihost.process_index_from_device(d) for d in jax.devices()
    }
  else:
    device_to_worker_ids_map = _device_to_worker_ids(dispatcher)
    # NOTE: We only transfer save shards with replica_id == replica_id, but we
    # are actually redundantly transferring all shards, thanks to remote python
    # / colcoated python. So we set replica_id to None, to estimate memory usage
    # for all replicas.
    replica_id = None

  def _no_worker_memory_usage() -> dict[int, int]:
    return {id: 0 for id in set(device_to_worker_ids_map.values())}

  current_batch = [(arrays[0], infos[0], args[0])]
  current_worker_memory_usage = _estimate_worker_memory_usage(
      arrays[0],
      replica_id=replica_id,
      device_to_worker_ids_map=device_to_worker_ids_map,
  )

  for arr, info, arg in zip(arrays[1:], infos[1:], args[1:]):
    incremented_worker_memory_usage = _increment_worker_memory_usage(
        arr,
        current_worker_memory_usage,
        replica_id=replica_id,
        device_to_worker_ids_map=device_to_worker_ids_map,
    )
    if _is_array_under_memory_budget(
        worker_memory_budget, incremented_worker_memory_usage
    ):
      current_batch.append((arr, info, arg))
      current_worker_memory_usage = incremented_worker_memory_usage
    else:
      assert current_batch
      logging.info(
          '[process=%d] Obtained a memory-limited batch (replica_id=%s) with'
          ' %d arrays. Per-worker, projected to cost: %s',
          multihost.process_index(),
          replica_id,
          len(current_batch),
          _humanize_worker_memory_usage(current_worker_memory_usage),
      )
      yield current_batch

      # Assumes arrays have completed saving and worker memory has been freed.
      # The current array still needs to be dealt with.
      current_batch = [(arr, info, arg)]
      current_worker_memory_usage = _increment_worker_memory_usage(
          arr,
          _no_worker_memory_usage(),
          replica_id=replica_id,
          device_to_worker_ids_map=device_to_worker_ids_map,
      )

  if current_batch:
    logging.info(
        '[process=%d] Obtained a memory-limited batch (replica_id=%s) with %d'
        ' arrays. Per-worker, projected to cost: %s',
        multihost.process_index(),
        replica_id,
        len(current_batch),
        _humanize_worker_memory_usage(current_worker_memory_usage),
    )
    yield current_batch
