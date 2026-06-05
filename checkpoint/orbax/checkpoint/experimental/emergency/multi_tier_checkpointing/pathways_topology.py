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

"""Pathways worker topology helpers for colocated MTC."""

from __future__ import annotations

import collections
from collections.abc import Sequence
import dataclasses
import functools

import jax
import numpy as np
from orbax.checkpoint._src.multihost import colocated_transport
from orbax.checkpoint._src.multihost import pathways

_WORKER_AXIS_NAME = 'worker'


def worker_key_sort_key(worker_key: tuple[int, ...]) -> tuple[int, ...]:
  """Sorts `(task, slice)` Pathways worker keys in slice-major order.

  Args:
    worker_key: The input worker key.

  Returns:
    The worker key reordered for sorting.
  """
  if len(worker_key) == 2:
    task_index, slice_index = worker_key
    return (slice_index, task_index)
  return worker_key


@dataclasses.dataclass(frozen=True)
class Worker:
  """A single Pathways worker in controller-assigned logical rank order.

  Attributes:
    rank: The controller-assigned logical rank of the worker.
    key: The worker key.
    device_ids: The local device ids for the worker.
    representative_device: The representative JAX device for the worker.
  """

  rank: int
  key: tuple[int, ...]
  device_ids: tuple[int, ...]
  representative_device: jax.Device


@dataclasses.dataclass(frozen=True)
class Topology:
  """Controller-side view of Pathways workers used by MTC.

  Attributes:
    workers: A tuple of `Worker` instances in the topology.
    num_slices: The number of data-parallel slices.
  """

  workers: tuple[Worker, ...]
  num_slices: int | None = None

  @classmethod
  def from_devices(
      cls,
      devices: Sequence[jax.Device],
      *,
      num_slices: int | None = None,
  ) -> 'Topology':
    """Builds a stable slice-major Pathways worker topology."""
    worker_groups = pathways.group_devices_by_worker(devices)
    sorted_worker_groups = sorted(
        worker_groups.items(),
        key=lambda item: worker_key_sort_key(item[0]),
    )
    workers = []
    for rank, (worker_key, worker_devices) in enumerate(sorted_worker_groups):
      workers.append(
          Worker(
              rank=rank,
              key=worker_key,
              device_ids=tuple(sorted(d.id for d in worker_devices)),
              representative_device=worker_devices[0],
          )
      )
    topology = cls(workers=tuple(workers), num_slices=num_slices)
    if num_slices is not None:
      topology.validate_num_slices()
    return topology

  @property
  def num_workers(self) -> int:
    """The total number of physical host workers in the topology."""
    return len(self.workers)

  @functools.cached_property
  def distributed_to_device_ids(self) -> tuple[tuple[int, ...], ...]:
    """The mapping of distributed device ids to local device ids."""
    return tuple(tuple(worker.device_ids) for worker in self.workers)

  def remap_distributed_device_ids(
      self,
      source_devices: Sequence[jax.Device],
      target_devices: Sequence[jax.Device],
  ) -> list[list[int]]:
    """Maps per-worker source device ids into another device-id namespace."""
    if len(source_devices) != len(target_devices):
      raise ValueError(
          'source_devices and target_devices must have the same length, got '
          f'{len(source_devices)} and {len(target_devices)}.'
      )
    target_id_by_source_id = {
        int(source.id): int(target.id)
        for source, target in zip(source_devices, target_devices)
    }
    missing_source_ids = []
    for worker in self.workers:
      for source_id in worker.device_ids:
        if source_id not in target_id_by_source_id:
          missing_source_ids.append(source_id)
    missing_source_ids.sort()
    if missing_source_ids:
      raise ValueError(
          'Topology contains device ids not present in source_devices: '
          f'{missing_source_ids}.'
      )
    return [
        [target_id_by_source_id[source_id] for source_id in worker.device_ids]
        for worker in self.workers
    ]

  @functools.cached_property
  def representative_devices(self) -> tuple[jax.Device, ...]:
    """The representative device for each worker."""
    return tuple(worker.representative_device for worker in self.workers)

  def worker_cpu_devices(self) -> tuple[jax.Device, ...]:
    """Returns one colocated CPU device per Pathways worker."""
    cpu_devices = colocated_transport.unique_colocated_cpu_devices(
        self.representative_devices
    )
    if len(cpu_devices) != len(self.representative_devices):
      raise ValueError(
          'Expected one colocated CPU device per Pathways worker, got '
          f'{len(cpu_devices)} CPU devices for '
          f'{len(self.representative_devices)} workers.'
      )
    return cpu_devices

  def validate_num_slices(self) -> None:
    """Validates that the worker topology is compatible with `num_slices`."""
    if self.num_slices is None:
      return
    if self.num_slices <= 0:
      raise ValueError(f'num_slices must be positive, got {self.num_slices}.')
    if self.num_workers <= 0:
      raise ValueError('Pathways MTC requires at least one worker.')
    if self.num_workers % self.num_slices != 0:
      raise ValueError(
          'num_workers must be divisible by num_slices, got '
          f'num_workers={self.num_workers}, num_slices={self.num_slices}.'
      )
    if self.num_slices == 1:
      return

    worker_keys = tuple(worker.key for worker in self.workers)
    if any(len(key) != 2 for key in worker_keys):
      raise ValueError(
          'Multi-slice Pathways MTC requires worker keys with task and slice '
          f'indices, got {worker_keys}.'
      )
    slices = sorted({key[1] for key in worker_keys})
    if len(slices) != self.num_slices:
      raise ValueError(
          'Pathways MTC expected '
          f'{self.num_slices} slices, got slice indices {slices}.'
      )
    tasks_by_slice: dict[int, set[int]] = collections.defaultdict(set)
    for task_index, slice_index in worker_keys:
      tasks_by_slice[slice_index].add(task_index)
    expected_tasks = tasks_by_slice[slices[0]]
    mismatched_slices = {
        slice_index: tasks
        for slice_index, tasks in tasks_by_slice.items()
        if tasks != expected_tasks
    }
    if mismatched_slices:
      raise ValueError(
          'Pathways MTC requires identical task sets on every slice, got '
          f'{tasks_by_slice}.'
      )

  @functools.cached_property
  def peer_ranks_by_worker_rank(self) -> tuple[tuple[int, ...], ...]:
    """Peer ranks for each worker rank across data-parallel slices.

    Raises:
      ValueError: If `num_slices` is None, as slices must be known to map peers.
    """
    if self.num_slices is None:
      raise ValueError('num_slices is required to compute peer ranks.')
    self.validate_num_slices()
    if self.num_slices == 1:
      return tuple(() for _ in self.workers)
    return _peer_ranks_by_worker_key(self.workers)

  def worker_rank_array(
      self, worker_cpu_devices: Sequence[jax.Device] | None = None
  ) -> jax.Array:
    """Builds a sharded array carrying each worker's logical rank."""
    if worker_cpu_devices is None:
      worker_cpu_devices = self.worker_cpu_devices()
    return worker_rank_array(worker_cpu_devices)


def _peer_ranks_by_worker_key(
    workers: Sequence[Worker],
) -> tuple[tuple[int, ...], ...]:
  """Returns peer ranks by matching Pathways workers across slices by task."""
  ranks_by_task_and_slice = {worker.key: worker.rank for worker in workers}
  slices = sorted({worker.key[1] for worker in workers})
  peers = []
  for worker in workers:
    task_index, slice_index = worker.key
    peers.append(
        tuple(
            ranks_by_task_and_slice[(task_index, peer_slice)]
            for peer_slice in slices
            if peer_slice != slice_index
        )
    )
  return tuple(peers)


def worker_rank_array(worker_cpu_devices: Sequence[jax.Device]) -> jax.Array:
  """Builds a one-element-per-worker logical rank array."""
  if not worker_cpu_devices:
    raise ValueError('worker_cpu_devices must be non-empty.')
  mesh = jax.sharding.Mesh(np.asarray(worker_cpu_devices), (_WORKER_AXIS_NAME,))
  sharding = jax.sharding.NamedSharding(
      mesh, jax.sharding.PartitionSpec(_WORKER_AXIS_NAME)
  )
  worker_ranks = np.arange(len(worker_cpu_devices), dtype=np.int32)
  return jax.make_array_from_callback(
      worker_ranks.shape,
      sharding,
      lambda index: worker_ranks if index is None else worker_ranks[index],
      dtype=worker_ranks.dtype,
  )


def worker_rank_from_array(rank_array: jax.Array) -> int:
  """Extracts a colocated worker's logical rank from a sharded rank array."""
  addressable_shards = tuple(getattr(rank_array, 'addressable_shards', ()))
  if len(addressable_shards) == 1:
    rank_value = np.asarray(addressable_shards[0].data)
  elif len(addressable_shards) > 1:
    raise ValueError(
        'Expected exactly one addressable logical worker rank shard for this '
        f'colocated worker, got {len(addressable_shards)}.'
    )
  else:
    rank_value = None

  is_fully_addressable = getattr(rank_array, 'is_fully_addressable', True)
  if callable(is_fully_addressable):
    is_fully_addressable = is_fully_addressable()
  if rank_value is None and is_fully_addressable:
    rank_value = np.asarray(rank_array)
  if rank_value is None:
    raise ValueError(
        'Expected exactly one addressable logical worker rank shard for this '
        'colocated worker, got 0.'
    )
  if rank_value.size != 1:
    raise ValueError(
        'Expected exactly one logical worker rank for this colocated worker, '
        f'got shape={rank_value.shape}.'
    )
  return int(rank_value.reshape(-1)[0])
