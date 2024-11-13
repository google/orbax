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

import dataclasses
from typing import Callable, Optional, Sequence

from absl import logging
import jax
import jax.numpy as jnp
import math
import numpy as np
from orbax.checkpoint._src.arrays import fragments
from orbax.checkpoint._src.arrays import numpy_utils
from orbax.checkpoint._src.arrays import types
from orbax.checkpoint._src.multihost import multihost


Shape = types.Shape
Index = types.Index


@dataclasses.dataclass(frozen=True)
class ReplicaSlice:
  """
  ReplicaSlice represents the part of a jax.Shard that a replica is uniquely
  responsible for. A replica slice can be either on-device (backed by a slice of
  a single-sharding array) or on-host (backed by a numpy ndarray).

  With single-replica checkpointing the entirety of each jax.Shard is owned by
  exactly one replica. (Currently the only option.)
  """

  replica_id: int
  index: Index
  data: jax.Array | np.ndarray

  @property
  def is_on_host(self):
    return isinstance(self.data, np.ndarray)


@dataclasses.dataclass(frozen=True)
class ReplicaSlices:
  """
  ReplicaSlices groups all the sliced data of one jax.Array that a replica is
  uniquely responsible for. Slices are either all on-device or all on-host.
  """

  global_shape: Shape
  local_shape: Shape
  sharding: jax.sharding.Sharding
  dtype: np.dtype
  is_on_host: bool
  replica_slices: list[ReplicaSlice]

  def __post_init__(self):
    assert all(
        rslice.is_on_host == self.is_on_host
        for rslice in self.replica_slices
    ), f'inconsistent is_on_host in {self!r}'

  @property
  def nbytes(self) -> int:
    slice_nbytes = math.prod(self.local_shape) * self.dtype.itemsize
    return slice_nbytes * len(self.replica_slices)

  def to_fragments(self) -> fragments.Fragments:
    assert self.is_on_host
    result = fragments.Fragments(
        shape=self.global_shape,
        dtype=self.dtype,
        fragments=[
            fragments.Fragment(
                index=numpy_utils.resolve_slice(
                    rslice.index,
                    self.global_shape
                ),
                value=rslice.data,
            )
            for rslice in self.replica_slices
        ],
    )
    if result.fragments:
      fragments.validate_fragments_can_be_stacked(result)
    if not result.is_degenerate():
      assert self.local_shape == result.fragments[0].shape
    return result


def get_replica_slices(
    arr: jax.Array,
    replica_id: Optional[int],
) -> ReplicaSlices:
  """Returns the replica slices a given replica is responsible for.
  Does not transfer allocate or transfer any data."""
  Result = tuple[list[ReplicaSlice], Shape]
  shard0 = arr.addressable_shards[0]

  # single-replica: a single replica saves an entire shard.
  def pick_single_replica() -> Result:
    # Omitting the replica id just picks the first addressable shard's replica
    # id so that the process writes each of its addressable shards exactly
    # once. (This is the desired behavior for local checkpointing.)
    target_replica_id = replica_id or shard0.replica_id
    rslices = [
        ReplicaSlice(
            replica_id=shard.replica_id,
            index=shard.index,
            data=shard.data,
        )
        for shard in arr.addressable_shards
        if shard.replica_id == target_replica_id
    ]
    local_shape = shard0.data.shape
    return rslices, local_shape

  shards_info = ', '.join(
      [
          f'Shard(index={shard.index}, replica_id={shard.replica_id})'
          for shard in arr.addressable_shards
      ]
  )
  logging.vlog(
      1,
      '[process=%d] get_replica_slices: replica_id=%d, shards=[%s]',
      multihost.process_index(),
      replica_id,
      shards_info,
  )

  # In order for all processes to agree on the right serialization metadata
  # we want to compute the correct local shape regardless of whether there
  # are any replica slices to save locally.
  rslices, local_shape = pick_single_replica()
  return ReplicaSlices(
      global_shape=arr.shape,
      local_shape=local_shape,
      sharding=arr.sharding,
      dtype=arr.dtype,
      is_on_host=False,
      replica_slices=rslices,
  )


def transfer_arrays_to_host(
    arrays: Sequence[jax.Array],
    replica_id: Optional[int],
    *,
    enable_pinned_host_transfer: bool = True,
) -> Sequence[ReplicaSlices]:
  """
  Transfers jax.Arrays to host memory and returns all the fragments to be
  serialized by the given replica, along with local shape. Blocks until
  completion.
  """

  def use_pinned_host_transfer(device):
    has_pinned_host = any(
        m.kind == 'pinned_host' for m in device.addressable_memories()
    )
    return (
        enable_pinned_host_transfer
        and has_pinned_host
        and jax._src.config.enable_memories.value  # pylint: disable=protected-access
    )

  def async_transfer_slice(rslice: ReplicaSlice) -> tuple[ReplicaSlice, jax.Array]:
    assert not rslice.is_on_host
    index = rslice.index
    data = rslice.data
    device = data.device
    # Start the asynchronous device-to-host copy
    if use_pinned_host_transfer(device):
      # If available, transfer to pinned host memory
      data = jax.device_put(
          data,
          jax.sharding.SingleDeviceSharding(device, memory_kind='pinned_host'),
      )
    else:
      data.copy_to_host_async()
    return rslice, data

  # Gather the replica slices to be saved for each array.
  rslices_per_array = [get_replica_slices(arr, replica_id) for arr in arrays]
  # Kick off transfers for all replica slices to be saved.
  transfers_per_array = [
      [async_transfer_slice(rslice) for rslice in rslices.replica_slices]
      for rslices in rslices_per_array
  ]
  # Wait for all the transferred data to be ready.
  return [
      dataclasses.replace(
          rslices,
          is_on_host=True,
          replica_slices=[
              dataclasses.replace(
                  rslice_on_device,
                  # Conversion to numpy arrays forces block_until_ready.
                  data=np.asarray(data),
              )
              for rslice_on_device, data in transfers
          ],
      )
      for rslices, transfers in zip(rslices_per_array, transfers_per_array)
  ]
