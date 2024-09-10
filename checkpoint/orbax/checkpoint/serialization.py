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

"""Array serialization and deserialization.

TODO(b/348434669): De-fork when possible.
"""

import asyncio
from collections.abc import Awaitable
import contextlib
import dataclasses
import functools
import os
import re
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Protocol, Sequence, Union

from absl import logging
import humanize
import jax
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint import multihost
from orbax.checkpoint._src.serialization import tensorstore_utils as ts_utils
import tensorstore as ts


TS_CONTEXT = ts.Context({'file_io_concurrency': {'limit': 128}})
_REMOVED_VALUE = 'Value removed'
_CHECKPOINT_SUCCESS = 'checkpoint_write_success'
_REMOTE_URL_PREFIXES = ['gs://', 's3://']
_REMOTE_DRIVER_VALIDATIONS = [
    {'driver': 'gcs', 'path_regex': None},
    {'driver': 's3', 'path_regex': None},
]


Shape = tuple[int, ...]
Index = Index = tuple[slice, ...]


async def create_async_array_from_callback(
    global_shape: Shape,
    inp_sharding: jax.sharding.Sharding,
    data_callback: Callable[[Index, jax.Device], Awaitable[jax.Array]],
):
  device_to_index_map = inp_sharding.devices_indices_map(global_shape)
  addressable_da = inp_sharding._addressable_device_assignment  # pylint: disable=protected-access
  future_arrays = [data_callback(device_to_index_map[d], d)
                   for d in addressable_da]
  dbs = await asyncio.gather(*future_arrays)
  return jax.make_array_from_single_device_arrays(
      global_shape, inp_sharding, dbs
  )


def _get_metadata(arr):
  local_shape = arr.addressable_data(0).shape
  return {
      'compressor': {'id': 'zstd'},
      'shape': arr.shape,
      'chunks': np.array(np.maximum(1, local_shape)),
  }


def _spec_has_metadata(tree):
  if not isinstance(tree, dict):
    return False
  return 'metadata' in tree or any(
      _spec_has_metadata(subtree) for _, subtree in tree.items()
  )


def _get_kvstore_for_gcs(ckpt_path: str):
  m = re.fullmatch('^gs://([^/]*)/(.*)$', ckpt_path, re.DOTALL)
  if m is None:
    raise ValueError(
        'The ckpt_path should contain the bucket name and the '
        f'file path inside the bucket. Got: {ckpt_path}'
    )
  gcs_bucket = m.group(1)
  path_without_bucket = m.group(2)
  return {'driver': 'gcs', 'bucket': gcs_bucket, 'path': path_without_bucket}


def get_tensorstore_spec(ckpt_path: str, ocdbt: bool = False):
  """Constructs a TensorStore spec for the given checkpoint path."""
  # Normalize path to exclude trailing '/'. In GCS path case, we will need to
  # fix the path prefix to add back the stripped '/'.
  ckpt_path = os.path.normpath(ckpt_path).replace('gs:/', 'gs://')
  is_gcs_path = ckpt_path.startswith('gs://')
  spec = {'driver': 'zarr', 'kvstore': {}}
  if ocdbt:
    if not is_gcs_path and not os.path.isabs(ckpt_path):
      raise ValueError(f'Checkpoint path should be absolute. Got {ckpt_path}')
    base_path = os.path.dirname(ckpt_path)
    base_driver_spec = (
        base_path
        if is_gcs_path
        else {'driver': ts_utils.DEFAULT_DRIVER, 'path': base_path}
    )
    spec['kvstore'] = {
        'driver': 'ocdbt',
        'base': base_driver_spec,
        'path': os.path.basename(ckpt_path),
    }
  else:
    if is_gcs_path:
      spec['kvstore'] = _get_kvstore_for_gcs(ckpt_path)
    else:
      spec['kvstore'] = {'driver': ts_utils.DEFAULT_DRIVER, 'path': ckpt_path}

  return spec


def is_remote_storage(tspec: Union[Dict[str, Any], str]) -> bool:
  """Detect if user is using remote storages.

  This can detect common defines and unable to detect some corner cases such as
  using gcsfuse.

  Args:
    tspec: Tensorstore spec.

  Returns:
    True if the spec is using remote storage.
  """
  if isinstance(tspec, str):
    # KvStoreUrl
    if re.match(rf'^({"|".join(_REMOTE_URL_PREFIXES)})', tspec):
      return True
    else:
      return False

  for key in ('base', 'kvstore'):
    if key in tspec:
      return is_remote_storage(tspec[key])

  if 'driver' in tspec:
    for rule in _REMOTE_DRIVER_VALIDATIONS:
      if tspec['driver'] == rule['driver']:
        if rule['path_regex'] is None:
          return True

        # check if path matches the regex.
        if re.match(rule['path_regex'], tspec['path']):
          return True

  return False


class ByteLimiter(Protocol):

  async def wait_for_bytes(self, requested_bytes: int):
    ...

  async def release_bytes(self, requested_bytes: int):
    ...


# Lifted from T5X.
class LimitInFlightBytes(ByteLimiter):
  """Limits in-flight bytes when reading/writing checkpoints per process."""

  def __init__(self, num_bytes: int):
    if num_bytes <= 0:
      raise ValueError(f'Must provide positive `num_bytes`. Found: {num_bytes}')
    self._max_bytes = num_bytes
    self._available_bytes = num_bytes
    self._cv = asyncio.Condition(lock=asyncio.Lock())

  async def wait_for_bytes(self, requested_bytes: int):
    """Reserve bytes."""
    if requested_bytes >= self._max_bytes:
      raise ValueError('Requested more bytes than we reserved space for: '
                       f'{requested_bytes} > {self._max_bytes}')
    async with self._cv:
      await self._cv.wait_for(lambda: self._available_bytes > requested_bytes)
      self._available_bytes -= requested_bytes
      logging.vlog(
          1,
          'Reserved bytes: %s | Remaining bytes: %s',
          humanize.naturalsize(requested_bytes, binary=True),
          humanize.naturalsize(self._available_bytes, binary=True),
      )
      assert self._available_bytes >= 0

  async def release_bytes(self, requested_bytes: int):
    async with self._cv:
      self._available_bytes += requested_bytes
      logging.vlog(
          1,
          'Releasing bytes: %s | Available bytes: %s',
          humanize.naturalsize(requested_bytes, binary=True),
          humanize.naturalsize(self._available_bytes, binary=True),
      )
      assert self._available_bytes <= self._max_bytes
      self._cv.notify_all()


class UnlimitedInFlightBytes(ByteLimiter):

  async def wait_for_bytes(self, requested_bytes: int):
    del requested_bytes
    return

  async def release_bytes(self, requested_bytes: int):
    del requested_bytes
    return


def get_byte_limiter(concurrent_bytes: Optional[int] = None) -> ByteLimiter:
  if concurrent_bytes is None:
    return UnlimitedInFlightBytes()
  if concurrent_bytes <= 0:
    raise ValueError(
        f'Must provide positive `concurrent_bytes`. Found: {concurrent_bytes}'
    )
  return LimitInFlightBytes(concurrent_bytes)


@contextlib.asynccontextmanager
async def reserved_bytes(
    byte_limiter: ByteLimiter,
    nbytes: int,
) -> AsyncIterator[None]:
  """Reserves some bytes for the duration of the context."""
  await byte_limiter.wait_for_bytes(nbytes)
  try:
    yield
  finally:
    await byte_limiter.release_bytes(nbytes)


@dataclasses.dataclass
class Shards:
  """Basic representation of host-local set of shards."""
  shards: List[jax.Shard]
  shape: Shape
  dtype: jnp.dtype
  sharding: jax.sharding.Sharding

  def block_until_ready(self):
    jax.block_until_ready([shard.data for shard in self.shards])
    # Ensure that jax.Array's internal numpy array can be zero-copied. This
    # guards against consumers like tensorstore that would otherwise copy
    # silently.
    for i, shard in enumerate(self.shards):
      self.shards[i] = jax.Shard(
          device=shard.device,
          data=np.array(shard.data, copy=False),
          sharding=self.sharding,
          global_shape=self.shape,
      )


def _transfer_shard_to_host(shard: jax.Shard) -> jax.Array:
  """Asynchronously transfers a shard to host memory. Does not block."""
  data = shard.data
  has_pinned_host = any(
      m.kind == 'pinned_host' for m in shard.device.addressable_memories()
  )
  if jax._src.config.enable_memories.value and has_pinned_host:  # pylint: disable=protected-access
    # If available, transfer to pinned host memory
    sharding = jax.sharding.SingleDeviceSharding(
        shard.device, memory_kind='pinned_host'
    )
    data = jax.device_put(data, sharding)
  else:
    data.copy_to_host_async()
  return data


def get_shards_for_host_transfer(
    arr: jax.Array, replica_id: int
) -> List[jax.Shard]:
  """Returns the list of jax.Shard that should be transferred to host memory."""
  return [
      shard
      for shard in arr.addressable_shards
      if shard.replica_id == replica_id
  ]


def transfer_array_to_host(arr: jax.Array, replica_id: int) -> Shards:
  """Transfers a jax.Array to host memory."""
  shard_data = []
  dedup_shards = get_shards_for_host_transfer(arr, replica_id)
  for shard in dedup_shards:
    shard_data.append(_transfer_shard_to_host(shard))

  return Shards(
      [
          jax.Shard(
              device=shard.device,
              data=data,
              sharding=arr.sharding,
              global_shape=arr.shape,
          )
          for data, shard in zip(shard_data, dedup_shards)
      ],
      shape=arr.shape,
      dtype=arr.dtype,
      sharding=arr.sharding,
  )


async def _write_shard(
    shard: jax.Shard,
    t: ts.TensorStore,
    replica_id: int,
    byte_limiter: ByteLimiter,
):
  """Writes a single array using TensorStore. No copy is performed."""
  assert shard.replica_id == replica_id
  assert isinstance(shard.data, np.ndarray)
  requested_bytes = estimate_write_memory_footprint(shard.data)
  async with reserved_bytes(byte_limiter, requested_bytes):
    await t[shard.index].write(
        shard.data,
        # Avoid additional copy of input array into the TensorStore chunk
        # cache. The data array of a shard is guaranteed to be immutable and
        # therefore it is safe to retain a reference indefinitely.
        can_reference_source_data_indefinitely=True,
    )


async def async_serialize(
    arr_inp: jax.Array,
    tensorstore_spec: Dict[str, Any],
    context: ts.Context = TS_CONTEXT,
    primary_host: Optional[int] = 0,
    replica_id: int = 0,
    transaction: Optional[ts.Transaction] = None,
    byte_limiter: Optional[ByteLimiter] = None,
):
  """Serialize an array using TensorStore.

  Performs a D2H transfer of the array. Prefer to use `async_serialize_shards`
  by separately performing a D2H transfer, and then starting the serialization
  in a background thread.

  Args:
    arr_inp: The array to serialize.
    tensorstore_spec: The tensorstore spec to use.
    context: ts.Context instance.
    primary_host: Primary host, which indicates the host that will be treated as
      the "leader". If None, all hosts are treated as the primary. DO NOT USE
      unless you are sure you know what you are doing.
    replica_id: Allows overriding the shard replica id that will be saved. DO
      NOT USE unless you are sure you know what you are doing.
    transaction: TensorStore transaction to use for opening and writing the
      array.  If not specified, a non-transactional write will be used.
    byte_limiter: A ByteLimiter instance that will be used to limit the number
      of bytes in flight when writing to TensorStore. If None, no limitation
      will be applied.
  """
  byte_limiter = byte_limiter or get_byte_limiter()
  # 'metadata' may not be present at the top level (for example, if we are using
  # a 'cast' driver).
  if not _spec_has_metadata(tensorstore_spec):
    tensorstore_spec['metadata'] = _get_metadata(arr_inp)
  # Set dtype if it's not in spec
  if 'dtype' not in tensorstore_spec:
    tensorstore_spec['dtype'] = jnp.dtype(arr_inp.dtype).name
  # Start D2H transfer in parallel for each array.
  host_shards = transfer_array_to_host(arr_inp, replica_id)
  host_shards.block_until_ready()
  await async_serialize_shards(
      host_shards,
      tensorstore_spec,
      context=context,
      primary_host=primary_host,
      replica_id=replica_id,
      transaction=transaction,
      byte_limiter=byte_limiter,
  )


async def async_serialize_shards(
    shards: Shards,
    tensorstore_spec: Dict[str, Any],
    *,
    context: ts.Context = TS_CONTEXT,
    primary_host: Optional[int] = 0,
    replica_id: int = 0,
    transaction: Optional[ts.Transaction] = None,
    byte_limiter: Optional[ByteLimiter] = None,
):
  """Serialize a host-local shards using TensorStore.

  Args:
    shards: A Shards object. Individual shards are expected to be host-local.
    tensorstore_spec: The tensorstore spec to use.
    context: ts.Context instance.
    primary_host: Primary host, which indicates the host that will be treated as
      the "leader". If None, all hosts are treated as the primary. DO NOT USE
      unless you are sure you know what you are doing.
    replica_id: Allows overriding the shard replica id that will be saved. DO
      NOT USE unless you are sure you know what you are doing.
    transaction: TensorStore transaction to use for opening and writing the
      array.  If not specified, a non-transactional write will be used.
    byte_limiter: A ByteLimiter instance that will be used to limit the number
      of bytes in flight when writing to TensorStore. If None, no limitation
      will be applied.

  Raises:
    KeyError: If `metadata` or `dtype` is not found in the tensorstore spec.
  """
  byte_limiter = byte_limiter or get_byte_limiter()
  if not _spec_has_metadata(tensorstore_spec):
    raise KeyError('`metadata` not found in tensorstore spec.')
  # Set dtype if it's not in spec
  if 'dtype' not in tensorstore_spec:
    raise KeyError('`dtype` not found in tensorstore spec.')

  # If primary_host is None, all hosts will checkpoint. This is used
  # for checkpointing to local filesystem.
  if primary_host is None or multihost.process_index() == primary_host:
    await ts.open(
        ts.Spec(tensorstore_spec),
        create=True,
        open=True,
        context=context,
        transaction=transaction,
    )

  # `ts.open` runs twice for process `primary_host` because for the first time,
  # we just get the future to be awaited upon in the background thread. The
  # second one runs with `assume_metadata=True` which does no I/O operation and
  # returns the tensorstore object.
  # For every process other than `primary_host`, we open with
  # `assume_metadata=True`.
  t = await ts.open(
      ts.Spec(tensorstore_spec),
      open=True,
      assume_metadata=True,
      context=context,
      transaction=transaction,
  )
  write_shard_coros = jax.tree.map(
      functools.partial(
          _write_shard,
          t=t,
          replica_id=replica_id,
          byte_limiter=byte_limiter,
      ),
      shards.shards,
  )
  await asyncio.gather(*write_shard_coros)


def estimate_write_memory_footprint(arr: jax.Array) -> int:
  return arr.size * arr.dtype.itemsize


def estimate_read_memory_footprint(t: ts.TensorStore,
                                   domain: ts.IndexDomain) -> int:
  """Estimates memory required to read a given domain."""
  rank = t.rank
  num_bytes = t.dtype.numpy_dtype.itemsize
  chunk_template = t.chunk_layout.read_chunk_template
  if domain is None:
    domain = t.domain
  origin = domain.origin
  shape = domain.shape
  chunk_origin = chunk_template.origin
  chunk_shape = chunk_template.shape

  # Some TensorStore drivers are not chunked, e.g. the inline 'array' driver.
  # For those, instead of returning a near-infinite memory footprint, estimate
  # the footprint as the entire shape.
  for i in range(rank):
    if not chunk_template[i].finite:
      return domain.size * num_bytes

  # Otherwise, if we have a chunked driver, estimate based on chunk size.
  for i in range(rank):
    origin_value = origin[i]
    chunk_origin_value = chunk_origin[i]
    chunk_size = chunk_shape[i]
    lower = origin_value - chunk_origin_value
    upper = origin_value + shape[i] - chunk_origin_value
    lower_aligned = lower // chunk_size * chunk_size
    upper_aligned = -(-upper // chunk_size) * chunk_size
    num_bytes *= (upper_aligned - lower_aligned)

  return num_bytes


async def _read_and_device_put_shard(
    device: jax.Device,
    t: ts.TensorStore,
    new_shard_shape: Sequence[int],
    dtype: jnp.dtype,
    requested_domain: ts.IndexDomain,
    restricted_domain: ts.IndexDomain,
) -> jax.Array:
  """Reads a single shard from TensorStore and places it on device."""
  # This maybe needed because the shape the array was saved with is smaller
  # than the requested shape of the array in which it will be reloaded. So
  # the extra values will be filled with 0s.
  out = np.zeros(new_shard_shape, dtype=t.dtype.numpy_dtype)
  await ts.array(out)[ts.d[:].translate_to[requested_domain.origin]][
      restricted_domain
  ].write(t[restricted_domain])
  if dtype is not None:
    # Cast while reloading on process to avoid 2 copies on device if the
    # casting is done on device.
    out = out.astype(dtype)
  # Convert to jnp array so that layouts are initialized properly for
  # sub-byte dtypes.
  # TODO(yashkatariya): This is a band-aid fix. Figure out a better way to
  # make this work.
  if out.dtype == jnp.int4:
    out = jnp.asarray(out)  # type: ignore
  return jax.device_put(out, jax.sharding.SingleDeviceSharding(device))


async def _read_array_index_callback(
    index: Index,
    device: jax.Device,
    t: ts.TensorStore,
    shape: Sequence[int],
    new_shard_shape: Sequence[int],
    dtype: jnp.dtype,
    byte_limiter: ByteLimiter,
) -> jax.Array:
  """Callback that reads an array index and places on device."""
  requested_domain = ts.IndexTransform(input_shape=shape)[index].domain
  restricted_domain = t.domain.intersect(requested_domain)
  requested_bytes = estimate_read_memory_footprint(t, restricted_domain)
  # Limit the bytes read for every shard.
  async with reserved_bytes(byte_limiter, requested_bytes):
    result = await _read_and_device_put_shard(
        device,
        t,
        new_shard_shape,
        dtype,
        requested_domain,
        restricted_domain,
    )
  return result


async def async_deserialize(
    user_in_sharding: jax.sharding.Sharding,
    tensorstore_spec: Union[ts.Spec, Dict[str, Any]],
    global_shape: Optional[Sequence[int]] = None,
    dtype: Optional[jnp.dtype] = None,
    byte_limiter: Optional[ByteLimiter] = None,
    context: ts.Context = TS_CONTEXT,
    assume_metadata: bool = False,
) -> jax.Array:
  """Reads an array using TensorStore."""
  byte_limiter = byte_limiter or get_byte_limiter()
  in_sharding = user_in_sharding
  if not isinstance(in_sharding, jax.sharding.Sharding):
    raise ValueError(
        'sharding passed to deserialization should be specified, concrete and'
        f' an instance of `jax.sharding.Sharding`. Got {in_sharding}')
  t = await ts.open(
      tensorstore_spec,
      open=True,
      assume_metadata=assume_metadata,
      context=context,
  )
  shape = t.shape if global_shape is None else global_shape
  new_shard_shape = in_sharding.shard_shape(tuple(shape))
  return await create_async_array_from_callback(
      tuple(shape),
      in_sharding,
      functools.partial(
          _read_array_index_callback,
          t=t,
          shape=shape,
          new_shard_shape=new_shard_shape,
          dtype=dtype,
          byte_limiter=byte_limiter,
      ),
  )
