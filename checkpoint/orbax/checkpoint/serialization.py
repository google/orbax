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
import functools
import os
import re
from typing import Any, Callable, Dict, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint import multihost
import tensorstore as ts


TS_CONTEXT = ts.Context({'file_io_concurrency': {'limit': 128}})
_REMOVED_VALUE = 'Value removed'
_CHECKPOINT_SUCCESS = 'checkpoint_write_success'
_DEFAULT_DRIVER = 'file'
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
        else {'driver': _DEFAULT_DRIVER, 'path': base_path}
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
      spec['kvstore'] = {'driver': _DEFAULT_DRIVER, 'path': ckpt_path}

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


# Lifted from T5X.
class _LimitInFlightBytes:
  """Limits in-flight bytes when reading/writing checkpoints per process."""

  def __init__(self, num_bytes):
    self._max_bytes = num_bytes
    self._available_bytes = num_bytes
    self._cv = asyncio.Condition(lock=asyncio.Lock())

  async def wait_for_bytes(self, requested_bytes):
    if requested_bytes >= self._max_bytes:
      raise ValueError('Requested more bytes than we reserved space for: '
                       f'{requested_bytes} > {self._max_bytes}')
    async with self._cv:
      await self._cv.wait_for(lambda: self._available_bytes > requested_bytes)
      self._available_bytes -= requested_bytes
      assert self._available_bytes >= 0

  async def release_bytes(self, requested_bytes):
    async with self._cv:
      self._available_bytes += requested_bytes
      assert self._available_bytes <= self._max_bytes
      self._cv.notify_all()


async def transfer_shard_to_host(shard: jax.Shard) -> np.ndarray:
  """Asynchronously transfers a shard to host memory."""
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
  # Allow other transfers to be scheduled simultaneously.
  await asyncio.sleep(0)
  # Ensure that jax.Array's internal numpy array can be zero-copied. This guards
  # against consumers like tensorstore that would otherwise copy silently.
  return np.array(data, copy=False)


def _get_copy_future(write_future):
  return write_future.copy


def _get_commit_future(write_future):
  return write_future.commit


async def _write_array(
    shard: jax.Shard,
    t: ts.TensorStore,
    commit_future: Optional[list[Any]],
    replica_id: int,
    can_reference_source_data_indefinitely: bool,
):
  """Writes a single array using TensorStore."""
  if shard.replica_id == replica_id:
    data = await transfer_shard_to_host(shard)
    write_future = t[shard.index].write(
        data,
        # Avoid additional copy of input array into the TensorStore chunk
        # cache.  If `arr_inp` is a jax.Array, the result of converting
        # it to a NumPy array, as is done internally by TensorStore, is
        # guaranteed to be immutable and therefore it is safe to retain a
        # reference indefinitely.
        can_reference_source_data_indefinitely=can_reference_source_data_indefinitely,
    )
    if commit_future is not None:
      assert isinstance(commit_future, list)
      commit_future.append(_get_commit_future(write_future))
      await _get_copy_future(write_future)
    else:
      await _get_commit_future(write_future)


async def async_serialize(
    arr_inp,
    tensorstore_spec,
    commit_future=None,
    context=TS_CONTEXT,
    primary_host: Optional[int] = 0,
    replica_id: int = 0,
    transaction: Optional[ts.Transaction] = None,
):
  """Serialize an array using TensorStore.

  Args:
    arr_inp: The array to serialize.
    tensorstore_spec: The tensorstore spec to use.
    commit_future: A list of futures that will be appended to. The futures can
      be awaited asynchronously. If None, the futures will be awaited
      synchronously by this method.
    context: ts.Context instance.
    primary_host: Primary host, which indicates the host that will be treated as
      the "leader". If None, all hosts are treated as the primary. DO NOT USE
      unless you are sure you know what you are doing.
    replica_id: Allows overriding the shard replica id that will be saved.
      DO NOT USE unless you are sure you know what you are doing.
    transaction: TensorStore transaction to use for opening and writing the
      array.  If not specified, a non-transactional write will be used.
  """
  if (
      isinstance(arr_inp, jax.Array)
      and jax.process_count() > 1
      and arr_inp.is_fully_addressable
  ):
    raise ValueError(
        f'Passing fully addressable arrays to a multiprocess '
        f'serialization is not allowed, as this may lead to a race condition '
        f'between processes. Serialization have failed for the array with '
        f'the path "{tensorstore_spec["kvstore"]["path"]}".')

  # 'metadata' may not be present at the top level (for example, if we are using
  # a 'cast' driver).
  if not _spec_has_metadata(tensorstore_spec):
    tensorstore_spec['metadata'] = _get_metadata(arr_inp)

  # Set dtype if it's not in spec
  if 'dtype' not in tensorstore_spec:
    tensorstore_spec['dtype'] = jnp.dtype(arr_inp.dtype).name

  # If primary_host is None, all hosts will checkpoint. This is used
  # for checkpointing to local filesystem.
  if primary_host is None or multihost.process_index() == primary_host:
    open_future = ts.open(
        ts.Spec(tensorstore_spec),
        create=True,
        open=True,
        context=context,
        transaction=transaction,
    )
    # Asynchronous case.
    if commit_future is not None:
      assert isinstance(commit_future, list)
      commit_future.append(open_future)
    else:
      await open_future

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
  local_shards = arr_inp.addressable_shards
  future_write_state = jax.tree_util.tree_map(
      functools.partial(
          _write_array,
          t=t,
          commit_future=commit_future,
          replica_id=replica_id,
          can_reference_source_data_indefinitely=isinstance(arr_inp, jax.Array),
      ),
      local_shards,
  )
  await asyncio.gather(*future_write_state)


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


async def async_deserialize(
    user_in_sharding: jax.sharding.Sharding,
    tensorstore_spec: Union[ts.Spec, Dict[str, Any]],
    global_shape: Optional[Sequence[int]] = None,
    dtype=None,
    byte_limiter: Optional[_LimitInFlightBytes] = None,
    context=TS_CONTEXT,
    assume_metadata: bool = False,
):
  """Reads an array using TensorStore."""
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

  async def cb(index: Index, device: jax.Device):
    requested_domain = ts.IndexTransform(input_shape=shape)[index].domain
    restricted_domain = t.domain.intersect(requested_domain)
    requested_bytes = estimate_read_memory_footprint(t, restricted_domain)
    # Limit the bytes read for every shard.
    if byte_limiter is not None:
      await byte_limiter.wait_for_bytes(requested_bytes)
    # This maybe needed because the shape the array was saved with is smaller
    # than the requested shape of the array in which it will be reloaded. So
    # the extra values will be filled with 0s.
    out = np.zeros(new_shard_shape, dtype=t.dtype.numpy_dtype)
    await ts.array(out)[ts.d[:].translate_to[requested_domain.origin]][
        restricted_domain].write(t[restricted_domain])
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
    result = jax.device_put(out, jax.sharding.SingleDeviceSharding(device))
    if byte_limiter is not None:
      # NB: `out` actually might not be ready for garbage collection by the
      # time we call release_bytes . Thus peak memory usage still might grow
      # beyond what byte_limiter limit suggests it should. The simplest option
      # would be to call  `result.block_until_ready()`` here. However it
      # also comes with ~15-20% perf penalty as we would be waiting for CPU->GPU
      # transfer instead of loading data. In the future, if memory pressure
      # becomes a problem, we can instead instrument  bytelimiter to
      # keep track of all in-flight tensors and only block_until_ready, if byte
      # limiter hits the limit to get reduced memory usage, without losing
      # performance in common use cases.
      await byte_limiter.release_bytes(requested_bytes)
    return result

  return await create_async_array_from_callback(tuple(shape), in_sharding, cb)
