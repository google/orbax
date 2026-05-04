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

"""Defines `SafetensorsLayout`, a class to handle Safetensors checkpoint formats."""

import asyncio
import concurrent
import dataclasses
import json
import os
import time
from typing import Any, Awaitable, cast

from absl import logging
import jax
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.path import async_path
from orbax.checkpoint._src.path import gcs_utils
from orbax.checkpoint._src.tree import utils as tree_utils
from orbax.checkpoint.experimental.v1._src.context import context as context_lib
from orbax.checkpoint.experimental.v1._src.layout import checkpoint_layout
from orbax.checkpoint.experimental.v1._src.metadata import types as metadata_types
from orbax.checkpoint.experimental.v1._src.path import types

CheckpointLayout = checkpoint_layout.CheckpointLayout
InvalidLayoutError = checkpoint_layout.InvalidLayoutError
Path = types.Path

HEADER_NUM_BYTES = 8
SAFETENSORS_SUFFIX = ".safetensors"


def _get_dtypes() -> dict[str, Any]:
  """Returns the mapping from safetensor `dtype` strings to NumPy `dtypes`."""
  return {
      "BOOL": np.bool_,
      "I8": np.int8,
      "U8": np.uint8,
      "I16": np.int16,
      "U16": np.uint16,
      "I32": np.int32,
      "U32": np.uint32,
      "I64": np.int64,
      "U64": np.uint64,
      "F16": np.float16,
      "F32": np.float32,
      "F64": np.float64,
      "BF16": jnp.bfloat16,
      "F8_E5M2": jnp.float8_e5m2,
      "F8_E4M3": jnp.float8_e4m3fn,
  }


def _get_array_properties(info: dict[str, Any]) -> tuple[tuple[int, ...], Any]:
  """Parses shape and `dtype` from a safetensors tensor header."""
  try:
    dtype_str = info["dtype"]
    dtype = _get_dtypes()[dtype_str]
  except KeyError as e:
    raise ValueError(f"Unsupported dtype in SafeTensors header: {e}") from e
  shape = tuple(info["shape"])
  return shape, dtype


def _should_replicate_array(shape: tuple[int, ...]) -> bool:
  """Returns True if the array should be replicated across devices."""
  return len(shape) == 0 or shape[0] < jax.local_device_count()


def _get_tensor_bundles(
    header: dict[str, Any], num_hosts: int
) -> list[list[str]]:
  """Partitions tensors in a file into contiguous bundles for each host.

  This method distributes tensors to hosts such that each host reads a
  contiguous block of bytes from the file. It tries to make the total byte
  size read by each host as close to 1/N as possible, where N is the number of
  hosts.

  TODO(b/496270336): Very large tensors should be subdivided among hosts. The
  current approach loads each tensor wholly onto its assigned host. This is
  efficient for checkpoints with lots of small tensors but can cause OOMs for
  very large tensors.

  Args:
    header: The header of the safetensors file.
    num_hosts: The number of hosts.

  Returns:
    A list of lists of tensor names, where each inner list contains the names
    of the tensors assigned to that host.
  """
  # Filter out metadata and sort tensors by their start offset in the file.
  tensors = {k: v for k, v in header.items() if k != "__metadata__"}
  sorted_tensors = sorted(
      tensors.items(), key=lambda item: item[1]["data_offsets"][0]
  )

  if not sorted_tensors:
    return [[] for _ in range(num_hosts)]

  # Calculate total data size based on the last tensor's end offset.
  total_size = sorted_tensors[-1][1]["data_offsets"][1]

  # Greedily assign tensors to hosts.
  bundles = [[] for _ in range(num_hosts)]
  current_bundle = 0
  cumulative_size = 0

  for name, info in sorted_tensors:
    start, end = info["data_offsets"]
    tensor_size = end - start

    if current_bundle < num_hosts - 1:
      # Calculate target cumulative size for current host.
      ideal_cumulative_size = (current_bundle + 1) * (total_size / num_hosts)

      # Decide whether to cut to next host or keep in current bundle.
      dist_if_cut = abs(cumulative_size - ideal_cumulative_size)
      dist_if_keep = abs(
          (cumulative_size + tensor_size) - ideal_cumulative_size
      )

      if dist_if_cut < dist_if_keep and cumulative_size > 0:
        current_bundle += 1

    bundles[current_bundle].append(name)
    cumulative_size += tensor_size

  return bundles


def _create_global_mesh() -> tuple[jax.sharding.Mesh, list[list[jax.Device]]]:
  """Creates a global mesh and returns it along with devices by host and count."""
  devices_by_host = np.asarray(jax.devices()).reshape(
      multihost.process_count(), jax.local_device_count()
  )

  for d in devices_by_host:
    if len(d) != jax.local_device_count():
      raise ValueError("Number of devices must be the same across all hosts.")

  global_mesh = jax.sharding.Mesh(
      np.array(devices_by_host), ("hosts", "devices")
  )
  return global_mesh, devices_by_host


def _get_abstract_transient_array(
    shape: tuple[int, ...],
    dtype: np.dtype,
    global_mesh: jax.sharding.Mesh,
    num_hosts: int,
) -> jax.ShapeDtypeStruct:
  """Determines the sharding strategy and shape for the transient array."""
  num_devices_per_host = jax.local_device_count()

  if _should_replicate_array(shape):
    # Cannot shard across devices, so replicate as a fallback.
    sharding = jax.sharding.NamedSharding(
        global_mesh, jax.sharding.PartitionSpec("hosts")
    )
    transient_shape = (num_hosts,) + shape
  else:
    # Enforce that the first dimension is divisible by the number of
    # devices per host. This allows us to shard the tensor across local
    # devices in host memory.
    # TODO(b/496270336): relax this constraint
    if shape[0] % num_devices_per_host != 0:
      raise ValueError(
          f"First dimension {shape[0]} is not divisible"
          f" by number of devices per host ({num_devices_per_host})."
      )
    # Fully shard across all devices. No replication. Keeps memory usage low.
    sharding = jax.sharding.NamedSharding(
        global_mesh, jax.sharding.PartitionSpec("hosts", "devices")
    )
    transient_shape = (num_hosts,) + shape

  return jax.ShapeDtypeStruct(
      shape=transient_shape, dtype=dtype, sharding=sharding
  )


def _get_current_shard_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
  """Calculates current shard shape."""
  num_devices_per_host = jax.local_device_count()

  if _should_replicate_array(shape):
    return (1,) + shape
  else:
    shard_size = shape[0] // num_devices_per_host
    return (1, shard_size) + shape[1:]


def _get_zero_shard_view(
    zero_buf: jax.Array, current_shard_shape: tuple[int, ...]
) -> jax.Array:
  """Returns a view of the shared zero buffer for a non-owner shard."""
  slices = [slice(0, 1)]
  for s_size in current_shard_shape[1:]:
    slices.append(slice(0, s_size))

  # Pad with 0 for remaining dimensions to reduce rank if needed.
  while len(slices) < len(zero_buf.shape):
    slices.append(0)

  return zero_buf[tuple(slices)]


def _calculate_max_shard_shapes(
    abstract_pytree: dict[str, Any], header: dict[str, Any]
) -> dict[np.dtype, list[int]]:
  """Calculates maximum shard shapes per dtype."""
  max_shard_shape_per_dtype = {}
  for name, _ in abstract_pytree.items():
    if name not in header:
      continue
    info = header[name]
    shape, dtype = _get_array_properties(info)
    current_shard_shape = _get_current_shard_shape(shape)

    if dtype not in max_shard_shape_per_dtype:
      max_shard_shape_per_dtype[dtype] = [1]

    max_shape = max_shard_shape_per_dtype[dtype]

    while len(max_shape) < len(current_shard_shape):
      max_shape.append(0)

    for i in range(1, len(current_shard_shape)):
      max_shape[i] = max(max_shape[i], current_shard_shape[i])
  return max_shard_shape_per_dtype


def _create_shared_zero_buffers(
    max_shard_shape_per_dtype: dict[np.dtype, list[int]],
    local_devices: list[jax.Device],
) -> dict[tuple[jax.Device, np.dtype], jax.Array]:
  """Creates shared zero buffers on local devices."""
  zero_buffers = {}
  for dtype, max_shape in max_shard_shape_per_dtype.items():
    for d in local_devices:
      zero_buffers[(d, dtype)] = jnp.zeros(
          tuple(max_shape), dtype=dtype, device=d
      )
  return zero_buffers


def _create_data_buffer(
    np_array: np.ndarray,
    device: jax.Device,
    shard_size: int | None,
    shard_index: int,
) -> jax.Array:
  """Creates a data buffer for a device."""
  if shard_size is None:
    return jnp.expand_dims(jax.device_put(np_array, device), axis=0)
  else:
    shard_start = shard_index * shard_size
    shard_end = (shard_index + 1) * shard_size
    tensor_shard = np_array[shard_start:shard_end]
    return jnp.expand_dims(jax.device_put(tensor_shard, device), axis=0)


def _extract_tensor_from_bundle(
    info: dict[str, Any],
    bundle_bytes: bytes,
    bundle_start_offset: int,
    shape: tuple[int, ...],
    dtype: np.dtype,
) -> np.ndarray:
  """Extracts tensor data from the read buffer."""
  start_offset, end_offset = info["data_offsets"]
  rel_start = start_offset - bundle_start_offset
  rel_end = end_offset - bundle_start_offset
  tensor_mv = memoryview(bundle_bytes)[rel_start:rel_end]
  return np.frombuffer(tensor_mv, dtype=dtype).reshape(shape)


def _reshard_transient_array(
    global_transient_array: jax.Array,
    target_sharding: jax.sharding.Sharding | None,
    global_mesh: jax.sharding.Mesh,
) -> jax.Array:
  """Reduces the transient array and reshards to the target sharding."""
  if target_sharding is not None:
    out_sharding = target_sharding
  else:
    out_sharding = jax.sharding.NamedSharding(
        global_mesh, jax.sharding.PartitionSpec()
    )

  return jax.jit(
      lambda x: jnp.sum(x, axis=0).astype(x.dtype), out_shardings=out_sharding
  )(global_transient_array)


@dataclasses.dataclass
class _LoadContext:
  host_id: int
  num_hosts: int
  global_mesh: jax.sharding.Mesh
  devices_by_host: list[list[jax.Device]]
  bundle_bytes: bytes
  bundle_start_offset: int
  zero_buffers: dict[tuple[jax.Device, np.dtype], jax.Array]


def _build_array_on_single_device(
    info: dict[str, Any], owner: int, ctx: _LoadContext
) -> jax.Array:
  """Builds a global JAX array placed on a single device."""
  shape, dtype = _get_array_properties(info)
  target_device = ctx.devices_by_host[owner][0]
  single_device_sharding = jax.sharding.SingleDeviceSharding(target_device)

  if ctx.host_id == owner:
    if not ctx.bundle_bytes:
      np_array = np.zeros(shape, dtype=dtype)
    else:
      np_array = _extract_tensor_from_bundle(
          info, ctx.bundle_bytes, ctx.bundle_start_offset, shape, dtype
      )
    device_array = jax.device_put(np_array, target_device)
    device_buffers = [device_array]
  else:
    device_buffers = []

  return jax.make_array_from_single_device_arrays(
      shape, single_device_sharding, device_buffers, dtype=dtype
  )


def _build_transient_array(
    name: str, info: dict[str, Any], owner: int, ctx: _LoadContext
) -> jax.Array:
  """Builds transient array for general resharding case."""
  shape, dtype = _get_array_properties(info)
  num_devices_per_host = jax.local_device_count()
  if _should_replicate_array(shape):
    shard_size = None
  else:
    shard_size = shape[0] // num_devices_per_host

  np_array = None
  if ctx.host_id == owner:
    np_array = _extract_tensor_from_bundle(
        info, ctx.bundle_bytes, ctx.bundle_start_offset, shape, dtype
    )

  device_buffers = []
  for i, d in enumerate(ctx.devices_by_host[ctx.host_id]):
    if ctx.host_id == owner:
      device_buffers.append(
          _create_data_buffer(
              np_array,
              d,
              shard_size,
              i,
          )
      )
    else:
      zero_buf = ctx.zero_buffers[(d, dtype)]
      current_shard_shape = _get_current_shard_shape(shape)
      device_buffers.append(_get_zero_shard_view(zero_buf, current_shard_shape))

  abstract_transient = _get_abstract_transient_array(
      shape, dtype, ctx.global_mesh, ctx.num_hosts
  )

  try:
    return jax.make_array_from_single_device_arrays(
        abstract_transient.shape,
        abstract_transient.sharding,
        device_buffers,
    )
  except Exception as e:
    logging.error(
        "Failed make_array_from_single_device_arrays for %s: %s", name, e
    )
    raise e


class _SingleFileLoader:
  """Single-file loader for Safetensors checkpoints."""

  def __init__(self, path: Path):
    self.path = path

  async def read_header(self) -> tuple[dict[str, Any], int]:
    """Reads a safetensors file header, returning the header and data start offset."""
    async with async_path.open_file(self.path, mode="rb") as f:
      header_size_bytes = await f.read(HEADER_NUM_BYTES)
      if not header_size_bytes:
        raise ValueError("Could not read header size from safetensors file.")

      header_size = int.from_bytes(header_size_bytes, byteorder="little")
      header_bytes = await f.read(header_size)
      if len(header_bytes) != header_size:
        raise ValueError("Could not read header content from safetensors file.")

      header = json.loads(header_bytes)
      data_start_offset = HEADER_NUM_BYTES + header_size
      return header, data_start_offset

  async def _read_bundle(
      self,
      bundles: list[list[str]],
      host_id: int,
      header: dict[str, Any],
      data_start_offset: int,
  ) -> tuple[bytes, int]:
    """Reads the assigned bundle of tensors in a single contiguous read."""
    my_bundle = bundles[host_id]
    if my_bundle:
      first_tensor = my_bundle[0]
      last_tensor = my_bundle[-1]
      bundle_start_offset = header[first_tensor]["data_offsets"][0]
      bundle_end_offset = header[last_tensor]["data_offsets"][1]
      bundle_num_bytes = bundle_end_offset - bundle_start_offset

      async with async_path.open_file(self.path, mode="rb") as f:
        await f.seek(data_start_offset + bundle_start_offset)
        bundle_bytes = cast(bytes, await f.read(bundle_num_bytes))
    else:
      bundle_bytes = b""
      bundle_start_offset = 0
    return bundle_bytes, bundle_start_offset

  async def load_single_host_gcsfuse(
      self, gcs_path_str: str, abstract_pytree: dict[str, Any] | None
  ) -> dict[str, np.ndarray]:
    """Downloads tensors from Google Cloud Storage using high-bandwidth parallel reads.

    This method uses `os.pread` with a thread pool to achieve high-bandwidth
    parallel downloads from GCS via gcsfuse. It first calculates the bounding
    box of the required tensor data and then reads chunks within that range.

    Args:
      gcs_path_str: The gcsfuse path to the safetensors file.
      abstract_pytree: A flat dictionary mapping tensor names to
        jax.ShapeDtypeStruct objects. Only tensors present in this dict will be
        loaded.

    Returns:
      A dictionary mapping tensor names to loaded NumPy arrays.

    Raises:
      EOFError: If the file is truncated or reading fails unexpectedly.
      ValueError: If non-finite values are found in a loaded tensor.
    """

    header, data_start_offset = await self.read_header()
    tensors = {}

    min_start = float("inf")
    max_end = 0

    # 1. Calculate the exact Bounding Box of the batch
    if abstract_pytree is None:
      tensor_names = header.keys()
    else:
      tensor_names = abstract_pytree.keys()
    for t_name in tensor_names:
      if t_name == "__metadata__":
        continue
      if t_name not in header:
        # Raise an error if the tensor is not found in the header.
        raise ValueError(f"Tensor {t_name} not found in header.")
      start, end = header[t_name]["data_offsets"]
      if start < min_start:
        min_start = start
      if end > max_end:
        max_end = end

    offset = data_start_offset + min_start
    length = max_end - min_start
    chunks = []
    bytes_read = 0
    chunk_size = 1024 * 1024 * 1024
    while bytes_read < length:
      current_chunk_size = min(chunk_size, length - bytes_read)
      current_offset = offset + bytes_read
      chunks.append(tuple([current_chunk_size, current_offset]))
      bytes_read += current_chunk_size

    def _read_single_chunk(chunk_data):
      chunk_size, offset = chunk_data
      with open(gcs_path_str, "rb") as f:
        bytes_read = 0
        chunk_pieces = []
        while bytes_read < chunk_size:
          piece = os.pread(
              f.fileno(), chunk_size - bytes_read, offset + bytes_read
          )
          if not piece:
            raise EOFError(
                f"Unexpected end of file at offset {offset + bytes_read} "
                f"in file {gcs_path_str}. Expected {chunk_size} bytes, "
                f"got {bytes_read}."
            )
          chunk_pieces.append(piece)
          bytes_read += len(piece)
        return b"".join(chunk_pieces)

    max_workers = 16
    # 2. Execute the parallel reads
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:
      read_chunks = list(executor.map(_read_single_chunk, chunks))

    data_bytes = b"".join(read_chunks)
    data_mv = memoryview(data_bytes)
    for name in tensor_names:
      if name == "__metadata__":
        continue
      shape, dtype = _get_array_properties(header[name])
      start_offset, end_offset = header[name]["data_offsets"]
      tensor_bytes = data_mv[
          start_offset - min_start : end_offset - min_start
      ]
      np_array = np.frombuffer(tensor_bytes, dtype=dtype).reshape(shape)
      if not np.isfinite(np_array).all():
        raise ValueError(f"Non-finite values found in tensor {name}.")
      tensors[name] = np_array
    return tensors

  async def load_single_host(
      self, abstract_pytree: dict[str, Any] | None
  ) -> dict[str, np.ndarray]:
    """Loads tensors from a safetensors file into host NumPy arrays."""
    if gcs_utils.is_gcs_path(self.path):
      gcs_path_str = gcs_utils.to_gcsfuse_path(self.path)
      return await self.load_single_host_gcsfuse(gcs_path_str, abstract_pytree)
    header, data_start_offset = await self.read_header()
    tensors = {}
    async with async_path.open_file(self.path, mode="rb") as f:
      await f.seek(data_start_offset)
      data_bytes = await f.read()
    for name, info in header.items():
      if name == "__metadata__":
        continue
      shape, dtype = _get_array_properties(info)
      start_offset, end_offset = info["data_offsets"]
      tensor_bytes = data_bytes[start_offset:end_offset]
      np_array = np.frombuffer(tensor_bytes, dtype=dtype).reshape(shape)
      if not np.isfinite(np_array).all():
        raise ValueError(f"Non-finite values found in tensor {name}.")
      tensors[name] = np_array
    return tensors

  async def load_multi_host(
      self, abstract_pytree: dict[str, Any]
  ) -> tuple[dict[str, Any], dict[str, float]]:
    """Loads tensors from a single safetensors file in multi-host mode.

    This method optimizes loading by:
    1. Partitioning tensors among hosts based on their file offsets to ensure
       each host performs a single contiguous read.
    2. Loading data into a transient array with a leading dummy 'hosts' axis
       to allow a whole tensor to be present on a single host and sharded
       exclusively across that host's local devices. This minimizes peak memory
       usage.
    3. Optional reduction of the global transient array to the target sharding.
       This can be skipped in cases like simple checkpoint conversion which
       requires no data processing.

    Args:
      abstract_pytree: A flat dictionary mapping tensor names to
        jax.ShapeDtypeStruct objects specifying target shape and sharding.

    Returns:
      A tuple containing:
        - A dictionary mapping tensor names to restored jax.Arrays.
        - A dictionary containing metrics (io_time, reshard_time).

    Raises:
      ValueError: If the number of devices is not uniform across hosts, or if a
        sharded tensor's first dimension is not divisible by the number of
        devices per host.
      KeyError: If a tensor in the abstract_pytree is not found in the
        safetensors file.
    """
    io_time = 0.0
    reshard_time = 0.0

    num_hosts = multihost.process_count()

    global_mesh, devices_by_host = _create_global_mesh()

    host_id = jax.process_index()

    t0 = time.time()
    header, data_start_offset = await self.read_header()
    io_time += time.time() - t0

    # Partition tensors among hosts based on file offsets for contiguous I/O.
    # TODO(b/496270336): Use the partial_load flag in Context to allow for
    # loading a subset of keys.
    bundles = _get_tensor_bundles(header, num_hosts)

    # Map each tensor to its owner host.
    tensor_to_owner = {}
    for h, bundle in enumerate(bundles):
      for name in bundle:
        tensor_to_owner[name] = h

    restored_pytree = {}

    t0 = time.time()
    bundle_bytes, bundle_start_offset = await self._read_bundle(
        bundles, host_id, header, data_start_offset
    )
    io_time += time.time() - t0

    if not context_lib.get_context().safetensors_options.ignore_load_sharding:
      max_shard_shape_per_dtype = _calculate_max_shard_shapes(
          abstract_pytree,
          header,
      )
      zero_buffers = _create_shared_zero_buffers(
          max_shard_shape_per_dtype, devices_by_host[host_id]
      )
    else:
      zero_buffers = {}

    ctx = _LoadContext(
        host_id=host_id,
        num_hosts=num_hosts,
        global_mesh=global_mesh,
        devices_by_host=devices_by_host,
        bundle_bytes=bundle_bytes,
        bundle_start_offset=bundle_start_offset,
        zero_buffers=zero_buffers,
    )

    # Process each tensor in the requested PyTree.
    for name, abstract_leaf in abstract_pytree.items():
      if name not in header:
        continue

      owner = tensor_to_owner.get(name, 0)
      info = header[name]
      target_sharding = abstract_leaf.sharding

      if context_lib.get_context().safetensors_options.ignore_load_sharding:
        restored_pytree[name] = _build_array_on_single_device(info, owner, ctx)
      else:
        global_transient_array = _build_transient_array(name, info, owner, ctx)
        t0 = time.time()
        restored_pytree[name] = _reshard_transient_array(
            global_transient_array, target_sharding, global_mesh
        )
        reshard_time += time.time() - t0

    return restored_pytree, {"io_time": io_time, "reshard_time": reshard_time}


class _MultiFileLoader:
  """Multi-file loader for Safetensors checkpoints."""

  def __init__(self, path: Path):
    self.path = path

  async def _get_loaders(self) -> list[_SingleFileLoader]:
    """Returns the list of SingleFileLoaders."""
    if await async_path.is_dir(self.path):
      paths = sorted(await async_path.glob(self.path, f"*{SAFETENSORS_SUFFIX}"))
    else:
      paths = [self.path]
    return [_SingleFileLoader(path) for path in paths]

  async def _load_single_host(self, abstract_pytree: dict[str, Any]) -> Any:
    """Loads a safetensors checkpoint on a single host."""
    # Return NumPy arrays.
    # Load from all files and merge.
    start = time.time()
    load_ops = []
    for loader in await self._get_loaders():
      load_ops.append(loader.load_single_host(abstract_pytree))

    restored_pytree = {}
    for file_tensors in await asyncio.gather(*load_ops):
      for name, arr in file_tensors.items():
        if name in restored_pytree:
          raise ValueError(f"Duplicate tensor {name} found in multiple files.")
        restored_pytree[name] = arr

    logging.info(
        "[safetensors][single-host] Loaded tensors in %.0fs",
        time.time() - start,
    )
    if not abstract_pytree:
      return restored_pytree

    start = time.time()
    for k in abstract_pytree:
      if k not in restored_pytree:
        raise KeyError(f"Tensor '{k}' not found in Safetensors checkpoint.")

    restored_pytree = {
        k: jax.device_put(
            restored_pytree[k],
            device=abstract_pytree[k].sharding,
        )
        for k in abstract_pytree
    }
    logging.info(
        "[safetensors][single-host] Host-to-device transfer in %.0fs",
        time.time() - start,
    )
    return restored_pytree

  async def _load_multi_host(
      self, abstract_pytree: dict[str, Any] | None
  ) -> Any:
    """Loads a safetensors checkpoint on multiple hosts."""
    if not abstract_pytree:
      raise ValueError(
          "abstract_pytree must be provided for multi-host loading."
      )

    loaders = await self._get_loaders()

    # Call load_multi_host on each loader concurrently.
    # Each loader handles loading from a single file.
    start = time.time()
    load_ops = []
    for loader in loaders:
      load_ops.append(loader.load_multi_host(abstract_pytree))

    restored_pytree = {}
    total_io_time = 0.0
    total_reshard_time = 0.0
    for file_tensors, metrics in await asyncio.gather(*load_ops):
      total_io_time += metrics["io_time"]
      total_reshard_time += metrics["reshard_time"]
      for name, arr in file_tensors.items():
        if name in restored_pytree:
          raise ValueError(f"Duplicate tensor {name} found in multiple files.")
        restored_pytree[name] = arr

    # Validate that all requested tensors were found in at least one file.
    for k in abstract_pytree:
      if k not in restored_pytree:
        raise KeyError(f"Tensor '{k}' not found in Safetensors checkpoint.")

    logging.info(
        "[safetensors][multi-host] Loaded and resharded %d tensors from %d"
        " files in %.2fs (io=%.2fs, reshard=%.2fs)",
        len(restored_pytree),
        len(loaders),
        time.time() - start,
        total_io_time,
        total_reshard_time,
    )

    return restored_pytree

  async def load_safetensors(
      self,
      abstract_pytree: dict[str, Any] | None = None,
  ) -> Any:
    """Calls the correct safetensors loading function."""
    if abstract_pytree is not None and not tree_utils.is_flat_dict(
        abstract_pytree
    ):
      raise ValueError("The PyTree is not a flat dictionary.")

    if multihost.process_count() > 1:
      return await self._load_multi_host(abstract_pytree)
    else:
      return await self._load_single_host(abstract_pytree)

  async def load_metadata(self):
    """Loads the metadata from a safetensors checkpoint."""
    start = time.time()
    metadata = {}
    custom_metadata = {}

    # Track the latest commit timestamp.
    commit_timestamp_nsecs = None

    for loader in await self._get_loaders():
      header, _ = await loader.read_header()
      stat = await async_path.async_stat(loader.path)
      ts = int(stat.mtime)
      if commit_timestamp_nsecs is None or ts > commit_timestamp_nsecs:
        commit_timestamp_nsecs = ts

      for name, info in header.items():
        if name == "__metadata__":
          # TODO(abhisekar): Consider warning on conflicting metadata keys.
          # If conflicting keys exist, last write wins.
          if info:
            custom_metadata.update(info)
          continue
        if name in metadata:
          raise ValueError(f"Duplicate tensor {name} found in multiple files.")
        shape, dtype = _get_array_properties(info)
        metadata[name] = jax.ShapeDtypeStruct(shape=shape, dtype=dtype)

    logging.info("[safetensors] Loaded metadata in %.0fs", time.time() - start)
    return metadata_types.CheckpointMetadata[dict[str, Any]](
        path=self.path,
        metadata={checkpoint_layout.PYTREE_CHECKPOINTABLE_KEY: metadata},
        commit_timestamp_nsecs=commit_timestamp_nsecs,
        custom_metadata=custom_metadata,
    )


class SafetensorsLayout(CheckpointLayout):
  """SafetensorsLayout.

  This class defines a class to handle Safetensors checkpoint formats. It
  inherits abstract methods from :py:class:`~.CheckpointLayout`.
  It performs a few core functions:
    - Resolves handlers for saving and loading.
    - Saves and loads checkpointables to/from individual subdirectories by
    delegating to the resolved handlers.
  """

  def __init__(self):
    if multihost.is_pathways_backend():
      raise ValueError(
          "SafetensorsLayout is not supported on Pathways backend."
      )

  async def metadata(
      self, path: Path
  ) -> metadata_types.CheckpointMetadata[dict[str, Any]]:
    """Returns the metadata of the SafeTensors checkpoint."""
    loader = _MultiFileLoader(path)
    return await loader.load_metadata()

  async def validate(self, path: Path):
    if await async_path.is_file(path):
      if path.suffix == SAFETENSORS_SUFFIX:
        return
      raise InvalidLayoutError(
          f"Failed to interpret path {path} as a SafeTensors checkpoint. A"
          " SafeTensors checkpoint must be a file with the"
          f" '{SAFETENSORS_SUFFIX}' suffix."
      )
    elif await async_path.is_dir(path):
      # Check if it contains any .safetensors files
      files = list(await async_path.glob(path, f"*{SAFETENSORS_SUFFIX}"))
      if not files:
        raise InvalidLayoutError(
            f"Directory {path} does not contain any '{SAFETENSORS_SUFFIX}'"
            " files."
        )
    else:
      raise InvalidLayoutError(
          f"Path {path} is neither a file nor a directory or does not exist."
      )

  async def get_checkpointable_names(self, path: Path) -> list[str]:
    return [checkpoint_layout.PYTREE_CHECKPOINTABLE_KEY]

  async def validate_pytree(
      self, path: Path, checkpointable_name: str | None
  ) -> None:
    return

  async def load_pytree(
      self,
      path: Path,
      checkpointable_name: str | None = None,
      abstract_pytree: Any | None = None,
  ) -> Awaitable[Any]:
    """Loads a NumPy checkpoint file.

    If `abstract_pytree` is provided, it attempts to load numpy arrays as
    sharded `jax.Arrays` onto devices.

    Args:
      path: The path to load the checkpoint from.
      checkpointable_name: The name of the pytree checkpointable to load,
        unsused in this case.
      abstract_pytree: An optional PyTree of abstract arrays specifying sharding
        information.

    Returns:
      An awaitable containing the loaded PyTree.
    """
    del checkpointable_name
    self._loader = _MultiFileLoader(path)
    return self._loader.load_safetensors(abstract_pytree)

  async def save(
      self,
      path: types.PathAwaitingCreation,
      *,
      checkpointables: dict[str, Any],
  ) -> Awaitable[None]:
    """Saves the checkpoint to the given directory."""
    raise NotImplementedError(
        "Saving to Safetensors format is not supported yet."
    )
