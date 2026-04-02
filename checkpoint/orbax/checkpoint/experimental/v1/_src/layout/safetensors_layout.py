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

"""Safetensors checkpoint format layout."""

import asyncio
import concurrent.futures
import json
import mmap
import os
from typing import Any, Awaitable, Sequence

from absl import logging
from google.cloud import storage
from google.cloud.storage import transfer_manager
import jax
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint._src.arrays import numpy_utils
from orbax.checkpoint._src.path import async_path
from orbax.checkpoint.experimental.v1._src.context import context
from orbax.checkpoint.experimental.v1._src.layout import checkpoint_layout
from orbax.checkpoint.experimental.v1._src.metadata import types as metadata_types
from orbax.checkpoint.experimental.v1._src.path import types


CheckpointLayout = checkpoint_layout.CheckpointLayout
InvalidLayoutError = checkpoint_layout.InvalidLayoutError
Path = types.Path

HEADER_NUM_BYTES = 8
SAFETENSORS_SUFFIX = ".safetensors"
_HEADER_CACHE = {}


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
      "F8_E8M0": jnp.float8_e8m0fnu,
      "F8_E4M3": jnp.float8_e4m3,
      "F8_E5M2": jax.numpy.float8_e5m2,
      "F4": "float4_e2m1fn_x2 (specialized ML dtype)",
  }


async def create_loading_plan(
    paths: Sequence[Path],
    abstract_pytree: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[float]]:
  """Creates a plan for loading tensors in batches based on memory limits."""

  async def _fetch_header(p):
    h, _ = await _read_safetensors_header(p)
    return p, h

  if abstract_pytree is None:
    abstract_pytree = {}
    total_tensors = 0
    for file_path in paths:
      header, _ = await _read_safetensors_header(file_path)
      for tensor in header:
        if tensor != "__metadata__":
          total_tensors += 1
          shape, dtype = _get_array_properties(header[tensor])
          abstract_pytree[tensor] = jax.ShapeDtypeStruct(
              shape=shape,
              dtype=dtype,
          )

  all_headers = await asyncio.gather(*[_fetch_header(p) for p in paths])
  batches = []
  batches_size = []
  current_batch = {}
  current_batch_size = 0
  ctx = context.get_context()
  max_bytes = ctx.array_options.loading.concurrent_bytes
  if max_bytes is None:
    max_bytes = float("inf")
  total_tensors = 0
  for _, header in all_headers:
    for tensor_name, leaf_meta in header.items():
      if tensor_name in abstract_pytree:
        total_tensors += 1
        if "shape" in leaf_meta and "dtype" in leaf_meta:
          shape, dtype = _get_array_properties(leaf_meta)
          dtype_size = np.dtype(dtype).itemsize
          size = np.prod(shape) * dtype_size
          if current_batch_size + size > max_bytes:
            logging.info(
                "Batch size: %.2f MB", current_batch_size / (1024 * 1024)
            )
            total_tensors = 0
            batches.append(current_batch)
            batches_size.append(current_batch_size / (1024 * 1024))
            current_batch = {}
            current_batch_size = 0
          current_batch[tensor_name] = jax.ShapeDtypeStruct(
              shape=shape,
              dtype=dtype,
          )
          current_batch_size += size
  if current_batch:
    logging.info("Batch size: %.2f MB", current_batch_size / (1024 * 1024))
    batches.append(current_batch)
    batches_size.append(current_batch_size / (1024 * 1024))
  return batches, batches_size


async def _read_safetensors_header(path: Path) -> tuple[dict[str, Any], int]:
  """Reads a safetensors file header, returning header and data start offset."""
  path_str = str(path)
  if path_str in _HEADER_CACHE:
    return _HEADER_CACHE[path_str]

  async with async_path.open_file(path, mode="rb") as f:
    header_size_bytes = await f.read(HEADER_NUM_BYTES)
    if not header_size_bytes:
      raise ValueError("Could not read header size from safetensors file.")

    header_size = int.from_bytes(header_size_bytes, byteorder="little")
    header_bytes = await f.read(header_size)
    if len(header_bytes) != header_size:
      raise ValueError("Could not read header content from safetensors file.")

    header = json.loads(header_bytes)
    data_start_offset = HEADER_NUM_BYTES + header_size

    _HEADER_CACHE[path_str] = header, data_start_offset
    return header, data_start_offset


def _get_array_properties(info: dict[str, Any]) -> tuple[tuple[int, ...], Any]:
  """Parses shape and `dtype` from a safetensors tensor header."""
  try:
    dtype_str = info["dtype"]
    dtype = _get_dtypes()[dtype_str]
  except KeyError as e:
    raise ValueError(f"Unsupported dtype in SafeTensors header: {e}") from e
  shape = tuple(info["shape"])
  return shape, dtype


def _process_bytes_to_jax(
    tensor_bytes: bytes,
    tensor_name: str,
    abstract_leaf: Any,
    header: dict[str, Any],
) -> jax.Array:
  """Universal helper to safely parse bytes into sharded or host JAX arrays."""
  stored_shape, stored_dtype = _get_array_properties(header[tensor_name])
  sharding = abstract_leaf.sharding
  target_shape = abstract_leaf.shape
  target_dtype = abstract_leaf.dtype
  np_array = np.frombuffer(tensor_bytes, dtype=stored_dtype).reshape(
      stored_shape
  )
  if np_array.dtype != target_dtype:
    np_array = np_array.astype(target_dtype)

  # Fallback to Host RAM if no sharding is provided
  if sharding is None:
    arr = jax.device_put(np_array.copy())
    del np_array
    return arr

  # Distributed Sharding
  device_indices_map = sharding.addressable_devices_indices_map(target_shape)
  device_map = list()

  for device in sharding.addressable_devices:
    if device in device_indices_map:
      idx = device_indices_map[device]
      resolved_idx = numpy_utils.resolve_slice(idx, stored_shape)
      shard_np = np_array[resolved_idx]
      device_map.append(jax.device_put(shard_np.copy(), device))
      del shard_np

  del np_array
  return jax.make_array_from_single_device_arrays(
      target_shape, sharding, device_map
  )


async def _load_safetensors_on_device_local(
    path: Path, abstract_pytree: dict[str, Any]
) -> dict[str, jax.Array]:
  """Fast path for local NVMe/SSD files using zero-copy memory mapping."""
  header, data_start_offset = await _read_safetensors_header(path)
  restored_tensors = {}
  logging.info("--- Starting local load for %s ---", path.name)

  with open(path, "rb") as f:
    with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
      for tensor_name, abstract_leaf in abstract_pytree.items():
        if tensor_name not in header:
          continue

        start_offset, end_offset = header[tensor_name]["data_offsets"]
        absolute_start = data_start_offset + start_offset
        absolute_end = data_start_offset + end_offset

        # Wrap in memoryview to ensure zero-copy bridging to NumPy
        tensor_bytes = memoryview(mm)[absolute_start:absolute_end]

        jax_array = _process_bytes_to_jax(
            tensor_bytes, tensor_name, abstract_leaf, header
        )
        restored_tensors[tensor_name] = jax_array

        del tensor_bytes

      # Ensure hardware has ingested the data before the mmap file lock is
      # released
      jax.block_until_ready(restored_tensors)

  return restored_tensors


def _gcs_bytes_to_jax_arrays(
    span_bytes: bytes | memoryview,
    tensors_to_load: dict[str, Any],
    min_start: int,
    header: dict[str, Any],
) -> dict[str, jax.Array]:
  """Converts bytes to JAX arrays based on tensor metadata."""
  restored_tensors = {}
  memory_span = memoryview(span_bytes)
  for t_name, (abstract_leaf, start, end) in tensors_to_load.items():
    rel_start = start - min_start
    rel_end = end - min_start
    tensor_bytes = memory_span[rel_start:rel_end]
    jax_array = _process_bytes_to_jax(
        tensor_bytes, t_name, abstract_leaf, header
    )
    restored_tensors[t_name] = jax_array
    del tensor_bytes
  jax.block_until_ready(restored_tensors)
  return restored_tensors


def load_safetensors_on_device_gcsfuse(
    gcs_path_str: str,
    tensors_to_load: dict[str, Any],
    data_start_offset: int,
    min_start: int,
    max_end: int,
    header: dict[str, Any],
) -> dict[str, jax.Array]:
  """High-bandwidth parallel downloader for Google Cloud Storage."""
  logging.info(
      "Loading data into JAX arrays from gcsfuse mounted file %s...",
      gcs_path_str,
  )
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
      chunk_bytes = os.pread(f.fileno(), chunk_size, offset)
      if not chunk_bytes:
        raise EOFError(
            f"Unexpected end of file at offset {offset}. Expected"
            f" {length} total bytes."
        )
      return chunk_bytes
  ctx = context.get_context()
  max_workers = ctx.multiprocessing_options.primary_host
  if max_workers == 0:
    max_workers = 16

  # 2. Execute the parallel reads
  with concurrent.futures.ThreadPoolExecutor(
      max_workers=max_workers
  ) as executor:
    read_chunks = list(executor.map(_read_single_chunk, chunks))

  span_bytes = b"".join(read_chunks)
  restored_tensors = _gcs_bytes_to_jax_arrays(
      span_bytes, tensors_to_load, min_start, header
  )
  del span_bytes
  return restored_tensors


def load_safetensors_on_device_gcs(
    bucket_name: str,
    blob_name: str,
    tensors_to_load: dict[str, Any],
    data_start_offset: int,
    min_start: int,
    max_end: int,
    header: dict[str, Any],
) -> dict[str, jax.Array]:
  """High-bandwidth parallel downloader for Google Cloud Storage."""
  logging.info(
      "Loading data into JAX arrays from GCS using transfer manager..."
  )
  client = storage.Client()
  blob = client.bucket(bucket_name).blob(blob_name)

  safe_temp_name = blob_name.replace("/", "_")
  ram_disk_path = f"/dev/shm/{safe_temp_name}_temp.bin"

  transfer_manager.download_chunks_concurrently(
      blob,
      ram_disk_path,
      chunk_size=1024 * 1024 * 1024,
      max_workers=16,
      worker_type="process",
  )
  with open(ram_disk_path, "rb") as f:
    with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
      span_bytes = memoryview(mm)[
          data_start_offset + min_start : data_start_offset + max_end
      ]
      restored_tensors = _gcs_bytes_to_jax_arrays(
          span_bytes, tensors_to_load, min_start, header
      )
      del span_bytes
  # 5. Clean up the RAM disk
  os.remove(ram_disk_path)
  return restored_tensors


async def _load_safetensors_on_device_cloud(
    path: Path, abstract_pytree: dict[str, Any]
) -> dict[str, jax.Array]:
  """High-bandwidth parallel downloader for Google Cloud Storage."""
  path_str = str(path)
  gcs_path_str = None
  if path_str.startswith("gs://"):
    gcs_path_str = "/gcs/" + path_str[5:]
  elif path_str.startswith("/big" + "store/"):
    gcs_path_str = "/gcs/" + path_str[10:]

  header, data_start_offset = await _read_safetensors_header(path)

  path_str = str(path)

  if path_str.startswith("/big" + "store/"):
    path_str = "gs://" + path_str[10:]

  if not path_str.startswith("gs://"):
    raise ValueError(f"Unsupported remote path format: {path_str}")

  bucket_name, blob_name = path_str[5:].split("/", 1)

  min_start = float("inf")
  max_end = 0
  tensors_to_load = {}

  # 1. Calculate the exact Bounding Box of the batch
  for t_name, abstract_leaf in abstract_pytree.items():
    if t_name in header:
      start, end = header[t_name]["data_offsets"]
      if start < min_start:
        min_start = start
      if end > max_end:
        max_end = end
      tensors_to_load[t_name] = (abstract_leaf, start, end)

  if not tensors_to_load:
    return {}

  if gcs_path_str and os.path.exists(gcs_path_str):
    return load_safetensors_on_device_gcsfuse(
        gcs_path_str,
        tensors_to_load,
        data_start_offset,
        min_start,
        max_end,
        header,
    )
  return load_safetensors_on_device_gcs(
      bucket_name,
      blob_name,
      tensors_to_load,
      data_start_offset,
      min_start,
      max_end,
      header,
  )


async def _load_safetensors_on_device(
    path: Path, abstract_pytree: dict[str, Any]
) -> dict[str, jax.Array]:
  """Intelligent Router to load Safetensors based on storage topology."""
  flat_abstract_with_path, _ = jax.tree.flatten_with_path(abstract_pytree)
  for key_path, _ in flat_abstract_with_path:
    if len(key_path) != 1 or not isinstance(
        key_path[0], jax.tree_util.DictKey
    ):
      raise ValueError(
          "The PyTree is not a flat dictionary. Key path: {key_path}"
      )

  is_local = False
  try:
    with open(path, "rb"):
      is_local = True
  except (FileNotFoundError, OSError):
    pass

  if is_local:
    result = await _load_safetensors_on_device_local(path, abstract_pytree)
  else:
    result = await _load_safetensors_on_device_cloud(path, abstract_pytree)
  return result


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
    pass

  async def metadata(
      self, path: Path
  ) -> metadata_types.CheckpointMetadata[dict[str, Any]]:
    """Returns the metadata of the SafeTensors checkpoint."""
    files = [path]
    if await async_path.is_dir(path):
      files = await async_path.glob(path, f"*{SAFETENSORS_SUFFIX}")
      files = sorted(files)

    metadata = {}
    custom_metadata = {}

    # Track the latest commit timestamp.
    commit_timestamp_nsecs = None
    async def _fetch_header(p):
      h, _ = await _read_safetensors_header(p)
      return p, h

    header_results = await asyncio.gather(*[_fetch_header(p) for p in files])

    for path, header in header_results:
      stat = await async_path.async_stat(path)
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

    return metadata_types.CheckpointMetadata[dict[str, Any]](
        metadata={checkpoint_layout.PYTREE_CHECKPOINTABLE_KEY: metadata},
        commit_timestamp_nsecs=commit_timestamp_nsecs,
        custom_metadata=custom_metadata,
    )

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

  async def validate_pytree(
      self, path: Path, checkpointable_name: str | None
  ) -> None:
    return

  async def load_pytree(
      self,
      path: Path | None = None,
      checkpointable_name: str | None = None,
      abstract_pytree: Any | None = None,
  ) -> Awaitable[dict[str, jax.Array]]:
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
    return _load_safetensors_on_device(path, abstract_pytree)

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
