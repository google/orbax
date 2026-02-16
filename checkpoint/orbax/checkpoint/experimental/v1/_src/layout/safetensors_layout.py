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
import collections
import json
import mmap
import time
from typing import Any, Awaitable, List, Sequence, Tuple, cast

from google.cloud import storage
import jax
import numpy as np
from orbax.checkpoint._src.arrays import numpy_utils
from orbax.checkpoint._src.path import async_path
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
      "BF16": jax.numpy.bfloat16,
      "F8_E4M3": jax.numpy.float8_e4m3fn,
      "F8_E5M2": jax.numpy.float8_e5m2,
      "F8_E8M0": "float8_e8m0fnu (specialized ML dtype)",
      "F4": "float4_e2m1fn_x2 (specialized ML dtype)",
  }


async def _get_safetensors_file_list(path: Path) -> Sequence[Path]:
  """Returns a list of safetensors files in the given path."""
  if await async_path.is_dir(path):
    files = await async_path.glob(path, f"*{SAFETENSORS_SUFFIX}")
    return sorted(files)
  return [path]


async def get_tensor_to_path_indexing(path):
  """Returns a mapping from tensor name to safetensors file."""
  path_ = Path(str(path) + "/model.safetensors.index.json")
  async with async_path.open_file(path_, mode="rb") as f:
    raw_data = await f.read()
    index_data = json.loads(raw_data)
  return index_data["weight_map"]


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


async def _read_non_contiguous_slice(
    f: async_path.AsyncFile,
    idx: tuple[slice, ...],
    stored_shape: tuple[int, ...],
    stored_dtype: np.dtype,
    tensor_file_offset: int,
) -> np.ndarray:
  """Reads a slice of a tensor from a file.

  This function solves the problem of reading a multi-dimensional slice from an
  array where the slice's data is not stored as a single, contiguous block in
  the file. It does so by recursively "walking" the dimensions of the slice.

  Args:
      f: The asynchronous file object (binary read mode)
      idx: A tuple of slice objects representing the n-dimensional slice to
        read.
      stored_shape: The shape of the tensor.
      stored_dtype: The `dtype` of the tensor.
      tensor_file_offset: The starting byte offset of the tensor's data within
        the file.

  Returns:
      The specific tensor slice.
  """
  # Handle 0-d scalar case
  if not idx:
    await f.seek(tensor_file_offset)
    num_bytes = np.dtype(stored_dtype).itemsize
    scalar_bytes = await f.read(num_bytes)
    # Reshape to () to create a 0-D NumPy array.
    return np.frombuffer(scalar_bytes, dtype=stored_dtype).reshape(())

  itemsize = np.dtype(stored_dtype).itemsize

  # Calculate the byte strides for the full tensor. The stride for a
  # dimension is the number of bytes to "jump" to get to the next element
  # in that dimension while keeping all other indices the same.
  global_strides = [itemsize] * len(stored_shape)
  for i in range(len(stored_shape) - 2, -1, -1):
    global_strides[i] = global_strides[i + 1] * stored_shape[i + 1]

  # Pre-calculate which dimensions are fully selected.
  is_full_dim = [False] * len(stored_shape)
  for i, s in enumerate(idx):
    if s.start == 0 and s.stop == stored_shape[i] and s.step == 1:
      is_full_dim[i] = True

  async def _read_slice_recursively(dim: int, base_offset: int) -> bytes:
    # If all remaining dimensions are fully selected, we can read the entire
    # contiguous block for the current dimension's slice.
    if dim == len(stored_shape) - 1 or all(is_full_dim[dim + 1 :]):
      s = idx[dim]
      start = base_offset + s.start * global_strides[dim]
      num_bytes = (s.stop - s.start) * global_strides[dim]

      await f.seek(tensor_file_offset + start)
      return cast(bytes, await f.read(num_bytes))

    # For all other dimensions, iterate through the indices
    # of the slice and make a recursive call for the next dimension.
    s = idx[dim]
    chunks = []
    for i in range(s.start, s.stop):
      offset = base_offset + i * global_strides[dim]
      chunk = await _read_slice_recursively(dim + 1, offset)
      chunks.append(chunk)

    return b"".join(chunks)

  # Start the recursive reading process from the first dimension.
  slice_bytes = await _read_slice_recursively(dim=0, base_offset=0)
  shard_shape = numpy_utils.slice_shape(idx)
  return np.frombuffer(slice_bytes, dtype=stored_dtype).reshape(shard_shape)


async def _load_safetensors_as_numpy(path: Path) -> dict[str, np.ndarray]:
  """Loads tensors from a safetensors file into host NumPy arrays."""
  header, data_start_offset = await _read_safetensors_header(path)
  tensors = {}
  async with async_path.open_file(path, mode="rb") as f:
    await f.seek(data_start_offset)
    data_bytes = await f.read()
  for name, info in header.items():
    if name == "__metadata__":
      continue
    shape, dtype = _get_array_properties(info)
    start_offset, end_offset = info["data_offsets"]
    tensor_bytes = data_bytes[start_offset:end_offset]
    np_array = np.frombuffer(tensor_bytes, dtype=dtype).reshape(shape)
    tensors[name] = np_array
  return tensors


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
    arr = jax.device_put(np_array)
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
  results_to_block = list()

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
        results_to_block.append(jax_array)

        del tensor_bytes

    # Ensure hardware has ingested the data before the mmap file lock is
    # released
    jax.block_until_ready(results_to_block)

  return restored_tensors


async def _load_safetensors_on_device_gcs(
    path: Path, abstract_pytree: dict[str, Any]
) -> dict[str, jax.Array]:
  """High-bandwidth parallel downloader for Google Cloud Storage."""
  header, data_start_offset = await _read_safetensors_header(path)
  restored_tensors = {}

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
      if start < min_start: min_start = start
      if end > max_end: max_end = end
      tensors_to_load[t_name] = (abstract_leaf, start, end)

  if not tensors_to_load:
    return restored_tensors

  span_size = max_end - min_start
  absolute_start = data_start_offset + min_start
  absolute_end = absolute_start + span_size - 1

  start_read_time = time.time()

  client = storage.Client()
  blob = client.bucket(bucket_name).blob(blob_name)
  span_buffer = blob.download_as_bytes(start=absolute_start, end=absolute_end)

  print(
      f"----- [{path.name}] Single-shot network read from GCS finished:"
      f" {span_size / (1024 * 1024):.2f} MB in"
      f" {time.time() - start_read_time:.2f}s Throughput:"
      f" ({span_size / (time.time() - start_read_time) / (1024 * 1024):.2f}"
      " MB/s)"
  )

  span_bytes = memoryview(span_buffer)
  for t_name, (abstract_leaf, start, end) in tensors_to_load.items():
    rel_start = start - min_start
    rel_end = end - min_start

    tensor_bytes = span_bytes[rel_start:rel_end]
    restored_tensors[t_name] = _process_bytes_to_jax(
        tensor_bytes, t_name, abstract_leaf, header
    )
    del tensor_bytes

  # 5. Destroy the large buffer immediately after the GPUs take over
  del span_bytes
  del span_buffer

  return restored_tensors


async def _load_safetensors_on_device(
    path: Path, abstract_pytree: dict[str, Any]
) -> dict[str, jax.Array]:
  """Intelligent Router to load Safetensors based on storage topology."""
  is_local = False
  try:
    with open(path, "rb"):
      is_local = True
  except (FileNotFoundError, OSError):
    pass

  if is_local:
    result = await _load_safetensors_on_device_local(path, abstract_pytree)
  else:
    result = await _load_safetensors_on_device_gcs(path, abstract_pytree)
  return result


async def _load_safetensors_on_device_old(
    path: Path, abstract_pytree: dict[str, Any]
) -> dict[str, jax.Array]:
  """Loads tensors from a safetensors file into on-device JAX arrays."""
  header, data_start_offset = await _read_safetensors_header(path)
  start_read_time = time.time()
  restored_pytree = {}
  async with async_path.open_file(path, mode="rb") as f:
    for tensor_name, abstract_leaf in abstract_pytree.items():
      if tensor_name not in header:
        raise KeyError(
            f"Tensor '{tensor_name}' not found in safetensors header of {path}."
        )

      stored_shape, stored_dtype = _get_array_properties(header[tensor_name])
      st_data_offsets = header[tensor_name]["data_offsets"]
      sharding = abstract_leaf.sharding
      target_shape = abstract_leaf.shape
      target_dtype = abstract_leaf.dtype

      if sharding is None:
        start_offset, end_offset = st_data_offsets
        num_bytes = end_offset - start_offset
        await f.seek(data_start_offset + start_offset)
        tensor_bytes = await f.read(num_bytes)
        np_array = np.frombuffer(tensor_bytes, dtype=stored_dtype).reshape(
            stored_shape
        )
        if np_array.dtype != target_dtype:
          np_array = np_array.astype(target_dtype)
        restored_pytree[tensor_name] = jax.device_put(np_array)
        continue

      device_indices_map = sharding.addressable_devices_indices_map(
          target_shape
      )

      device_map = []
      for device in device_indices_map:
        idx = device_indices_map[device]
        resolved_idx = numpy_utils.resolve_slice(idx, stored_shape)
        shard_shape = numpy_utils.slice_shape(resolved_idx)

        shard_np = await _read_non_contiguous_slice(
            f,
            resolved_idx,
            stored_shape,
            stored_dtype,
            st_data_offsets[0] + data_start_offset,
        )
        shard_np = shard_np.reshape(shard_shape)  # pytype: disable=attribute-error

        if shard_np.dtype != target_dtype:
          shard_np = shard_np.astype(target_dtype)

        device_map.append(jax.device_put(shard_np, device))

      restored_pytree[tensor_name] = jax.make_array_from_single_device_arrays(
          target_shape, sharding, device_map
      )
  print(
      f"----- [{path.name}] Network read finished, fetched"
      f" {time.time() - start_read_time:.2f}s"
  )
  return restored_pytree


async def _load_safetensors(
    paths: Sequence[Path], abstract_pytree: dict[str, Any] | None = None
) -> Any:
  """Calls the correct safetensors loading function."""

  if abstract_pytree is None:
    # Return NumPy arrays.
    # Load from all files and merge.
    tensors = {}
    for path in paths:
      file_tensors = await _load_safetensors_as_numpy(path)
      for name, arr in file_tensors.items():
        if name in tensors:
          raise ValueError(f"Duplicate tensor {name} found in multiple files.")
        tensors[name] = arr
    restored_pytree = tensors
  else:
    # Return on-device JAX arrays.
    flat_abstract_with_path, _ = jax.tree.flatten_with_path(abstract_pytree)
    for key_path, _ in flat_abstract_with_path:
      if len(key_path) != 1 or not isinstance(
          key_path[0], jax.tree_util.DictKey
      ):
        raise ValueError(
            "The PyTree is not a flat dictionary. Key path: {key_path}"
        )

    # 1. Map tensor names to files
    # TODO(abhisekar): This could be improved - it is fairly common to only read
    # a few weights for debugging purposes, and this should not require opening
    # every safetensors file, many of which may be discarded. The index.json
    # (https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/model.safetensors.index.json)
    # has a `weight_map` that maps tensor names to file paths. From the
    # abstract tree, we can look up only the keys that are actually needed to
    # load using the index.json.

    start_time = time.time()
    tensor_to_path = {}
    file_to_path = {}

    for file_ in paths:
      file_to_path[str(Path(file_).name)] = file_

    indexing_results = await get_tensor_to_path_indexing(paths[0].parent)

    for name, path in indexing_results.items():
      if name in tensor_to_path:
        raise ValueError(f"Duplicate tensor {name} found in multiple files.")
      tensor_to_path[name] = file_to_path[str(path)]

    end_time = time.time()
    print(
        f"----- Mapping tensor names to files took"
        f" {end_time - start_time:.2f} seconds."
    )

    # 2. Split abstract_pytree by file
    file_abstract_trees = collections.defaultdict(
        dict
    )  # Path -> dict[str, abstract_leaf]
    for key_path, abstract_leaf in flat_abstract_with_path:
      tensor_name = str(key_path[0].key)
      if tensor_name not in tensor_to_path:
        raise KeyError(
            f"Tensor '{tensor_name}' not found in any safetensors file."
        )

      path = tensor_to_path[tensor_name]
      file_abstract_trees[path][tensor_name] = abstract_leaf

    # 3. Load from each file
    restored_pytree = {}
    for path, sub_tree in file_abstract_trees.items():
      sub_restored = await _load_safetensors_on_device(path, sub_tree)
      restored_pytree.update(sub_restored)

    end_time = time.time()
    print(
        f"----- [{paths[0].name}] Loading {len(file_abstract_trees)} files took"
        f" {end_time - start_time:.2f} seconds."
    )

  return restored_pytree


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

  async def create_loading_plan(
      self,
      path: Path,
      max_batch_size_gb: float,
  ) -> Tuple[List[List[Tuple[str, ...]]], List[float]]:
    """Saves the checkpoint to the given directory."""
    files = await _get_safetensors_file_list(path)

    async def _fetch_header(p):
      h, _ = await _read_safetensors_header(p)
      return p, h

    all_headers = await asyncio.gather(*[_fetch_header(p) for p in files])
    batches = []
    batches_size = []
    current_batch = []
    current_batch_size = 0
    max_bytes = max_batch_size_gb * (1024**3)
    for _, header in all_headers:
      for tensor_name, leaf_meta in header.items():
        if tensor_name == "__metadata__":
          continue
        else:
          if "shape" in leaf_meta and "dtype" in leaf_meta:
            shape, dtype = _get_array_properties(leaf_meta)
            dtype_size = np.dtype(dtype).itemsize
            size = np.prod(shape) * dtype_size
          else:
            size = 0
          if current_batch_size + size > max_bytes and current_batch:
            batches.append(current_batch)
            batches_size.append(current_batch_size)
            current_batch = []
            current_batch_size = 0

          current_batch.append(tensor_name)
          current_batch_size += size

    if current_batch:
      batches.append(current_batch)
      batches_size.append(current_batch_size)
    return batches, batches_size

  async def metadata(
      self, path: Path
  ) -> metadata_types.CheckpointMetadata[dict[str, Any]]:
    """Returns the metadata of the SafeTensors checkpoint."""
    files = await _get_safetensors_file_list(path)
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
    files = await _get_safetensors_file_list(path)
    result = await _load_safetensors(files, abstract_pytree)
    return result

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
