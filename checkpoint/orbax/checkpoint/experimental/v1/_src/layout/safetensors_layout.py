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
import concurrent
import json
import mmap
import os
import time
from typing import Any, Awaitable, Sequence

from absl import logging
from google.cloud import storage
from google.cloud.storage import transfer_manager
import jax
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
      "BF16": jax.numpy.bfloat16,
      "F8_E4M3": jax.numpy.float8_e4m3fn,
      "F8_E5M2": jax.numpy.float8_e5m2,
      "F8_E8M0": "float8_e8m0fnu (specialized ML dtype)",
      "F4": "float4_e2m1fn_x2 (specialized ML dtype)",
  }


async def _get_safetensors_file_list(
    path: Path, abstract_pytree: Any | None = None
) -> tuple[Sequence[Path], dict[str, Any]]:
  """Returns a list of safetensors files in the given path."""
  files = [path]
  if await async_path.is_dir(path):
    files = await async_path.glob(path, f"*{SAFETENSORS_SUFFIX}")
    files = sorted(files)
  tensor_to_path = await get_tensor_to_path_indexing(files)
  if abstract_pytree is None:
    print("--- No abstract pytree provided, inferring from files ---")
    abstract_pytree = {}
    total_tensors = 0
    for file_path in files:
      header, _ = await _read_safetensors_header(file_path)
      for tensor in header:
        if tensor != "__metadata__":
          total_tensors += 1
          shape, dtype = _get_array_properties(header[tensor])
          abstract_pytree[tensor] = jax.ShapeDtypeStruct(
              shape=shape,
              dtype=dtype,
          )
    print(f"--- Total tensors found: {total_tensors} ---")
    return files, abstract_pytree
  files_to_load = set()
  for tensor_name, _ in abstract_pytree.items():
    if tensor_name in tensor_to_path:
      files_to_load.add(tensor_to_path[tensor_name])
  # logging.info("--- Found safetensors file: %s ---", list(files_to_load))
  print("--- Found safetensors file: %s ---", list(files_to_load))
  return list(files_to_load), abstract_pytree


async def get_tensor_to_path_indexing(paths: Sequence[Path]):
  """Returns a mapping from tensor name to safetensors file."""

  file_to_path = {}
  for file_ in paths:
    file_to_path[str(Path(file_).name)] = file_

  path_ = Path(str(paths[0].parent) + "/model.safetensors.index.json")

  tensor_to_path = {}
  if not await async_path.exists(path_):
    for path in paths:
      header, _ = await _read_safetensors_header(path)
      for name in header:
        if name == "__metadata__":
          continue
        if name in tensor_to_path:
          raise ValueError(f"Duplicate tensor {name} found in multiple files.")
        tensor_to_path[name] = path
    return tensor_to_path

  async with async_path.open_file(path_, mode="rb") as f:
    raw_data = await f.read()
    index_data = json.loads(raw_data)

  for name, path in index_data["weight_map"].items():
    if name in tensor_to_path:
      raise ValueError(f"Duplicate tensor {name} found in multiple files.")
    tensor_to_path[name] = file_to_path[str(path)]
  return tensor_to_path


async def create_loading_plan(
    paths: Sequence[Path],
    abstract_pytree: dict[str, Any] | None = None,
) -> tuple[list[list[str]], list[float]]:
  """Creates a plan for loading tensors in batches based on memory limits."""

  async def _fetch_header(p):
    h, _ = await _read_safetensors_header(p)
    return p, h

  all_headers = await asyncio.gather(*[_fetch_header(p) for p in paths])
  batches = []
  batches_size = []
  current_batch = []
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
        size = 0
        if "shape" in leaf_meta and "dtype" in leaf_meta:
          shape, dtype = _get_array_properties(leaf_meta)
          dtype_size = np.dtype(dtype).itemsize
          size = np.prod(shape) * dtype_size
        if current_batch_size + size > max_bytes:
          # logging.info(
          #     "Batch size: %.2f MB", current_batch_size / (1024 * 1024)
          # )
          print(
              "Batch index: %d, Batch size: %.2f MB (Total tensors: %d)"
              % (
                  len(batches),
                  current_batch_size / (1024 * 1024),
                  total_tensors,
              )
          )
          total_tensors = 0
          batches.append(current_batch)
          batches_size.append(current_batch_size / (1024 * 1024))
          current_batch = []
          current_batch_size = 0
        current_batch.append(tensor_name)
        current_batch_size += size
  if current_batch:
    # logging.info("Batch size: %.2f MB", current_batch_size / (1024 * 1024))
    print(
        "Batch index: %d, Batch size: %.2f MB (Total tensors: %d)"
        % (len(batches), current_batch_size / (1024 * 1024), total_tensors)
    )
    batches.append(current_batch)
    batches_size.append(current_batch_size / (1024 * 1024))
  return batches, batches_size


async def create_loading_plan_by_file(
    paths: Sequence[Path],
    abstract_pytree: dict[str, Any] | None = None,
) -> tuple[list[list[str]], list[float]]:
  """Creates a plan for loading tensors in batches based on memory limits."""

  async def _fetch_header(p):
    h, _ = await _read_safetensors_header(p)
    return p, h

  all_headers = await asyncio.gather(*[_fetch_header(p) for p in paths])
  batches = []
  batches_size = []
  current_batch = []
  current_batch_size = 0
  total_tensors = 0
  ctx = context.get_context()
  max_bytes = ctx.array_options.loading.concurrent_bytes
  if max_bytes is None:
    max_bytes = float("inf")
  for _, header in all_headers:
    current_file_size = 0
    current_file_tensors = []
    for tensor_name, leaf_meta in header.items():
      if tensor_name in abstract_pytree:
        size = 0
        total_tensors += 1
        if "shape" in leaf_meta and "dtype" in leaf_meta:
          shape, dtype = _get_array_properties(leaf_meta)
          dtype_size = np.dtype(dtype).itemsize
          size = np.prod(shape) * dtype_size
        current_file_tensors.append(tensor_name)
        current_file_size += size
    if current_batch_size + current_file_size > max_bytes:
      # logging.info("Batch size: %.2f MB", current_batch_size / (1024 * 1024))
      print(
          "Batch index: %d, Batch size: %.2f MB (Total tensors: %d)"
          % (
              len(batches),
              current_batch_size / (1024 * 1024),
              total_tensors,
          )
      )
      total_tensors = 0
      batches.append(current_batch)
      batches_size.append(current_batch_size)
      current_batch = current_file_tensors
      current_batch_size = current_file_size
    else:
      current_batch.extend(current_file_tensors)
      current_batch_size += current_file_size
  if current_batch:
    # logging.info("Batch size: %.2f MB", current_batch_size / (1024 * 1024))
    print(
        "Batch index: %d, Batch size: %.2f MB (Total tensors: %d)"
        % (len(batches), current_batch_size / (1024 * 1024), total_tensors)
    )
    batches.append(current_batch)
    batches_size.append(current_batch_size)
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
  dim1, dim2 = stored_shape
  try:
    np_array = np.frombuffer(tensor_bytes, dtype=stored_dtype).reshape(
        stored_shape
    )
  except ValueError as e:
    raise ValueError(
        f"Failed to convert bytes to JAX array for tensor {tensor_name}:"
        f" stored shape {stored_shape}, stored dtype {stored_dtype}"
        f" target shape {target_shape}, target dtype {target_dtype}"
        f" stored bytes length {len(tensor_bytes)}"
        f" stored dtype itemsize {np.dtype(stored_dtype).itemsize}"
        f" dim1 {dim1}, dim2 {dim2}"
        f" {e}"
    ) from e
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
  results_to_block = list()
  # logging.info("--- Starting local load for %s ---", path.name)

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


async def _load_safetensors_on_device_cns(
    path: Path, abstract_pytree: dict[str, Any]
) -> dict[str, jax.Array]:
  """High-bandwidth reader for CNS."""
  header, data_start_offset = await _read_safetensors_header(path)
  restored_tensors = {}

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
    return restored_tensors

  async with async_path.open_file(path, "rb") as f:
    await f.seek(data_start_offset + min_start)
    span_bytes = await f.read(max_end - min_start)

  # 2. Load into JAX arrays
  logging.info("Loading data into JAX arrays...")
  results_to_block = []
  for t_name, (abstract_leaf, start, end) in tensors_to_load.items():
    rel_start = start - min_start
    rel_end = end - min_start

    tensor_bytes = span_bytes[rel_start:rel_end]
    jax_array = _process_bytes_to_jax(
        tensor_bytes, t_name, abstract_leaf, header
    )
    restored_tensors[t_name] = jax_array
    results_to_block.append(jax_array)
    del tensor_bytes

  jax.block_until_ready(results_to_block)

  # 3. Destroy the large buffer immediately after the GPUs take over
  del span_bytes

  return restored_tensors


async def _gcs_bytes_to_jax_arrays(
    span_bytes: bytes | memoryview,
    tensors_to_load: dict[str, Any],
    min_start: int,
    header: dict[str, Any],
) -> dict[str, jax.Array]:
  """Converts bytes to JAX arrays based on tensor metadata."""
  restored_tensors = {}
  results_to_block = []
  for t_name, (abstract_leaf, start, end) in tensors_to_load.items():
    rel_start = start - min_start
    rel_end = end - min_start
    tensor_bytes = span_bytes[rel_start:rel_end]
    jax_array = _process_bytes_to_jax(
        tensor_bytes, t_name, abstract_leaf, header
    )
    restored_tensors[t_name] = jax_array
    results_to_block.append(jax_array)
    del tensor_bytes
  jax.block_until_ready(results_to_block)
  return restored_tensors


async def _load_safetensors_on_device_gcs(
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
    # logging.info(
    #     "Loading data into JAX arrays from gcsfuse mounted file %s...",
    #     gcs_path_str,
    # )
    print(
        "Loading data into JAX arrays from gcsfuse mounted file %s...",
        gcs_path_str,
    )
    offset = data_start_offset + min_start
    length = max_end - min_start
    with open(gcs_path_str, "rb") as f:
      start_time = time.time()
      span_bytes = os.pread(f.fileno(), length, offset)
      print(
          "gcsfuse Read time: %s seconds" % (time.time() - start_time),
          "throughput: %s MB/s"
          % (len(span_bytes) / (1024 * 1024) / (time.time() - start_time)),
      )
      start_time = time.time()
      restored_tensors = await _gcs_bytes_to_jax_arrays(
          span_bytes, tensors_to_load, min_start, header
      )
      print(
          "Conversion to JAX arrays time: %s seconds"
          % (time.time() - start_time),
          "throughput: %s MB/s"
          % (len(span_bytes) / (1024 * 1024) / (time.time() - start_time)),
      )
      del span_bytes
      return restored_tensors
  # logging.info(
  #     "Loading data into JAX arrays from GCS using transfer manager..."
  # )
  print("Loading data into JAX arrays from GCS using transfer manager...")
  client = storage.Client()
  blob = client.bucket(bucket_name).blob(blob_name)

  safe_temp_name = blob_name.replace("/", "_")
  ram_disk_path = f"/dev/shm/{safe_temp_name}_temp.bin"
  start_time = time.time()

  transfer_manager.download_chunks_concurrently(
      blob,
      ram_disk_path,
      chunk_size=512 * 1024 * 1024,
      max_workers=32,
      worker_type="process",
  )
  print(
      "Transfer manager download time: %s seconds" % (time.time() - start_time),
      "throughput: %s MB/s"
      % (blob.size / (1024 * 1024) / (time.time() - start_time)),
  )
  start_time = time.time()
  with open(ram_disk_path, "rb") as f:
    with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
      span_bytes = memoryview(mm)[
          data_start_offset + min_start : data_start_offset + max_end
      ]
      print(
          "Memory mapping time: %s seconds" % (time.time() - start_time),
          "throughput: %s MB/s"
          % (len(span_bytes) / (1024 * 1024) / (time.time() - start_time)),
      )
      start_time = time.time()
      restored_tensors = await _gcs_bytes_to_jax_arrays(
          span_bytes, tensors_to_load, min_start, header
      )
      print(
          "Conversion to JAX arrays time: %s seconds"
          % (time.time() - start_time),
          "throughput: %s MB/s"
          % (len(span_bytes) / (1024 * 1024) / (time.time() - start_time)),
      )
      del span_bytes
  # 5. Clean up the RAM disk
  os.remove(ram_disk_path)
  return restored_tensors


async def _load_safetensors_on_device(
    path: Path, abstract_pytree: dict[str, Any]
) -> dict[str, jax.Array]:
  """Intelligent Router to load Safetensors based on storage topology."""
  if str(path).startswith("/cn"+"s/"):
    return await _load_safetensors_on_device_cns(path, abstract_pytree)

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


def _load_single_batch(
    file_abstract_trees: Any,
    batch_index: int,
) -> float:
  """Loads a single batch of tensors."""

  print(f"\033[1mStarting load for batch {batch_index} at {time.time()}\033[0m")

  load_start_time = time.time()
  restored_pytree = {}
  for path, sub_tree in file_abstract_trees.items():
    # logging.info("Loading path %s of %d tensors...", path, len(sub_tree))
    print(f"Loading path {path} of {len(sub_tree)} tensors...")
    time_start = time.time()
    sub_restored = asyncio.run(_load_safetensors_on_device(path, sub_tree))
    restored_pytree.update(sub_restored)
    sub_restored_size = sum(
        leaf.nbytes
        for leaf in jax.tree_util.tree_leaves(sub_restored)
        if isinstance(leaf, jax.Array)
    )
    # logging.info(
    #     "Done loading path %s of %d tensors and size is %s MB",
    #     path, len(sub_tree), sub_restored_size / (1024 * 1024),
    # )
    print(
        f"Done loading path {path} of {len(sub_tree)} tensors and size is"
        f" {sub_restored_size / (1024 * 1024):.2f} MB Load time:"
        f" {time.time() - time_start:.2f} seconds Load throughput:"
        f" {sub_restored_size / (1024 * 1024) / (time.time() - time_start):.2f}"
        " MB/s"
    )
  # logging.info(
  #     "Batch %d load size: %.2f MB", i, total_load_size / (1024 * 1024)
  # )
  total_load_size = sum(
      leaf.nbytes
      for leaf in jax.tree_util.tree_leaves(restored_pytree)
      if isinstance(leaf, jax.Array)
  )
  print(
      f"\033[1mBatch {batch_index} load size:"
      f" {total_load_size / (1024 * 1024):.2f} MB Load time:"
      f" {time.time() - load_start_time:.2f} seconds Load throughput:"
      f" {total_load_size / (1024 * 1024) / (time.time() - load_start_time):.2f}"
      " MB/s \033[0m"
  )
  del restored_pytree
  return total_load_size / (1024 * 1024)


async def _load_safetensors(
    paths: Sequence[Path], abstract_pytree: dict[str, Any] | None = None
) -> Any:
  """Calls the correct safetensors loading function."""

  start_time = time.time()
  flat_abstract_with_path, _ = jax.tree.flatten_with_path(abstract_pytree)
  for key_path, _ in flat_abstract_with_path:
    if len(key_path) != 1 or not isinstance(key_path[0], jax.tree_util.DictKey):
      raise ValueError(
          "The PyTree is not a flat dictionary. Key path: {key_path}"
      )

  batches, _ = await create_loading_plan(paths, abstract_pytree)
  # batches, batch_sizes = await create_loading_plan_by_file(
  #     paths, abstract_pytree
  # )
  tensor_to_path = await get_tensor_to_path_indexing(paths)
  batches_to_load = []
  for batch_abstract_pytree in batches:
    file_abstract_trees = collections.defaultdict(dict)
    for tensor_name in batch_abstract_pytree:
      path = tensor_to_path[tensor_name]
      file_abstract_trees[path][tensor_name] = abstract_pytree[tensor_name]
    batches_to_load.append(file_abstract_trees)
  restored_pytree = {}
  ctx = context.get_context()
  max_workers = ctx.multiprocessing_options.primary_host
  with concurrent.futures.ThreadPoolExecutor(
      max_workers=max_workers
  ) as executor:
    futures = [
        executor.submit(_load_single_batch, file_abstract_trees, i)
        for i, file_abstract_trees in enumerate(batches_to_load)
    ]
    final_load_size = 0
    for future in concurrent.futures.as_completed(futures):
      print(f"Batch completed with result {future.result()}")
      final_load_size += future.result()

    print(
        "\033[1mFinal load size: %.2f MB" % (final_load_size),
        "Final load time: %.2f seconds" % (time.time() - start_time),
        "Final load throughput: %.2f MB/s \033[0m"
        % (final_load_size / (time.time() - start_time)),
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

  async def metadata(
      self, path: Path
  ) -> metadata_types.CheckpointMetadata[dict[str, Any]]:
    """Returns the metadata of the SafeTensors checkpoint."""
    files, _ = await _get_safetensors_file_list(path)
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
    files, abstract_pytree = await _get_safetensors_file_list(
        path, abstract_pytree
    )
    return await _load_safetensors(files, abstract_pytree)

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
