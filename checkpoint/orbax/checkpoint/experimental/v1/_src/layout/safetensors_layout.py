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
import collections
import json
from typing import Any, Awaitable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint._src.multihost import multihost as multihost_v0
from orbax.checkpoint._src.path import async_path
from orbax.checkpoint.experimental.v1._src.layout import checkpoint_layout
from orbax.checkpoint.experimental.v1._src.metadata import types as metadata_types
from orbax.checkpoint.experimental.v1._src.path import types
from orbax.checkpoint.experimental.v1._src.synchronization import multihost

CheckpointLayout = checkpoint_layout.CheckpointLayout
InvalidLayoutError = checkpoint_layout.InvalidLayoutError
Path = types.Path

HEADER_NUM_BYTES = 8
SAFETENSORS_SUFFIX = ".safetensors"
MAX_GAP_SIZE_BYTES = (
    32 * 1024 * 1024
)  # 32 MB gap allowed between tensors in a coalesced read block


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
  }


async def _get_safetensors_file_list(path: Path) -> Sequence[Path]:
  """Returns a list of safetensors files in the given path."""
  if await async_path.is_dir(path):
    files = await async_path.glob(path, f"*{SAFETENSORS_SUFFIX}")
    return sorted(files)
  return [path]


async def _read_safetensors_header(path: Path) -> tuple[dict[str, Any], int]:
  """Reads a safetensors file header, returning the header and data start offset."""
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


def _create_non_sharded_array(
    raw_data: memoryview | bytes,
    abstract_leaf: Any,
    stored_shape: tuple[int, ...],
    stored_dtype: Any,
) -> jax.Array:
  """Creates a non-sharded JAX array from raw bytes."""
  np_array = np.frombuffer(raw_data, dtype=stored_dtype).reshape(stored_shape)
  target_dtype = abstract_leaf.dtype
  if np_array.dtype != target_dtype:
    np_array = np_array.astype(target_dtype)
  return jax.device_put(np_array)


def _create_sharded_array(
    raw_data: memoryview | bytes,
    abstract_leaf: Any,
    stored_shape: tuple[int, ...],
    stored_dtype: Any,
    num_hosts: int,
    host_id: int,
    flat_sharding: jax.sharding.NamedSharding,
) -> jax.Array:
  """Creates a sharded JAX array from raw bytes."""
  sharding = abstract_leaf.sharding
  target_dtype = abstract_leaf.dtype

  # Use 1D flat contiguous read + reshard logic for maximum IO throughput.
  total_elements = int(np.prod(stored_shape)) if stored_shape else 1

  # Calculate padding
  elements_per_host = (total_elements + num_hosts - 1) // num_hosts
  padded_elements = elements_per_host * num_hosts

  start_idx = host_id * elements_per_host
  end_idx = min((host_id + 1) * elements_per_host, total_elements)
  num_elements_to_read = max(0, end_idx - start_idx)

  local_data = np.frombuffer(raw_data, dtype=stored_dtype)
  if local_data.dtype != target_dtype:
    local_data = local_data.astype(target_dtype)

  if num_elements_to_read < elements_per_host:
    local_data = np.pad(
        local_data, (0, elements_per_host - num_elements_to_read)
    )

  # Put local data on all addressable devices in the flat sharding
  local_arrays = [
      jax.device_put(local_data, d) for d in flat_sharding.addressable_devices
  ]

  # Create the 1D sharded array
  flat_array = jax.make_array_from_single_device_arrays(
      (padded_elements,), flat_sharding, local_arrays
  )

  # Slice off the padding and reshape
  if padded_elements > total_elements:
    flat_array = flat_array[:total_elements]

  reshaped_array = flat_array.reshape(stored_shape)

  # Reshard to the target sharding
  target_array = jax.device_put(reshaped_array, sharding)

  return target_array


async def _load_non_sharded_array(
    path: Path,
    abstract_leaf: Any,
    header_info: dict[str, Any],
    data_start_offset: int,
) -> jax.Array:
  """Loads a single non-sharded array from a safetensors file."""
  stored_shape, stored_dtype = _get_array_properties(header_info)
  st_data_offsets = header_info["data_offsets"]

  start_offset, end_offset = st_data_offsets
  num_bytes = end_offset - start_offset
  async with async_path.open_file(path, mode="rb") as f:
    await f.seek(data_start_offset + start_offset)
    tensor_bytes = await f.read(num_bytes)

  return _create_non_sharded_array(
      tensor_bytes, abstract_leaf, stored_shape, stored_dtype
  )


async def _load_sharded_array(
    path: Path,
    abstract_leaf: Any,
    header_info: dict[str, Any],
    data_start_offset: int,
    num_hosts: int,
    host_id: int,
    flat_sharding: jax.sharding.NamedSharding,
) -> jax.Array:
  """Loads a single sharded array from a safetensors file."""
  stored_shape, stored_dtype = _get_array_properties(header_info)
  st_data_offsets = header_info["data_offsets"]

  total_elements = int(np.prod(stored_shape)) if stored_shape else 1
  elements_per_host = (total_elements + num_hosts - 1) // num_hosts
  start_idx = host_id * elements_per_host
  end_idx = min((host_id + 1) * elements_per_host, total_elements)
  num_elements_to_read = max(0, end_idx - start_idx)
  itemsize = np.dtype(stored_dtype).itemsize

  start_byte = st_data_offsets[0] + data_start_offset + start_idx * itemsize
  num_bytes = num_elements_to_read * itemsize

  async with async_path.open_file(path, mode="rb") as f:
    await f.seek(start_byte)
    raw_data = await f.read(num_bytes)

  return _create_sharded_array(
      raw_data,
      abstract_leaf,
      stored_shape,
      stored_dtype,
      num_hosts,
      host_id,
      flat_sharding,
  )


async def _load_safetensors_on_device(
    path: Path, abstract_pytree: dict[str, Any]
) -> dict[str, jax.Array]:
  """Loads tensors from a safetensors file into on-device JAX arrays."""
  header, data_start_offset = await _read_safetensors_header(path)
  restored_pytree = {}

  num_hosts = multihost.process_count()
  host_id = jax.process_index()

  # Build an initial mesh grouping all global devices by host
  devices_by_host = []
  for i in range(num_hosts):
    devices_by_host.append([
        d
        for d in jax.devices()
        if multihost_v0.process_index_from_device(d) == i
    ])

  # Ensure uniform mesh shape (in case of uneven device counts, which is rare)
  num_devices_per_host = len(devices_by_host[0])
  for d in devices_by_host:
    if len(d) != num_devices_per_host:
      raise ValueError("Number of devices must be the same across all hosts.")

  initial_mesh = jax.sharding.Mesh(
      np.array(devices_by_host), ("hosts", "devices")
  )
  flat_sharding = jax.sharding.NamedSharding(
      initial_mesh, jax.sharding.PartitionSpec("hosts")
  )

  async def _load_tensor(
      tensor_name: str, abstract_leaf: Any
  ) -> tuple[str, jax.Array]:
    if abstract_leaf.sharding is None:
      tensor = await _load_non_sharded_array(
          path,
          abstract_leaf,
          header[tensor_name],
          data_start_offset,
      )
    else:
      # We have a target sharding.
      tensor = await _load_sharded_array(
          path,
          abstract_leaf,
          header[tensor_name],
          data_start_offset,
          num_hosts,
          host_id,
          flat_sharding,
      )
    return tensor_name, tensor

  tasks = []
  for tensor_name, abstract_leaf in abstract_pytree.items():
    if tensor_name not in header:
      raise KeyError(
          f"Tensor '{tensor_name}' not found in safetensors header of {path}."
      )
    tasks.append(_load_tensor(tensor_name, abstract_leaf))

  results = await asyncio.gather(*tasks)
  for tensor_name, tensor in results:
    restored_pytree[tensor_name] = tensor

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
    tensor_to_path = {}
    for path in paths:
      header, _ = await _read_safetensors_header(path)
      for name in header:
        if name == "__metadata__":
          continue
        if name in tensor_to_path:
          raise ValueError(f"Duplicate tensor {name} found in multiple files.")
        tensor_to_path[name] = path

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
    files = await _get_safetensors_file_list(path)
    metadata = {}
    custom_metadata = {}

    # Track the latest commit timestamp.
    commit_timestamp_nsecs = None

    for path in files:
      header, _ = await _read_safetensors_header(path)
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
    return _load_safetensors(files, abstract_pytree)

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
