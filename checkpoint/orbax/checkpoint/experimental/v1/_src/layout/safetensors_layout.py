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

import collections
import json
from typing import Any, Awaitable, Sequence, cast

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
      "F8_E8M0": "float8_e8m0fnu (specialized ML dtype)",
      "F4": "float4_e2m1fn_x2 (specialized ML dtype)",
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

  async def _read_slice_recursively(dim: int, base_offset: int) -> bytes:
    # TODO(b/438763866) - @zachmeyers to consider alternative methods.
    s = idx[dim]  # The slice for the current dimension.

    # If we are at the last dimension, the data is contiguous.
    if dim == len(stored_shape) - 1:
      start = base_offset + s.start * global_strides[dim]
      num_bytes = (s.stop - s.start) * itemsize
      await f.seek(tensor_file_offset + start)
      return cast(bytes, await f.read(num_bytes))

    # For all other dimensions, iterate through the indices
    # of the slice and make a recursive call for the next dimension.
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


async def _load_safetensors_on_device(
    path: Path, abstract_pytree: dict[str, Any]
) -> dict[str, jax.Array]:
  """Loads tensors from a safetensors file into on-device JAX arrays."""
  header, data_start_offset = await _read_safetensors_header(path)
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
