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
import json
import time
from typing import Any, Awaitable

from absl import logging
import jax
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.path import async_path
from orbax.checkpoint._src.tree import utils as tree_utils
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
      "F8_E8M0": jnp.float8_e8m0fnu,
      "F8_E4M3": jnp.float8_e4m3,
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

  async def load_single_host(self) -> dict[str, np.ndarray]:
    """Loads tensors from a safetensors file into host NumPy arrays."""
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
      tensors[name] = np_array
    return tensors


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
      load_ops.append(loader.load_single_host())

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

  async def load_safetensors(
      self, abstract_pytree: dict[str, Any] | None = None
  ) -> Any:
    """Calls the correct safetensors loading function."""
    if abstract_pytree is not None and not tree_utils.is_flat_dict(
        abstract_pytree
    ):
      raise ValueError("The PyTree is not a flat dictionary.")
    if multihost.process_count() > 1:
      raise ValueError("Multi-host loading is not supported yet.")
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
