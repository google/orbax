# Copyright 2025 The Orbax Authors.
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

import json
import os
from typing import Any, Awaitable

import aiofiles
import jax
import numpy as np
from orbax.checkpoint.experimental.v1._src.layout import checkpoint_layout
from orbax.checkpoint.experimental.v1._src.metadata import types as metadata_types
from orbax.checkpoint.experimental.v1._src.path import format_utils
from orbax.checkpoint.experimental.v1._src.path import types


CheckpointLayout = checkpoint_layout.CheckpointLayout
InvalidLayoutError = checkpoint_layout.InvalidLayoutError
Path = types.Path


def _get_dtypes() -> dict[str, Any]:
  """Returns the mapping from safetensor dtype strings to numpy dtypes."""
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
      "F8_E8M0": "float8_e8m0fnu (specialized ML dtype)",
      "F4": "float4_e2m1fn_x2 (specialized ML dtype)",
  }


async def _read_safetensors_file(path: Path) -> tuple[dict[str, Any], bytes]:
  """Reads a safetensors file, returning the header and data bytes."""
  async with aiofiles.open(path, mode="rb") as f:
    header_size_bytes = await f.read(8)
    if not header_size_bytes:
      raise ValueError("Could not read header size from safetensors file.")

    header_size = int.from_bytes(header_size_bytes, byteorder="little")
    header_bytes = await f.read(header_size)
    if len(header_bytes) != header_size:
      raise ValueError("Could not read header content from safetensors file.")

    header = json.loads(header_bytes)
    data_bytes = await f.read()
    return header, data_bytes


def _get_array_properties(info: dict[str, Any]) -> tuple[tuple[int, ...], Any]:
  """Parses shape and dtype from a safetensors tensor header."""
  try:
    dtype_str = info["dtype"]
    dtype = _get_dtypes()[dtype_str]
  except KeyError as e:
    raise ValueError(f"Unsupported dtype in SafeTensors header: {e}") from e
  shape = tuple(info["shape"])
  return shape, dtype


async def _load_safetensors(path: Path) -> dict[str, Any]:
  """Reads a safetensors file and constructs the tensor dictionary."""
  header, data_bytes = await _read_safetensors_file(path)
  tensors = {}
  for name, info in header.items():
    if name == "__metadata__":
      continue
    shape, dtype = _get_array_properties(info)
    start_offset, end_offset = info["data_offsets"]
    tensor_bytes = data_bytes[start_offset:end_offset]
    np_array = np.frombuffer(tensor_bytes, dtype=dtype).reshape(shape)
    tensors[name] = np_array
  return {format_utils.PYTREE_CHECKPOINTABLE_KEY: tensors}


class SafetensorsLayout(CheckpointLayout):
  """SafetensorsLayout.

  This class defines a class to handle Safetensors checkpoint formats. It
  inherits
  abstract methods from CheckpointLayout. It performs a few core functions:
    - Resolves handlers for saving and loading.
    - Saves and loads checkpointables to/from individual subdirectories by
    delegating to the resolved handlers.
  """

  def __init__(self, path: Path):
    self._path = path

  @property
  def path(self) -> Path:
    """Returns the path of the SafeTensors checkpoint."""
    return self._path

  async def metadata(self) -> metadata_types.CheckpointMetadata[dict[str, Any]]:
    """Returns the metadata of the SafeTensors checkpoint."""
    header, _ = await _read_safetensors_file(self._path)

    metadata = {}
    for name, info in header.items():
      if name == "__metadata__":
        continue
      shape, dtype = _get_array_properties(info)
      metadata[name] = jax.ShapeDtypeStruct(shape=shape, dtype=dtype)

    custom_metadata = header.get("__metadata__")
    commit_timestamp_nsecs = int(os.stat(self._path).st_mtime)

    return metadata_types.CheckpointMetadata[dict[str, Any]](
        metadata={format_utils.PYTREE_CHECKPOINTABLE_KEY: metadata},
        commit_timestamp_nsecs=commit_timestamp_nsecs,
        custom_metadata=custom_metadata,
    )

  def validate(self):
    if self._path.is_file() and self._path.suffix == ".safetensors":
      return
    else:
      raise InvalidLayoutError(
          f"Failed to interpret path {self._path} as a SafeTensors checkpoint."
          " A SafeTensors checkpoint must be a file with the '.safetensors'"
          " suffix."
      )

  def validate_pytree(self, checkpointable_name: str | None) -> None:
    return

  async def load(
      self,
      abstract_checkpointables: dict[str, Any] | None = None,
  ) -> Awaitable[dict[str, Any]]:
    del abstract_checkpointables
    # TODO(b/430388193) - Add support for abstract_checkpointables.
    return _load_safetensors(self._path)
