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

"""Defines `NumpyLayout` for loading NumPy checkpoint files."""

import asyncio
from typing import Any, Awaitable, IO
import zipfile

import jax
import jax.tree_util
import numpy as np
from numpy.lib import format as np_format
from orbax.checkpoint._src.path import async_path
from orbax.checkpoint.experimental.v1._src.layout import checkpoint_layout
from orbax.checkpoint.experimental.v1._src.metadata import types as metadata_types
from orbax.checkpoint.experimental.v1._src.path import types
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types


CheckpointLayout = checkpoint_layout.CheckpointLayout
InvalidLayoutError = checkpoint_layout.InvalidLayoutError
Path = types.Path


def _get_npy_info(fp: IO[bytes]) -> tuple[tuple[int, ...], np.dtype]:
  """Reads shape and dtype from npy file header."""
  try:
    version = np_format.read_magic(fp)
  except ValueError as e:
    raise ValueError('File does not start with npy magic') from e

  if version == (1, 0):
    shape, _, dtype = np_format.read_array_header_1_0(fp)
  elif version == (2, 0):
    shape, _, dtype = np_format.read_array_header_2_0(fp)
  elif version == (3, 0):
    if not hasattr(np_format, 'read_array_header_3_0'):
      raise ValueError(
          'NumPy checkpoint uses .npy version 3.0, but support for this'
          ' format requires NumPy version 1.17 or later.'
      )
    shape, _, dtype = np_format.read_array_header_3_0(fp)
  else:
    raise ValueError(f'Unsupported npy format version: {version}')
  return shape, dtype


def _reconstruct_npz_contents(npz_file: Any) -> tree_types.PyTreeOf[np.ndarray]:
  """Reconstructs nested PyTree from npz file contents."""
  result = {}
  for k in npz_file.files:
    if npz_file[k].ndim == 0 and npz_file[k].dtype == object:
      result[k] = npz_file[k].item()
    else:
      result[k] = npz_file[k]
  return result


def _load_numpy_on_device(
    numpy_pytree: tree_types.PyTreeOf[np.ndarray],
    abstract_pytree: tree_types.PyTreeOf[jax.ShapeDtypeStruct],
) -> tree_types.PyTreeOf[jax.Array]:
  """Loads arrays from numpy_pytree into on-device JAX arrays."""

  def _load_leaf(leaf: Any, abstract_leaf: jax.ShapeDtypeStruct):
    if not isinstance(leaf, np.ndarray):
      raise ValueError(f'Expected np.ndarray, got {type(leaf)}.')
    sharding = abstract_leaf.sharding
    target_shape = abstract_leaf.shape
    target_dtype = abstract_leaf.dtype

    device_indices_map = sharding.addressable_devices_indices_map(target_shape)
    device_arrays = []
    for device, idx in device_indices_map.items():
      shard_np = leaf[idx]
      if shard_np.dtype != target_dtype:
        shard_np = shard_np.astype(target_dtype)
      device_arrays.append(jax.device_put(shard_np, device))
    return jax.make_array_from_single_device_arrays(
        target_shape, sharding, device_arrays
    )

  return jax.tree.map(_load_leaf, numpy_pytree, abstract_pytree)


async def _load_numpy(
    path: Path,
    abstract_pytree: tree_types.PyTreeOf[jax.ShapeDtypeStruct] | None = None,
) -> Any:
  """Loads numpy checkpoint as numpy arrays or sharded jax arrays."""
  npz_file = await asyncio.to_thread(np.load, path, allow_pickle=True)
  try:
    numpy_pytree = _reconstruct_npz_contents(npz_file)
    if abstract_pytree is None:
      # Return NumPy arrays.
      restored_pytree = numpy_pytree
    else:
      # Return on-device JAX arrays.
      restored_pytree = _load_numpy_on_device(numpy_pytree, abstract_pytree)
  finally:
    npz_file.close()

  return restored_pytree


class NumpyLayout(CheckpointLayout):
  """Layout for loading NumPy checkpoints (.npz)."""

  def __init__(self, path: Path):
    self._path = path

  @property
  def path(self) -> Path:
    """Returns the path of the NumPy checkpoint file."""
    return self._path

  def _check_zip_structure(self):
    """Sync helper to check zip file."""
    try:
      with zipfile.ZipFile(self._path, 'r') as zf:
        if not zf.namelist():
          raise InvalidLayoutError(f"'{self._path}' is an empty zip archive.")
        if not any(name.endswith('.npy') for name in zf.namelist()):
          raise InvalidLayoutError(
              f"'{self._path}' is not a valid NumPy archive "
              '(missing .npy files).'
          )
    except zipfile.BadZipFile as e:
      raise InvalidLayoutError(
          f"'{self._path}' is not a valid ZIP file."
      ) from e
    except Exception as e:
      raise InvalidLayoutError(
          f"Failed to read '{self._path}' as zip file: {e}"
      ) from e

  async def validate(self) -> None:
    """Checks if the path is a file and a valid NumPy ZIP archive."""
    if not await async_path.is_file(self._path):
      raise InvalidLayoutError(f'Path is not a file: {self._path}')
    if self._path.suffix not in ['.npz']:
      raise InvalidLayoutError(
          f'File {self._path} must have a .npz suffix to be loaded as a'
          ' NumPy checkpoint.'
      )
    try:
      await asyncio.to_thread(self._check_zip_structure)
    except OSError as e:
      raise InvalidLayoutError(
          f'Failed to validate {self._path} as NumPy checkpoint: {e}'
      ) from e

  async def validate_pytree(self, checkpointable_name: str | None) -> None:
    """No-op, as NumpyLayout treats the entire file as the 'pytree' item."""
    return

  async def metadata(
      self,
  ) -> metadata_types.CheckpointMetadata[dict[str, tree_types.PyTreeOf[Any]]]:
    """Extracts ShapeDtypeStruct metadata without loading array data."""

    def _read_metadata_sync():
      metadata = {}
      try:
        with zipfile.ZipFile(self._path, 'r') as zf:
          for name in zf.namelist():
            if not name.endswith('.npy'):
              continue
            arr_name = name[:-4]
            with zf.open(name) as f:
              shape, dtype = _get_npy_info(f)
              metadata[arr_name] = jax.ShapeDtypeStruct(
                  shape=shape, dtype=dtype
              )
      except zipfile.BadZipFile as e:
        raise InvalidLayoutError(
            f"'{self._path}' is not a valid ZIP file."
        ) from e
      except Exception as e:
        raise InvalidLayoutError(
            f'Failed to read metadata from {self._path}'
        ) from e
      return metadata

    metadata_tree = await asyncio.to_thread(_read_metadata_sync)
    stat_result = await async_path.async_stat(self._path)
    commit_timestamp_nsecs = int(stat_result.mtime * 1e9)

    return metadata_types.CheckpointMetadata[dict[str, Any]](
        metadata={checkpoint_layout.PYTREE_CHECKPOINTABLE_KEY: metadata_tree},
        commit_timestamp_nsecs=commit_timestamp_nsecs,
    )

  async def load_pytree(
      self,
      checkpointable_name: str | None = None,
      abstract_pytree: Any | None = None,
  ) -> Awaitable[Any]:
    del checkpointable_name
    load_awaitable = _load_numpy(self._path, abstract_pytree)
    return load_awaitable

  async def load_checkpointables(
      self,
      abstract_checkpointables: (
          dict[str, tree_types.PyTreeOf[jax.ShapeDtypeStruct]] | None
      ) = None,
  ) -> Awaitable[dict[str, tree_types.PyTreeOf[Any]]]:
    """Loads a NumPy checkpoint file."""
    abstract_pytree = None
    if abstract_checkpointables:
      abstract_pytree = abstract_checkpointables.get(
          checkpoint_layout.PYTREE_CHECKPOINTABLE_KEY
      )

    async def _loader():
      restored_pytree = await _load_numpy(self._path, abstract_pytree)
      return {checkpoint_layout.PYTREE_CHECKPOINTABLE_KEY: restored_pytree}

    load_awaitable = _loader()
    return load_awaitable
