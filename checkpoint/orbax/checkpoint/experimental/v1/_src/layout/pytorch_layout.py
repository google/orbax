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

"""Defines `PyTorchLayout` for loading PyTorch checkpoint files."""

import asyncio
import dataclasses
import io
import pickle
from typing import Any, Awaitable
import zipfile

from etils import epath
import jax
import numpy as np
from orbax.checkpoint._src.path import async_path
from orbax.checkpoint.experimental.v1._src.layout import checkpoint_layout
from orbax.checkpoint.experimental.v1._src.metadata import types as metadata_types
from orbax.checkpoint.experimental.v1._src.path import types
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types


CheckpointLayout = checkpoint_layout.CheckpointLayout
InvalidLayoutError = checkpoint_layout.InvalidLayoutError
Path = types.Path


_PICKLE_FILENAME = "data.pkl"
_STORAGE_PREFIX = "data"

# Maps torch.dtype to an equivalent numpy dtype.
_TORCH_TO_NP_DTYPE = {
    "torch.float16": np.float16,
    "torch.float32": np.float32,
    "torch.float64": np.float64,
    # JAX's numpy supports bfloat16, but we use a string to avoid a direct
    # dependency on a specific numpy implementation having np.bfloat16.
    "torch.bfloat16": "bfloat16",
    "torch.uint8": np.uint8,
    "torch.int8": np.int8,
    "torch.int16": np.int16,
    "torch.int32": np.int32,
    "torch.int64": np.int64,
    "torch.bool": np.bool_,
    "torch.complex64": np.complex64,
    "torch.complex128": np.complex128,
    # Map quantized types to their numpy equivalents. Note that this loses
    # quantization information (scale and zero-point).
    "torch.qint8": np.int8,
    "torch.quint8": np.uint8,
    "torch.qint32": np.int32,
}


# The format of persistent_id for torch.Storage objects in torch pickle files:
# ('storage', torch.LongStorage, '0', 'cpu', 8)
StoragePID = tuple[
    str,  # 'storage'
    type[Any],  # StorageType
    str,  # ObjKey
    str,  # Location
    int,  # Size
]


def _parse_storage_pid(
    pid: StoragePID,
) -> tuple[Any, str]:
  """Parses a PyTorch storage persistent ID.

  Args:
    pid: The persistent id.

  Returns:
    A tuple of (storage_type, key).

  Raises:
    pickle.UnpicklingError: If the pid is not a valid storage pid.
  """
  # pid is typically a tuple like:
  # ('storage', torch.LongStorage, '0', 'cpu', 8)
  if not isinstance(pid, tuple) or pid[0] != "storage":
    raise pickle.UnpicklingError(f"Unsupported persistent id object: {pid}")
  storage_type, key = pid[1], pid[2]
  return storage_type, key


class CustomTorchUnpickler(pickle.Unpickler):
  """An unpickler that can handle PyTorch's 'storage' persistent IDs.

  by looking up data in an externally provided dictionary of bytes.
  """

  def __init__(
      self,
      file: io.BytesIO,
      storage_data: dict[str, bytes],
  ):
    super().__init__(file)
    self._storage_data = storage_data

  def persistent_load(self, pid: StoragePID) -> Any:
    """Handles persistent loads by returning a torch.Storage object for `pid`."""
    storage_type, key = _parse_storage_pid(pid)
    if key not in self._storage_data:
      raise pickle.UnpicklingError(
          f"Storage key '{key}' not found in checkpoint archive."
      )

    storage_bytes = self._storage_data[key]
    return storage_type.from_buffer(storage_bytes, "little")


@dataclasses.dataclass
class _StorageMetadata:
  """A placeholder for torch.Storage metadata, containing only the dtype."""

  dtype: str

  def __init__(self, dtype: str):
    self.dtype = dtype


@dataclasses.dataclass
class _NumpySdsMaker:
  """Placeholder for numpy array metadata."""

  shape: tuple[int, ...] | None = None
  dtype: np.dtype | None = None

  def __setstate__(self, state):
    # state: version, shape, dtype, isfortran, rawdata
    self.shape = state[1]
    self.dtype = state[2]


def _rebuild_tensor_as_sds(
    storage: Any,
    storage_offset: int,
    size: tuple[int, ...],
    stride: tuple[int, ...],
    requires_grad: bool = False,
    backward_hooks: Any = (),
) -> jax.ShapeDtypeStruct:
  """Pickle reduction function to rebuild a tensor as a ShapeDtypeStruct."""
  del storage_offset, stride, requires_grad, backward_hooks  # Unused.
  if not isinstance(storage, _StorageMetadata):
    # This error indicates that the unpickler's persistent_load did not return
    # the expected placeholder. This can happen with unsupported PyTorch
    # versions or corrupted files.
    raise pickle.UnpicklingError(
        "Expected to find _StorageMetadata, but got"
        f" {type(storage).__name__}. This may indicate an unsupported PyTorch"
        " version."
    )
  if storage.dtype not in _TORCH_TO_NP_DTYPE:
    raise pickle.UnpicklingError(
        f"Unsupported torch dtype for conversion to numpy: {storage.dtype}"
    )
  numpy_dtype = np.dtype(_TORCH_TO_NP_DTYPE[storage.dtype])
  return jax.ShapeDtypeStruct(shape=tuple(size), dtype=numpy_dtype)


def _rebuild_numpy_as_sds(
    subtype: Any,
    shape: tuple[int, ...],
    dtype: Any,
) -> _NumpySdsMaker:
  """Pickle reduction function to rebuild a numpy ndarray as a _NumpySdsMaker."""
  del subtype, shape, dtype  # Unused for metadata
  return _NumpySdsMaker()


class MetadataUnpickler(pickle.Unpickler):
  """An unpickler that reconstructs tensors as ShapeDtypeStructs."""

  def find_class(self, module: str, name: str) -> Any:
    """Overrides class lookup to intercept tensor creation."""
    if (module == "torch._utils" and name == "_rebuild_tensor_v2") or (
        module == "torch" and name == "_rebuild_tensor"
    ):
      return _rebuild_tensor_as_sds
    elif (
        module in ("numpy._core.multiarray", "numpy.core.multiarray")
        and name == "_reconstruct"
    ):
      return _rebuild_numpy_as_sds
    return super().find_class(module, name)

  def persistent_load(self, pid: StoragePID) -> _StorageMetadata:
    """Handles persistent load calls for torch.Storage."""
    storage_type, _ = _parse_storage_pid(pid)
    # For metadata, we only need the dtype from the storage type.
    return _StorageMetadata(dtype=str(storage_type.dtype))


def _unpickle_metadata(pickle_bytes: bytes) -> tree_types.PyTreeOf[Any]:
  """Unpickles metadata using MetadataUnpickler."""
  data_stream = io.BytesIO(pickle_bytes)
  unpickler = MetadataUnpickler(data_stream)
  tree = unpickler.load()

  def _replace_numpy_sds_maker(x: Any) -> Any:
    if isinstance(x, _NumpySdsMaker):
      return jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype)
    return x

  return jax.tree.map(_replace_numpy_sds_maker, tree)


def _read_zip_contents_sync(path: Path) -> tuple[bytes, dict[str, bytes]]:
  """Sync helper for `_read_zip_contents`."""
  pickle_bytes = None
  storage_data = {}
  with zipfile.ZipFile(path, "r") as zf:
    for name in zf.namelist():
      if name.endswith(_PICKLE_FILENAME):
        pickle_bytes = zf.read(name)
      else:
        p = epath.Path(name)
        if p.parent.name == _STORAGE_PREFIX:
          storage_id = p.name
          # Accommodate different key formats. Some PyTorch versions may use
          # storage keys with underscores.
          if storage_id.isdigit() or "_" in storage_id:
            storage_data[storage_id] = zf.read(name)
  if pickle_bytes is None:
    raise FileNotFoundError(f"{_PICKLE_FILENAME} not found in {path}")
  return pickle_bytes, storage_data


async def _read_zip_contents(path: Path) -> tuple[bytes, dict[str, bytes]]:
  """Reads pickle data and all storage files from a PyTorch zip archive."""
  return await asyncio.to_thread(_read_zip_contents_sync, path)


def _structure_to_numpy(
    pytorch_data: tree_types.PyTreeOf[Any],
) -> tree_types.PyTreeOf[Any]:
  """Converts torch.Tensors in pytorch_data to NumPy arrays."""

  def _to_numpy(leaf: Any) -> Any:
    if hasattr(leaf, "numpy"):
      return leaf.numpy()
    return leaf

  return jax.tree.map(_to_numpy, pytorch_data)


def _load_pytorch_on_device(
    pytorch_data: tree_types.PyTreeOf[Any],
    abstract_pytree: tree_types.PyTreeOf[jax.ShapeDtypeStruct],
) -> tree_types.PyTreeOf[jax.Array]:
  """Loads tensors from pytorch_data into on-device JAX arrays based on abstract_pytree."""

  def _load_leaf(leaf: Any, abstract_leaf: Any) -> jax.Array:
    if not hasattr(leaf, "numpy"):
      raise ValueError(
          "Item in PyTorch checkpoint is not a tensor-like object with a"
          " 'numpy' method or is missing from the checkpoint."
      )

    sharding = abstract_leaf.sharding
    target_shape = abstract_leaf.shape
    target_dtype = abstract_leaf.dtype

    # TODO(abhisekar): Optimize sharded data loading. Currently, the entire
    # global array is loaded on each host, and data is discarded if sharding
    # is used. Investigate reading only the piece of the array required by
    # each host based on its sharding.
    device_indices_map = sharding.addressable_devices_indices_map(target_shape)
    device_arrays = []
    for device in device_indices_map:
      idx = device_indices_map[device]
      shard_tensor = leaf[idx]
      shard_np = shard_tensor.numpy()
      if shard_np.dtype != target_dtype:
        shard_np = shard_np.astype(target_dtype)
      device_arrays.append(jax.device_put(shard_np, device))

    return jax.make_array_from_single_device_arrays(
        target_shape, sharding, device_arrays
    )

  return jax.tree.map(_load_leaf, pytorch_data, abstract_pytree)


def _unpickle_structure(
    pickle_bytes: bytes, storage_data: dict[str, bytes]
) -> tree_types.PyTreeOf[Any]:
  """Unpickles the structure using CustomTorchUnpickler."""
  data_stream = io.BytesIO(pickle_bytes)
  unpickler = CustomTorchUnpickler(data_stream, storage_data)
  return unpickler.load()


async def _load_pytorch(
    path: Path,
    abstract_pytree: tree_types.PyTreeOf[jax.ShapeDtypeStruct] | None = None,
) -> dict[str, Any]:
  """Loads pytorch checkpoint as numpy arrays or sharded jax arrays."""
  pickle_bytes, storage_data = await _read_zip_contents(path)

  pytorch_data = _unpickle_structure(pickle_bytes, storage_data)

  if abstract_pytree is None:
    # Return NumPy arrays.
    restored_pytree = _structure_to_numpy(pytorch_data)
  else:
    # Return on-device JAX arrays.
    restored_pytree = _load_pytorch_on_device(pytorch_data, abstract_pytree)

  return {checkpoint_layout.PYTREE_CHECKPOINTABLE_KEY: restored_pytree}


class PyTorchLayout(CheckpointLayout):
  """Layout for loading PyTorch checkpoints (.pt, .pth).

  Uses zipfile and a custom unpickler to handle torch.Tensors
  without calling torch.load().
  """

  def __init__(self, path: Path):
    self._path = path

  @property
  def path(self) -> Path:
    """Returns the path of the PyTorch checkpoint file."""
    return self._path

  def _check_zip_structure(self):
    """Sync helper to check zip file contents."""
    try:
      with zipfile.ZipFile(self._path, "r") as zf:
        if not any(name.endswith(_PICKLE_FILENAME) for name in zf.namelist()):
          raise InvalidLayoutError(
              f"'{self._path}' is not a valid PyTorch zip archive"
              " (missing data.pkl)."
          )
    except zipfile.BadZipFile as e:
      raise InvalidLayoutError(
          f"'{self._path}' is not a valid ZIP file."
      ) from e

  async def validate(self) -> None:
    """Checks if the path is a file and a valid PyTorch ZIP archive."""
    if not await async_path.is_file(self._path):
      raise InvalidLayoutError(f"Path is not a file: {self._path}")
    if self._path.suffix not in [".pt", ".pth"]:
      raise InvalidLayoutError(
          f"File {self._path} must have a .pt or .pth suffix to be loaded as a"
          " PyTorch checkpoint."
      )
    try:
      await asyncio.to_thread(self._check_zip_structure)
    except InvalidLayoutError as e:
      raise e
    except OSError as e:
      raise InvalidLayoutError(
          f"Failed to validate {self._path} as PyTorch checkpoint: {e}"
      ) from e

  async def validate_pytree(self, checkpointable_name: str | None) -> None:
    """No-op, as PyTorchLayout treats the entire file as the 'pytree' item."""
    return

  async def metadata(
      self,
  ) -> metadata_types.CheckpointMetadata[dict[str, tree_types.PyTreeOf[Any]]]:
    """Extracts ShapeDtypeStruct metadata without loading tensor data."""
    pickle_bytes, _ = await _read_zip_contents(self._path)
    metadata_tree = _unpickle_metadata(pickle_bytes)
    stat_result = await async_path.async_stat(self._path)
    commit_timestamp_nsecs = int(stat_result.mtime * 1e9)

    return metadata_types.CheckpointMetadata[dict[str, Any]](
        metadata={checkpoint_layout.PYTREE_CHECKPOINTABLE_KEY: metadata_tree},
        commit_timestamp_nsecs=commit_timestamp_nsecs,
    )

  async def load(
      self,
      abstract_checkpointables: (
          dict[str, tree_types.PyTreeOf[jax.ShapeDtypeStruct]] | None
      ) = None,
  ) -> Awaitable[dict[str, tree_types.PyTreeOf[Any]]]:
    """Loads a PyTorch checkpoint file.

    If abstract_checkpointables are provided, it attempts to load tensors as
    sharded jax.Arrays onto devices. Otherwise, it loads tensors as host
    NumPy arrays.

    Args:
      abstract_checkpointables: An optional PyTree of abstract arrays specifying
        sharding information.

    Returns:
      An awaitable of a dictionary containing the loaded PyTree.
    """
    abstract_pytree = None
    if abstract_checkpointables:
      abstract_pytree = abstract_checkpointables.get(
          checkpoint_layout.PYTREE_CHECKPOINTABLE_KEY
      )
    return _load_pytorch(self._path, abstract_pytree)
