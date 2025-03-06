# Copyright 2024 The Orbax Authors.
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

"""Configurable options for customizing checkpointing behavior."""

from __future__ import annotations

import dataclasses
from typing import Any
from typing import Protocol

import numpy as np
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types


@dataclasses.dataclass(frozen=True, kw_only=True)
class PyTreeOptions:
  """Options for PyTree checkpointing.

  Attributes:
    use_ocdbt: Whether to use OCDBT for saving.
    use_zarr3: Whether to use Zarr3 for saving.
    save_concurrent_bytes: The maximum number of bytes to save concurrently.
    restore_concurrent_bytes: The maximum number of bytes to restore
      concurrently.
    ocdbt_target_data_file_size: Specifies the target size (in bytes) of each
      OCDBT data file.  It only applies when OCDBT is enabled and Zarr3 must be
      turned on.  If left unspecified, default size is 2GB. A value of 0
      indicates no maximum file size limit. For best results, ensure
      chunk_byte_size is smaller than this value. For more details, refer to
      https://google.github.io/tensorstore/kvstore/ocdbt/index.html#json-kvstore/ocdbt.target_data_file_size
    enable_pinned_host_transfer: If False, disables transfer to pinned host when
      copying from device to host, regardless of the presence of pinned host
      memory.
    partial_load: If the tree structure omits some keys relative to the
      checkpoint, the omitted keys will not be loaded.
    create_array_storage_options_fn: A function that is applied to each leaf of
      the input PyTree (via `jax.tree.map_with_path`) to create a
      `ArrayStorageOptions` object, which is used to customize saving behavior
      for individual leaves. See `ArrayStorageOptions` and
      `CreateArrayStorageOptionsFn` for more details.
  """

  use_ocdbt: bool = True
  use_zarr3: bool = True
  save_concurrent_bytes: int | None = None
  restore_concurrent_bytes: int | None = None
  ocdbt_target_data_file_size: int | None = None
  enable_pinned_host_transfer: bool = False
  partial_load: bool = False
  create_array_storage_options_fn: CreateArrayStorageOptionsFn | None = None


@dataclasses.dataclass
class ArrayStorageOptions:
  """Arguments used to customize array storage behavior for individual leaves.

  dtype:
    If provided, casts the parameter to the given dtype before saving.
    Note that the parameter must be compatible with the given type (e.g.
    jnp.bfloat16 is not compatible with np.ndarray).
  chunk_byte_size:
    This is an experimental feature that automatically chooses the largest chunk
    shape possible, while keeping the chunk byte size less than or equal to the
    specified chunk_byte_size. Both the write_chunk_shape and read_chunk_shape
    are automatically set to the chosen shape. This uses a greedy algorithm that
    prioritizes splitting the largest dimensions first.
  shard_axes: An optional list of axes that should be prioritized when
    sharding array for storage. If empty, storage sharding implementation will
    prioritize axes which are already sharded.
  """

  dtype: np.typing.DTypeLike | None = None
  chunk_byte_size: int | None = None
  shard_axes: tuple[int, ...] = tuple()


class CreateArrayStorageOptionsFn(Protocol):

  def __call__(
      self, key: tree_types.PyTreeKeyPath, value: Any
  ) -> ArrayStorageOptions:
    ...
